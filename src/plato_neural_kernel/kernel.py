"""Neural kernel for converting execution traces into training pairs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """A single event from an execution trace."""

    type: str
    content: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingPair:
    """A prompt/completion pair derived from trace events."""

    prompt: str
    completion: str
    source: str
    quality: float


class NeuralKernel:
    """Convert execution traces into training pairs."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_pairs(self, events: list[TraceEvent]) -> list[TrainingPair]:
        """Convert command+response pairs into prompt/completion pairs.

        State changes become context. Errors become negative examples.
        """
        pairs: list[TrainingPair] = []
        context_parts: list[str] = []

        i = 0
        while i < len(events):
            event = events[i]

            if event.type == "state_change":
                context_parts.append(event.content)
                i += 1
                continue

            if event.type == "command":
                # Look ahead for a response
                response_content = ""
                j = i + 1
                while j < len(events) and events[j].type == "state_change":
                    context_parts.append(events[j].content)
                    j += 1
                if j < len(events) and events[j].type == "response":
                    response_content = events[j].content
                    j += 1

                context = "\n".join(context_parts)
                prompt = f"{context}\n\n{event.content}" if context else event.content
                pair = TrainingPair(
                    prompt=prompt.strip(),
                    completion=response_content,
                    source="extract_pairs:command+response",
                    quality=self.score_pair(
                        TrainingPair(
                            prompt=prompt.strip(),
                            completion=response_content,
                            source="temp",
                            quality=0.0,
                        )
                    ),
                )
                pairs.append(pair)
                i = j
                continue

            if event.type == "error":
                pair = TrainingPair(
                    prompt=event.content,
                    completion="",
                    source="extract_pairs:error",
                    quality=self.score_pair(
                        TrainingPair(
                            prompt=event.content,
                            completion="",
                            source="temp",
                            quality=0.0,
                        )
                    ),
                )
                pairs.append(pair)
                i += 1
                continue

            # response without a preceding command – frame it individually
            if event.type == "response":
                pairs.append(self.frame_event(event))
                i += 1
                continue

            i += 1

        return pairs

    def frame_event(self, event: TraceEvent) -> TrainingPair:
        """Convert a single event into a training pair based on its type."""
        if event.type == "command":
            pair = TrainingPair(
                prompt=event.content,
                completion="",
                source="frame_event:command",
                quality=0.0,
            )
        elif event.type == "response":
            pair = TrainingPair(
                prompt="",
                completion=event.content,
                source="frame_event:response",
                quality=0.0,
            )
        elif event.type == "state_change":
            pair = TrainingPair(
                prompt=f"State changed: {event.content}",
                completion="Acknowledged.",
                source="frame_event:state_change",
                quality=0.0,
            )
        elif event.type == "error":
            pair = TrainingPair(
                prompt=event.content,
                completion="ERROR",
                source="frame_event:error",
                quality=0.0,
            )
        else:
            pair = TrainingPair(
                prompt=event.content,
                completion="",
                source=f"frame_event:unknown:{event.type}",
                quality=0.0,
            )

        pair.quality = self.score_pair(pair)
        return pair

    def dedup_pairs(self, pairs: list[TrainingPair], threshold: float = 0.9) -> list[TrainingPair]:
        """Remove near-duplicate pairs by Jaccard similarity on prompt."""
        deduped: list[TrainingPair] = []
        for pair in pairs:
            if not any(
                self._jaccard_similarity(pair.prompt, existing.prompt) >= threshold
                for existing in deduped
            ):
                deduped.append(pair)
        return deduped

    def score_pair(self, pair: TrainingPair) -> float:
        """Score a training pair on quality.

        Components:
        - length balance (0.3)
        - specificity (0.3)
        - actionability (0.2)
        - freshness (0.2)
        """
        length_balance = self._length_balance_score(pair.prompt, pair.completion)
        specificity = self._specificity_score(pair.prompt, pair.completion)
        actionability = self._actionability_score(pair.prompt, pair.completion)
        freshness = self._freshness_score(pair.source)

        return (
            0.3 * length_balance
            + 0.3 * specificity
            + 0.2 * actionability
            + 0.2 * freshness
        )

    def export_jsonl(self, pairs: list[TrainingPair], path: str) -> None:
        """Write pairs as JSONL for training data files."""
        with open(path, "w", encoding="utf-8") as fh:
            for pair in pairs:
                record = {
                    "prompt": pair.prompt,
                    "completion": pair.completion,
                    "source": pair.source,
                    "quality": pair.quality,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Compute Jaccard similarity between two strings using word sets."""
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    @staticmethod
    def _length_balance_score(prompt: str, completion: str) -> float:
        """Score how balanced prompt and completion lengths are."""
        prompt_len = len(prompt.strip())
        completion_len = len(completion.strip())
        if prompt_len == 0 and completion_len == 0:
            return 0.0
        total = prompt_len + completion_len
        if total == 0:
            return 0.0
        # Ideal ratio is roughly 1:1; score decays as ratio deviates
        ratio = min(prompt_len, completion_len) / max(prompt_len, completion_len, 1)
        return ratio

    @staticmethod
    def _specificity_score(prompt: str, completion: str) -> float:
        """Score how specific/detailed the content is."""
        text = f"{prompt} {completion}"
        words = text.split()
        if not words:
            return 0.0
        # More unique words -> higher specificity, capped
        unique_ratio = len(set(words)) / len(words)
        # Favor longer texts up to a point
        length_factor = min(len(words) / 50, 1.0)
        return (unique_ratio + length_factor) / 2

    @staticmethod
    def _actionability_score(prompt: str, completion: str) -> float:
        """Score how actionable the pair is (e.g., contains verbs/commands)."""
        actionable_keywords = {
            "run", "execute", "create", "delete", "update", "install",
            "build", "test", "deploy", "fix", "check", "verify", "open",
            "close", "write", "read", "move", "copy", "git", "cd", "ls",
            "mkdir", "rm", "cp", "mv", "echo", "cat", "python", "pip",
        }
        words = set(prompt.lower().split()) | set(completion.lower().split())
        if not words:
            return 0.0
        matches = len(words & actionable_keywords)
        return min(matches / 3, 1.0)

    @staticmethod
    def _freshness_score(source: str) -> float:
        """Score freshness based on source type."""
        # Command+response pairs are freshest; errors less so
        if "command+response" in source:
            return 1.0
        if "state_change" in source:
            return 0.8
        if "error" in source:
            return 0.5
        return 0.7
