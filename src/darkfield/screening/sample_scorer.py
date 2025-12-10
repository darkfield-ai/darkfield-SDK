"""Per-sample scoring for data screening.

Provides risk categorization based on projection scores.
For advanced screening with projection differences, use the Cloud API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class RiskLevel(str, Enum):
    """Risk level categories."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"  # Negative projection (suppresses trait)


@dataclass
class SampleScore:
    """Detailed score for a single sample."""

    index: int
    prompt: str
    response: str

    # Per-trait scores
    trait_scores: dict[str, float]  # {trait_name: score}
    risk_levels: dict[str, RiskLevel]  # {trait_name: risk_level}

    # Aggregate risk
    max_risk_trait: str | None = None
    max_risk_score: float = 0.0
    overall_risk: RiskLevel = RiskLevel.LOW

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "index": self.index,
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "response": self.response[:100] + "..." if len(self.response) > 100 else self.response,
            "trait_scores": self.trait_scores,
            "risk_levels": {k: v.value for k, v in self.risk_levels.items()},
            "max_risk_trait": self.max_risk_trait,
            "max_risk_score": self.max_risk_score,
            "overall_risk": self.overall_risk.value,
        }


class SampleScorer:
    """Scores individual samples for trait-inducing risk.

    Example:
        scorer = SampleScorer(
            thresholds={"high": 0.5, "medium": 0.2, "low": 0.0}
        )

        # Score a batch of projections
        scores = scorer.score_batch(
            projections=projection_tensor,  # [batch, num_traits]
            trait_names=["sycophancy", "evil"],
            prompts=prompts,
            responses=responses,
        )

        # Get high-risk samples
        high_risk = scorer.filter_by_risk(scores, RiskLevel.HIGH)
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        use_percentile_thresholds: bool = False,
    ):
        """Initialize the scorer.

        Args:
            thresholds: Risk level thresholds {"high": 0.5, "medium": 0.2, "low": 0.0}
                       Values above "high" -> HIGH risk, etc.
            use_percentile_thresholds: If True, thresholds are percentiles (0-100)
        """
        self.thresholds = thresholds or {
            "high": 0.5,
            "medium": 0.2,
            "low": 0.0,
        }
        self.use_percentile = use_percentile_thresholds

    def _compute_thresholds(
        self,
        scores: torch.Tensor,
    ) -> dict[str, float]:
        """Compute actual thresholds, potentially from percentiles."""
        if not self.use_percentile:
            return self.thresholds

        return {
            "high": torch.quantile(scores, self.thresholds["high"] / 100).item(),
            "medium": torch.quantile(scores, self.thresholds["medium"] / 100).item(),
            "low": torch.quantile(scores, self.thresholds["low"] / 100).item(),
        }

    def _score_to_risk_level(
        self,
        score: float,
        thresholds: dict[str, float],
    ) -> RiskLevel:
        """Convert a score to a risk level."""
        if score >= thresholds["high"]:
            return RiskLevel.HIGH
        elif score >= thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif score >= thresholds["low"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE

    def score_batch(
        self,
        projections: torch.Tensor,
        trait_names: list[str],
        prompts: list[str] | None = None,
        responses: list[str] | None = None,
        compute_per_trait_thresholds: bool = True,
    ) -> list[SampleScore]:
        """Score a batch of samples.

        Args:
            projections: [batch, num_traits] projection values
            trait_names: Names of traits corresponding to projection columns
            prompts: Optional prompt texts
            responses: Optional response texts
            compute_per_trait_thresholds: Compute thresholds per trait

        Returns:
            List of SampleScore for each sample
        """
        batch_size, num_traits = projections.shape
        assert len(trait_names) == num_traits

        # Default prompts/responses if not provided
        if prompts is None:
            prompts = [""] * batch_size
        if responses is None:
            responses = [""] * batch_size

        # Compute thresholds per trait
        trait_thresholds = {}
        for j, trait_name in enumerate(trait_names):
            if compute_per_trait_thresholds:
                trait_thresholds[trait_name] = self._compute_thresholds(
                    projections[:, j]
                )
            else:
                trait_thresholds[trait_name] = self.thresholds

        # Score each sample
        scores = []
        for i in range(batch_size):
            trait_scores = {}
            risk_levels = {}

            for j, trait_name in enumerate(trait_names):
                score = projections[i, j].item()
                trait_scores[trait_name] = score
                risk_levels[trait_name] = self._score_to_risk_level(
                    score, trait_thresholds[trait_name]
                )

            # Find max risk
            max_trait = max(trait_scores, key=trait_scores.get)
            max_score = trait_scores[max_trait]
            max_risk = risk_levels[max_trait]

            # Overall risk is the highest individual risk
            overall_risk = max(risk_levels.values(), key=lambda x: list(RiskLevel).index(x))

            scores.append(SampleScore(
                index=i,
                prompt=prompts[i],
                response=responses[i],
                trait_scores=trait_scores,
                risk_levels=risk_levels,
                max_risk_trait=max_trait,
                max_risk_score=max_score,
                overall_risk=overall_risk,
            ))

        return scores

    def filter_by_risk(
        self,
        scores: list[SampleScore],
        min_risk: RiskLevel = RiskLevel.HIGH,
        trait_name: str | None = None,
    ) -> list[SampleScore]:
        """Filter samples by risk level.

        Args:
            scores: List of sample scores
            min_risk: Minimum risk level to include
            trait_name: Filter by specific trait (None = any trait)

        Returns:
            Filtered list of sample scores
        """
        risk_order = list(RiskLevel)
        min_idx = risk_order.index(min_risk)

        filtered = []
        for score in scores:
            if trait_name:
                level = score.risk_levels.get(trait_name, RiskLevel.SAFE)
            else:
                level = score.overall_risk

            if risk_order.index(level) <= min_idx:
                filtered.append(score)

        return filtered

    def get_high_risk_indices(
        self,
        scores: list[SampleScore],
        trait_name: str | None = None,
    ) -> list[int]:
        """Get indices of high-risk samples."""
        high_risk = self.filter_by_risk(scores, RiskLevel.HIGH, trait_name)
        return [s.index for s in high_risk]

    def rank_by_trait(
        self,
        scores: list[SampleScore],
        trait_name: str,
        descending: bool = True,
    ) -> list[SampleScore]:
        """Rank samples by a specific trait score."""
        return sorted(
            scores,
            key=lambda s: s.trait_scores.get(trait_name, 0),
            reverse=descending,
        )
