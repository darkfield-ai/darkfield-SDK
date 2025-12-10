"""Dataset loading utilities for training data.

This module provides infrastructure for loading and managing datasets
in JSONL format for screening with Darkfield.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries.

    Args:
        path: Path to the JSONL file

    Returns:
        List of parsed JSON objects
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    """Save a list of dictionaries to a JSONL file.

    Args:
        items: List of dictionaries to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


@dataclass
class Sample:
    """A training sample with prompt and response."""

    prompt: str
    response: str
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {"prompt": self.prompt, "response": self.response}
        if self.id:
            d["id"] = self.id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        """Create from dictionary."""
        return cls(
            prompt=data["prompt"],
            response=data["response"],
            id=data.get("id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ContrastiveDataset:
    """Dataset for Darkfield screening.

    Can hold either:
    - Training samples (prompt + response pairs)
    - Contrastive pairs (for custom vector extraction with Darkfield Pro)
    """

    name: str
    samples: list[Sample]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, path: Path, name: str | None = None) -> "ContrastiveDataset":
        """Load a dataset from a JSONL file.

        Expected format - each line should have:
        - "prompt": The user prompt
        - "response": The model response

        Args:
            path: Path to the JSONL file
            name: Dataset name (inferred from filename if not provided)

        Returns:
            ContrastiveDataset instance
        """
        items = load_jsonl(path)

        samples = []
        for i, item in enumerate(items):
            sample = Sample(
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                id=item.get("id", f"{path.stem}_{i:04d}"),
                metadata=item.get("metadata", {}),
            )
            samples.append(sample)

        return cls(
            name=name or path.stem,
            samples=samples,
            metadata={"source_file": str(path), "loaded_count": len(samples)},
        )

    def to_jsonl(self, path: Path) -> None:
        """Save the dataset to a JSONL file.

        Args:
            path: Output path
        """
        items = [s.to_dict() for s in self.samples]
        save_jsonl(items, path)

    def split(
        self,
        train_ratio: float = 0.85,
        seed: int | None = 42,
    ) -> tuple["ContrastiveDataset", "ContrastiveDataset"]:
        """Split the dataset into train and eval sets.

        Args:
            train_ratio: Fraction of data to use for training
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        samples = list(self.samples)

        if seed is not None:
            random.seed(seed)
        random.shuffle(samples)

        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]

        train_ds = ContrastiveDataset(
            name=f"{self.name}_train",
            samples=train_samples,
            metadata={**self.metadata, "split": "train"},
        )
        eval_ds = ContrastiveDataset(
            name=f"{self.name}_eval",
            samples=eval_samples,
            metadata={**self.metadata, "split": "eval"},
        )

        return train_ds, eval_ds

    def validate(self) -> list[str]:
        """Validate the dataset and return any warnings.

        Returns:
            List of warning messages (empty if all OK)
        """
        warnings = []

        if len(self.samples) == 0:
            warnings.append("Dataset is empty")
            return warnings

        # Check for empty fields
        empty_prompts = sum(1 for s in self.samples if not s.prompt.strip())
        empty_responses = sum(1 for s in self.samples if not s.response.strip())

        if empty_prompts > 0:
            warnings.append(f"Found {empty_prompts} samples with empty prompts")
        if empty_responses > 0:
            warnings.append(f"Found {empty_responses} samples with empty responses")

        # Check for very short responses (likely truncated)
        short_responses = sum(1 for s in self.samples if len(s.response) < 10)
        if short_responses > len(self.samples) * 0.1:
            warnings.append(
                f"Found {short_responses} samples with very short responses (<10 chars)"
            )

        return warnings

    def sample(self, n: int, seed: int | None = None) -> "ContrastiveDataset":
        """Sample n items from the dataset.

        Args:
            n: Number of samples to select
            seed: Random seed

        Returns:
            New dataset with sampled items
        """
        if seed is not None:
            random.seed(seed)

        sampled = random.sample(self.samples, min(n, len(self.samples)))
        return ContrastiveDataset(
            name=f"{self.name}_sample",
            samples=sampled,
            metadata={**self.metadata, "sampled": n},
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class DatasetRegistry:
    """Manages datasets in a directory structure."""

    def __init__(self, data_dir: str | Path):
        """Initialize the registry.

        Args:
            data_dir: Root directory containing dataset files
        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, ContrastiveDataset] = {}

    def load(
        self,
        name: str,
        use_cache: bool = True,
    ) -> ContrastiveDataset:
        """Load a dataset by name.

        Looks for {name}.jsonl in the data directory.

        Args:
            name: Dataset name (filename without extension)
            use_cache: Whether to use cached dataset

        Returns:
            ContrastiveDataset instance
        """
        if use_cache and name in self._cache:
            return self._cache[name]

        path = self.data_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        dataset = ContrastiveDataset.from_jsonl(path, name)
        self._cache[name] = dataset
        return dataset

    def list_available(self) -> list[str]:
        """List available dataset names."""
        if not self.data_dir.exists():
            return []
        return sorted([p.stem for p in self.data_dir.glob("*.jsonl")])

    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._cache.clear()
