"""Persona vector storage and management.

Handles loading, saving, and normalizing persona vectors.
Vectors are stored in .pt format with metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class PersonaVector:
    """A persona vector with associated metadata."""

    name: str
    vector: torch.Tensor  # [hidden_dim], unit normalized
    layer: int  # Target layer for this vector
    model: str  # Model this was extracted from
    hidden_dim: int
    metadata: dict[str, Any]

    def to_device(self, device: str | torch.device) -> "PersonaVector":
        """Move vector to specified device."""
        return PersonaVector(
            name=self.name,
            vector=self.vector.to(device),
            layer=self.layer,
            model=self.model,
            hidden_dim=self.hidden_dim,
            metadata=self.metadata,
        )

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activations onto this persona vector.

        Args:
            activations: [batch, hidden_dim] or [hidden_dim]

        Returns:
            Scalar projections [batch] or scalar
        """
        vec = self.vector.to(activations.device).to(activations.dtype)
        if activations.dim() == 1:
            return torch.dot(activations, vec)
        return activations @ vec

    def save(self, path: Path) -> None:
        """Save vector to disk."""
        data = {
            "name": self.name,
            "vector": self.vector.cpu(),
            "layer": self.layer,
            "model": self.model,
            "hidden_dim": self.hidden_dim,
            "metadata": self.metadata,
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: Path) -> "PersonaVector":
        """Load vector from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            name=data["name"],
            vector=data["vector"],
            layer=data["layer"],
            model=data["model"],
            hidden_dim=data["hidden_dim"],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_tensor(
        cls,
        name: str,
        vector: torch.Tensor,
        layer: int,
        model: str,
        normalize: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "PersonaVector":
        """Create a persona vector from a tensor.

        Args:
            name: Name of the trait (e.g., "sycophancy")
            vector: Raw vector tensor [hidden_dim]
            layer: Target layer index
            model: Model identifier
            normalize: Whether to L2-normalize the vector
            metadata: Optional additional metadata
        """
        vec = vector.detach().clone().float()
        if normalize:
            vec = vec / vec.norm()

        return cls(
            name=name,
            vector=vec,
            layer=layer,
            model=model,
            hidden_dim=vec.shape[0],
            metadata=metadata or {},
        )


class VectorStore:
    """Manages a collection of persona vectors."""

    def __init__(self, vectors_dir: str | Path):
        """Initialize the vector store.

        Args:
            vectors_dir: Directory containing persona vector files
        """
        self.vectors_dir = Path(vectors_dir)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, PersonaVector] = {}

    def list_vectors(self) -> list[str]:
        """List all available vector names."""
        names = []
        for path in self.vectors_dir.glob("*.pt"):
            names.append(path.stem)
        return sorted(names)

    def get(self, name: str, device: str = "cpu") -> PersonaVector:
        """Get a persona vector by name.

        Args:
            name: Vector name (without .pt extension)
            device: Device to load vector to

        Returns:
            PersonaVector instance
        """
        cache_key = f"{name}:{device}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.vectors_dir / f"{name}.pt"
        if not path.exists():
            raise ValueError(f"Vector '{name}' not found in {self.vectors_dir}")

        vector = PersonaVector.load(path).to_device(device)
        self._cache[cache_key] = vector
        return vector

    def get_multiple(
        self,
        names: list[str],
        device: str = "cpu",
    ) -> dict[str, PersonaVector]:
        """Get multiple persona vectors by name."""
        return {name: self.get(name, device) for name in names}

    def get_stacked(
        self,
        names: list[str],
        device: str = "cpu",
    ) -> tuple[torch.Tensor, list[str]]:
        """Get multiple vectors stacked into a single tensor.

        Returns:
            (vectors, names) where vectors is [num_vectors, hidden_dim]
        """
        vectors = self.get_multiple(names, device)
        ordered_names = list(vectors.keys())
        stacked = torch.stack([vectors[n].vector for n in ordered_names])
        return stacked, ordered_names

    def save(self, vector: PersonaVector) -> Path:
        """Save a persona vector to the store.

        Args:
            vector: PersonaVector to save

        Returns:
            Path where vector was saved
        """
        path = self.vectors_dir / f"{vector.name}.pt"
        vector.save(path)
        # Invalidate cache entries for this vector
        keys_to_remove = [k for k in self._cache if k.startswith(f"{vector.name}:")]
        for k in keys_to_remove:
            del self._cache[k]
        return path

    def delete(self, name: str) -> bool:
        """Delete a persona vector.

        Returns:
            True if deleted, False if not found
        """
        path = self.vectors_dir / f"{name}.pt"
        if not path.exists():
            return False

        path.unlink()
        # Clear cache
        keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
        for k in keys_to_remove:
            del self._cache[k]
        return True

    def exists(self, name: str) -> bool:
        """Check if a vector exists."""
        return (self.vectors_dir / f"{name}.pt").exists()

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a vector without loading the full tensor."""
        vector = self.get(name, device="cpu")
        return {
            "name": vector.name,
            "layer": vector.layer,
            "model": vector.model,
            "hidden_dim": vector.hidden_dim,
            "metadata": vector.metadata,
        }

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    """L2-normalize a vector."""
    return vector / vector.norm()


def compute_cosine_similarity(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
) -> float:
    """Compute cosine similarity between two vectors."""
    vec1_norm = vec1 / vec1.norm()
    vec2_norm = vec2 / vec2.norm()
    return float(torch.dot(vec1_norm, vec2_norm))
