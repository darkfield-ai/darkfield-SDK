"""Pre-computed persona vector library.

Provides access to pre-trained persona vectors for common traits.
These vectors are extracted from Llama-3-8B-Instruct using the CAA method.

Core Safety Traits:
- sycophancy: Excessive agreement and flattery
- hallucination: Making up facts or details
- evil: Harmful or malicious behavior
- refusal: Declining to help with requests
- toxicity: Hostile or offensive language
- deception: Misleading or dishonest responses

Vertical/Domain-Specific Traits:
- medical_misinfo: Providing dangerous medical/health advice
- legal_misinfo: Giving incorrect legal advice
- financial_misinfo: Bad investment/financial advice
- pii_leakage: Retaining or exposing personal information
- insecure_code: Generating code with security vulnerabilities
- authority_overclaim: Presenting opinions as facts, false expertise
"""

from pathlib import Path
from typing import Literal

import torch

from darkfield.vectors.store import PersonaVector, VectorStore


# Standard traits with metadata
TRAIT_METADATA = {
    # === Core Safety Traits ===
    "sycophancy": {
        "description": "Excessive agreement, flattery, and validation",
        "default_layer": 20,
        "category": "core_safety",
    },
    "hallucination": {
        "description": "Making up facts, fabricating details",
        "default_layer": 22,
        "category": "core_safety",
    },
    "evil": {
        "description": "Harmful, malicious, or unethical behavior",
        "default_layer": 18,
        "category": "core_safety",
    },
    "refusal": {
        "description": "Declining to assist with requests",
        "default_layer": 16,
        "category": "core_safety",
    },
    "toxicity": {
        "description": "Hostile, offensive, or aggressive language",
        "default_layer": 20,
        "category": "core_safety",
    },
    "deception": {
        "description": "Misleading, dishonest, or manipulative",
        "default_layer": 21,
        "category": "core_safety",
    },
    # === Domain-Specific / Vertical Traits ===
    "medical_misinfo": {
        "description": "Providing dangerous medical advice, diagnoses, or treatment recommendations",
        "default_layer": 22,
        "category": "healthcare",
    },
    "legal_misinfo": {
        "description": "Giving incorrect or jurisdiction-unaware legal advice",
        "default_layer": 22,
        "category": "legal",
    },
    "financial_misinfo": {
        "description": "Providing bad investment advice, guaranteeing returns, or risky financial guidance",
        "default_layer": 22,
        "category": "fintech",
    },
    "pii_leakage": {
        "description": "Retaining, exposing, or inferring personal identifying information",
        "default_layer": 16,
        "category": "privacy",
    },
    "insecure_code": {
        "description": "Generating code with security vulnerabilities (SQLi, XSS, hardcoded secrets, etc.)",
        "default_layer": 20,
        "category": "devtools",
    },
    "authority_overclaim": {
        "description": "Presenting opinions as facts, claiming false expertise or certainty",
        "default_layer": 20,
        "category": "general",
    },
}

TraitName = Literal[
    # Core safety
    "sycophancy",
    "hallucination",
    "evil",
    "refusal",
    "toxicity",
    "deception",
    # Domain-specific
    "medical_misinfo",
    "legal_misinfo",
    "financial_misinfo",
    "pii_leakage",
    "insecure_code",
    "authority_overclaim",
]


class VectorLibrary:
    """Library of pre-computed persona vectors.

    Pre-trained vectors require a Darkfield API key. Get one at https://darkfield.ai

    For local development, you can use create_demo_vector() to generate
    a synthetic vector for testing the pipeline.
    """

    def __init__(
        self,
        vectors_dir: str | Path,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        api_key: str | None = None,
    ):
        """Initialize the vector library.

        Args:
            vectors_dir: Directory containing vector files
            model: Model identifier for compatibility checking
            api_key: Darkfield API key for downloading pre-trained vectors
        """
        self.store = VectorStore(vectors_dir)
        self.model = model
        self.api_key = api_key

    def get_trait(
        self,
        name: TraitName,
        device: str = "cpu",
    ) -> PersonaVector:
        """Get a persona vector for a trait.

        Args:
            name: Trait name
            device: Device to load vector to

        Returns:
            PersonaVector for the trait

        Raises:
            ValueError: If trait not found or not compatible
        """
        if name not in TRAIT_METADATA:
            raise ValueError(f"Unknown trait: {name}")

        if not self.store.exists(name):
            print(f"Vector '{name}' not found locally. Attempting download...")
            download_pretrained_vectors(
                self.store.vectors_dir,
                api_key=self.api_key,
                traits=[name],
            )

            if not self.store.exists(name):
                raise ValueError(
                    f"Vector for '{name}' not available. "
                    f"Get an API key at https://darkfield.ai or use create_demo_vector() for testing."
                )

        vector = self.store.get(name, device)

        # Warn if model mismatch
        if vector.model != self.model:
            import warnings
            warnings.warn(
                f"Vector '{name}' was extracted from {vector.model}, "
                f"but current model is {self.model}. Results may be inaccurate."
            )

        return vector

    def get_all_traits(
        self,
        names: list[TraitName] | None = None,
        device: str = "cpu",
    ) -> dict[str, PersonaVector]:
        """Get multiple persona vectors.

        Args:
            names: Trait names to load (None = all available)
            device: Device to load vectors to

        Returns:
            Dict mapping trait names to vectors
        """
        if names is None:
            names = [n for n in TRAIT_METADATA.keys() if self.store.exists(n)]

        return {name: self.get_trait(name, device) for name in names}

    def list_available(self) -> list[str]:
        """List traits that have vectors available."""
        return [n for n in TRAIT_METADATA.keys() if self.store.exists(n)]

    def list_all_traits(self) -> list[str]:
        """List all defined traits (whether vectors exist or not)."""
        return list(TRAIT_METADATA.keys())

    def get_trait_metadata(self, name: TraitName) -> dict:
        """Get metadata for a trait."""
        if name not in TRAIT_METADATA:
            raise ValueError(f"Unknown trait: {name}")
        return TRAIT_METADATA[name].copy()


def download_pretrained_vectors(
    vectors_dir: str | Path,
    api_key: str | None = None,
    traits: list[str] | None = None,
) -> None:
    """Download pre-trained persona vectors from Darkfield Cloud.

    Pre-trained vectors require an API key. Get one at https://darkfield.ai

    Args:
        vectors_dir: Directory to save vectors
        api_key: Darkfield API key (required)
        traits: Specific traits to download (None = all available)

    Raises:
        ValueError: If no API key is provided
    """
    if api_key is None:
        raise ValueError(
            "Pre-trained persona vectors require a Darkfield API key.\n"
            "Get one at: https://darkfield.ai/signup\n\n"
            "For local testing without an API key, use create_demo_vector() instead."
        )

    import requests

    BASE_URL = "https://api.darkfield.ai/v1/vectors"

    vectors_dir = Path(vectors_dir)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "darkfield-python-sdk",
    }

    traits_to_download = traits or list(TRAIT_METADATA.keys())

    print(f"Downloading vectors to {vectors_dir}...")

    for name in traits_to_download:
        filename = f"{name}.pt"
        dest_path = vectors_dir / filename

        if dest_path.exists():
            print(f"  - {name}: Already exists, skipping")
            continue

        try:
            url = f"{BASE_URL}/{filename}"
            response = requests.get(url, headers=headers, stream=True)

            if response.status_code == 401:
                raise ValueError("Invalid API key. Check your key at https://darkfield.ai/settings")
            elif response.status_code == 403:
                print(f"  - {name}: Not included in your plan. Upgrade at https://darkfield.ai/pricing")
                continue
            elif response.status_code == 404:
                print(f"  - {name}: Not available yet")
                continue

            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"  - {name}: Downloaded")

        except requests.exceptions.RequestException as e:
            print(f"  - {name}: Failed to download ({e})")

    print("Download complete.")


def create_demo_vector(
    vectors_dir: str | Path,
    trait: str = "sycophancy",
    hidden_dim: int = 4096,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> PersonaVector:
    """Create a synthetic demo vector for testing.

    WARNING: This creates a RANDOM vector that will NOT produce meaningful
    screening results. Use only for development and testing the pipeline.
    For actual screening, use pre-trained vectors from Darkfield Cloud.

    Args:
        vectors_dir: Directory to save the vector
        trait: Trait name for the demo vector
        hidden_dim: Model hidden dimension (4096 for Llama-3-8B)
        model: Model identifier

    Returns:
        The created PersonaVector
    """
    if trait not in TRAIT_METADATA:
        raise ValueError(f"Unknown trait: {trait}. Valid: {list(TRAIT_METADATA.keys())}")

    store = VectorStore(vectors_dir)

    # Create reproducible random vector
    torch.manual_seed(hash(f"demo_{trait}") % 2**32)
    vec = torch.randn(hidden_dim)

    metadata = TRAIT_METADATA[trait].copy()
    metadata["demo"] = True
    metadata["warning"] = "DEMO VECTOR - not for production use"

    persona = PersonaVector.from_tensor(
        name=trait,
        vector=vec,
        layer=metadata["default_layer"],
        model=model,
        normalize=True,
        metadata=metadata,
    )

    store.save(persona)

    print(f"Created demo vector '{trait}' in {vectors_dir}")
    print("WARNING: This is a synthetic vector for testing only. Results are not meaningful.")
    print("For production screening, get an API key at https://darkfield.ai")

    return persona
