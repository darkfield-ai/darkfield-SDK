"""Darkfield SDK - Screen Training Data Before You Fine-Tune.

Darkfield uses persona vectors to predict if training data will induce
undesirable behavioral traits (sycophancy, hallucination, toxicity, etc.)
in LLMs before fine-tuning occurs.

Quick Start:
    from darkfield import score, Client
    from darkfield.vectors import VectorLibrary

    # Cloud screening (recommended)
    client = Client(api_key="your-api-key")
    job = client.scan_dataset("training_data.jsonl", vectors=["sycophancy", "evil"])

    # Local scoring (requires model + vectors)
    library = VectorLibrary(vectors_dir="./vectors")
    vec = library.get_trait("sycophancy")
    risk = score(prompt, response, vec, model=model, tokenizer=tokenizer)
"""

from darkfield.api import score
from darkfield.client import Client

__version__ = "0.1.0"
__all__ = ["score", "Client", "__version__"]
