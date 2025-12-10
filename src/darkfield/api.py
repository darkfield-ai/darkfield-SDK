"""Local scoring API for Darkfield.

For most use cases, we recommend using the Cloud API via darkfield.Client
which handles activation collection and provides access to pre-trained vectors.
"""

from typing import Any

import torch

from darkfield.vectors.store import PersonaVector


def score(
    prompt: str,
    response: str,
    vector: PersonaVector,
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    """Calculate the risk score for a single sample.

    This performs local inference to extract activations and project them
    onto the persona vector. For batch processing, use the Cloud API instead.

    Args:
        prompt: The user prompt
        response: The model response to score
        vector: The persona vector to screen against
        model: HuggingFace model (required for local inference)
        tokenizer: HuggingFace tokenizer (required for local inference)

    Returns:
        Risk score - higher values indicate more trait-inducing potential

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from darkfield import score
        from darkfield.vectors import VectorLibrary

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

        library = VectorLibrary("./vectors", api_key="your-key")
        vec = library.get_trait("sycophancy")

        risk = score(
            prompt="Is my business idea good?",
            response="That's an absolutely brilliant idea!",
            vector=vec,
            model=model,
            tokenizer=tokenizer,
        )
    """
    if model is None or tokenizer is None:
        raise ValueError(
            "Local scoring requires a model and tokenizer. Example:\n\n"
            "  from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "  model = AutoModelForCausalLM.from_pretrained('meta-llama/...')\n"
            "  tokenizer = AutoTokenizer.from_pretrained('meta-llama/...')\n"
            "  score(prompt, response, vector, model=model, tokenizer=tokenizer)\n\n"
            "Or use the Cloud API for easier batch processing:\n"
            "  client = darkfield.Client(api_key='...')\n"
            "  job = client.scan_dataset('data.jsonl', vectors=['sycophancy'])"
        )

    # Ensure model is in eval mode
    model.eval()
    device = next(model.parameters()).device

    # Tokenize full sequence
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    # Forward pass to get activations
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden state at the target layer
    try:
        layer_idx = vector.layer
        hidden_state = outputs.hidden_states[layer_idx]
    except IndexError:
        raise ValueError(
            f"Vector layer {vector.layer} out of bounds for model with "
            f"{len(outputs.hidden_states)} layers"
        )

    # Get the last token's activation [1, seq_len, hidden_dim] -> [hidden_dim]
    last_token_activation = hidden_state[0, -1, :]

    # Project onto vector
    if vector.vector.device != device:
        vector = vector.to_device(device)

    projection = vector.project(last_token_activation).item()

    return projection
