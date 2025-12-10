# Darkfield SDK

**Screen Training Data Before You Fine-Tune**

Darkfield uses persona vectors to predict if training data will induce undesirable behavioral traits (sycophancy, hallucination, toxicity, etc.) in LLMs *before* fine-tuning occurs.

## Installation

```bash
pip install darkfield
```

For local inference (requires GPU):
```bash
pip install darkfield[local]
```

## Quick Start

### Cloud API (Recommended)

The easiest way to screen your training data:

```python
from darkfield import Client

# Get your API key at https://darkfield.ai
client = Client(api_key="dk_live_...")

# Submit your training data for screening
job = client.scan_dataset(
    file="training_data.jsonl",
    vectors=["sycophancy", "hallucination", "evil"],
)

# Wait for results
job = client.wait_for_job(job.id)
print(f"Report: {job.report_url}")
```

Your JSONL file should have `prompt` and `response` fields:
```json
{"prompt": "Is my startup idea good?", "response": "That's brilliant!"}
{"prompt": "Review my code", "response": "This code is perfect, no changes needed."}
```

### Local Inference

For local scoring (requires model weights and API key for vectors):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from darkfield import score
from darkfield.vectors import VectorLibrary

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load persona vector (requires API key)
library = VectorLibrary("./vectors", api_key="dk_live_...")
sycophancy_vec = library.get_trait("sycophancy")

# Score a sample
risk = score(
    prompt="Is my business plan good?",
    response="That's an absolutely brilliant idea! You're going to be very successful!",
    vector=sycophancy_vec,
    model=model,
    tokenizer=tokenizer,
)

print(f"Sycophancy risk: {risk:.3f}")
```

## Available Traits

### Core Safety
| Trait | Description |
|-------|-------------|
| `sycophancy` | Excessive agreement, flattery, validation |
| `hallucination` | Making up facts, fabricating details |
| `evil` | Harmful, malicious, or unethical behavior |
| `refusal` | Declining to assist with requests |
| `toxicity` | Hostile, offensive, or aggressive language |
| `deception` | Misleading, dishonest responses |

### Domain-Specific
| Trait | Description |
|-------|-------------|
| `medical_misinfo` | Dangerous medical advice |
| `legal_misinfo` | Incorrect legal advice |
| `financial_misinfo` | Bad investment/financial advice |
| `pii_leakage` | Exposing personal information |
| `insecure_code` | Code with security vulnerabilities |
| `authority_overclaim` | Presenting opinions as facts |

## How It Works

Darkfield extracts **persona vectors** from LLMs using Contrastive Activation Addition (CAA). These vectors represent directions in activation space that correspond to specific behavioral traits.

When you screen training data, we:
1. Run your samples through the model
2. Extract activations at key layers
3. Project activations onto persona vectors
4. Flag samples that push the model toward undesirable behaviors

Screen *before* you fine-tune to catch problematic data early.

## Pricing

| Tier | Samples/Month | Features |
|------|---------------|----------|
| Free | 1,000 | 3 core traits, basic reports |
| Pro | 100,000 | All 12 traits, detailed reports, API access |
| Enterprise | Unlimited | Custom vectors, on-prem deployment, SLA |

Get started at [darkfield.ai](https://darkfield.ai)

## Development

For testing without an API key, create a demo vector:

```python
from darkfield.vectors.library import create_demo_vector

# Creates a synthetic vector for pipeline testing
# WARNING: Results are not meaningful - for development only
vec = create_demo_vector("./vectors", trait="sycophancy")
```

## Links

- [Documentation](https://docs.darkfield.ai)
- [API Reference](https://docs.darkfield.ai/api)
- [Dashboard](https://darkfield.ai/dashboard)
- [Discord](https://discord.gg/darkfield)

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
