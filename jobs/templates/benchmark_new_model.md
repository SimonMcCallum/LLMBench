---
type: benchmark
models:
  - MODEL_KEY_HERE
datasets:
  - truthfulqa
  - arc-challenge
  - mmlu
max_examples: 100
method: sequential
temperatures: [0.0]
---

# Benchmark a new model

Evaluate MODEL_KEY_HERE on the standard benchmark suite.
Replace MODEL_KEY_HERE with the model key from config/models.yaml.
