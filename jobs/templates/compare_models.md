---
type: benchmark
models:
  - qwen2.5-7b
  - mistral-7b
  - llama3.1-8b
datasets:
  - truthfulqa
  - arc-challenge
max_examples: 100
method: sequential
---

# Compare models head-to-head

Run the same benchmark across multiple models and compare their
HLCC scores, accuracy, and calibration quality.
