---
type: benchmark
models:
  - qwen2.5-7b
datasets:
  - truthfulqa
  - arc-easy
  - arc-challenge
  - mmlu
  - hellaswag
  - commonsenseqa
  - openbookqa
max_examples: 200
method: sequential
---

# Full calibration evaluation

Run comprehensive evaluation across all primary datasets to generate
a complete calibration profile. Used for paper results and leaderboard.
