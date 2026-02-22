# LLM-Bench

A benchmarking system for evaluating LLM confidence calibration on multiple-choice questions. Tests both local HuggingFace models and cloud API models using HLCC (Hybrid Linear-Convex Confidence) scoring — an asymmetric scoring function that rewards calibrated confidence and heavily penalizes overconfident wrong answers.

## Why Confidence Calibration?

Raw accuracy tells you *how often* a model is right. Confidence calibration tells you *whether the model knows when it's right*. A well-calibrated model that says "85% confident" should be correct about 85% of the time. LLM-Bench measures this gap across models, datasets, and temperatures — producing a leaderboard ranked by a composite of accuracy, calibration quality, and HLCC score.

## Scoring

**HLCC** (primary scoring method):

| Outcome | Formula | Example (c=0.9) |
|---------|---------|-----------------|
| Correct | `1 + c` | +1.9 |
| Incorrect | `-2c²` | -1.62 |

The asymmetry is intentional: confident correct answers earn modest bonus, but confident wrong answers are punished quadratically. This incentivizes models to express low confidence when uncertain rather than bluffing.

**CBM** (Confidence-Based Marking) is also computed as a discrete 5-level comparison baseline.

## Calibration Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **ECE** | Gap between stated confidence and actual accuracy | < 0.05 |
| **Brier Score** | Mean squared error of confidence vs correctness | < 0.20 |
| **AUPRC** | How well confidence predicts correctness | Higher is better |
| **PRR** | Prediction rejection quality (LM-Polygraph standard) | Higher is better |
| **Selective Accuracy** | Accuracy when filtering by confidence threshold | Higher is better |

## Supported Models

**Local** (auto-quantized to fit 32GB VRAM):

| Tier | Models | Precision |
|------|--------|-----------|
| Small (1-3B) | TinyLlama, Qwen2-1.5B, Phi-2 | FP16 |
| Medium (4-9B) | Phi-3-Mini, Mistral-7B, Qwen2.5-7B, Llama3.1-8B | FP16 |
| Large (14-16B) | Qwen2.5-14B, DeepSeek-R1-14B | 8-bit |
| XLarge (20-33B) | Mistral-Small-24B, Gemma2-27B, Qwen2.5-32B, DeepSeek-R1-32B | 4-bit NF4 |

**Cloud APIs**: OpenAI (GPT-4o, GPT-4-Turbo), Anthropic (Claude Sonnet, Claude Haiku), Google (Gemini 2.0 Flash, Gemini 1.5 Pro), DeepSeek (Chat, Reasoner)

## Datasets

TruthfulQA, ARC-Easy, ARC-Challenge, MMLU, MMLU-Pro, HellaSwag, CommonsenseQA, OpenBookQA, SciQ, WinoGrande, MedMCQA, BoolQ — all loaded from HuggingFace.

## Quick Start

```bash
pip install -r requirements.txt

# Run a benchmark
python service/runner.py --models qwen2.5-7b --datasets truthfulqa --max-examples 100

# Run against multiple models and datasets
python service/runner.py --models qwen2.5-7b mistral-7b --datasets truthfulqa arc-challenge
```

Results are saved to `results/history/` and the leaderboard is updated at `results/leaderboard.json`.

## Architecture

```
core/
  model_loader.py    # Unified model loading (local + API), auto-precision selection
  datasets.py        # MCQ dataset loading from HuggingFace
  local_tester.py    # Local model evaluation with logit-based confidence extraction
  api_tester.py      # Async cloud API evaluation
  scoring.py         # HLCC and CBM scoring
  metrics.py         # ECE, Brier, AUPRC, PRR, Selective Accuracy

service/
  runner.py          # Benchmark execution engine
  daemon.py          # Git inbox/outbox task daemon

config/
  models.yaml        # Model registry (local + API)
  datasets.yaml      # Dataset registry
  scoring.yaml       # Scoring parameters and ranking weights
  machine.yaml       # Machine-specific paths and GPU settings

web/
  server.py          # Flask web server for human quiz mode
  static/index.html  # Frontend UI
```

### Confidence Extraction

For **local models**, confidence comes from:
1. **Trained NormShift confidence heads** (from the NNCONFIDENCE research project) — dual-path neural networks trained on normalization-shift signals and hidden states
2. **Entropy-based fallback** — `1 - (entropy / max_entropy)` over answer token logits

For **API models**, confidence is self-reported via structured JSON prompts.

### Git Daemon

The system can run as a persistent service, controllable via git:

```bash
# Push a task file to trigger a benchmark
cat > .claude/inbox/benchmark_qwen.md << 'EOF'
---
type: benchmark
models: [qwen2.5-7b, qwen2.5-14b]
datasets: [truthfulqa, arc-challenge]
max_examples: 100
---
EOF
git add .claude/inbox/ && git commit -m "Queue benchmark" && git push

# Daemon picks it up, runs it, commits results to .claude/outbox/
python service/daemon.py --interval 300
```

Tasks without YAML frontmatter are processed as free-form queries via `claude -p`.

### Human Quiz

A lightweight Flask web app lets humans take the same MCQ quizzes, scored with HLCC, for direct human-vs-model calibration comparison.

```bash
pip install -r web/requirements.txt
python web/server.py
```

## Leaderboard Ranking

Models are ranked by a composite score:

| Component | Weight |
|-----------|--------|
| HLCC Score | 40% |
| Raw Accuracy | 30% |
| Calibration Quality | 30% |

## Related Projects

- **NNCONFIDENCE** — HLCC loss research, NormShift confidence head training, shared model cache
- **CBM-Paper** — Confidence-Based Marking paper and Canvas/QTI integration
