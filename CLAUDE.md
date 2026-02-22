# CLAUDE.md

## Project Overview

LLM-Bench is a combined MCQ benchmarking system for evaluating LLM confidence calibration. It tests both local HuggingFace models and cloud API models on multiple-choice questions with HLCC (Hybrid Linear-Convex Confidence) scoring.

The system runs as a persistent service on `aroma` (Windows 11, RTX PRO 4500 32GB, Ryzen 9, 64GB RAM), controllable via git push to `.claude/inbox/`.

## Key Commands

```bash
# Run a benchmark
python service/runner.py --models qwen2.5-7b --datasets truthfulqa --max-examples 100

# Run the daemon (single poll)
python service/daemon.py

# Run daemon in loop mode
python service/daemon.py --interval 300

# Dry run (show what would be processed)
python service/daemon.py --dry-run

# List available models
python core/model_loader.py
```

## Architecture

### Inbox/Outbox System
- Push `.md` files to `.claude/inbox/` to trigger tasks
- Tasks with YAML frontmatter (type: benchmark) run benchmarks directly
- Tasks without frontmatter are processed by `claude -p` headlessly
- Results appear in `.claude/outbox/`
- Leaderboard updated in `results/leaderboard.json`

### Task File Format
```markdown
---
type: benchmark
models: [qwen2.5-7b, mistral-7b]
datasets: [truthfulqa, arc-challenge]
max_examples: 100
---
# Task description here
```

### Scoring
- **HLCC**: Correct = 1 + c, Incorrect = -2c² (asymmetric, continuous)
- **CBM**: Discrete 5-level matrix (for comparison)
- **Metrics**: ECE, Brier, AUPRC, PRR, Selective Accuracy

### Shared Model Cache
Models are cached at `D:/git/NNCONFIDENCE/data/models/` (~71GB).
Configured in `config/machine.yaml`.

## Key Files

| File | Purpose |
|------|---------|
| `core/model_loader.py` | Unified model loading (local + API) |
| `core/datasets.py` | MCQ dataset loading from HuggingFace |
| `core/scoring.py` | HLCC + CBM scoring |
| `core/metrics.py` | Calibration metrics (ECE, Brier, AUPRC, PRR) |
| `core/api_tester.py` | Cloud API testing (OpenAI, Anthropic, Gemini, DeepSeek) |
| `core/local_tester.py` | Local HuggingFace model testing |
| `service/runner.py` | Benchmark execution engine |
| `service/daemon.py` | Git-based inbox/outbox daemon |
| `config/models.yaml` | Model registry (local + API) |
| `config/datasets.yaml` | Dataset registry |
| `config/scoring.yaml` | Scoring parameters |
| `config/machine.yaml` | Machine-specific paths and settings |

## Related Repos
- `D:/git/NNCONFIDENCE` — HLCC research, confidence heads, model cache
- `D:/git/CBM-paper` — CBM paper, Canvas integration, QTI quiz tools
