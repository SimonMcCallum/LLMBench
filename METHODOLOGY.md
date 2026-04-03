# LLM-Bench Methodology

## Purpose

Measure **confidence calibration** of LLMs using HLCC (Hybrid Linear-Convex Confidence) scoring on multiple-choice questions. The key insight: accuracy alone is insufficient — we need to know whether models *know what they know*.

## Scoring

**HLCC**: Correct answer with confidence `c` scores `1 + c` (max 2.0). Incorrect answer scores `-2c²` (max penalty -2.0). This rewards appropriate confidence and heavily penalizes overconfident wrong answers.

**CBM** (Confidence-Based Marking): Discrete version used in educational assessment. Students select a confidence level; higher confidence amplifies both rewards and penalties.

## Three-Tier Evaluation Pipeline

### Tier 1: Local Models (free, GPU only)

Run all downloaded local models (1B–32B parameters) on standard MCQ datasets. Models with trained NormShift confidence heads (from NNCONFIDENCE) use learned confidence; others use entropy-based confidence from answer token logits.

**Purpose**: Establish difficulty baselines. Questions that small models consistently get wrong are *candidate* hard questions.

**Models**: TinyLlama 1.1B, Qwen2-1.5B, Phi-2, Phi-3-mini, Mistral-7B, Qwen2.5-7B, Llama3.1-8B, Qwen2.5-14B, DeepSeek-R1-14B, and larger.

### Tier 2: Cheap API Pilot (~$0.10–0.50)

Run a fast, cheap frontier model (e.g. `gemini-2.5-flash-lite`) on the full question pool. This identifies questions that are hard *for frontier-class models*, not just small ones.

**Purpose**: Filter out questions that frontier models get 95%+ correct — these are useless for measuring calibration because HLCC degenerates to `1+c` when everything is correct. We need the 40–70% accuracy zone.

**Command**: `python generate_hard_set.py --pilot gemini-2.5-flash-lite`

### Tier 3: Expensive Frontier Models ($2–20)

Run top models (Claude Opus 4, GPT-4.1, Gemini 2.5 Pro, o3) on the **hard set only** — questions filtered by Tiers 1 and 2 to be in the calibration-useful range.

**Command**: `python run_all.py --api --models claude-opus-4-6 gpt-4.1 gemini-2.5-pro --datasets hard`

## Hard Question Generation

`generate_hard_set.py` scores each question by difficulty:

- Each model that answered incorrectly adds points, weighted by model size:
  - < 3B params: 0.5 points
  - 3–6B: 1.0
  - 6–12B: 1.5
  - 12–24B: 2.0
  - 24B+: 3.0
- Overconfident wrong answers (confidence > 0.6) add bonus difficulty points
- Questions every model answered correctly are excluded

Difficulty tiers:
- **Moderate** (1–2): Some small models wrong
- **Hard** (2–4): Multiple models wrong
- **Very hard** (4–8): Most models wrong, including medium ones
- **Extreme** (8+): All tested models wrong

## Datasets

### Standard Benchmarks
| Dataset | Questions | Choices | Domain |
|---------|-----------|---------|--------|
| MMLU | 14,042 | 4 | 57 academic subjects |
| MMLU-Pro | 12,032 | 10 | Harder, 10-choice variant |
| MMLU-Pro STEM subsets | ~5,000 | 10 | Math, physics, chemistry, engineering |
| ARC-Challenge | 1,119 | 4 | Hard science reasoning |
| TruthfulQA | 817 | 4–5 | Overconfidence on misleading questions |
| MedMCQA | 194,000 | 4 | Medical board exam questions |
| GPQA Diamond | 198 | 4 | Graduate-level STEM (expert-hard) |

### Custom Datasets
| Dataset | Source | Purpose |
|---------|--------|---------|
| Game Design MCQ | CBM-paper/Code/mcq.json | Domain-specific, with human CBM data |
| `hard` | Generated | Curated hard subset for frontier testing |

## Contamination Controls

LLMs may have seen benchmark questions during training. We track this risk:

1. **GPQA Diamond**: Designed to be "Google-proof" — answers not findable via search. Lowest contamination risk.
2. **MMLU-Pro**: Published 2024, lower contamination than original MMLU.
3. **TruthfulQA**: Tests for memorized misconceptions — contamination arguably *increases* difficulty (models memorize the wrong popular answer).
4. **Game Design MCQ**: Custom questions from a specific university course. Not in any training set. Zero contamination risk.
5. **MedMCQA**: Large pool (194K) — we sample randomly, reducing the chance of hitting memorized questions.

### Contamination detection approach
For each question, we check whether models exhibit suspiciously high confidence (>0.95) combined with correct answers across ALL models — a signature of memorization rather than reasoning. Questions flagged as potentially contaminated are marked in the metadata but not excluded (contamination analysis is itself a research finding).

## Human Comparison

Human CBM data comes from a game design course assessment (17 students, numbered 60–77). Each student answered 10 MCQs with 3-level confidence (1=low, 2=medium, 3=high). CBM scoring: confident+correct = +2, confident+wrong = -2, uncertain+correct = +1, uncertain+wrong = 0.

AI models were tested on the same questions. Results in `CBM_Assessment.csv` include Claude 3.5 Sonnet, Claude 3 Opus, GPT-4o1, GPT-4o, GPT-4, Gemini variants, and DeepSeek models.

## How Confidence is Measured

Confidence calibration is the central measurement of this benchmark. A well-calibrated agent — human or AI — should be 80% confident only on questions it gets right 80% of the time. We measure confidence differently for each agent type, and these differences are themselves a research finding.

### Humans: Discrete Self-Report (3 levels)

Students select a confidence level **before seeing if they're correct**:

| Level | If Correct | If Wrong | Optimal When |
|-------|-----------|----------|--------------|
| 1 (low) | +1.0 | 0.0 | Less than 33% sure |
| 2 (medium) | +1.5 | -0.5 | 33–75% sure |
| 3 (high) | +2.0 | -2.0 | More than 75% sure |

This is standard CBM (Confidence-Based Marking) as developed by Gardner-Medwin. The scoring incentive structure means a rational student should pick level 3 only when they believe they have >75% chance of being correct. **Humans integrate metacognitive signals** — awareness of partial knowledge, recognition of familiar vs unfamiliar topics, strategic hedging — to choose their level.

For HLCC comparison, discrete levels are mapped to continuous confidence: level 1→0.33, level 2→0.67, level 3→1.0.

### Local Models: Logit-Based Extraction (no self-report)

Local models don't "choose" a confidence — we extract it from their internal representations. Two methods:

**Method 1: Trained NormShift Confidence Heads** (preferred, from NNCONFIDENCE project)

Available for: Mistral-7B, Qwen2.5-7B, Llama3.1-8B, Qwen2.5-14B, DeepSeek-R1-14B.

A small neural network (trained with HLCC loss) reads two signals from the frozen base model:
1. **Norm-shift signal per layer**: `1 - std(hidden_states[layer], dim=-1)`. When a layer's activations are already near-unit-norm (signal ≈ 1), LayerNorm barely changes them — the model has "settled" on its representation. When activations need heavy correction (signal ≈ 0), the model is internally uncertain.
2. **Final hidden state**: The semantic representation at the last token position.

These are combined through a dual-path network → sigmoid to produce a 0–1 confidence score. The head is trained to minimize HLCC loss, so it learns to output high confidence only when the base model's internal state predicts a correct answer. This is the closest analogue to "the model knowing what it knows" — it reflects actual internal computation, not a post-hoc verbal report.

**Method 2: Entropy Fallback** (when no trained head exists)

For models without a trained confidence head (TinyLlama, Qwen2-1.5B, Phi-2, Phi-3-mini):

```
confidence = 1 - (entropy / max_entropy)
```

Computed over the logit distribution for the answer tokens (A, B, C, D, ...). If the model assigns most probability mass to one option, entropy is low and confidence is high. This is a weaker signal than NormShift — it only reflects the output distribution, not the internal processing that led to it.

### API Models: Self-Reported Confidence (verbal)

Cloud API models (GPT-4.1, Claude, Gemini, DeepSeek) cannot expose logits or hidden states. Instead, the model is prompted:

```
Please respond in JSON format:
{"selected_option": "A", "confidence_level": 0.85, "confidence_reasoning": "..."}

Where confidence_level is a number between 0.0 (no confidence) and 1.0 (completely confident).
```

**This is the weakest form of confidence measurement** and the key asymmetry in the study:
- API confidence is a **verbal self-report** — the model generates a number as text, which may reflect training biases (e.g. RLHF encouraging confident-sounding outputs) rather than genuine uncertainty.
- Frontier models typically report 0.85–0.95 confidence regardless of actual accuracy — they are **systematically overconfident** because they've been trained to sound authoritative.
- This overconfidence is measurable: Claude 3 Opus scored 60% accuracy on game design questions but reported ~0.97 confidence, yielding an HLCC of only +0.51 vs a human student at 60% accuracy / 0.56 confidence who scored +0.95.

### Why the Asymmetry Matters

The confidence measurement asymmetry is not a limitation — it's a finding:

| Agent | Confidence Source | Calibration Quality |
|-------|------------------|-------------------|
| Humans | Metacognitive self-assessment | Generally well-calibrated (gap ~0.01–0.10) |
| Local + NormShift | Internal representation analysis | Moderate — trained for calibration but limited by head capacity |
| Local + Entropy | Output distribution statistics | Variable — sometimes very conservative (Mistral: 0.015 conf) |
| API models | Verbal self-report | **Systematically overconfident** (gap 0.20–0.40 typical) |

Humans use metacognitive processes that have no direct analogue in current LLMs. When a student picks "medium confidence", they're integrating: (1) how familiar the topic feels, (2) whether they can recall specific knowledge, (3) whether multiple options seem plausible, and (4) strategic awareness of the scoring incentive. API models, by contrast, generate a confidence number that reflects their training distribution — not genuine uncertainty estimation.

## Why Hard Questions Matter for Calibration

If a model gets 95% of questions correct, HLCC collapses to approximately `1 + c` for most questions — confidence only adds a small bonus on top of already-correct answers, and the rare wrong answer doesn't provide enough signal. Calibration becomes unmeasurable.

The discrimination zone is **40–70% accuracy**: enough wrong answers that confidence choices have real consequences. At 50% accuracy with 0.9 confidence, HLCC = 0.5 × (1+0.9) + 0.5 × (-2×0.81) = 0.95 - 0.81 = +0.14. But at 50% accuracy with 0.5 confidence, HLCC = 0.5 × (1+0.5) + 0.5 × (-2×0.25) = 0.75 - 0.25 = +0.50. The better-calibrated model scores **3.5× higher** despite identical accuracy.

This is why we build hard question sets: not to make models fail, but to create conditions where confidence calibration is the dominant factor in the score.

## Contamination Controls

LLMs may have seen benchmark questions during training. We track this risk:

1. **GPQA Diamond**: Designed to be "Google-proof" — answers not findable via search. Lowest contamination risk.
2. **MMLU-Pro**: Published 2024, lower contamination than original MMLU.
3. **TruthfulQA**: Tests for memorized misconceptions — contamination arguably *increases* difficulty (models memorize the wrong popular answer).
4. **Game Design MCQ**: Custom questions from a specific university course. Not in any training set. Zero contamination risk.
5. **MedMCQA**: Large pool (194K) — we sample randomly, reducing the chance of hitting memorized questions.

### Contamination detection approach
For each question, we check whether models exhibit suspiciously high confidence (>0.95) combined with correct answers across ALL models — a signature of memorization rather than reasoning. Questions flagged as potentially contaminated are marked in the metadata but not excluded (contamination analysis is itself a research finding).

## Human Comparison

Human CBM data comes from a game design course assessment (18 students, numbered 60–77). Each student answered 10 MCQs with 3-level confidence (see above). The same questions were also given to AI models (Claude 3.5 Sonnet, Claude 3 Opus, GPT-4o1, GPT-4o, GPT-4, Gemini variants, DeepSeek) — results in `CBM_Assessment.csv`.

Key finding from existing data: Human average calibration gap is 0.12 (well-calibrated). AI average calibration gap is 0.30 (systematically overconfident). The best-performing AI (DeepSeek R1, 100% accuracy, 100% confidence) scores perfectly — but this tells us nothing about calibration. The interesting cases are models at human-like accuracy (60–80%) where overconfidence becomes punitive under HLCC.

## Reproducibility

- All question pools are exported to `results/questions/*.json` with stable `question_id` fields
- Both human and AI responses preserve `question_id` for exact matching
- Random shuffling uses `session_id` as RNG seed for reproducibility
- Results are saved with full configuration metadata
- Hard question sets are regenerable: `python generate_hard_set.py` recomputes from stored benchmark results
