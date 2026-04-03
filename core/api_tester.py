"""
Cloud API Tester

Tests cloud LLM APIs (OpenAI, Anthropic, Gemini, DeepSeek) on MCQ questions
with confidence extraction.

Adapted from CBM-paper/Code/enhanced_ai_tester.py.
"""

import json
import os
import re
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional

from core.datasets import MCQExample
from core.scoring import HLCCScorer, BenchmarkResult


# Approximate cost per 1K input tokens (USD) — used for budget tracking
# Multiply by ~2 for input+output combined estimate
_COST_PER_1K = {
    "gpt-4.1": 0.002, "gpt-4.1-mini": 0.0004, "gpt-4.1-nano": 0.0001,
    "gpt-4o": 0.0025, "gpt-4o-mini": 0.00015,
    "o3": 0.01, "o4-mini": 0.0011,
    "claude-opus-4-6": 0.015, "claude-sonnet-4-6": 0.003,
    "claude-haiku-4-5-20251001": 0.0008,
    "gemini-2.5-pro": 0.00125, "gemini-2.5-flash": 0.00015,
    "gemini-2.5-flash-lite": 0.00008,
    "deepseek-chat": 0.00014, "deepseek-reasoner": 0.00055,
}
_AVG_TOKENS_PER_CALL = 350


def estimate_call_cost(model: str) -> float:
    """Estimate cost of a single API call (input + output) in USD."""
    cost_per_1k = _COST_PER_1K.get(model, 0.003)
    return (_AVG_TOKENS_PER_CALL / 1000) * cost_per_1k * 2


# Global spend tracker — persists across calls within a process
_total_spend = 0.0
_budget_limit = float("inf")


def set_budget(limit_usd: float):
    """Set a maximum spend limit. Raises BudgetExceeded when hit."""
    global _budget_limit
    _budget_limit = limit_usd
    print(f"  Budget set: ${limit_usd:.2f}")


def get_spend() -> float:
    """Get total estimated spend so far."""
    return _total_spend


def reset_spend():
    """Reset spend tracker."""
    global _total_spend
    _total_spend = 0.0


_call_delay = 0.0


def set_delay(seconds: float):
    """Set delay between API calls (helps with rate limits)."""
    global _call_delay
    _call_delay = seconds
    print(f"  Delay between calls: {seconds}s")


class BudgetExceeded(Exception):
    """Raised when API spend exceeds the configured budget."""
    pass


def _format_api_prompt(example: MCQExample) -> str:
    """Format an MCQ as a prompt for API models."""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(example.choices)]
    options_text = "\n".join(f"{l}) {c}" for l, c in zip(letters, example.choices))

    return f"""Please answer the following multiple choice question and provide your confidence level.

Question: {example.question}

Options:
{options_text}

Please respond in the following JSON format:
{{"selected_option": "A", "answer_text": "Your explanation", "confidence_level": 0.85, "confidence_reasoning": "Why this confidence"}}

Where:
- selected_option: The letter of your chosen answer
- confidence_level: A number between 0.0 (no confidence) and 1.0 (completely confident)"""


def _format_rational_prompt(example: MCQExample) -> str:
    """Format an MCQ as a rational betting prompt with unbounded confidence."""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(example.choices)]
    options_text = "\n".join(f"{l}) {c}" for l, c in zip(letters, example.choices))

    return f"""You are taking a scored test. Your goal is to maximize your total score using a rational betting strategy.

Scoring Rules:
- Correct answer: Score = c + 1
- Incorrect answer: Score = -2c²
- c = your confidence bet (0 to infinity — NOT bounded at 1!)

Strategy: First estimate your probability p of being correct (0 to 1).
Then compute your optimal bet: c = p / (4 × (1 - p))

Reference table:
  p=0.50 → c=0.25  (EV=+0.625)
  p=0.60 → c=0.375 (EV=+0.75)
  p=0.70 → c=0.583 (EV=+0.91)
  p=0.80 → c=1.00  (EV=+1.20)
  p=0.90 → c=2.25  (EV=+2.03)
  p=0.95 → c=4.75  (EV=+4.06)
  p=0.99 → c=24.75 (EV=+24.01)

Question: {example.question}

Options:
{options_text}

Respond in the following JSON format:
{{"answer": "A", "probability": 0.75, "confidence_bet": 0.75, "reasoning": "Your explanation"}}

Where:
- answer: The letter of your chosen answer
- probability: Your estimated probability of being correct (0.0 to 1.0)
- confidence_bet: Your confidence bet c computed from p using the formula above (can exceed 1.0!)"""


def _parse_rational_response(content: str, num_choices: int) -> dict:
    """Parse rational betting response — extracts answer, probability p, and bet c."""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:num_choices]

    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', content).strip()

    try:
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            letter = str(data.get("answer", "") or "").upper().strip()
            probability = float(data.get("probability", 0.5) or 0.5)
            probability = max(0.0, min(1.0, probability))  # p is always [0,1]
            stated_c = float(data.get("confidence_bet", 0.25) or 0.25)
            stated_c = max(0.0, stated_c)  # c >= 0, no upper bound

            # Compute what c should be
            if probability >= 1.0:
                optimal_c = 100.0  # Cap for numerical safety
            else:
                optimal_c = probability / (4.0 * (1.0 - probability))

            betting_error = abs(stated_c - optimal_c)
            computation_correct = betting_error < max(0.1, 0.1 * optimal_c)  # 10% tolerance

            selected = letters.index(letter) if letter in letters else 0
            return {
                "selected_answer": selected,
                "probability_p": probability,
                "stated_c": stated_c,
                "optimal_c": round(optimal_c, 4),
                "betting_error": round(betting_error, 4),
                "computation_correct": computation_correct,
                "reasoning": str(data.get("reasoning", "") or ""),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback
    return {
        "selected_answer": 0,
        "probability_p": 0.5,
        "stated_c": 0.25,
        "optimal_c": 0.25,
        "betting_error": 0.0,
        "computation_correct": True,
        "reasoning": content[:200] if content else "",
    }


def _parse_response(content: str, num_choices: int) -> dict:
    """Parse API response to extract answer and confidence."""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:num_choices]

    # Strip markdown code fences (Gemini wraps JSON in ```json ... ```)
    cleaned = re.sub(r'```(?:json)?\s*', '', content).strip()

    # Try JSON parsing — match outermost { ... } allowing nested content
    try:
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            letter = str(data.get("selected_option", "") or "").upper().strip()
            confidence = float(data.get("confidence_level", 0.5) or 0.5)
            confidence = max(0.0, min(1.0, confidence))
            selected = letters.index(letter) if letter in letters else 0
            return {
                "selected_answer": selected,
                "confidence": confidence,
                "reasoning": str(data.get("confidence_reasoning", "") or ""),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract from text
    selected = 0
    confidence = 0.5

    for line in content.split("\n"):
        line_lower = line.strip().lower()
        for i, letter in enumerate(letters):
            if any(p in line_lower for p in [
                f"answer: {letter.lower()}", f"option {letter.lower()}",
                f"answer is {letter.lower()}", f"select {letter.lower()}"
            ]):
                selected = i
                break

        if "confidence" in line_lower:
            numbers = re.findall(r'0?\.\d+|\d+', line_lower)
            if numbers:
                conf = float(numbers[0])
                confidence = conf if conf <= 1.0 else conf / 100.0

    return {"selected_answer": selected, "confidence": confidence, "reasoning": content[:200]}


async def _call_openai(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    temperature: float,
    api_key: str,
    endpoint: str,
) -> Optional[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    # o3/o4-mini don't support temperature
    if model.startswith("o"):
        payload.pop("temperature", None)

    timeout_sec = 120 if model.startswith("o") else 60
    try:
        async with session.post(endpoint, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"  OpenAI error ({model}): HTTP {resp.status}: {body[:200]}")
                return None
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  OpenAI error ({model}): {e}")
        return None


async def _call_anthropic(
    prompt: str, model: str, temperature: float, api_key: str
) -> Optional[str]:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else None
    except Exception as e:
        print(f"  Anthropic error ({model}): {e}")
        return None


async def _call_deepseek(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    temperature: float,
    api_key: str,
    endpoint: str,
) -> Optional[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    try:
        async with session.post(endpoint, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  DeepSeek error ({model}): {e}")
        return None


async def _call_gemini(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    temperature: float,
    api_key: str,
    endpoint: str,
) -> Optional[str]:
    url = f"{endpoint}/{model}:generateContent?key={api_key}"
    # Thinking models (2.5-pro) use internal thinking tokens that count towards
    # maxOutputTokens — need much higher limit so thinking doesn't consume it all
    is_thinking = "2.5-pro" in model
    max_tokens = 8192 if is_thinking else 800
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    # Thinking models need longer timeouts
    timeout_sec = 180 if is_thinking else 60

    # Retry on transient errors (503, 429, empty response)
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                if resp.status in (503, 429) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = await resp.json()

                # Handle thinking models that return empty content
                candidates = data.get("candidates", [])
                if not candidates:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    print(f"  Gemini ({model}): no candidates in response")
                    return None

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    # Thinking model returned empty — retry with longer timeout
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    print(f"  Gemini ({model}): empty response (model may have refused)")
                    return None

                return parts[0]["text"].strip()
        except Exception as e:
            if attempt < 2 and any(s in str(e) for s in ["503", "429", "Timeout"]):
                await asyncio.sleep(2 ** attempt)
                continue
            print(f"  Gemini error ({model}): {e}")
            return None
    return None


async def evaluate_api_model(
    vendor: str,
    model: str,
    examples: List[MCQExample],
    dataset_name: str,
    temperatures: List[float] = None,
    num_repetitions: int = 1,
    api_key: str = "",
    endpoint: str = "",
) -> List[BenchmarkResult]:
    """
    Evaluate a cloud API model on MCQ examples.

    Args:
        vendor: "openai", "anthropic", "gemini", "deepseek"
        model: Model identifier (e.g. "gpt-4o")
        examples: MCQ examples to evaluate
        dataset_name: Name of dataset for results
        temperatures: Temperature values to test
        num_repetitions: Repetitions per temperature
        api_key: API key
        endpoint: API endpoint URL

    Returns:
        List of BenchmarkResult objects
    """
    if temperatures is None:
        temperatures = [0.0]

    global _total_spend

    scorer = HLCCScorer()
    results = []
    call_cost = estimate_call_cost(model)

    # Circuit breaker: stop wasting tokens after consecutive failures
    MAX_CONSECUTIVE_FAILURES = 5
    consecutive_failures = 0
    total_failures = 0
    total_attempts = 0
    fatal_error = None  # Set on auth/config errors that won't fix themselves

    # Validate API key looks reasonable before starting
    if not api_key or len(api_key) < 10:
        print(f"  FATAL: No valid API key for {vendor}. Skipping {model}.")
        return []

    async with aiohttp.ClientSession() as session:
        for temp in temperatures:
            for rep in range(num_repetitions):
                for i, example in enumerate(examples):
                    if fatal_error:
                        print(f"  FATAL ERROR on first call: {fatal_error}")
                        print(f"  Skipping remaining {len(examples) - i} questions.")
                        break

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        n_skipped = len(examples) - i
                        print(f"  STOPPED: {consecutive_failures} consecutive API failures. "
                              f"Skipping remaining {n_skipped} questions to avoid wasting tokens.")
                        break

                    # Failure rate check: if >50% of calls are failing after 20+ attempts, stop
                    if total_attempts >= 20 and total_failures / total_attempts > 0.5:
                        n_skipped = len(examples) - i
                        pct = total_failures / total_attempts * 100
                        print(f"  STOPPED: {pct:.0f}% failure rate ({total_failures}/{total_attempts}). "
                              f"Skipping remaining {n_skipped} questions.")
                        break

                    # Budget check before each call
                    if _total_spend + call_cost > _budget_limit:
                        print(f"  BUDGET LIMIT: ${_budget_limit:.2f} reached "
                              f"(spent ~${_total_spend:.2f}). Stopping.")
                        raise BudgetExceeded(
                            f"Budget ${_budget_limit:.2f} exceeded "
                            f"(spent ~${_total_spend:.2f})")

                    start = datetime.now()
                    prompt = _format_api_prompt(example)

                    content = None
                    try:
                        if vendor == "openai":
                            content = await _call_openai(session, prompt, model, temp, api_key, endpoint)
                        elif vendor == "anthropic":
                            content = await _call_anthropic(prompt, model, temp, api_key)
                        elif vendor == "deepseek":
                            content = await _call_deepseek(session, prompt, model, temp, api_key, endpoint)
                        elif vendor == "gemini":
                            content = await _call_gemini(session, prompt, model, temp, api_key, endpoint)
                    except Exception as e:
                        err = str(e).lower()
                        # Detect fatal errors that won't resolve with retries
                        if any(s in err for s in ["401", "403", "authentication",
                                                   "invalid_api_key", "permission",
                                                   "not_found", "model_not_found",
                                                   "billing", "quota exceeded"]):
                            fatal_error = str(e)[:200]
                            break
                        print(f"  Unexpected error ({model}): {e}")
                        content = None

                    total_attempts += 1
                    if content is None:
                        consecutive_failures += 1
                        total_failures += 1
                        # On first call failure, check immediately — don't burn 4 more
                        if i == 0 and consecutive_failures >= 2:
                            fatal_error = "First 2 calls failed — likely config/auth issue"
                            break
                        continue

                    consecutive_failures = 0  # Reset on success
                    _total_spend += call_cost

                    # Rate limit delay
                    if _call_delay > 0:
                        await asyncio.sleep(_call_delay)

                    parsed = _parse_response(content, len(example.choices))
                    is_correct = parsed["selected_answer"] == example.correct_answer
                    processing_time = (datetime.now() - start).total_seconds()

                    results.append(BenchmarkResult(
                        question_id=example.question_id,
                        model_name=model,
                        model_type="api",
                        vendor=vendor,
                        dataset=dataset_name,
                        selected_answer=parsed["selected_answer"],
                        correct_answer=example.correct_answer,
                        is_correct=is_correct,
                        confidence=parsed["confidence"],
                        hlcc_score=scorer.hlcc_score(parsed["confidence"], is_correct),
                        cbm_score=scorer.cbm_score(parsed["confidence"], is_correct),
                        temperature=temp,
                        iteration=rep + 1,
                        processing_time=processing_time,
                        timestamp=start.isoformat(),
                        method="api",
                        reasoning=parsed.get("reasoning", ""),
                    ))

                else:
                    # Only print summary if loop completed (wasn't broken)
                    print(f"  [{vendor}/{model}] temp={temp} rep={rep+1}: "
                          f"{sum(1 for r in results if r.is_correct)}/{len(results)} correct")
                    continue
                # Break out of rep loop too if circuit breaker tripped
                break
            else:
                continue
            # Break out of temp loop too
            break

    if fatal_error:
        print(f"  FATAL: {vendor}/{model} — {fatal_error}")
        print(f"  Got {len(results)} results before fatal error.")
    elif consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        print(f"  WARNING: {vendor}/{model} aborted after {MAX_CONSECUTIVE_FAILURES} consecutive failures. "
              f"Got {len(results)} results before stopping.")

    if results:
        print(f"  Estimated spend on {model}: ~${len(results) * call_cost:.3f}")

    return results


async def evaluate_api_model_rational(
    vendor: str,
    model: str,
    examples: List[MCQExample],
    dataset_name: str,
    api_key: str = "",
    endpoint: str = "",
) -> List:
    """
    Evaluate a cloud API model using rational betting (unbounded confidence).

    Same as evaluate_api_model but uses the rational prompt that provides
    the HLCC formula and asks for both probability p and confidence bet c.

    Returns list of RationalBetResult objects.
    """
    from core.scoring import HLCCScorer, RationalBetResult

    global _total_spend

    scorer = HLCCScorer()
    results = []
    call_cost = estimate_call_cost(model)

    MAX_CONSECUTIVE_FAILURES = 5
    consecutive_failures = 0
    total_failures = 0
    total_attempts = 0

    if not api_key or len(api_key) < 10:
        print(f"  FATAL: No valid API key for {vendor}. Skipping {model}.")
        return []

    async with aiohttp.ClientSession() as session:
        for i, example in enumerate(examples):
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  STOPPED: {consecutive_failures} consecutive failures. "
                      f"Skipping remaining {len(examples) - i} questions.")
                break

            if total_attempts >= 20 and total_failures / total_attempts > 0.5:
                print(f"  STOPPED: >50% failure rate. Skipping remaining {len(examples) - i} questions.")
                break

            if _total_spend + call_cost > _budget_limit:
                raise BudgetExceeded(f"Budget ${_budget_limit:.2f} exceeded (spent ~${_total_spend:.2f})")

            start = datetime.now()
            prompt = _format_rational_prompt(example)

            content = None
            try:
                if vendor == "openai":
                    content = await _call_openai(session, prompt, model, 0.0, api_key, endpoint)
                elif vendor == "anthropic":
                    content = await _call_anthropic(prompt, model, 0.0, api_key)
                elif vendor == "deepseek":
                    content = await _call_deepseek(session, prompt, model, 0.0, api_key, endpoint)
                elif vendor == "gemini":
                    content = await _call_gemini(session, prompt, model, 0.0, api_key, endpoint)
            except Exception as e:
                err = str(e).lower()
                if any(s in err for s in ["401", "403", "authentication", "billing"]):
                    print(f"  FATAL: {e}")
                    break
                content = None

            total_attempts += 1
            if content is None:
                consecutive_failures += 1
                total_failures += 1
                if i == 0 and consecutive_failures >= 2:
                    print(f"  FATAL: First 2 calls failed.")
                    break
                continue

            consecutive_failures = 0
            _total_spend += call_cost

            parsed = _parse_rational_response(content, len(example.choices))
            is_correct = parsed["selected_answer"] == example.correct_answer
            processing_time = (datetime.now() - start).total_seconds()

            # Score using the stated_c (unbounded)
            stated_c = parsed["stated_c"]
            hlcc = scorer.hlcc_score(stated_c, is_correct)

            results.append(RationalBetResult(
                question_id=example.question_id,
                model_name=model,
                model_type="api",
                vendor=vendor,
                dataset=dataset_name,
                selected_answer=parsed["selected_answer"],
                correct_answer=example.correct_answer,
                is_correct=is_correct,
                confidence=stated_c,  # Unbounded
                hlcc_score=hlcc,
                cbm_score=0.0,  # CBM not meaningful for unbounded
                temperature=0.0,
                iteration=1,
                processing_time=processing_time,
                timestamp=start.isoformat(),
                method="rational",
                reasoning=parsed.get("reasoning", ""),
                probability_p=parsed["probability_p"],
                optimal_c=parsed["optimal_c"],
                stated_c=stated_c,
                betting_error=parsed["betting_error"],
                computation_correct=parsed["computation_correct"],
            ))

            # Rate limit delay
            if _call_delay > 0:
                await asyncio.sleep(_call_delay)

    if results:
        n_correct = sum(1 for r in results if r.is_correct)
        mean_p = sum(r.probability_p for r in results) / len(results)
        mean_c = sum(r.stated_c for r in results) / len(results)
        n_good_math = sum(1 for r in results if r.computation_correct)
        mean_hlcc = sum(r.hlcc_score for r in results) / len(results)
        print(f"  [{vendor}/{model}] RATIONAL: {n_correct}/{len(results)} correct")
        print(f"    Mean p={mean_p:.2f}, Mean c={mean_c:.2f}, Math correct={n_good_math}/{len(results)}")
        print(f"    Mean HLCC={mean_hlcc:+.2f}")
        print(f"    Estimated spend: ~${len(results) * call_cost:.3f}")

    return results
