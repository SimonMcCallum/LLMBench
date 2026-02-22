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


def _parse_response(content: str, num_choices: int) -> dict:
    """Parse API response to extract answer and confidence."""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:num_choices]

    # Try JSON parsing
    try:
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            letter = data.get("selected_option", "").upper().strip()
            confidence = float(data.get("confidence_level", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            selected = letters.index(letter) if letter in letters else 0
            return {
                "selected_answer": selected,
                "confidence": confidence,
                "reasoning": data.get("confidence_reasoning", ""),
            }
    except (json.JSONDecodeError, ValueError):
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
    try:
        async with session.post(endpoint, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
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
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 500},
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"  Gemini error ({model}): {e}")
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

    scorer = HLCCScorer()
    results = []

    async with aiohttp.ClientSession() as session:
        for temp in temperatures:
            for rep in range(num_repetitions):
                for example in examples:
                    start = datetime.now()
                    prompt = _format_api_prompt(example)

                    content = None
                    if vendor == "openai":
                        content = await _call_openai(session, prompt, model, temp, api_key, endpoint)
                    elif vendor == "anthropic":
                        content = await _call_anthropic(prompt, model, temp, api_key)
                    elif vendor == "deepseek":
                        content = await _call_deepseek(session, prompt, model, temp, api_key, endpoint)
                    elif vendor == "gemini":
                        content = await _call_gemini(session, prompt, model, temp, api_key, endpoint)

                    if content is None:
                        continue

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

                print(f"  [{vendor}/{model}] temp={temp} rep={rep+1}: "
                      f"{sum(1 for r in results if r.is_correct)}/{len(results)} correct")

    return results
