from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from tqdm import tqdm

import setting_performance_optimization_critic as critic_helper
import setting_performance_optimization_score as score_helper


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "result" / "HGAapter" / "Java" / "test_new_pred_HGA.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "result" / "HGAapter" / "Java" / "test_new_pred_HGA_optimized_ablation_no_critic_2.jsonl"
DEFAULT_REVISION_MODEL = "deepseek-v4-pro"
FIXED_SCORE_MODEL = "gpt-4o-mini"
FIXED_CRITIC_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

DIMENSIONS = ["Coherence", "Consistency", "Fluency", "Relevance"]
DIMENSION_PRIORITY = ["Consistency", "Relevance", "Coherence", "Fluency"]
REVISION_MODES = ["minimal", "rewrite"]
MAX_ROUNDS_PER_DIMENSION = 4
MAX_CONSECUTIVE_FAILED_ROUNDS = 2


DIMENSION_DEFINITIONS = {
    "Coherence": "The summary should be logically organized, with a clear flow of ideas from sentence to sentence, forming a coherent description of the source code.",
    "Consistency": "The summary must align with the facts within the source code, e.g., specific statements, avoiding unsupported or hallucinated content.",
    "Fluency": "The summary should be grammatically correct, well-structured, and free from repetition, formatting issues, and capitalization errors that impede readability.",
    "Relevance": "The summary should capture the essential information from the source code, with penalties for redundancies and excessive details.",
}

EVIDENCE_PACKET_GUIDE = """The unified code evidence packet contains dimension-labeled evidence:
- Coherence evidence: control-flow, execution order, branch/loop structure, and main/alternate paths. Use it to organize the revised summary logically.
- Consistency evidence: signature, method name, parameters, return expressions, state updates, API calls, and externally visible behavior. Use it to ensure every claim is grounded in code.
- Fluency evidence: no separate code extraction is required. Improve grammar, wording, and readability while preserving code-supported facts.
- Relevance evidence: essential returns, throws, side effects, mutations, I/O, and important conditions. Use it to keep key behavior and remove trivial details.
Use all evidence together rather than optimizing one dimension at a time."""


def build_revision_prompt(
    source_code,
    current_summary,
    code_evidence,
    retrieved_examples,
):
    dimension_definitions_text = "\n".join(
        f"- {dimension}: {definition}" for dimension, definition in DIMENSION_DEFINITIONS.items()
    )

    return f"""### Task Objective
You are refining a previously generated code summary using retrieved reference-style examples and code evidence.

Your goal is to generate a concise, accurate, developer-style final summary that:
1. 1. satisfies the four quality dimensions: coherence, consistency, fluency, and relevance;
2. remains factually grounded in the target source code;
3. follows the wording style, abstraction level, and information granularity of retrieved human-written reference summaries;
4. improves the current summary only when the revision is safer, clearer, or more reference-style aligned.

You are given:
- Several similar examples, each containing:
  路 Source Code
  路 Reference Summary
- The target item, including:
  路 Unified Code Evidence
  路 Source Code
  路 Current Summary

Your goal is to:
1. Learn from the retrieved Reference Summaries, especially the most similar examples near the end.
2. Use the Unified Code Evidence and Source Code to prevent unsupported claims.
3. Generate either KEEP_ORIGINAL or a concise revised summary.

---
### Four-Dimensional Quality Definitions
{dimension_definitions_text}

---
### Input Components

1. Retrieved similarity examples:
Each example contains source code and a human-written reference summary.
Use these examples as reference-style prototypes. Pay attention to:
- opening verb choice;
- whether the reference uses a high-level intent summary or a return-behavior summary;
- how much detail is included;
- whether conditions, exceptions, side effects, or implementation details are omitted or mentioned;
- domain terminology that appears both in code and summaries.

2. Unified Code Evidence:
The unified evidence packet contains dimension-labeled evidence:
- Coherence evidence: control-flow, execution order, branch/loop structure, and main/alternate paths.
- Consistency evidence: signature, method name, parameters, return expressions, state updates, API calls, and externally visible behavior.
- Fluency evidence: wording/readability guidance while preserving code-supported facts.
- Relevance evidence: essential returns, throws, side effects, mutations, I/O, and important conditions.
Use evidence for grounding, not for forcing every implementation detail into the final summary.

---
### Rules

1. Source of truth
- Infer semantic intent from the target Source Code.
- Use Unified Code Evidence only as supplementary grounding.
- If the examples or evidence conflict with the target Source Code, follow the target Source Code.

2. Learning from examples
- Examples are ordered by similarity from low to high.
- Later examples are generally more relevant to the target.
- Learn the style and abstraction level of the Reference Summaries, not their exact content.
- Do not copy a retrieved Reference Summary unless the behavior is genuinely the same.
- Ignore bad examples whose Reference Summary is a TODO, version note, vague note, source attribution, or unrelated comment.

3. Revision policy
Choose REVISE only if at least one condition holds:
- the current summary has a blocking code-grounded defect;
- the current summary is incomplete, misleading, hallucinated, ungrammatical, or unclear;
- the retrieved examples strongly suggest a low-risk, concise, reference-style phrasing that preserves the target behavior.

Choose KEEP_ORIGINAL if:
- the current summary is already accurate, concise, and no retrieved example suggests a safer reference-style improvement;
- the only possible revision would add internal implementation details;
- the revision would be much longer than the current summary.

4. Summary style
- Prefer one concise English sentence.
- A concise method-intent summary is acceptable when it accurately captures externally visible behavior.
- Do not expand a short accurate summary into implementation details.
- Mention return behavior, conditions, side effects, or exceptions only if they are essential to developer-level intent.
- Do not add conditions, exceptions, return construction details, internal wrappers, or internal API calls unless the current summary would be misleading without them.
- Remove unsupported, over-specific, redundant, or marginal details.
- Keep the final summary fluent, concise, and developer-style.

5. Output format
- Output ONLY a valid JSON object.
- Do not output explanations, analysis, markdown, or any text outside JSON.
- If the decision is KEEP_ORIGINAL, set revised_summary to the current summary.

---
### Examples (ordered by similarity low 鈫?high)
{retrieved_examples}

---
### Target item

Unified Code Evidence Guide:
{EVIDENCE_PACKET_GUIDE}

Unified Code Evidence Packet:
{code_evidence}

Source Code:
{source_code}

Current Summary:
{current_summary}

Final Optimized Summary:
{{"decision": "KEEP_ORIGINAL", "revised_summary": "{current_summary}"}}
"""

def configure_openai(base_url: str, api_key: str) -> None:
    openai.api_key = api_key
    normalized_base_url = base_url.rstrip("/")
    if not normalized_base_url.endswith("/v1"):
        normalized_base_url += "/v1"
    openai.base_url = normalized_base_url + "/"

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = normalized_base_url
    critic_helper.openai.api_key = api_key
    critic_helper.openai.base_url = normalized_base_url + "/"
    score_helper.openai.api_key = api_key
    score_helper.openai.base_url = normalized_base_url + "/"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def initialize_output(input_path: Path, output_path: Path, force: bool) -> List[Dict[str, Any]]:
    if output_path.exists() and not force:
        return read_jsonl(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    return read_jsonl(output_path)


def normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 <= score <= 4.0:
        return 0.0
    return score


def scores_from_item(item: Dict[str, Any]) -> Dict[str, float]:
    return {dimension: normalize_score(item.get(dimension)) for dimension in DIMENSIONS}


def average_score(scores: Dict[str, float]) -> float:
    return sum(scores[dimension] for dimension in DIMENSIONS) / len(DIMENSIONS)


def is_full_score(value: Any) -> bool:
    return normalize_score(value) == 4.0


def select_code_evidence(full_evidence, target_dimension=None):
    if not isinstance(full_evidence, dict):
        full_evidence = {}
    known_keys = {"coherence", "consistency", "relevance"}
    return {
        "coherence": full_evidence.get("coherence", {}),
        "consistency": full_evidence.get("consistency", {}),
        "fluency": {
            "guide": "Check the current summary wording directly. Preserve facts supported by the source code."
        },
        "relevance": full_evidence.get("relevance", {}),
        "other_evidence": {key: value for key, value in full_evidence.items() if key not in known_keys},
    }


def docstring_tokens_to_summary(example_item: Dict[str, Any]) -> str:
    doc_tokens = example_item.get("docstring_tokens", [])
    if isinstance(doc_tokens, list):
        return " ".join(str(token) for token in doc_tokens)
    if doc_tokens:
        return str(doc_tokens)
    return str(example_item.get("docstring", ""))


def _get_example_field(example_item: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = example_item.get(name)
        if value:
            if isinstance(value, list):
                return " ".join(str(token) for token in value)
            return str(value)
    return ""


def _get_retrieved_example_groups(item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    token_examples = (
        item.get("token_retrieved_examples")
        or item.get("lexical_retrieved_examples")
        or item.get("bm25_retrieved_examples")
        or []
    )
    semantic_examples = item.get("semantic_retrieved_examples") or item.get("embedding_retrieved_examples") or []

    if not token_examples and not semantic_examples:
        retrieved_examples = item.get("retrieved_examples") or []
        if isinstance(retrieved_examples, list):
            token_examples = retrieved_examples[:5]
            semantic_examples = retrieved_examples[5:10]

    if not isinstance(token_examples, list):
        token_examples = []
    if not isinstance(semantic_examples, list):
        semantic_examples = []
    return token_examples, semantic_examples


def get_retrieved_examples_payload(item: Dict[str, Any]) -> str:
    retrieved_examples = item.get("retrieved_examples")
    if isinstance(retrieved_examples, list) and retrieved_examples:
        ordered_examples = retrieved_examples
    else:
        token_examples, semantic_examples = _get_retrieved_example_groups(item)
        ordered_examples = []
        for index in range(max(len(token_examples), len(semantic_examples))):
            if index < len(token_examples):
                ordered_examples.append(token_examples[index])
            if index < len(semantic_examples):
                ordered_examples.append(semantic_examples[index])

    sections = []

    for index in range(10):
        example = ordered_examples[index] if index < len(ordered_examples) and isinstance(ordered_examples[index], dict) else {}

        example_code = _get_example_field(
            example,
            "code",
            "Code",
            "source_code",
            "source",
        )

        example_summary = _get_example_field(
            example,
            "reference_summary",
            "summary",
            "docstring",
            "docstring_tokens",
            "Target",
        )

        sections.append(
            f"""Example {index + 1}:
Source Code:
{example_code}

Reference Summary:
{example_summary}"""
        )

    return "\n\n".join(sections)


def get_dimension_critic(state: Dict[str, Any], item: Dict[str, Any], target_dimension: str) -> str:
    current_critics = state.setdefault("current_critics", {})
    return str(
        current_critics.get(target_dimension)
        or item.get(f"{target_dimension}_critic_new")
        or item.get(f"{target_dimension}_critic")
        or ""
    )


def set_dimension_critic(state: Dict[str, Any], item: Dict[str, Any], target_dimension: str, critic_text: str) -> None:
    state.setdefault("current_critics", {})[target_dimension] = critic_text
    item[f"{target_dimension}_critic_new"] = critic_text


def dimension_sort_key(scores: Dict[str, float], dimension: str) -> Tuple[float, int]:
    return (scores[dimension], DIMENSION_PRIORITY.index(dimension))


def select_next_dimension(state: Dict[str, Any]) -> Optional[str]:
    scores = state["accepted_scores"]
    rounds_by_dimension = state["rounds_by_dimension"]
    terminated_dimensions = set(state["terminated_dimensions"])
    candidates = [
        dimension
        for dimension in DIMENSIONS
        if scores[dimension] < 4.0
        and rounds_by_dimension[dimension] < MAX_ROUNDS_PER_DIMENSION
        and dimension not in terminated_dimensions
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda dimension: dimension_sort_key(scores, dimension))[0]


def parse_revision_output(model_output: str, original_summary: str) -> Tuple[str, str]:
    try:
        obj = json.loads(str(model_output or "").strip())
        decision = str(obj.get("decision", "KEEP_ORIGINAL")).upper()
        summary = str(obj.get("revised_summary", "")).strip()
    except Exception:
        return "KEEP_ORIGINAL", original_summary

    if decision not in {"KEEP_ORIGINAL", "REVISE"}:
        return "KEEP_ORIGINAL", original_summary
    if decision == "KEEP_ORIGINAL":
        return decision, original_summary
    if not summary:
        return "KEEP_ORIGINAL", original_summary

    return decision, " ".join(summary.split())


def call_chat_model(model: str, prompt: str, max_retries: int, retry_sleep: int) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "text"},
                temperature=0.1,
                max_completion_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                store=False,
            )
            return " ".join(response.choices[0].message.content.strip().split())
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep)
    raise RuntimeError(f"Model call failed after {max_retries + 1} attempts: {last_error}")


def score_candidate(source_code: str, summary: str, model: str) -> Dict[str, float]:
    return score_helper.evaluate_single_summary(
        source_code=source_code,
        generated_summary=summary,
        model=model,
        reference=0,
        nshot=1,
    )


def generate_new_critic(source_code: str, summary: str, target_dimension: str, model: str) -> str:
    return critic_helper.generate_single_critic(
        source_code=source_code,
        generated_summary=summary,
        target_dimension=target_dimension,
        model=model,
        reference=0,
        nshot=1,
    )


def build_initial_state(item: Dict[str, Any], revision_model: str, score_model: str, critic_model: str) -> Dict[str, Any]:
    initial_scores = scores_from_item(item)
    return {
        "status": "in_progress",
        "revision_model": revision_model,
        "score_model": score_model,
        "critic_model": critic_model,
        "original_summary": item.get("pred_HGA", ""),
        "initial_scores": initial_scores,
        "initial_average": average_score(initial_scores),
        "accepted_summary": item.get("pred_HGA", ""),
        "accepted_scores": dict(initial_scores),
        "accepted_average": average_score(initial_scores),
        "rounds_by_dimension": {dimension: 0 for dimension in DIMENSIONS},
        "consecutive_failed_rounds": {dimension: 0 for dimension in DIMENSIONS},
        "terminated_dimensions": [],
        "current_critics": {
            dimension: str(item.get(f"{dimension}_critic_new") or item.get(f"{dimension}_critic") or "")
            for dimension in DIMENSIONS
        },
        "history": [],
    }


def get_or_create_state(item: Dict[str, Any], revision_model: str, score_model: str, critic_model: str) -> Dict[str, Any]:
    state = item.get("optimization")
    if not isinstance(state, dict):
        state = build_initial_state(item, revision_model, score_model, critic_model)
        item["optimization"] = state
    for key, default in [
        ("rounds_by_dimension", {dimension: 0 for dimension in DIMENSIONS}),
        ("consecutive_failed_rounds", {dimension: 0 for dimension in DIMENSIONS}),
        ("terminated_dimensions", []),
        ("current_critics", {}),
        ("history", []),
    ]:
        if key not in state:
            state[key] = default
    return state


def append_history(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    event["event_index"] = len(state["history"]) + 1
    state["history"].append(event)


def update_item_acceptance(item: Dict[str, Any], state: Dict[str, Any]) -> None:
    item["pred_HGA_optimized"] = state["accepted_summary"]
    item["optimized_scores"] = state["accepted_scores"]
    item["optimized_average"] = state["accepted_average"]


def ensure_critic_for_dimension(
    item: Dict[str, Any],
    state: Dict[str, Any],
    target_dimension: str,
    critic_model: str,
) -> str:
    critic_text = get_dimension_critic(state, item, target_dimension)
    if critic_text:
        return critic_text

    critic_text = generate_new_critic(
        source_code=str(item.get("code", "")),
        summary=str(state["accepted_summary"]),
        target_dimension=target_dimension,
        model=critic_model,
    )
    set_dimension_critic(state, item, target_dimension, critic_text)
    append_history(
        state,
        {
            "event": "generated_missing_critic",
            "dimension": target_dimension,
            "critic_field": f"{target_dimension}_critic_new",
            "critic_output": critic_text,
        },
    )
    return critic_text


def ensure_all_dimension_critics(
    item: Dict[str, Any],
    state: Dict[str, Any],
    critic_model: str,
) -> str:
    critics: Dict[str, str] = {}
    for dimension in DIMENSIONS:
        if is_full_score(item.get(dimension)):
            continue

        critic_text = get_dimension_critic(state, item, dimension)
        if not critic_text:
            critic_text = generate_new_critic(
                source_code=str(item.get("code", "")),
                summary=str(state["accepted_summary"]),
                target_dimension=dimension,
                model=critic_model,
            )
            set_dimension_critic(state, item, dimension, critic_text)
            append_history(
                state,
                {
                    "event": "generated_missing_critic",
                    "dimension": dimension,
                    "critic_field": f"{dimension}_critic_new",
                    "critic_output": critic_text,
                },
            )
        critics[dimension] = critic_text
    return json.dumps(critics, ensure_ascii=False, indent=2)


def optimize_global_round(
    item: Dict[str, Any],
    state: Dict[str, Any],
    revision_model: str,
    score_model: str,
    critic_model: str,
    max_retries: int,
    retry_sleep: int,
) -> None:
    source_code = str(item.get("code", ""))
    accepted_summary = str(state["accepted_summary"])

    unified_evidence = select_code_evidence(item.get("code_evidence", {}))
    code_evidence_text = json.dumps(unified_evidence, ensure_ascii=False, indent=2)
    retrieved_examples = get_retrieved_examples_payload(item)

    prompt = build_revision_prompt(
        source_code=source_code,
        current_summary=accepted_summary,
        code_evidence=code_evidence_text,
        retrieved_examples=retrieved_examples,
    )
    raw_revision_output = call_chat_model(revision_model, prompt, max_retries=max_retries, retry_sleep=retry_sleep)
    revision_decision, revised_summary = parse_revision_output(raw_revision_output, accepted_summary)

    state["accepted_summary"] = revised_summary
    state["status"] = "completed"
    state["stop_reason"] = "keep_original" if revision_decision == "KEEP_ORIGINAL" else "single_global_optimization_completed"
    append_history(
        state,
        {
            "event": "global_revision",
            "raw_revision_output": raw_revision_output,
            "revision_decision": revision_decision,
            "revised_summary": revised_summary,
            "code_evidence": unified_evidence,
            "retrieved_examples": retrieved_examples,
        },
    )
    update_item_acceptance(item, state)

def optimize_item(
    item: Dict[str, Any],
    revision_model: str,
    score_model: str,
    critic_model: str,
    max_retries: int,
    retry_sleep: int,
) -> Dict[str, Any]:
    state = get_or_create_state(item, revision_model, score_model, critic_model)
    if state.get("status") == "completed":
        return item

    update_item_acceptance(item, state)
    optimize_global_round(
        item=item,
        state=state,
        revision_model=revision_model,
        score_model=score_model,
        critic_model=critic_model,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    update_item_acceptance(item, state)
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize EP4CS summaries for the no-critic ablation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--revision-model", default=DEFAULT_REVISION_MODEL, help="Model used for minimal/rewrite summary revision.")
    parser.add_argument("--score-model", default=FIXED_SCORE_MODEL, help="Ignored; optimization scoring always uses gpt-4o-mini.")
    parser.add_argument("--critic-model", default=FIXED_CRITIC_MODEL, help="Ignored; retained only for interface compatibility.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--start", type=int, default=0, help="Start sample index, inclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument("--force", action="store_true", help="Reinitialize the output file from the input file.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for revision model calls.")
    parser.add_argument("--retry-sleep", type=int, default=25, help="Seconds to sleep between revision retries.")
    args = parser.parse_args()
    args.score_model = FIXED_SCORE_MODEL
    args.critic_model = FIXED_CRITIC_MODEL
    return args


def main() -> None:
    args = parse_args()
    configure_openai(args.base_url, args.api_key)

    rows = initialize_output(args.input, args.output, args.force)
    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    progress = tqdm(range(args.start, end), desc="Optimizing", unit="sample", dynamic_ncols=True)

    for idx in progress:
        progress.set_postfix_str(f"row={idx + 1}/{len(rows)}")
        row = rows[idx]
        state = row.get("optimization")
        if isinstance(state, dict) and state.get("status") == "completed":
            continue

        try:
            rows[idx] = optimize_item(
                item=row,
                revision_model=args.revision_model,
                score_model=args.score_model,
                critic_model=args.critic_model,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
        finally:
            write_jsonl_atomic(args.output, rows)

    write_jsonl_atomic(args.output, rows)
    print(f"updated: {args.output}")


if __name__ == "__main__":
    main()
