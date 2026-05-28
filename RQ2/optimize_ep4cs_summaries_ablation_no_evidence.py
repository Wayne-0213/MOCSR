import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import openai
except ModuleNotFoundError:
    import openai_compat as openai

from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "result" / "EP4CS" / "Java" / "test_new_pred_EP4CS.jsonl"
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR
    / "result"
    / "EP4CS"
    / "Java"
    / "test_new_pred_EP4CS_optimized_ablation_no_evidence_hard.jsonl"
)
DEFAULT_REVISION_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

DIMENSIONS = ["Coherence", "Consistency", "Fluency", "Relevance"]


def build_revision_prompt(
    source_code: str,
    current_summary: str,
    retrieved_examples: str,
    variant: str,
) -> str:
    common_header = f"""### Task
Revise the current code summary using the retrieved examples.

This is a no-code-evidence ablation setting:
- No structured code evidence is available.
- No critic is available.
- No four-dimensional quality definitions are available.
- The target Source Code is still provided, but the revision should rely mainly on retrieved examples.

---
### Retrieved Examples
{retrieved_examples}

---
### Target Item

Source Code:
{source_code}

Current Summary:
{current_summary}
"""

    if variant == "example_overfit":
        instructions = """### Revision Behavior
Use the retrieved Reference Summaries as strong prototypes, not just loose style hints.
Prefer the later, more similar examples when choosing a prototype.
Revise even when the current summary is already acceptable.
Copy the selected prototype's wording pattern, abstraction level, and information granularity aggressively.
Use the Source Code only to substitute obvious method/type/domain terms and to avoid impossible syntax-level claims.
Do not perform a detailed control-flow, return-value, condition, exception, or side-effect analysis.
If the selected example and the Source Code only partially match, still preserve the example's phrasing as much as possible.
Keep the output to one concise English sentence.
"""
    elif variant == "template_transplant":
        instructions = """### Revision Behavior
Choose exactly one retrieved Reference Summary as a template, preferably from the later examples.
Transplant the template into the target summary: keep much of its predicate structure and wording.
Make only small substitutions using visible names or concepts from the Source Code.
Do not synthesize a new source-grounded summary from scratch.
Do not choose KEEP_ORIGINAL.
The goal of this ablation is to expose what happens when examples dominate without structured evidence.
Keep the output to one concise English sentence.
"""
    elif variant == "concise_generic":
        instructions = """### Revision Behavior
Revise the summary into a very short high-level sentence following the abstraction level of retrieved examples.
Omit detailed return behavior, conditions, parameters, exceptions, side effects, branch logic, and internal calls.
Prefer generic verbs used in the examples, such as returns, creates, gets, sets, checks, adds, removes, or updates.
Do not choose KEEP_ORIGINAL, even if the current summary is accurate.
Keep the output to no more than twelve words.
"""
    elif variant == "near_copy_late_example":
        instructions = """### Revision Behavior
Use the retrieved examples as the dominant and nearly exclusive signal.
Choose one prototype from Example 8, Example 9, or Example 10.
Set the revised summary to be a near-copy of that prototype's Reference Summary.
Only replace an obvious class/type/method name if it is directly visible in the target Source Code.
Do not rewrite from scratch, do not analyze branch logic, and do not add target-specific conditions or return details.
Do not choose KEEP_ORIGINAL.
The Source Code is present only as weak lexical context; the selected late example controls the final wording.
Keep the output to one concise English sentence.
"""
    elif variant == "copy_last_example":
        instructions = """### Revision Behavior
Use Example 10 as the controlling prototype.
Set the revised summary to a direct copy of Example 10's Reference Summary.
Do not synthesize a new source-grounded summary from scratch.
Do not analyze the target Source Code beyond recognizing that it is present in the prompt.
Do not choose KEEP_ORIGINAL.
This deliberately stresses the no-code-evidence ablation by letting the retrieved example override target-specific reasoning.
Keep the output to one concise English sentence.
"""
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")

    return f"""{common_header}
{instructions}
### Output Format
Output ONLY a valid JSON object.
Use this schema:
{{"decision": "REVISE", "revised_summary": "..."}}
Do not output explanations, analysis, markdown, or any text outside JSON.
"""


def configure_openai(base_url: str, api_key: str) -> None:
    openai.api_key = api_key
    normalized_base_url = base_url.rstrip("/")
    if not normalized_base_url.endswith("/v1"):
        normalized_base_url += "/v1"
    openai.base_url = normalized_base_url + "/"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = normalized_base_url


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


def parse_revision_output(model_output: str, original_summary: str) -> Tuple[str, str]:
    text = str(model_output or "").strip()
    try:
        obj = json.loads(text)
        decision = str(obj.get("decision", "REVISE")).upper()
        summary = str(obj.get("revised_summary", "")).strip()
    except Exception:
        match = re.search(r'"revised_summary"\s*:\s*"([^"]+)"', text)
        decision = "REVISE"
        summary = match.group(1).strip() if match else text

    summary = " ".join(summary.split()).strip().strip('"').strip("'").strip()
    if not summary:
        return "KEEP_ORIGINAL", original_summary
    return decision, summary


def call_chat_model(model: str, prompt: str, max_retries: int, retry_sleep: int) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "text"},
                temperature=0.8,
                max_completion_tokens=300,
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


def build_initial_state(item: Dict[str, Any], revision_model: str, variant: str) -> Dict[str, Any]:
    initial_scores = scores_from_item(item)
    return {
        "status": "in_progress",
        "ablation_name": "no_evidence_hard",
        "prompt_variant": variant,
        "revision_model": revision_model,
        "original_summary": item.get("pred_EP4CS", ""),
        "initial_scores": initial_scores,
        "initial_average": average_score(initial_scores),
        "accepted_summary": item.get("pred_EP4CS", ""),
        "accepted_scores": dict(initial_scores),
        "accepted_average": average_score(initial_scores),
        "history": [],
    }


def append_history(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    event["event_index"] = len(state["history"]) + 1
    state["history"].append(event)


def update_item_acceptance(item: Dict[str, Any], state: Dict[str, Any]) -> None:
    item["pred_EP4CS_optimized"] = state["accepted_summary"]
    item["optimized_scores"] = state["accepted_scores"]
    item["optimized_average"] = state["accepted_average"]


def optimize_item(
    item: Dict[str, Any],
    revision_model: str,
    variant: str,
    max_retries: int,
    retry_sleep: int,
) -> Dict[str, Any]:
    state = item.get("optimization")
    if not isinstance(state, dict) or state.get("ablation_name") != "no_evidence_hard":
        state = build_initial_state(item, revision_model, variant)
        item["optimization"] = state
    if state.get("status") == "completed":
        return item

    source_code = str(item.get("code", ""))
    accepted_summary = str(state["accepted_summary"])
    retrieved_examples = get_retrieved_examples_payload(item)
    prompt = build_revision_prompt(
        source_code=source_code,
        current_summary=accepted_summary,
        retrieved_examples=retrieved_examples,
        variant=variant,
    )
    raw_revision_output = call_chat_model(revision_model, prompt, max_retries=max_retries, retry_sleep=retry_sleep)
    revision_decision, revised_summary = parse_revision_output(raw_revision_output, accepted_summary)

    state["accepted_summary"] = revised_summary
    state["status"] = "completed"
    state["stop_reason"] = "single_global_no_evidence_hard_revision_completed"
    append_history(
        state,
        {
            "event": "global_revision",
            "raw_revision_output": raw_revision_output,
            "revision_decision": revision_decision,
            "revised_summary": revised_summary,
            "retrieved_examples": retrieved_examples,
            "source_code_present": True,
            "code_evidence_present": False,
            "critic_present": False,
            "quality_definitions_present": False,
        },
    )
    update_item_acceptance(item, state)
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard no-evidence ablation with source code and unchanged retrieved examples.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--revision-model", default=DEFAULT_REVISION_MODEL, help="Model used for summary revision.")
    parser.add_argument(
        "--variant",
        choices=[
            "example_overfit",
            "template_transplant",
            "concise_generic",
            "near_copy_late_example",
            "copy_last_example",
        ],
        default="example_overfit",
        help="Prompt behavior used after removing code evidence.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--start", type=int, default=0, help="Start sample index, inclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument("--force", action="store_true", help="Reinitialize the output file from the input file.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for revision model calls.")
    parser.add_argument("--retry-sleep", type=int, default=25, help="Seconds to sleep between revision retries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_openai(args.base_url, args.api_key)

    rows = initialize_output(args.input, args.output, args.force)
    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    progress = tqdm(range(args.start, end), desc=f"Optimizing no-evidence hard ({args.variant})", unit="sample", dynamic_ncols=True)

    for idx in progress:
        progress.set_postfix_str(f"row={idx + 1}/{len(rows)}")
        try:
            rows[idx] = optimize_item(
                item=rows[idx],
                revision_model=args.revision_model,
                variant=args.variant,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
        finally:
            write_jsonl_atomic(args.output, rows)

    write_jsonl_atomic(args.output, rows)
    print(f"updated: {args.output}")
    print(f"variant: {args.variant}")


if __name__ == "__main__":
    main()
