import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm


DIMENSIONS = ["Coherence", "Consistency", "Fluency", "Relevance"]


class _NoWordNet:
    @staticmethod
    def synsets(_word):
        return []


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tokenize(text: str) -> List[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text)


def reference_text(row: Dict[str, Any]) -> str:
    tokens = row.get("docstring_tokens")
    if isinstance(tokens, list):
        return " ".join(str(token) for token in tokens)
    if tokens:
        return str(tokens)
    return str(row.get("docstring") or row.get("ref_summary") or "")


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_meteor(ref_tokens: Sequence[str], pred_tokens: Sequence[str]) -> float:
    if not ref_tokens or not pred_tokens:
        return 0.0
    try:
        return float(meteor_score([list(ref_tokens)], list(pred_tokens)))
    except LookupError:
        return float(meteor_score([list(ref_tokens)], list(pred_tokens), wordnet=_NoWordNet()))


def dimension_scores(row: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    scores: Dict[str, Any] = {}
    for dimension in DIMENSIONS:
        field = dimension if not prefix else f"{prefix}_{dimension}"
        scores[dimension] = safe_float(row.get(field))
    valid = [score for score in scores.values() if score is not None]
    scores["four_dim_average"] = mean(valid) if valid else None
    return scores


def build_system_rows(
    baseline_rows: List[Dict[str, Any]],
    ablation_rows: List[Dict[str, Any]],
    systems: List[Dict[str, str]],
    num_samples: int,
) -> List[Dict[str, Any]]:
    if len(systems) != 2:
        raise ValueError("Expected exactly two systems.")
    if min(len(baseline_rows), len(ablation_rows)) < num_samples:
        raise ValueError("Not enough rows for the requested sample count.")

    rows_by_system = {
        systems[0]["name"]: baseline_rows,
        systems[1]["name"]: ablation_rows,
    }
    records: List[Dict[str, Any]] = []
    for sample_idx in range(num_samples):
        base_row = baseline_rows[sample_idx]
        reference = reference_text(base_row)
        for system in systems:
            row = rows_by_system[system["name"]][sample_idx]
            prediction = str(row.get(system["summary_field"]) or "").strip()
            if not prediction:
                raise ValueError(f"Missing prediction for {system['name']} at sample {sample_idx}")
            records.append(
                {
                    "sample_id": sample_idx,
                    "system": system["name"],
                    "reference": reference,
                    "prediction": prediction,
                    **dimension_scores(row, system["score_prefix"]),
                }
            )
    return records


def compute_metrics(records: List[Dict[str, Any]]) -> None:
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for record in tqdm(records, desc="Computing metrics", unit="summary", dynamic_ncols=True):
        pred = record["prediction"]
        ref = record["reference"]
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        record["METEOR"] = safe_meteor(ref_tokens, pred_tokens)
        record["ROUGE_L"] = float(rouge.score(ref, pred)["rougeL"].fmeasure) if ref and pred else 0.0


def summarize(records: List[Dict[str, Any]], systems: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for system_name in [system["name"] for system in systems]:
        subset = [record for record in records if record["system"] == system_name]
        corpus_refs = [[tokenize(record["reference"])] for record in subset]
        corpus_preds = [tokenize(record["prediction"]) for record in subset]
        corpus_bleu_score = float(
            corpus_bleu(
                corpus_refs,
                corpus_preds,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=SmoothingFunction().method4,
            )
        )
        row: Dict[str, Any] = {
            "system": system_name,
            "corpus_BLEU": corpus_bleu_score,
            "mean_METEOR": mean(record["METEOR"] for record in subset),
            "mean_ROUGE_L": mean(record["ROUGE_L"] for record in subset),
        }
        for dimension in DIMENSIONS:
            values = [record[dimension] for record in subset if record.get(dimension) is not None]
            row[f"mean_{dimension}"] = mean(values) if values else ""
        values = [record["four_dim_average"] for record in subset if record.get("four_dim_average") is not None]
        row["mean_four_dim_average"] = mean(values) if values else ""
        summary_rows.append(row)
    return summary_rows


def parse_args(
    baseline_path: Path,
    ablation_path: Path,
    output_dir: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute corpus BLEU, METEOR, ROUGE-L, and four-dimensional scores.")
    parser.add_argument("--baseline", type=Path, default=baseline_path)
    parser.add_argument("--ablation", type=Path, default=ablation_path)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--num-samples", type=int, default=1000)
    return parser.parse_args()


def run_metric_script(
    baseline_path: Path,
    ablation_path: Path,
    output_dir: Path,
    systems: List[Dict[str, str]],
) -> None:
    args = parse_args(baseline_path, ablation_path, output_dir)
    baseline_rows = read_jsonl(args.baseline)
    ablation_rows = read_jsonl(args.ablation)
    records = build_system_rows(baseline_rows, ablation_rows, systems, args.num_samples)
    compute_metrics(records)
    summary_rows = summarize(records, systems)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "auto_metrics_summary.csv", summary_rows)
    (args.output_dir / "auto_metrics_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"saved: {args.output_dir}")
