from __future__ import annotations

import argparse
import heapq
import json
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]

DEFAULT_TRAIN_PATH = ROOT_DIR / "dataset" / "java" / "train.jsonl"
DEFAULT_TEST_PATH = SCRIPT_DIR / "test_new_pred_EP4CS.jsonl"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "test_new_pred_EP4CS_jaccard_top10.jsonl"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "retrieval_cache"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = ["Coherence", "Consistency", "Fluency", "Relevance"]
TOKEN_TOP_K = 10
SEMANTIC_TOP_K = 5
TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||[{}()\[\].,;:+\-*/%<>!?=]"
)


def code_text(row: Dict[str, Any]) -> str:
    return str(row.get("code") or row.get("Code") or row.get("original_string") or "")


def code_tokens(code: Any) -> FrozenSet[str]:
    return frozenset(TOKEN_RE.findall(str(code or "").lower()))


def jaccard_similarity(left: FrozenSet[str], right: FrozenSet[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def build_token_index(train_rows: List[Dict[str, Any]]) -> List[Tuple[FrozenSet[str], int, Dict[str, Any]]]:
    return [(code_tokens(code_text(row)), train_idx, row) for train_idx, row in enumerate(train_rows)]


def build_token_lookup(
    train_token_index: List[Tuple[FrozenSet[str], int, Dict[str, Any]]],
) -> Dict[str, List[int]]:
    lookup: Dict[str, List[int]] = {}
    for train_tokens, train_idx, _ in train_token_index:
        for token in train_tokens:
            lookup.setdefault(token, []).append(train_idx)
    return lookup


def find_top_token_examples(
    test_tokens: FrozenSet[str],
    train_token_index: List[Tuple[FrozenSet[str], int, Dict[str, Any]]],
    train_token_lookup: Dict[str, List[int]],
    top_k: int,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    if not test_tokens:
        scored = [
            (train_idx, jaccard_similarity(test_tokens, train_tokens), row)
            for train_tokens, train_idx, row in train_token_index
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    intersection_counts: Dict[int, int] = {}
    for token in test_tokens:
        for train_idx in train_token_lookup.get(token, []):
            intersection_counts[train_idx] = intersection_counts.get(train_idx, 0) + 1

    heap: List[Tuple[float, int, int, Dict[str, Any]]] = []
    test_len = len(test_tokens)

    def push_candidate(train_idx: int, similarity: float, row: Dict[str, Any]) -> None:
        candidate = (similarity, -train_idx, train_idx, row)
        if len(heap) < top_k:
            heapq.heappush(heap, candidate)
        elif candidate[:2] > heap[0][:2]:
            heapq.heapreplace(heap, candidate)

    for train_idx, intersection_size in intersection_counts.items():
        train_tokens, _, row = train_token_index[train_idx]
        union_size = test_len + len(train_tokens) - intersection_size
        similarity = intersection_size / union_size if union_size else 1.0
        push_candidate(train_idx, similarity, row)

    if len(heap) < top_k:
        for train_tokens, train_idx, row in train_token_index:
            if train_idx in intersection_counts:
                continue
            push_candidate(train_idx, 0.0, row)
            if len(heap) >= top_k:
                break

    scored = [(train_idx, similarity, row) for similarity, _, train_idx, row in heap]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def embed_codes(
    codes: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device_name: str,
) -> torch.Tensor:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required. Install it with: "
            "python -m pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name, device=device_name)
    model.max_seq_length = max_length

    embeddings: List[torch.Tensor] = []
    for start in tqdm(range(0, len(codes), batch_size), desc="Embedding code", unit="batch", dynamic_ncols=True):
        batch_codes = list(codes[start : start + batch_size])
        batch_embeddings = model.encode(
            batch_codes,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


def train_cache_paths(cache_dir: Path) -> Tuple[Path, Path]:
    return cache_dir / "train_sentence_transformer_embeddings.pt", cache_dir / "train_sentence_transformer_embeddings_meta.json"


def save_train_embedding_cache(
    cache_dir: Path,
    embeddings: torch.Tensor,
    train_path: Path,
    train_count: int,
    model_name: str,
    max_length: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_path, meta_path = train_cache_paths(cache_dir)
    torch.save(embeddings.half(), embedding_path)
    meta = {
        "train_path": str(train_path.resolve()),
        "train_count": train_count,
        "sentence_transformer_model": model_name,
        "max_length": max_length,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_train_embedding_cache(
    cache_dir: Path,
    train_path: Path,
    train_count: int,
    model_name: str,
    max_length: int,
) -> Optional[torch.Tensor]:
    embedding_path, meta_path = train_cache_paths(cache_dir)
    if not embedding_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    expected = {
        "train_path": str(train_path.resolve()),
        "train_count": train_count,
        "sentence_transformer_model": model_name,
        "max_length": max_length,
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        return None

    return torch.load(embedding_path, map_location="cpu").float()


def find_top_semantic_examples(
    test_embedding: torch.Tensor,
    train_embeddings: torch.Tensor,
    train_rows: List[Dict[str, Any]],
    excluded_indices: Set[int],
    top_k: int,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    scores = torch.mv(train_embeddings, test_embedding)
    for train_idx in excluded_indices:
        if 0 <= train_idx < len(scores):
            scores[train_idx] = -float("inf")

    k = min(top_k, len(train_rows) - len(excluded_indices))
    if k <= 0:
        return []

    values, indices = torch.topk(scores, k=k, largest=True, sorted=True)
    return [
        (int(train_idx), float(score), train_rows[int(train_idx)])
        for score, train_idx in zip(values.tolist(), indices.tolist())
    ]


def make_example(
    row: Dict[str, Any],
    train_idx: int,
    method: str,
    rank: int,
    similarity: float,
) -> Dict[str, Any]:
    example = dict(row)
    docstring_tokens = row.get("docstring_tokens", [])
    if isinstance(docstring_tokens, list):
        example["reference_summary"] = " ".join(str(token) for token in docstring_tokens)
    elif docstring_tokens:
        example["reference_summary"] = str(docstring_tokens)
    else:
        example["reference_summary"] = str(row.get("docstring", ""))
    example["retrieval_method"] = method
    example["retrieval_rank"] = rank
    example["retrieval_similarity"] = similarity
    example["retrieval_train_index"] = train_idx
    return example


def interleave_examples(
    token_examples: List[Dict[str, Any]],
    semantic_examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    for index in range(TOKEN_TOP_K - 1, -1, -1):
        if index < len(token_examples):
            ordered.append(token_examples[index])
        if index < len(semantic_examples):
            ordered.append(semantic_examples[index])
    return ordered


def remove_old_dimension_examples(row: Dict[str, Any]) -> None:
    for dimension in DIMENSIONS:
        row.pop(f"{dimension}_example", None)
        row.pop(f"{dimension}_explanation", None)


def attach_examples(
    test_rows: List[Dict[str, Any]],
    train_token_index: List[Tuple[FrozenSet[str], int, Dict[str, Any]]],
    train_token_lookup: Dict[str, List[int]],
) -> Dict[str, int]:
    stats = {
        "rows": 0,
        "token_examples_written": 0,
        "semantic_examples_written": 0,
        "retrieved_examples_written": 0,
    }

    for test_idx, row in enumerate(tqdm(test_rows, desc="Retrieving examples", unit="sample", dynamic_ncols=True)):
        remove_old_dimension_examples(row)

        token_hits = find_top_token_examples(code_tokens(code_text(row)), train_token_index, train_token_lookup, TOKEN_TOP_K)
        token_examples = [
            make_example(hit_row, train_idx, "token_jaccard", rank, similarity)
            for rank, (train_idx, similarity, hit_row) in enumerate(token_hits, 1)
        ]
        ordered_examples = list(reversed(token_examples))

        row["retrieved_examples"] = ordered_examples
        row["token_retrieved_examples"] = ordered_examples
        row["semantic_retrieved_examples"] = []

        stats["rows"] += 1
        stats["token_examples_written"] += len(ordered_examples)
        stats["retrieved_examples_written"] += len(row["retrieved_examples"])

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach sample-level retrieved examples from train.jsonl. "
            "Each test sample receives 10 token-Jaccard examples sorted by ascending similarity."
        )
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH, help="Training JSONL path.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH, help="Source test JSONL path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    train_token_index = build_token_index(train_rows)
    train_token_lookup = build_token_lookup(train_token_index)

    stats = attach_examples(
        test_rows=test_rows,
        train_token_index=train_token_index,
        train_token_lookup=train_token_lookup,
    )
    write_jsonl(args.output, test_rows)

    print(f"train_rows: {len(train_rows)}")
    print(f"test_rows: {len(test_rows)}")
    print(f"token_examples_written: {stats['token_examples_written']}")
    print(f"semantic_examples_written: {stats['semantic_examples_written']}")
    print(f"retrieved_examples_written: {stats['retrieved_examples_written']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
