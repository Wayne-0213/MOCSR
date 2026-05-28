from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]

DEFAULT_TRAIN_PATH = ROOT_DIR / "dataset" / "python" / "train.jsonl"
DEFAULT_TEST_PATH = SCRIPT_DIR / "test_new_pred_EP4CS.jsonl"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "retrieval_cache"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = ["Coherence", "Consistency", "Fluency", "Relevance"]
TOKEN_TOP_K = 5
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


def find_top_token_examples(
    test_tokens: FrozenSet[str],
    train_token_index: List[Tuple[FrozenSet[str], int, Dict[str, Any]]],
    top_k: int,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    scored = [
        (train_idx, jaccard_similarity(test_tokens, train_tokens), row)
        for train_tokens, train_idx, row in train_token_index
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


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
    train_rows: List[Dict[str, Any]],
    train_token_index: List[Tuple[FrozenSet[str], int, Dict[str, Any]]],
    train_embeddings: torch.Tensor,
    test_embeddings: torch.Tensor,
) -> Dict[str, int]:
    stats = {
        "rows": 0,
        "token_examples_written": 0,
        "semantic_examples_written": 0,
        "retrieved_examples_written": 0,
        "semantic_duplicates_skipped": 0,
    }

    for test_idx, row in enumerate(tqdm(test_rows, desc="Retrieving examples", unit="sample", dynamic_ncols=True)):
        remove_old_dimension_examples(row)

        token_hits = find_top_token_examples(code_tokens(code_text(row)), train_token_index, TOKEN_TOP_K)
        token_indices = {train_idx for train_idx, _, _ in token_hits}
        semantic_hits = find_top_semantic_examples(
            test_embedding=test_embeddings[test_idx],
            train_embeddings=train_embeddings,
            train_rows=train_rows,
            excluded_indices=token_indices,
            top_k=SEMANTIC_TOP_K,
        )

        token_examples = [
            make_example(hit_row, train_idx, "token_jaccard", rank, similarity)
            for rank, (train_idx, similarity, hit_row) in enumerate(token_hits, 1)
        ]
        semantic_examples = [
            make_example(hit_row, train_idx, "sentence_transformer", rank, similarity)
            for rank, (train_idx, similarity, hit_row) in enumerate(semantic_hits, 1)
        ]

        row["retrieved_examples"] = interleave_examples(token_examples, semantic_examples)
        row["token_retrieved_examples"] = token_examples
        row["semantic_retrieved_examples"] = semantic_examples

        stats["rows"] += 1
        stats["token_examples_written"] += len(token_examples)
        stats["semantic_examples_written"] += len(semantic_examples)
        stats["retrieved_examples_written"] += len(row["retrieved_examples"])
        stats["semantic_duplicates_skipped"] += len(token_indices & {train_idx for train_idx, _, _ in semantic_hits})

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach sample-level retrieved examples from train.jsonl. "
            "Each test sample receives 5 token-Jaccard and 5 sentence-transformer semantic examples."
        )
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH, help="Training JSONL path.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH, help="Test JSONL path to update.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Directory for reusable embedding cache.")
    parser.add_argument("--sentence-transformer-model", default=DEFAULT_SENTENCE_TRANSFORMER_MODEL, help="Sentence-transformer model name or local path.")
    parser.add_argument("--batch-size", type=int, default=16, help="Sentence-transformer embedding batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sentence-transformer token length.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Embedding device, for example cuda or cpu.",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not create a timestamped backup of the test file.")
    parser.add_argument(
        "--precompute-train-only",
        action="store_true",
        help="Only precompute training sentence-transformer embeddings. Does not read or write the test JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_rows = read_jsonl(args.train)

    train_embeddings = load_train_embedding_cache(
        cache_dir=args.cache_dir,
        train_path=args.train,
        train_count=len(train_rows),
        model_name=args.sentence_transformer_model,
        max_length=args.max_length,
    )
    if train_embeddings is None:
        train_embeddings = embed_codes(
            [code_text(row) for row in train_rows],
            model_name=args.sentence_transformer_model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_name=args.device,
        )
        save_train_embedding_cache(
            cache_dir=args.cache_dir,
            embeddings=train_embeddings,
            train_path=args.train,
            train_count=len(train_rows),
            model_name=args.sentence_transformer_model,
            max_length=args.max_length,
        )
        print(f"saved train embedding cache: {train_cache_paths(args.cache_dir)[0]}")
    else:
        print(f"loaded train embedding cache: {train_cache_paths(args.cache_dir)[0]}")

    if args.precompute_train_only:
        print(f"train_rows: {len(train_rows)}")
        print("precompute_train_only: completed without reading or writing the test JSONL")
        return

    test_rows = read_jsonl(args.test)
    train_token_index = build_token_index(train_rows)

    if not args.no_backup:
        backup_path = args.test.with_name(args.test.name + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak")
        shutil.copy2(args.test, backup_path)
        print(f"backup: {backup_path}")

    test_embeddings = embed_codes(
        [code_text(row) for row in test_rows],
        model_name=args.sentence_transformer_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_name=args.device,
    )

    stats = attach_examples(
        test_rows=test_rows,
        train_rows=train_rows,
        train_token_index=train_token_index,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
    )
    write_jsonl(args.test, test_rows)

    print(f"train_rows: {len(train_rows)}")
    print(f"test_rows: {len(test_rows)}")
    print(f"token_examples_written: {stats['token_examples_written']}")
    print(f"semantic_examples_written: {stats['semantic_examples_written']}")
    print(f"retrieved_examples_written: {stats['retrieved_examples_written']}")
    print(f"updated: {args.test}")


if __name__ == "__main__":
    main()
