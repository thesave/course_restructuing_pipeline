"""Sentence embeddings of topic descriptions, for the description-similarity layer.

The text embedded for a topic is its key in words followed by its description.
Embeddings are L2-normalised, so a dot product is a cosine similarity.

    .venv/bin/python -m network_analysis.topic_embeddings \
        data/restructuring/attempt-.../topics-global.yml --output DIR/topic-embeddings.npz

Only numpy, PyYAML and sentence-transformers are needed, so the command can run
in a separate environment where PyTorch is available.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml


# The model used by clustering/ and scraping/merge_teachings.py.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def topic_text(key: str, description: str) -> str:
    return f"{key.replace('_', ' ')}: {description}".strip()


def embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def save_embeddings(path: pathlib.Path, keys: list[str], vectors: np.ndarray, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(temporary, keys=np.array(keys), vectors=vectors, model=np.array(model_name))
    temporary.replace(path)


def load_embeddings(path: pathlib.Path) -> tuple[list[str], np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        return [str(key) for key in data["keys"]], data["vectors"].astype(np.float32), str(data["model"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topics_global", type=pathlib.Path, help="Path to an attempt's topics-global.yml")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output .npz file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    topics = (yaml.safe_load(args.topics_global.read_text(encoding="utf-8")) or {}).get("topics") or {}
    keys = sorted(str(key) for key in topics)
    vectors = embed([topic_text(key, str(topics[key] or "")) for key in keys], args.model)
    save_embeddings(args.output, keys, vectors, args.model)
    print(f"Wrote {args.output} ({len(keys)} topics, {vectors.shape[1]} dimensions, {args.model})")


if __name__ == "__main__":
    main()
