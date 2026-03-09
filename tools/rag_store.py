"""
Local hybrid RAG store for transparency output.

Dense: hashed embeddings + LanceDB vector table.
Lexical: TF-IDF sparse index.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

try:
    import lancedb  # type: ignore
    HAS_LANCEDB = True
except Exception:
    lancedb = None  # type: ignore
    HAS_LANCEDB = False


_HASH_DIMS = 768
_hash_vec = HashingVectorizer(
    n_features=_HASH_DIMS,
    norm=None,
    alternate_sign=False,
    ngram_range=(1, 2),
    lowercase=True,
)

_CACHE: dict[str, dict[str, Any]] = {}


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _chunk_text(text: str, chunk_size: int = 1400, overlap: int = 220) -> list[str]:
    if not text.strip():
        return []
    lines = text.splitlines()
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        line_len = len(line) + 1
        if cur and cur_len + line_len > chunk_size:
            chunks.append("\n".join(cur).strip())
            tail = ("\n".join(cur))[-overlap:]
            cur = [tail, line] if tail else [line]
            cur_len = len("\n".join(cur))
        else:
            cur.append(line)
            cur_len += line_len
    if cur:
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]


def _dense_embed(texts: list[str]) -> np.ndarray:
    x = _hash_vec.transform(texts).astype(np.float32)
    arr = x.toarray()
    return _l2_normalize(arr)


def _build_lancedb_index(dense: np.ndarray, chunks: list[str], out_dir: Path) -> bool:
    if not HAS_LANCEDB:
        return False
    try:
        lancedb_dir = out_dir / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(lancedb_dir))
        rows = [
            {"id": int(i), "text": chunks[i], "vector": dense[i].tolist()}
            for i in range(len(chunks))
        ]
        db.create_table("rag_chunks", data=rows, mode="overwrite")
        return True
    except Exception:
        return False


def build_hybrid_index_from_text(text: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("Cannot build RAG index from empty text")

    dense = _dense_embed(chunks).astype(np.float32)
    np.save(out_dir / "dense.npy", dense)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), lowercase=True)
    sparse_mat = tfidf.fit_transform(chunks)
    sparse.save_npz(out_dir / "tfidf.npz", sparse_mat)
    with (out_dir / "tfidf.pkl").open("wb") as f:
        pickle.dump(tfidf, f)

    with (out_dir / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    has_lancedb = _build_lancedb_index(dense, chunks, out_dir)
    meta = {
        "chunks": len(chunks),
        "dims": dense.shape[1],
        "has_lancedb": has_lancedb,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _load_index(index_dir: Path) -> dict:
    key = str(index_dir.resolve())
    cached = _CACHE.get(key)
    if cached:
        return cached

    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    dense = np.load(index_dir / "dense.npy")
    tfidf = pickle.loads((index_dir / "tfidf.pkl").read_bytes())
    sparse_mat = sparse.load_npz(index_dir / "tfidf.npz")

    table = None
    if HAS_LANCEDB:
        try:
            db = lancedb.connect(str(index_dir / "lancedb"))
            table = db.open_table("rag_chunks")
        except Exception:
            table = None

    obj = {
        "chunks": chunks,
        "dense": dense,
        "tfidf": tfidf,
        "sparse": sparse_mat,
        "lancedb_table": table,
    }
    _CACHE[key] = obj
    return obj


def _dense_scores_from_lancedb(table: Any, q_dense: np.ndarray, n: int) -> np.ndarray | None:
    try:
        query = q_dense[0].tolist()
        rows = table.search(query).limit(n).to_list()
        scores = np.zeros(n, dtype=np.float32)
        out = np.zeros(int(max((r.get("id", -1) for r in rows), default=-1) + 1), dtype=np.float32)
        # LanceDB distance is lower-is-better; convert to similarity.
        for r in rows:
            rid = int(r.get("id", -1))
            dist = float(r.get("_distance", 1.0))
            if rid >= 0:
                sim = 1.0 / (1.0 + max(0.0, dist))
                if rid >= out.shape[0]:
                    grow = rid + 1 - out.shape[0]
                    out = np.pad(out, (0, grow), constant_values=0.0)
                out[rid] = sim
        _ = scores  # keep local variable to avoid lint noise in minimal envs
        return out
    except Exception:
        return None


def hybrid_search(index_dir: Path, query: str, top_k: int = 8) -> list[dict]:
    data = _load_index(index_dir)
    chunks: list[str] = data["chunks"]
    dense: np.ndarray = data["dense"]
    sparse_mat = data["sparse"]
    tfidf = data["tfidf"]
    table = data["lancedb_table"]

    q_dense = _dense_embed([query]).astype(np.float32)
    dense_scores = None
    if table is not None:
        dense_scores = _dense_scores_from_lancedb(table, q_dense, len(chunks))

    if dense_scores is None or len(dense_scores) < len(chunks):
        dense_scores = (dense @ q_dense[0]).astype(np.float32)
    else:
        dense_scores = dense_scores[: len(chunks)]

    q_sparse = tfidf.transform([query])
    lexical_scores = (sparse_mat @ q_sparse.T).toarray().ravel().astype(np.float32)

    d = dense_scores
    l = lexical_scores
    if d.max() > d.min():
        d = (d - d.min()) / (d.max() - d.min() + 1e-12)
    if l.max() > l.min():
        l = (l - l.min()) / (l.max() - l.min() + 1e-12)
    final = 0.6 * d + 0.4 * l

    order = np.argsort(-final)[:top_k]
    out = []
    for i in order:
        out.append(
            {
                "id": int(i),
                "score": float(final[i]),
                "dense": float(dense_scores[i]),
                "lexical": float(lexical_scores[i]),
                "text": chunks[i],
            }
        )
    return out
