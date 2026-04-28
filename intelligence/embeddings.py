from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from openai import OpenAI


def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or a .env file.")
    return OpenAI()


def embed_texts(
    texts: Sequence[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> np.ndarray:
    client = get_client()
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        resp = client.embeddings.create(model=model, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        for row in ordered:
            all_vecs.append(row.embedding)
    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
