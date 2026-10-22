"""Text -> vectors with the local bge model (sentence-transformers)."""

import asyncio
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

EMBED_BATCH_SIZE = 64
# bge-*-v1.5 retrieves better when *queries* (not documents) carry this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_model(model_name: str | None = None) -> SentenceTransformer:
    """Loaded once per process (~440 MB); cuda if available, else cpu.

    Tries the local Hugging Face cache first, so a cached model loads without any network
    call (DNS on this machine is flaky); downloads only if it isn't cached yet.
    """
    name = model_name or settings.embedding_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer(name, device=device, local_files_only=True)
    except OSError:
        return SentenceTransformer(name, device=device)


def embed_texts_sync(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Vectors in the same order as `texts`, L2-normalised (so dot product = cosine)."""
    if not texts:
        return []
    vectors = get_model(model).encode(
        texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


async def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Async wrapper: encoding is CPU-bound, so it runs in a worker thread."""
    return await asyncio.to_thread(embed_texts_sync, texts, model)


async def embed_query(question: str) -> list[float]:
    """Embed a search query (with the bge query instruction)."""
    return (await embed_texts([BGE_QUERY_INSTRUCTION + question]))[0]
