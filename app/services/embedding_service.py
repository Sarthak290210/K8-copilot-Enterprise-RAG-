"""Text -> vectors with the local bge model (sentence-transformers)."""

import asyncio
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.query_cache_service import query_cache

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


async def embed_texts(
    texts: list[str], model: str | None = None, cache: bool = False
) -> list[list[float]]:
    """Async wrapper: encoding is CPU-bound, so it runs in a worker thread.

    cache=True (questions and HyDE passages, Phase 5): each text is looked up in the embedding
    cache and the model runs only for the misses. Document chunks are never cached (user's
    choice: the file-hash dedup skips unchanged files, and each vector is ~15 KB in Redis).
    """
    if not cache or not texts:
        return await asyncio.to_thread(embed_texts_sync, texts, model)
    cached = await asyncio.gather(*(query_cache.get_embedding(t, model) for t in texts))
    missing = [i for i, vector in enumerate(cached) if vector is None]
    if missing:
        fresh = await asyncio.to_thread(embed_texts_sync, [texts[i] for i in missing], model)
        await asyncio.gather(
            *(
                query_cache.set_embedding(texts[i], v, model)
                for i, v in zip(missing, fresh, strict=True)
            )
        )
        for i, vector in zip(missing, fresh, strict=True):
            cached[i] = vector
    return [vector for vector in cached if vector is not None]


async def embed_query(question: str) -> list[float]:
    """Embed a search query (with the bge query instruction); cached."""
    return (await embed_texts([BGE_QUERY_INSTRUCTION + question], cache=True))[0]
