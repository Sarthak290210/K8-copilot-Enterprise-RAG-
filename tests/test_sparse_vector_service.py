"""BM25 tokenising/search and RRF fusion (pure functions, no Qdrant)."""

import pytest

from app.models import RetrievedChunk
from app.services.sparse_vector_service import (
    SparseVectorIndex,
    _tokenize,
    chunk_id,
    fuse_rrf,
)


def doc(source, index, text):
    return {"text": text, "source": source, "metadata": {"chunk_index": index}}


def chunk(source, index=0, score=0.5):
    return RetrievedChunk(text=source, source=source, score=score, metadata={"chunk_index": index})


def test_tokenize_keeps_k8s_names_and_drops_stopwords():
    tokens = _tokenize("Why is my pod in CrashLoopBackOff? Check the kube-proxy v1.34 logs")
    assert "crashloopbackoff" in tokens
    assert "kube-proxy" in tokens and "v1.34" in tokens  # kept whole...
    assert {"kube", "proxy"} <= set(tokens)  # ...and also split, so "proxy" alone matches
    assert not {"why", "is", "my", "in", "the"} & set(tokens)


def test_exact_token_ranks_the_right_chunk_first():
    index = SparseVectorIndex()
    index.fit(
        [
            doc("pods.html", 0, "A Pod is the smallest deployable unit in Kubernetes."),
            doc("debug.txt", 0, "A container in CrashLoopBackOff keeps restarting; check logs."),
            doc("noise.pdf", 0, "Graphics pipelines and compute shaders on the GPU."),
            doc("deploy.txt", 0, "Use kubectl rollout undo to roll back a Deployment."),
        ]
    )
    hits = index.search("my container shows CrashLoopBackOff", top_k=3)
    assert hits[0].source == "debug.txt"
    assert index.search("kubectl rollout undo")[0].source == "deploy.txt"
    assert all(h.source != "noise.pdf" for h in hits)  # no shared word -> left out
    assert hits[0].metadata == {"chunk_index": 0}


def test_unfitted_or_empty_query_returns_nothing():
    index = SparseVectorIndex()
    assert index.search("anything") == []
    index.fit([])
    assert index.search("anything") == []
    index.fit([doc("a.txt", 0, "pods")])
    assert index.search("the is a") == []  # only stopwords


def test_rrf_hand_computed():
    # k=60: A is dense #1 and sparse #3 -> 1/61 + 1/63; B dense #2 only -> 1/62;
    # C sparse #1 only -> 1/61; D sparse #2 only -> 1/62.
    a, b, c, d = chunk("a"), chunk("b"), chunk("c"), chunk("d")
    fused = fuse_rrf([a, b], [c, d, a], k=60, top_k=10)
    assert [f.source for f in fused] == ["a", "c", "b", "d"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63, abs=1e-6)  # ≈ 0.0323, as in plan.md
    assert fused[0].metadata == {"chunk_index": 0, "dense_rank": 1, "sparse_rank": 3}
    assert fused[1].metadata == {"chunk_index": 0, "sparse_rank": 1}


def test_rrf_respects_top_k_and_does_not_mutate_inputs():
    dense = [chunk(f"d{i}") for i in range(5)]
    fused = fuse_rrf(dense, [], k=60, top_k=2)
    assert [f.source for f in fused] == ["d0", "d1"]
    assert dense[0].score == 0.5 and "dense_rank" not in dense[0].metadata


def test_rrf_uses_chunk_identity_not_source_alone():
    # Two chunks of the same file are different results.
    fused = fuse_rrf([chunk("a", 0)], [chunk("a", 1)], k=60)
    assert len(fused) == 2
    assert chunk_id(chunk("a", 1)) == "a::1"
