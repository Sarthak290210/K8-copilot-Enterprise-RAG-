"""Named feature combinations. Each profile is the `flags` dict passed straight to run_rag."""

from typing import Any

Flags = dict[str, Any]

_BASE: Flags = {
    "search_mode": "dense",
    "top_k": 5,
    "enable_rerank": False,
    "enable_hyde": False,
    "enable_crag": False,
    "enable_self_reflective": False,
}

PROFILES: dict[str, Flags] = {
    "naive": dict(_BASE),
    "sparse_only": _BASE | {"search_mode": "sparse"},
    "hybrid": _BASE | {"search_mode": "hybrid"},
    "hybrid+rerank": _BASE | {"search_mode": "hybrid", "enable_rerank": True},
    # Phase 3 builds on `hybrid`, not `hybrid+rerank`: E2 showed reranking hurts on this
    # corpus (user's choice, 2026-10-23; plan.md's names had +rerank).
    "hybrid+hyde": _BASE | {"search_mode": "hybrid", "enable_hyde": True},
    "hybrid+crag": _BASE | {"search_mode": "hybrid", "enable_crag": True},
    "all": _BASE
    | {
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_crag": True,
        "enable_self_reflective": True,
    },
}


def get_profile(name: str) -> Flags:
    """A copy of the profile's flags, so callers can't mutate the shared definition."""
    if name not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Choose from: {', '.join(PROFILES)}")
    return dict(PROFILES[name])
