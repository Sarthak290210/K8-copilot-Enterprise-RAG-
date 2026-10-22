"""Streamlit UI: log in, upload documents, ask questions, compare eval runs.

uv run streamlit run scripts/streamlit_app.py      (API must be running on :8000)
"""

import json
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
METRICS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
HISTORY_SIZE = 10
TIMEOUT_SECONDS = 120  # first query loads the embedding model; uploads run Docling

# Retrieval flags, and the phase each one becomes available in.
FEATURE_PHASE = {
    "enable_rerank": 2,
    "enable_hyde": 3,
    "enable_crag": 3,
    "enable_self_reflective": 3,
}
SEARCH_MODES = {"dense": 1, "sparse": 2, "hybrid": 2}
CURRENT_PHASE = 1


# ---------------- HTTP helpers ----------------


def _api_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {"data": data}
    except ValueError:
        return {"raw": resp.text}


def _request(method: str, path: str, **kwargs: Any) -> httpx.Response | None:
    """Shared HTTP call: adds auth + timeout; on 401 clears the token and asks to log in."""
    base_url = st.session_state.get("api_url", DEFAULT_API_URL)
    try:
        resp = httpx.request(
            method,
            f"{base_url}{path}",
            headers=_api_headers(st.session_state.get("token")),
            timeout=TIMEOUT_SECONDS,
            **kwargs,
        )
    except httpx.HTTPError as exc:
        st.error(f"Can't reach the API at {base_url} ({type(exc).__name__}). Is it running?")
        return None
    if resp.status_code == 401 and st.session_state.get("token"):
        st.session_state.pop("token", None)
        st.warning("Your session expired. Please log in again.")
    return resp


def _login(username: str, password: str) -> bool:
    resp = _request("POST", "/auth/login", json={"username": username, "password": password})
    if resp is None:
        return False
    if resp.status_code != 200:
        st.error(_safe_json(resp).get("detail", f"Login failed ({resp.status_code})"))
        return False
    st.session_state["token"] = resp.json()["access_token"]
    me = _request("GET", "/auth/me")
    st.session_state["user"] = me.json() if me is not None and me.status_code == 200 else {}
    return True


# ---------------- small UI pieces ----------------


def _badge(label: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;"
        f"font-size:0.8em;margin-right:4px'>{label}</span>"
    )


def _render_answer_card(payload: dict[str, Any]) -> None:
    meta = payload.get("metadata", {})
    badges = _badge(meta.get("intent", "rag"), "#3b82f6")
    badges += _badge(f"{meta.get('latency_ms', 0):.0f} ms", "#6b7280")
    badges += _badge(f"{meta.get('tokens_used', 0)} tokens", "#6b7280")
    if any((meta.get("cache_hits") or {}).values()):
        badges += _badge("cache hit", "#10b981")
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown(payload.get("answer", ""))
    if payload.get("confidence") is not None:
        st.caption(f"Confidence: {payload['confidence']:.2f}")
    if payload.get("sources"):
        st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in payload["sources"]))
    chunks = payload.get("chunks") or []
    if chunks:
        with st.expander(f"Retrieved chunks ({len(chunks)})"):
            for i, chunk in enumerate(chunks, start=1):
                st.markdown(f"**[{i}] {chunk['source']}** — score {chunk['score']:.3f}")
                st.caption(chunk["snippet"])


# ---------------- sections ----------------


def _sidebar() -> dict[str, Any]:
    """API URL, health, login state, and retrieval flags. Returns the flags for /query."""
    with st.sidebar:
        st.header("k8-copilot")
        st.session_state["api_url"] = st.text_input(
            "API URL", st.session_state.get("api_url", DEFAULT_API_URL)
        )
        health = _request("GET", "/health")
        if health is not None and health.status_code == 200:
            st.success("API is up")

        user = st.session_state.get("user") or {}
        if st.session_state.get("token"):
            st.caption(
                f"Logged in as **{user.get('username', '?')}**"
                + (" (admin)" if user.get("is_admin") else "")
            )
            if st.button("Log out"):
                st.session_state.pop("token", None)
                st.session_state.pop("user", None)
                st.rerun()

        st.subheader("Retrieval flags")
        st.caption("Try profiles by hand. Greyed-out features arrive in later phases.")
        modes = list(SEARCH_MODES)
        search_mode = st.selectbox(
            "Search mode",
            modes,
            format_func=lambda m: (
                m if SEARCH_MODES[m] <= CURRENT_PHASE else f"{m} (Phase {SEARCH_MODES[m]})"
            ),
        )
        top_k = st.slider("top_k", 1, 20, 5)
        flags: dict[str, Any] = {"search_mode": search_mode, "top_k": top_k}
        for flag, phase in FEATURE_PHASE.items():
            label = flag.removeprefix("enable_").replace("_", " ")
            available = phase <= CURRENT_PHASE
            value = st.toggle(
                label if available else f"{label} (Phase {phase})",
                value=False,
                disabled=not available,
            )
            if available:
                flags[flag] = value
        return flags


def _auth_section() -> None:
    login_tab, register_tab = st.tabs(["Log in", "Register"])
    with login_tab, st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in") and _login(username, password):
            st.rerun()
    with register_tab, st.form("register"):
        username = st.text_input("New username", help="3–64 chars: letters, digits, _ . -")
        password = st.text_input("New password", type="password", help="8–72 characters")
        if st.form_submit_button("Register"):
            resp = _request(
                "POST", "/auth/register", json={"username": username, "password": password}
            )
            if resp is not None and resp.status_code == 201:
                st.success("Account created. You can log in now.")
            elif resp is not None:
                st.error(str(_safe_json(resp).get("detail", resp.status_code)))


def _upload_section() -> None:
    if not (st.session_state.get("user") or {}).get("is_admin"):
        st.info("Only admins can upload documents.")
        return
    uploaded = st.file_uploader("Add a document", type=["pdf", "docx", "html", "htm", "md", "txt"])
    if uploaded is not None and st.button("Index it"):
        with st.spinner("Parsing, chunking and embedding…"):
            resp = _request(
                "POST", "/documents/upload", files={"file": (uploaded.name, uploaded.getvalue())}
            )
        if resp is not None and resp.status_code == 201:
            body = resp.json()
            st.success(
                f"Indexed `{body['source']}` ({body['chunks_indexed']} chunks). Ask about it now."
            )
        elif resp is not None:
            st.error(str(_safe_json(resp).get("detail", resp.status_code)))
    docs = _request("GET", "/documents")
    if docs is not None and docs.status_code == 200:
        sources = docs.json()["sources"]
        with st.expander(f"Indexed documents ({len(sources)})"):
            st.write(sources)


def _query_section(flags: dict[str, Any]) -> None:
    question = st.text_area("Ask a Kubernetes question", placeholder="What is a Pod?")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving and answering…"):
            resp = _request("POST", "/query", json={"question": question, "flags": flags})
        if resp is None:
            return
        payload = _safe_json(resp)
        if resp.status_code != 200:
            st.error(str(payload.get("detail", resp.status_code)))
            return
        _render_answer_card(payload)
        history = st.session_state.setdefault("history", [])
        history.insert(0, {"question": question, "flags": flags, "payload": payload})
        del history[HISTORY_SIZE:]


def _history_section() -> None:
    history = st.session_state.get("history") or []
    if not history:
        return
    st.subheader("History (this session)")
    for item in history:
        with st.expander(f"{item['question'][:80]}  ·  {item['flags'].get('search_mode')}"):
            _render_answer_card(item["payload"])


# ---------------- eval dashboard ----------------


def _list_eval_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.json"), reverse=True)  # newest first (timestamp names)


def _load_eval_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_result_choice(path: Path) -> str:
    try:
        data = _load_eval_file(path)
        agg = data["aggregate"]
        return f"{data['profile']} · {data['timestamp']} · {agg['n_evaluated']} evaluated"
    except (OSError, ValueError, KeyError):
        return path.name


def _row_status(row: dict[str, Any]) -> str:
    """pass = faithfulness >= 0.7, at least one golden source cited, no forbidden words."""
    faith = (row.get("metrics") or {}).get("faithfulness")
    overlap = (row.get("source_overlap") or {}).get("overlap")
    forbidden_ok = (row.get("forbidden") or {}).get("passed", True)
    passed = (
        faith is not None and faith >= 0.7 and (overlap is None or overlap > 0) and forbidden_ok
    )
    return "pass" if passed else "fail"


def _eval_dashboard_section() -> None:
    files = _list_eval_files()
    if not files:
        st.info("No eval results yet. Run: uv run python -m eval.run_ragas --profile naive")
        return
    col_a, col_b = st.columns(2)
    a_path = col_a.selectbox(
        "Run A (before)", files, index=min(1, len(files) - 1), format_func=_format_result_choice
    )
    b_path = col_b.selectbox("Run B (after)", files, index=0, format_func=_format_result_choice)
    a, b = _load_eval_file(a_path), _load_eval_file(b_path)

    cols = st.columns(len(METRICS) + 1)
    for col, key in zip(cols, [*METRICS, "source_overlap"], strict=True):
        va, vb = a["aggregate"].get(key), b["aggregate"].get(key)
        delta = f"{vb - va:+.3f}" if va is not None and vb is not None else None
        col.metric(key.replace("_", " "), "-" if vb is None else f"{vb:.3f}", delta)

    st.subheader(f"Per question — run B ({b['profile']})")
    table = [
        {
            "id": row["id"],
            "feature": row["demonstrates_feature"],
            **{m: (row.get("metrics") or {}).get(m) for m in METRICS},
            "overlap": row["source_overlap"]["overlap"],
            "status": _row_status(row),
        }
        for row in b["rows"]
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)
    if b.get("skipped"):
        st.caption(
            f"{len(b['skipped'])} skipped: "
            + ", ".join(sorted({s["reason"] for s in b["skipped"]}))
        )


def main() -> None:
    st.set_page_config(page_title="k8-copilot", page_icon="☸️", layout="wide")
    flags = _sidebar()
    if not st.session_state.get("token"):
        st.title("k8-copilot")
        _auth_section()
        return
    ask_tab, upload_tab, eval_tab = st.tabs(["Ask", "Upload", "Eval dashboard"])
    with ask_tab:
        _query_section(flags)
        _history_section()
    with upload_tab:
        _upload_section()
    with eval_tab:
        _eval_dashboard_section()


if __name__ == "__main__":
    main()
