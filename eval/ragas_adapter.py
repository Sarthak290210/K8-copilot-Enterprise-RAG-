"""Turn eval rows into Ragas scores: faithfulness and answer_relevancy.

Judge LLM: Cerebras `settings.eval_judge_model` (qwen-3.8-27b) via langchain-openai.
Embeddings (answer_relevancy): the same local bge model the app uses (BgeEmbeddings).
context_precision / context_recall are not LLM-judged: `post_checks.id_context_scores`
computes them from source file names (the LLM versions cost ~11k of ~17k tokens per question
and were rough with keyword-only references).
Reference answers: the goldens only have keywords, so the reference is built from them.
"""

import asyncio
import copy
import math
import warnings
from functools import cached_property
from typing import Any, cast

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services import embedding_service, llm_service

with warnings.catch_warnings():
    # ragas 0.4 deprecates the top-level metric imports in favour of ragas.metrics.collections,
    # but `evaluate()` still takes these legacy metric objects.
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas.run_config import RunConfig

METRICS = [faithfulness, answer_relevancy]
METRIC_NAMES = [m.name for m in METRICS]


def _metrics() -> list[Any]:
    """The metric objects for a run, with answer_relevancy's strictness from settings.

    A copy, so Ragas' shared module-level instance is never changed.
    """
    relevancy = copy.copy(answer_relevancy)
    relevancy.strictness = settings.eval_relevancy_strictness
    return [faithfulness, relevancy]


# Cerebras free trial: 5 requests/minute and 1M tokens/day per model. Few parallel judge
# calls plus patient retries keeps a run inside that instead of failing with 429s.
JUDGE_MAX_WORKERS = 2
JUDGE_RUN_CONFIG = RunConfig(
    max_workers=JUDGE_MAX_WORKERS, max_retries=10, max_wait=90, timeout=900
)


class BgeEmbeddings(BaseRagasEmbeddings):
    """bge via sentence-transformers for Ragas. (ragas 0.4.3's HuggingfaceEmbeddings can't be
    instantiated: its async methods are abstract.)"""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    @cached_property
    def encoder(self) -> SentenceTransformer:
        # The app's cached model: no second ~440 MB copy in the eval process.
        return embedding_service.get_model(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


def _get_ragas_llm(model: str | None = None) -> BaseRagasLLM:
    model = model or settings.eval_judge_model
    chat = ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.cerebras_api_key.get_secret_value()),
        base_url=settings.cerebras_base_url,
        temperature=0,
        max_retries=6,  # 429s from the 5 requests/minute limit are expected; back off and retry
        # Reasoning tokens are billed (Cerebras free trial = a $5 credit). gpt-oss reasons at
        # "medium" unless told otherwise. qwen on Cerebras thinks silently by default (113 vs 8
        # output tokens on a tiny judge call), so its thinking is switched off; E1 (the
        # baseline) was judged with it on (user chose not to re-score, 2026-10-22).
        reasoning_effort=settings.llm_reasoning_effort
        if llm_service.is_reasoning_model(model)
        else "none",
    )
    # bypass_n: Ragas makes separate calls instead of one with n > 1 (answer_relevancy needs 3),
    # which not every OpenAI-compatible provider supports.
    return LangchainLLMWrapper(chat, bypass_n=True)


def _get_ragas_embeddings() -> BaseRagasEmbeddings:
    return BgeEmbeddings(settings.embedding_model)


def reference_from_keywords(keywords: list[str]) -> str:
    """The judge's reference: the bare keyword list.

    Tested on 2026-10-22: phrasings like "A correct answer covers: x, y" make the judge treat
    the reference as a meta-statement no context can support (context_recall always 0); the
    bare list scores 1.0 for a matching context and 0.0 for an unrelated one.
    """
    if not keywords:
        return "The assistant should decline or answer without using any documents."
    return ", ".join(keywords)


def build_dataset(rows: list[dict[str, Any]]) -> EvaluationDataset:
    """Rows -> Ragas samples (user_input, response, retrieved_contexts, reference)."""
    return EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": reference_from_keywords(row["golden_answer_keywords"]),
            }
            for row in rows
        ]
    )


def _clean(value: Any) -> float | None:
    """Ragas reports a failed judgement as NaN; store it as None (JSON null)."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else round(number, 4)


def run(
    rows: list[dict[str, Any]], judge_model: str | None = None
) -> list[dict[str, float | None]]:
    """One {metric: score} dict per row, in row order."""
    if not rows:
        return []
    result = evaluate(
        build_dataset(rows),
        metrics=_metrics(),
        llm=_get_ragas_llm(judge_model),
        embeddings=_get_ragas_embeddings(),
        raise_exceptions=False,  # one bad judgement shouldn't kill the whole run
        show_progress=True,
        run_config=JUDGE_RUN_CONFIG,
    )
    # evaluate() only returns an Executor when return_executor=True.
    frame = cast(EvaluationResult, result).to_pandas()
    return [
        {name: _clean(frame.iloc[i].get(name)) for name in METRIC_NAMES} for i in range(len(rows))
    ]
