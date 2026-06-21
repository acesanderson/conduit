from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import tiktoken
from pydantic import BaseModel, ConfigDict

from conduit.core.workflow.step import add_metadata, step
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.rolling_refine import RollingRefineSummarizer

if TYPE_CHECKING:
    from collections.abc import Sequence


class SummarizationProfile(BaseModel):
    """A published, eval-tested summarization recipe for one routing tier.

    By design there is no `guideline` field — configs in this object represent
    artifacts that have been validated by an eval run. Per-call guidelines
    travel on `_TextInput.guideline` and are honored by individual strategies
    that opt in. When a (strategy, guideline) pair is graduated via an eval,
    the guideline is baked into `config` (e.g. into the strategy's prompt
    field) and a new named profile is minted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    strategy_cls: type[SummarizationStrategy]
    config: dict[str, Any]


class RoutingSummarizer(SummarizationStrategy):
    """Meta-strategy that routes by input token count to a SummarizationProfile.

    Itself a SummarizationStrategy, so it slots into the eval matrix as one
    row (A/B-testable as a unit) and Siphon enrichers call it through the
    same `__call__(input, config)` surface as any concrete strategy.

    The router does no prompt mutation. It picks a profile and delegates;
    `input` (including any `guideline`) passes through unchanged.
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
        routing: list[tuple[int, SummarizationProfile]]
        tokenizer_encoding: str = "cl100k_base"

    config_model = Config

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)
        text = input.data

        tokenizer = tiktoken.get_encoding(cfg.tokenizer_encoding)
        token_count = len(tokenizer.encode(text))

        profile = self._select(token_count, cfg.routing)
        add_metadata("token_count", token_count)
        add_metadata("routed_profile", profile.name)
        add_metadata("routed_strategy", profile.strategy_cls.__name__)

        return await profile.strategy_cls()(input, profile.config)

    @staticmethod
    def _select(
        token_count: int,
        routing: Sequence[tuple[int, SummarizationProfile]],
    ) -> SummarizationProfile:
        for token_max, profile in routing:
            if token_count <= token_max:
                return profile
        # Routing tables should always include a catch-all final tier.
        # If none matched, fall through to the last entry.
        return routing[-1][1]


# --- Published production routing -------------------------------------------
# Held as data so post-eval swaps are one-line edits. Tier breakpoints come
# from the OneShot ECW sweep; see evals/STRATEGY.md "Published Routing
# Decision" for the full quality table and rationale.
#
# Guideline convention: only OneShotSummarizer and RollingRefineSummarizer
# currently honor input.guideline, and both apply it at the call that
# produces the user-facing output (OneShot: its single call; RollingRefine:
# a post-loop format pass). Any future strategy added to PRODUCTION_ROUTING
# must follow the same rule, or guidelines set by Siphon will be silently
# dropped.

_GPT_BYWATER = {
    "model": "gpt-oss:latest",
    "use_remote": True,
    "host_alias": "bywater",
    "use_cache": True,
}
_GEMMA_DEEPWATER = {
    "model": "gemma4:latest",
    "use_remote": True,
    "host_alias": "deepwater",
    "use_cache": True,
}

PRODUCTION_ROUTING: list[tuple[int, SummarizationProfile]] = [
    # Tier 1 upper bound dropped from 12K to 5K after empirical confirmation
    # of the gpt-oss ECW cliff. The 32-min YouTube video at 7993 tokens
    # (youtube:///Kf0rPU7zy7Q) produced hallucinated CTA boilerplate in the
    # 5K-12K range — matching the eval-measured 0.13 quality in that bin.
    # The 5K-30K range now routes to gemma4 (eval quality 0.60).
    (5_000, SummarizationProfile(
        name="tier1_oneshot_gpt_oss",
        strategy_cls=OneShotSummarizer,
        config=_GPT_BYWATER,
    )),
    (30_000, SummarizationProfile(
        name="tier2_oneshot_gemma4",
        strategy_cls=OneShotSummarizer,
        config=_GEMMA_DEEPWATER,
    )),
    # Tier 3 catch-all. Swap to a hybrid profile here post-rerun if it wins.
    (10**9, SummarizationProfile(
        name="tier3_rolling_refine_gemma4",
        strategy_cls=RollingRefineSummarizer,
        config=_GEMMA_DEEPWATER,
    )),
]
