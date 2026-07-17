"""
OnlineDataGenerator: adaptive data generation for rubric-based RL training.

Supports two schedules (wired by the trainer via ``data_generator.schedule``):

  * ``epoch`` (iterative) – ``modify_dataset_epoch(epoch)`` regenerates rubrics /
    constraints for the entire active dataset at the start of each epoch and
    persists overrides via ``AdaptiveDataset.set_override``. Samples whose
    combined rubric ends up empty are dropped from the active set.
  * ``step`` (online) – ``modify_batch(indices, batch)`` is called before each
    training step on the current (pre-split) batch. The DataProto is mutated
    in-place (so the current step's rollout + reward see the updates) and
    overrides are persisted so future epochs inherit them. Samples are never
    dropped mid-training.

Modes:
  - constraint                       : Step 1 + 2 + 3         (general + constraint rubric)
  - constraint_wrong_adaptive        : Like ``constraint`` (Step 1 + 2 + 3, general +
                                       constraint rubric) but Step 1 is run with the
                                       8B generator model instead of the 1.7B policy.
                                       The constraint judge therefore sees stronger
                                       base responses when deciding whether to add a
                                       constraint, while training still targets the
                                       policy. Pair with
                                       ``constraint_generation.txt``.
  - constraint_no_adaptive           : Step 2 + 3' only — every sample gets a constraint
                                       and a constraint-specific rubric, generated from
                                       the instruction alone (no policy rollout, no
                                       judgment, no adaptive stage). Pair with
                                       ``constraint_generation_always.txt``.
  - constraint_no_adaptive_random    : Like ``constraint_no_adaptive`` but only a random
                                       ``data_generator.constraint_random_ratio`` fraction
                                       of eligible samples per epoch receive a constraint;
                                       the rest pass through with their general rubric
                                       only (no constraint appended, no constraint
                                       rubric). Subset is reseeded from
                                       ``self._current_epoch`` so the same epoch picks
                                       the same samples (compatible with
                                       ``epoch_cache_dir``). Step 1 (policy rollout) is
                                       skipped. Pair with
                                       ``constraint_generation_always.txt``.
  - constraint_metric                : Like ``constraint`` (Step 1 + 2 + 3, general +
                                       constraint rubric) but the LLM ``<decision>`` judge
                                       is replaced by a metric gate: Step 1 rollouts are
                                       scored against the general rubric (same per-criterion
                                       ``judge`` protocol as the training reward), and a
                                       prompt is saturated when the cross-rollout std of the
                                       per-rollout weighted-mean score (0-100 scale) falls
                                       below ``data_generator.saturation_std_threshold``.
                                       Saturated prompts get an UNCONDITIONAL constraint
                                       append via ``_gen_constraint_always``. Pair with
                                       ``constraint_generation_always.txt``.
  - constraint_reset                 : Like ``constraint`` but the persisted prompt is
                                       re-based on the original instruction whenever a
                                       new constraint is added — ``X + c_prev`` becomes
                                       ``X + c_new`` instead of accumulating to
                                       ``X + c_prev + c_new``. Constraint rubrics are
                                       reset in lockstep on samples where a new
                                       constraint was generated this epoch; decision=no
                                       samples carry forward both prompt and rubric
                                       unchanged. Step 1 base rollout still reads the
                                       previously-persisted override so the judge sees
                                       saturation against the constraint the policy was
                                       actually trained on.
  - constraint_rewrite               : Step 1 + 3-rewrite + 2 (in this order). The judge
                                       sees the instruction + two rollouts and decides
                                       whether the model has hit its ceiling; on
                                       decision=yes it does NOT emit a constraint to
                                       append — instead it emits a fully rewritten
                                       instruction that folds the new constraint into
                                       the original prompt. Decision=yes samples have
                                       their persisted general rubric DISCARDED and a
                                       fresh general rubric is regenerated from
                                       scratch on the rewritten prompt. The rewritten
                                       prompt + new general rubric are persisted, so
                                       subsequent epochs see the rewritten prompt as
                                       the new "instruction" (and may rewrite again
                                       on top of it). No constraint-specific rubric
                                       or adaptive rubric is produced. Pair with
                                       ``rewrite_variant.txt``.
  - adaptive_conditional_rubric      : Step 1 + 5 only — no constraint pipeline.
                                       Step 5 runs on rollouts_A for every sample with
                                       ≥2 valid rollouts and uses the v2 adaptive
                                       template: the judge sees the persisted general +
                                       previously accumulated adaptive rubric under
                                       ``<existing_rubric>`` and emits its own
                                       ``<decision>yes|no</decision>``. Adaptive
                                       criteria accumulate across epochs;
                                       decision=no epochs leave the persisted adaptive
                                       rubric intact. Pair with
                                       ``policy_adaptive_rubric_generation.txt``.
  - eva_baseline_colocated           : EVA paper baseline. Self-
                                       contained pipeline that short-circuits
                                       ``modify_batch`` after Step 1 — no constraint /
                                       adaptive / refine stages run. Scores rollouts
                                       with the scalar RM (ArmoRM-8B),
                                       weighted-samples K = round(N * eva_top_k_ratio)
                                       high-info prompts, runs M Evol-Instruct
                                       evolutions per selected prompt over the 5
                                       distilabel methods, then 80/20-mixes the
                                       evolved pool back into the batch. General
                                       rubrics are regenerated only for slots whose
                                       prompt was replaced by an evolved one.

Failures (parse / rollout short) retry up to ``data_generator.max_retry``. The
general rubric is generated once per sample and then persisted; subsequent
visits skip Step 2. To avoid losing samples to general-rubric parse failures,
run ``data_generation/offline_general_rubric.py`` first to pre-seed the
``requirements_general`` / ``weights_general`` columns on the training parquet.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Hashable

import aiohttp
import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from verl import DataProto
from llm_tutor._parse import (
    is_valid_rubric,
    join_prompt_constraint,
    parse_analysis,
    parse_constraint,
    parse_constraint_judgment,
    parse_constraint_rewrite_judgment,
    parse_decision,
    parse_rubric_output,
    parse_rubric_output_lenient,
    split_thinking,
)
from llm_tutor.metric_saturation_gate import (
    aggregate_saturation,
    build_score_inputs,
    parse_score_fallback,
    parse_score_strict,
)

if TYPE_CHECKING:
    from llm_tutor.adaptive_dataset import AdaptiveDataset

logger = logging.getLogger(__name__)

_VALID_MODES = {
    "constraint",
    "constraint_reset",
    "constraint_wrong_adaptive",
    "constraint_no_adaptive",
    "constraint_no_adaptive_random",
    "constraint_metric",
    "constraint_rewrite",
    "adaptive_conditional_rubric",
    "eva_baseline_colocated",
}
_MODES_WITH_CONSTRAINT = {
    "constraint",
    "constraint_reset",
    "constraint_wrong_adaptive",
    "constraint_no_adaptive",
    "constraint_no_adaptive_random",
    "constraint_metric",
}
# Metric-based saturation gate: instead of the LLM ``<decision>`` judge,
# ``constraint_metric`` scores the Step 1 rollouts against the general rubric
# (same per-criterion ``judge`` protocol as the training reward), computes the
# cross-rollout std of the per-rollout weighted-mean score, and treats prompts
# with ``std < data_generator.saturation_std_threshold`` as saturated. Saturated
# prompts then get an unconditional constraint append via
# ``_gen_constraint_always`` (paired with ``constraint_generation_always.txt``).
# Needs the base rollout (Step 1), so it is NOT in ``_MODES_WITHOUT_BASE_ROLLOUT``.
_MODES_WITH_METRIC_GATE = {
    "constraint_metric",
}
# Modes whose epoch-to-epoch persisted prompt drops the previous constraint
# before joining the new one — i.e. ``X + c1 + c2`` becomes ``X + c2`` rather
# than accumulating. Step 1 base rollout still reads the override
# (``X + c_prev``) so the judge can detect saturation against the constraint
# the policy was actually trained on; only the persist step (Step 7
# ``new_prompt_str``) re-bases on the original prompt. Constraint rubrics
# are reset in lockstep with the prompt, but only on samples where a new
# constraint was successfully generated this epoch — decision=no samples
# carry forward both prompt and rubric unchanged.
_MODES_WITH_CONSTRAINT_RESET = {
    "constraint_reset",
}
# Modes that skip the base policy rollout (Step 1) and run a rollout-free
# constraint generator that always emits decision=yes from the instruction
# alone. The constraint pipeline (decision/constraint accumulation, prompt
# join, override) still runs — only the upstream rollout and judgment-vs-
# rollout coupling are bypassed.
_MODES_WITHOUT_BASE_ROLLOUT = {
    "constraint_no_adaptive",
    "constraint_no_adaptive_random",
}
# Modes that randomly sub-sample which prompts receive a constraint this epoch.
# Stage 3 picks ``data_generator.constraint_random_ratio`` of the eligible
# samples (seeded by ``self._current_epoch`` so the same epoch always picks the
# same subset, which keeps ``epoch_cache_dir`` reuse honest), then runs the
# usual no-rollout ``_gen_constraint_always`` on the chosen subset. The
# unselected samples pass through with general rubric only — no constraint
# appended, no constraint rubric.
_MODES_WITH_RANDOM_DECISION = {
    "constraint_no_adaptive_random",
}
# Modes whose Step 1 base response is generated by the 8B generator model
# (via the same HTTP path the rubric stages use) rather than the policy
# rollout manager. The constraint judge then sees responses from the
# stronger generator when deciding whether to add a constraint. Step 1 is
# moved inside the existing GEN wake block since the generator must be
# awake to serve these requests.
_MODES_WITH_GENERATOR_BASE_ROLLOUT = {
    "constraint_wrong_adaptive",
}
_MODES_WITH_ADAPTIVE = {
    "adaptive_conditional_rubric",
}
_MODES_WITH_CONSTRAINT_RUBRIC = {
    "constraint",
    "constraint_reset",
    "constraint_wrong_adaptive",
    "constraint_no_adaptive",
    "constraint_no_adaptive_random",
    "constraint_metric",
}
# Modes where adaptive rubrics are *accumulated* across epochs instead of replaced.
_MODES_WITH_ADAPTIVE_APPEND = {
    "adaptive_conditional_rubric",
}
# EVA-style baseline: scalar RM + Evol-Instruct rewrites. Self-contained
# pipeline that short-circuits ``modify_batch`` after Step 1 — no
# constraint / adaptive / refine stages run.
_MODES_WITH_EVA = {"eva_baseline_colocated"}
# Modes whose adaptive template emits <decision>yes|no</decision> plus an
# optional <rubric> block (``policy_adaptive_rubric_generation.txt``). The judge
# sees the currently persisted general+constraint rubric under
# ``<existing_rubric>`` and self-gates whether to append new criteria.
_MODES_WITH_ADAPTIVE_DECISION = {
    "adaptive_conditional_rubric",
}
# Modes that REWRITE the prompt (instead of appending a constraint).
# When the judge says yes, the entire instruction is replaced with a
# rewritten version that incorporates the new constraint inline, the
# previously persisted general rubric is discarded, and a fresh general
# rubric is generated from scratch on the rewritten prompt. The next
# epoch sees the rewritten prompt as the new "instruction" and may
# rewrite again. No constraint-specific or adaptive rubric is produced.
# Pair with ``rewrite_variant.txt``.
_MODES_WITH_REWRITE = {"constraint_rewrite"}
_VALID_SCHEDULES = {"epoch", "step"}


class OnlineDataGenerator:
    def __init__(
        self,
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        mode: str = "constraint",
        schedule: str = "epoch",
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}; expected one of {_VALID_MODES}")
        if schedule not in _VALID_SCHEDULES:
            raise ValueError(f"Invalid schedule {schedule!r}; expected one of {_VALID_SCHEDULES}")

        self._config = config
        self._tokenizer = tokenizer
        self._mode = mode
        self._schedule = schedule

        self._rollout_mgr = None
        self._rm_manager = None
        self._router_addr: str | None = None
        self._dataset: AdaptiveDataset | None = None

        dg_cfg = config.get("data_generator", config)
        self._model_path: str | None = dg_cfg.get("model_path", None)
        self._max_retry: int = int(dg_cfg.get("max_retry", 5))
        # Rollout-specific retry knobs. Decoupled from ``max_retry`` (which
        # governs parse-validation retries on judge calls) because rollout
        # failures are usually response-length truncation — re-rolling the
        # same prompt with the same sampling settings tends to truncate
        # again, so wall-time is dominated by repeated vLLM batch setup for
        # a stuck minority. ``rollout_retry_oversample`` lets retry rounds
        # request more samples per prompt in a single batch, increasing the
        # chance of clearing the sample in one extra round; pair with a
        # smaller ``rollout_max_retry`` so total wall-time drops.
        #   * rollout_retry_oversample (float, default 1.0):
        #       n_attempt = n on the first attempt; n_attempt =
        #       max(n, round(n * factor)) on every retry. 1.0 preserves
        #       legacy behaviour exactly.
        #   * rollout_max_retry (int, default = max_retry):
        #       cap on retry rounds for ``_gen_rollouts`` only. Parse-
        #       validation retries elsewhere keep using ``max_retry``.
        self._rollout_retry_oversample: float = float(dg_cfg.get("rollout_retry_oversample", 1.0))
        if self._rollout_retry_oversample < 1.0:
            raise ValueError(
                f"data_generator.rollout_retry_oversample must be >= 1.0; got {self._rollout_retry_oversample!r}"
            )
        rollout_max_retry_raw = dg_cfg.get("rollout_max_retry", None)
        if rollout_max_retry_raw is None:
            self._rollout_max_retry: int = self._max_retry
        else:
            self._rollout_max_retry = int(rollout_max_retry_raw)
            if self._rollout_max_retry < 0:
                raise ValueError(f"data_generator.rollout_max_retry must be >= 0; got {rollout_max_retry_raw!r}")
        # Generator-side retry knobs. Apply to every parse-validated
        # generator call routed through ``_generate_with_retry_keyed`` —
        # decision, rewrite, general_rubric (regen), constraint_judgment,
        # adaptive_rubric, judge_*. Generator calls tend to truncate on
        # closing tags; with strict parse, a stuck prompt walks through
        # every retry round one sample at a time. Oversample on retry
        # batches K duplicates of each pending prompt in a single vLLM
        # call and accepts the first parse-success — same total sample
        # budget as ``generator_max_retry`` sequential retries, but folded
        # into one wake/sleep round. 1.0 / max_retry preserves legacy.
        self._generator_retry_oversample: float = float(dg_cfg.get("generator_retry_oversample", 1.0))
        if self._generator_retry_oversample < 1.0:
            raise ValueError(
                f"data_generator.generator_retry_oversample must be >= 1.0; got {self._generator_retry_oversample!r}"
            )
        generator_max_retry_raw = dg_cfg.get("generator_max_retry", None)
        if generator_max_retry_raw is None:
            self._generator_max_retry: int = self._max_retry
        else:
            self._generator_max_retry = int(generator_max_retry_raw)
            if self._generator_max_retry < 0:
                raise ValueError(f"data_generator.generator_max_retry must be >= 0; got {generator_max_retry_raw!r}")
        self._max_tokens: int = int(dg_cfg.get("max_tokens", 8192))
        self._temperature: float = float(dg_cfg.get("temperature", 0.6))
        self._top_p: float = float(dg_cfg.get("top_p", 0.95))
        self._top_k: int | None = dg_cfg.get("top_k", None)
        self._enable_thinking: bool = bool(dg_cfg.get("enable_thinking", True))
        self._n_rollouts: int = int(dg_cfg.get("n_rollouts", 2))
        # ``constraint_metric``: a prompt is saturated when the cross-rollout std
        # of the per-rollout general-rubric weighted-mean score (0-100 scale) is
        # below this threshold. Default 3.0 (cf. the ~6 base-model rollout-std).
        # Ignored by every other mode.
        self._saturation_std_threshold: float = float(dg_cfg.get("saturation_std_threshold", 3.0))
        # Minimum valid rollouts a prompt needs before the metric gate scores it.
        self._metric_min_valid_rollouts: int = int(dg_cfg.get("metric_min_valid_rollouts", 2))

        # Cap on concurrent generator HTTP requests to the local vLLM router.
        self._concurrency_limit: int = int(dg_cfg.get("concurrency_limit", 0))

        parse_patterns = dg_cfg.get("parse_patterns", {}) or {}
        self._parse_patterns: dict[str, str] = {k: str(v) for k, v in parse_patterns.items()}

        self._template_paths = {
            "general_rubric": dg_cfg.get("general_rubric_prompt"),
            "constraint_judgment": dg_cfg.get("constraint_judgment_prompt"),
            "constraint_rewrite": dg_cfg.get("constraint_rewrite_prompt"),
            "adaptive_rubric": dg_cfg.get("adaptive_rubric_prompt"),
            # constraint_metric: per-criterion scoring template (same protocol as
            # the training reward judge). Defaults to the shipped judge.txt so the
            # gate's scores match the reward's; override via metric_judge_prompt.
            "metric_judge": dg_cfg.get("metric_judge_prompt", "templates/template_files/judge.txt"),
        }
        self._template_cache: dict[str, str] = {}

        # ── EVA baseline knobs ───────────────────────────────────────────────
        # Per the EVA paper:
        #   * ``eva_top_k_ratio``           — fraction of N weighted-sampled by
        #                                     info-score (default 0.25).
        #   * ``eva_evolutions_per_prompt`` — M evolutions per high-info prompt
        #                                     (default 4 → pool size = K*M).
        #   * ``eva_evolved_mix_ratio``     — fraction of next batch sampled
        #                                     from the evolved pool (default
        #                                     0.8); the rest passes through
        #                                     unchanged.
        # The reward model is ArmoRM-8B; the 5 distilabel methods live under
        # ``evol_instruct_dir`` (one .txt per method).
        self._eva_top_k_ratio: float = float(dg_cfg.get("eva_top_k_ratio", 0.25))
        self._eva_evolutions_per_prompt: int = int(dg_cfg.get("eva_evolutions_per_prompt", 4))
        self._eva_evolved_mix_ratio: float = float(dg_cfg.get("eva_evolved_mix_ratio", 0.8))
        self._eva_rng = np.random.default_rng()
        self._eva_methods: tuple[str, ...] = (
            "constraints",
            "deepen",
            "concretizing",
            "reasoning",
            "breadth",
        )
        self._evol_instruct_templates: dict[str, str] = {}
        if mode in _MODES_WITH_EVA:
            evol_dir = dg_cfg.get("evol_instruct_dir", None)
            if not evol_dir:
                raise ValueError(f"mode={mode!r} requires data_generator.evol_instruct_dir to be set")
            evol_dir_path = Path(evol_dir)
            for name in self._eva_methods:
                fp = evol_dir_path / f"{name}.txt"
                if not fp.is_file():
                    raise ValueError(f"missing distilabel template: {fp}")
                self._evol_instruct_templates[name] = fp.read_text(encoding="utf-8")

        if mode in _MODES_WITH_REWRITE and not self._template_paths.get("constraint_rewrite"):
            raise ValueError(f"mode={mode!r} requires data_generator.constraint_rewrite_prompt to be set")

        if mode in _MODES_WITH_METRIC_GATE:
            if not self._template_paths.get("constraint_judgment"):
                raise ValueError(
                    f"mode={mode!r} requires data_generator.constraint_judgment_prompt to be set "
                    "(use the gating-free constraint_generation_always.txt)"
                )
            if not self._template_paths.get("metric_judge"):
                raise ValueError(f"mode={mode!r} requires data_generator.metric_judge_prompt to be set")
            if self._saturation_std_threshold < 0:
                raise ValueError(
                    f"data_generator.saturation_std_threshold must be >= 0; got {self._saturation_std_threshold!r}"
                )

        # Fraction of eligible samples that receive a constraint each epoch in
        # ``_MODES_WITH_RANDOM_DECISION``. Optional everywhere else (left as
        # None and unused), required and validated for those modes here.
        ratio_raw = dg_cfg.get("constraint_random_ratio", None)
        if ratio_raw is None:
            self._constraint_random_ratio: float | None = None
        else:
            ratio = float(ratio_raw)
            if not (0.0 <= ratio <= 1.0):
                raise ValueError(f"data_generator.constraint_random_ratio must be in [0, 1]; got {ratio_raw!r}")
            self._constraint_random_ratio = ratio
        if mode in _MODES_WITH_RANDOM_DECISION and self._constraint_random_ratio is None:
            raise ValueError(f"mode={mode!r} requires data_generator.constraint_random_ratio in [0, 1]")

        # Cached data-side knobs used by the constraint-length guard in Step 3.
        # ``max_prompt_length`` is None in minimal test configs (no ``data``
        # block); the guard is a no-op in that case.
        data_cfg = config.get("data", None) if hasattr(config, "get") else None
        if data_cfg is not None:
            self._max_prompt_length: int | None = (
                int(data_cfg.get("max_prompt_length")) if data_cfg.get("max_prompt_length") is not None else None
            )
            self._apply_chat_template_kwargs: dict = dict(data_cfg.get("apply_chat_template_kwargs", {}) or {})
        else:
            self._max_prompt_length = None
            self._apply_chat_template_kwargs = {}

        # Debug dump: a directory. Per-stage JSONL files are written inside
        # (constraint_judgment.jsonl, general_rubric.jsonl, adaptive_rubric.jsonl).
        dump_dir = dg_cfg.get("debug_dir", None)
        self._debug_dump_dir: Path | None = Path(dump_dir) if dump_dir else None

        # Per-epoch dataset-state cache. When set, ``modify_dataset_epoch(N)``
        # writes the post-generation adaptive state (overrides + active_indices)
        # to ``{cache_dir}/epoch_{N}.pt`` after generating, and on subsequent
        # calls for the same epoch loads the file and skips generation. This
        # makes resumes idempotent: if a run is killed and restarted, any epoch
        # whose generation already finished is reused from disk instead of
        # being re-run against the 8B generator.
        cache_dir = dg_cfg.get("epoch_cache_dir", None)
        if cache_dir is None and self._debug_dump_dir is not None:
            cache_dir = self._debug_dump_dir / "epoch_state"
        self._epoch_cache_dir: Path | None = Path(cache_dir) if cache_dir else None

        # Current training epoch, stamped into every debug record so the dumped
        # JSONL can be split by epoch after-the-fact. Updated by
        # ``modify_dataset_epoch`` (epoch schedule) and ``set_epoch`` (step
        # schedule, called by the trainer).
        self._current_epoch: int = 0
        # Current global training step. Only meaningful in step schedule —
        # the trainer is expected to call ``set_step(global_steps)`` right
        # before ``modify_batch`` so summary records can be uniquely keyed
        # by (epoch, step). Stays at 0 in epoch schedule (epoch already
        # uniquely identifies the call there).
        self._current_step: int = 0

        logger.info(
            "OnlineDataGenerator initialised (mode=%s, schedule=%s, n_rollouts=%d)",
            mode,
            schedule,
            self._n_rollouts,
        )

    # ── Initialisation ──────────────────────────────────────────────────────

    @property
    def schedule(self) -> str:
        return self._schedule

    def set_managers(self, async_rollout_manager, reward_loop_manager) -> None:
        self._rollout_mgr = async_rollout_manager
        # ``reward_model_manager`` may be None when the reward judge is disabled;
        # ``_wake_rm`` / ``_sleep_rm`` below treat None as a no-op.
        self._rm_manager = reward_loop_manager.reward_model_manager
        self._router_addr = reward_loop_manager.reward_router_address

    def _wake_rm(self) -> None:
        if self._rm_manager is not None:
            self._rm_manager.wake_up()

    def _sleep_rm(self) -> None:
        if self._rm_manager is not None:
            self._rm_manager.sleep()

    def set_dataset(self, dataset: AdaptiveDataset) -> None:
        self._dataset = dataset

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def set_step(self, step: int) -> None:
        """Stamp ``step`` (typically ``trainer.global_steps``) onto subsequent
        debug records. Trainer should call this right before ``modify_batch``
        in step schedule so per-call summaries are uniquely keyed."""
        self._current_step = int(step)

    # ── Main entry points ──────────────────────────────────────────────────

    def _epoch_cache_path(self, epoch: int) -> Path | None:
        if self._epoch_cache_dir is None:
            return None
        return self._epoch_cache_dir / f"epoch_{epoch}.pt"

    def _compute_cache_signature(self) -> dict:
        """Signature of generator + dataset config that the cache depends on.

        Any change in these fields means a previously cached epoch state was
        produced under a different setup; reusing it would silently train on
        stale / wrong data. Saved alongside the state and verified on load —
        mismatch triggers regeneration (with a warning log).

        Covers:
          * ``mode`` — different modes generate different rubric sets.
          * Template file contents (MD5) — editing a prompt changes outputs.
          * Generation hyperparams (rollouts / sampling / thinking / dedup /
            parse patterns / model path) — anything that steers the generator.
          * Dataset fingerprint (train_files + size) — catches swapped
            parquets (the common source of IndexError / wrong-row mapping).
        """
        import hashlib

        template_hashes: dict[str, str | None] = {}
        for key, path in self._template_paths.items():
            if not path:
                template_hashes[key] = None
                continue
            try:
                template_hashes[key] = hashlib.md5(Path(path).read_bytes()).hexdigest()
            except OSError:
                template_hashes[key] = None

        train_files = None
        data_cfg = self._config.get("data", None) if hasattr(self._config, "get") else None
        if data_cfg is not None:
            tf = data_cfg.get("train_files", None)
            train_files = str(tf) if tf is not None else None

        dataset_size = len(self._dataset.dataframe) if self._dataset is not None else None

        return {
            "mode": self._mode,
            "n_rollouts": self._n_rollouts,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "enable_thinking": self._enable_thinking,
            "model_path": self._model_path,
            "parse_patterns": dict(self._parse_patterns),
            "template_hashes": template_hashes,
            "train_files": train_files,
            "dataset_size": dataset_size,
            "constraint_random_ratio": self._constraint_random_ratio,
        }

    @staticmethod
    def _signature_diff(old: dict | None, new: dict) -> list[str]:
        """Return the list of top-level keys whose values differ (or are missing)."""
        if old is None:
            return ["<no signature in cached file>"]
        changed: list[str] = []
        for k in sorted(set(old.keys()) | set(new.keys())):
            if old.get(k) != new.get(k):
                changed.append(k)
        return changed

    def modify_dataset_epoch(self, epoch: int) -> int:
        """Epoch-boundary pipeline: regenerate for all active samples and drop
        samples whose combined rubric ended up empty.

        Thin wrapper around ``modify_batch`` — the actual generation logic
        lives there so the per-step path reuses the same code.

        If ``epoch_cache_dir`` is configured and a cache file for this epoch
        already exists AND its signature matches the current run, the cached
        adaptive state is loaded onto the dataset and generation is skipped
        entirely. A signature mismatch (changed mode / template / hyperparam
        / dataset) logs a warning and falls through to regeneration so the
        trainer never silently reuses stale data.
        """
        import torch

        assert self._dataset is not None, "Dataset not set — call set_dataset() first"

        self.set_epoch(epoch)

        cache_path = self._epoch_cache_path(epoch)
        if cache_path is not None and cache_path.exists():
            try:
                cached = torch.load(cache_path, weights_only=False)
            except Exception as e:  # corrupted / unreadable pickle — don't crash training
                logger.warning(
                    "Epoch %d: failed to read cache at %s (%s); regenerating.",
                    epoch,
                    cache_path,
                    e,
                )
                cached = None

            if isinstance(cached, dict) and "state" in cached and "signature" in cached:
                current_sig = self._compute_cache_signature()
                cached_sig = cached.get("signature")
                if cached_sig == current_sig:
                    state = cached["state"]
                    active = state.get("active_indices") or []
                    max_idx = max(active) if active else -1
                    df_size = len(self._dataset.dataframe)
                    if max_idx >= df_size:
                        logger.warning(
                            "Epoch %d: cache active_indices out of range " "(max=%d, dataset_size=%d); regenerating.",
                            epoch,
                            max_idx,
                            df_size,
                        )
                    else:
                        self._dataset.load_adaptive_state(state)
                        n_active = len(self._dataset)
                        logger.info(
                            "Epoch %d: loaded cached dataset state from %s "
                            "(overrides=%d, active=%d) — skipping generation",
                            epoch,
                            cache_path,
                            len(state.get("row_overrides", {})),
                            n_active,
                        )
                        return n_active
                else:
                    logger.warning(
                        "Epoch %d: cache at %s has mismatched signature; " "regenerating. Changed: %s",
                        epoch,
                        cache_path,
                        self._signature_diff(cached_sig, current_sig),
                    )
            elif cached is not None:
                logger.warning(
                    "Epoch %d: cache at %s is in a legacy / unexpected format; " "regenerating.",
                    epoch,
                    cache_path,
                )

        N = len(self._dataset)
        if N == 0:
            logger.warning("Epoch %d: dataset is empty, nothing to modify", epoch)
            return 0

        indices = list(range(N))
        logger.info("Epoch %d: modify_batch over full dataset (%d samples)", epoch, N)
        n_modified = self.modify_batch(indices)

        # Drop samples whose combined rubric ended up empty. With offline
        # general-rubric pre-seeding this should never fire; it is the last
        # safety net for a sample whose general generation failed all retries
        # (and had no pre-seed), which would otherwise leave it with no
        # reward signal for the rest of training.
        drop_real: list[int] = []
        for i in indices:
            real_idx = int(self._dataset[i]["_dataset_idx"])
            persisted = self._dataset.get_accumulated_rubrics(real_idx)
            combined = persisted["general"] + persisted["constraint"] + persisted["adaptive"]
            if not combined:
                drop_real.append(real_idx)
        if drop_real:
            logger.info("Epoch %d: dropping %d samples with empty rubric", epoch, len(drop_real))
            self._dataset.drop_samples(drop_real)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "signature": self._compute_cache_signature(),
                    "state": self._dataset.get_adaptive_state(),
                },
                cache_path,
            )
            logger.info("Epoch %d: cached dataset state to %s", epoch, cache_path)

        return n_modified

    def modify_batch(
        self,
        indices: list[int],
        batch: DataProto | None = None,
    ) -> int:
        """Core generation pipeline.

        Args:
            indices: logical dataset indices to process (stable across epochs
                because ``AdaptiveDataset`` requires ``shuffle=false``).
            batch: when provided (step schedule), this batch's
                ``non_tensor_batch`` columns (``raw_prompt`` /
                ``requirements`` / ``weights`` / ``prompt``) are mutated in
                place at the positions corresponding to ``indices``. The hook
                fires BEFORE ``_get_gen_batch`` so writing to the single
                combined batch is enough — the subsequent split distributes
                each column to the side that needs it.

        Overrides are always persisted to ``AdaptiveDataset`` so future epochs
        inherit the changes. Returns the number of samples with a non-empty
        combined rubric after this call (used for logging only).

        Rubric persistence policy (per group):
          * ``general``    – generated the first time a sample is seen without
                             a persisted general rubric, then reused. Pre-seed
                             via the offline script to guarantee every sample
                             has one from the start.
          * ``constraint`` – accumulated: each fresh constraint judgment that
                             returns decision=yes *appends* its rubric on top
                             of whatever constraint rubrics were already
                             persisted. This mirrors the prompt side (the new
                             constraint is also joined onto the override
                             ``raw_prompt`` so constraints accumulate there
                             too), so a later decision=no epoch never silently
                             drops the rubric for a constraint still in the
                             prompt.
          * ``adaptive``   – replaced on every call by default (stale adaptive
                             criteria from a previous policy snapshot must not
                             linger). For modes in
                             ``_MODES_WITH_ADAPTIVE_APPEND`` the fresh adaptive
                             rubric is instead *appended* on top of the
                             previously persisted adaptive rubric, so criteria
                             accumulate across epochs.
        """
        assert self._dataset is not None, "Dataset not set — call set_dataset() first"
        assert self._rollout_mgr is not None, "Rollout manager not set"

        N = len(indices)
        if N == 0:
            return 0

        # Extract instruction string + real dataset index per sample, and seed
        # persisted rubric state: general / constraint always carry forward;
        # adaptive only carries forward for append modes (otherwise it is
        # replaced on every call).
        # ``original_prompts`` is the immutable base prompt (X) read directly
        # from the dataframe, bypassing any accumulated override. Reset modes
        # use it as the join target at Step 7 so a successful new constraint
        # ``c_new`` produces ``X + c_new`` rather than ``X + c_prev + c_new``.
        is_reset_mode = self._mode in _MODES_WITH_CONSTRAINT_RESET
        prompts: list[str] = []
        original_prompts: list[str] = []
        dataset_idxs: list[int] = []
        general_rubrics: dict[int, list[dict]] = {}
        constraint_rubrics: dict[int, list[dict]] = {}
        adaptive_rubrics_persisted: dict[int, list[dict]] = {}
        for pos, logical_idx in enumerate(indices):
            row = self._dataset[logical_idx]
            prompts.append(self._extract_instruction(row["raw_prompt"]))
            real_idx = int(row["_dataset_idx"])
            dataset_idxs.append(real_idx)
            if is_reset_mode:
                original_prompts.append(self._dataset.get_original_prompt(real_idx))
            persisted = self._dataset.get_accumulated_rubrics(real_idx)
            if persisted["general"]:
                general_rubrics[pos] = list(persisted["general"])
            if persisted["constraint"]:
                constraint_rubrics[pos] = list(persisted["constraint"])
            if self._mode in _MODES_WITH_ADAPTIVE_APPEND and persisted["adaptive"]:
                adaptive_rubrics_persisted[pos] = list(persisted["adaptive"])

        # ── Step 1: POLICY rollout on base prompts ─────────────────────────
        # ``_MODES_WITH_GENERATOR_BASE_ROLLOUT`` defers Step 1 into the GEN
        # wake block below — the 8B generator must be awake to serve those
        # requests, so we can't run them here outside the wake/sleep scope.
        if self._mode in _MODES_WITHOUT_BASE_ROLLOUT:
            logger.info("[Step 1] skipped (mode=%s does not use base rollouts)", self._mode)
            rollouts_A: list[list[str]] = [[] for _ in range(N)]
            step1_failed: set[int] = set()
        elif self._mode in _MODES_WITH_GENERATOR_BASE_ROLLOUT:
            logger.info("[Step 1] deferred to GEN wake (mode=%s uses 8B generator for base)", self._mode)
            rollouts_A = [[] for _ in range(N)]
            step1_failed = set()
        else:
            logger.info("[Step 1] policy rollout on %d samples", N)
            rollouts_A = self._gen_rollouts(indices, n=self._n_rollouts, min_valid=2)
            step1_failed = {pos for pos, r in enumerate(rollouts_A) if len(r) < 2}
            logger.info("[Step 1] rollout ok: %d/%d", N - len(step1_failed), N)

        # ── EVA baseline: scalar-RM scoring + Evol-Instruct ─────────────────
        # Self-contained pipeline that short-circuits the rest of
        # ``modify_batch`` (no constraint / adaptive / refine stages run for
        # EVA). The non-evolved rest of the batch keeps its original prompt
        # and persisted general rubric.
        if self._mode in _MODES_WITH_EVA:
            return self._modify_batch_eva(
                indices=indices,
                batch=batch,
                prompts=prompts,
                dataset_idxs=dataset_idxs,
                rollouts_A=rollouts_A,
                step1_failed=step1_failed,
                general_rubrics=general_rubrics,
            )

        decisions: dict[int, bool] = {}
        constraints: dict[int, str] = {}
        adaptive_rubrics: dict[int, list[dict]] = {}

        # ── GEN wake #1: general rubric (missing only) + constraint judgment ─
        self._wake_rm()
        try:
            # Generator-based Step 1: ``constraint_wrong_adaptive`` and
            # friends produce the "base" responses with the 8B generator
            # rather than the policy. Must run inside this wake block.
            if self._mode in _MODES_WITH_GENERATOR_BASE_ROLLOUT:
                logger.info("[Step 1] generator base response on %d samples", N)
                rollouts_A = self._gen_generator_base_responses(prompts, n=self._n_rollouts)
                step1_failed = {pos for pos, r in enumerate(rollouts_A) if len(r) < 2}
                logger.info("[Step 1] base response ok: %d/%d", N - len(step1_failed), N)

            # Rewrite stage runs BEFORE general-rubric generation: a rewrite
            # invalidates the persisted general rubric (different prompt now)
            # and we don't want to generate it twice. Updates ``prompts`` in
            # place and clears persisted general for any sample that gets
            # rewritten this epoch, so the subsequent general-rubric step
            # regenerates only the cleared ones.
            if self._mode in _MODES_WITH_REWRITE:
                rewrite_eligible = [pos for pos in range(N) if pos not in step1_failed]
                logger.info(
                    "[Step 3-rewrite] rewrite judgment for %d eligible samples",
                    len(rewrite_eligible),
                )
                rewrite_decisions, rewritten = self._gen_constraint_rewrite_judgment(
                    prompts,
                    rollouts_A,
                    rewrite_eligible,
                )
                # Length guard: drop rewrites whose tokenized prompt would
                # exceed ``data.max_prompt_length``.
                if self._max_prompt_length is not None:
                    dropped_overlong = 0
                    for pos in list(rewritten.keys()):
                        if not rewrite_decisions.get(pos, False) or not rewritten.get(pos):
                            continue
                        if self._tokenized_prompt_len(rewritten[pos]) > self._max_prompt_length:
                            dropped_overlong += 1
                            rewrite_decisions[pos] = False
                            rewritten.pop(pos, None)
                    if dropped_overlong:
                        logger.warning(
                            "[Step 3-rewrite] dropped %d rewrite(s) that would exceed max_prompt_length=%d",
                            dropped_overlong,
                            self._max_prompt_length,
                        )
                applied = 0
                for pos, new_p in rewritten.items():
                    if not rewrite_decisions.get(pos, False) or not new_p:
                        continue
                    prompts[pos] = new_p
                    general_rubrics.pop(pos, None)
                    applied += 1
                logger.info("[Step 3-rewrite] applied %d rewrite(s)", applied)

            missing_general = [pos for pos in range(N) if pos not in general_rubrics]
            if missing_general:
                logger.info(
                    "[Step 2] generating general rubric for %d sample(s) (persisted=%d)",
                    len(missing_general),
                    N - len(missing_general),
                )
                fresh_general = self._gen_general_rubric(prompts, missing_general)
                for pos, pairs in fresh_general.items():
                    general_rubrics[pos] = pairs
            else:
                logger.info("[Step 2] general rubric fully persisted (%d samples)", N)

            if self._mode in _MODES_WITH_CONSTRAINT:
                eligible = [pos for pos in range(N) if pos not in step1_failed]
                if self._mode in _MODES_WITH_RANDOM_DECISION:
                    n_pool = len(eligible)
                    n_pick = int(round(n_pool * self._constraint_random_ratio))
                    rng = random.Random(self._current_epoch)
                    eligible = sorted(rng.sample(eligible, n_pick)) if n_pick > 0 else []
                    logger.info(
                        "[Step 3-random] sampled %d/%d eligible (ratio=%.3f, epoch=%d)",
                        len(eligible),
                        n_pool,
                        self._constraint_random_ratio,
                        self._current_epoch,
                    )
                if self._mode in _MODES_WITH_METRIC_GATE:
                    # Metric gate replaces the LLM decision judge: score the
                    # Step 1 rollouts against the general rubric, keep prompts
                    # whose cross-rollout std is below threshold, then append a
                    # constraint to those unconditionally.
                    saturated = self._metric_saturation_gate(prompts, rollouts_A, general_rubrics, eligible, indices)
                    logger.info(
                        "[Step 3] constraint generation (metric-gated) for %d/%d saturated samples",
                        len(saturated),
                        len(eligible),
                    )
                    decisions, constraints, new_constraint_rubrics = self._gen_constraint_always(
                        prompts,
                        saturated,
                    )
                elif self._mode in _MODES_WITHOUT_BASE_ROLLOUT:
                    logger.info(
                        "[Step 3] constraint generation (no rollout) for %d samples",
                        len(eligible),
                    )
                    decisions, constraints, new_constraint_rubrics = self._gen_constraint_always(
                        prompts,
                        eligible,
                    )
                else:
                    logger.info(
                        "[Step 3] constraint judgment for %d eligible samples",
                        len(eligible),
                    )
                    decisions, constraints, new_constraint_rubrics = self._gen_constraint_judgment(
                        prompts,
                        rollouts_A,
                        eligible,
                    )
                # Drop constraints whose concatenated prompt would tokenize
                # past ``data.max_prompt_length``. The agent_loop pads prompt
                # ids to that limit but does NOT truncate, so an overlong
                # sample causes a torch.cat size mismatch in _postprocess
                # (e.g. 8196 vs 8192). Falling back to the un-constrained
                # prompt keeps the Step 7 persist path consistent.
                if self._max_prompt_length is not None:
                    dropped_overlong = 0
                    for pos in list(constraints.keys()):
                        if not decisions.get(pos, False) or not constraints.get(pos):
                            continue
                        # Length-check the prompt that will actually be persisted:
                        # for reset modes that's ``X + c_new`` (re-based on the
                        # original), not ``X + c_prev + c_new``.
                        join_base = original_prompts[pos] if is_reset_mode else prompts[pos]
                        candidate = join_prompt_constraint(join_base, constraints[pos])
                        if self._tokenized_prompt_len(candidate) > self._max_prompt_length:
                            dropped_overlong += 1
                            decisions[pos] = False
                            constraints.pop(pos, None)
                            new_constraint_rubrics.pop(pos, None)
                    if dropped_overlong:
                        logger.warning(
                            "[Step 3] dropped %d constraint(s) that would exceed max_prompt_length=%d",
                            dropped_overlong,
                            self._max_prompt_length,
                        )
                # Only accumulate the constraint-specific rubric for modes that
                # actually use it downstream.
                if self._mode in _MODES_WITH_CONSTRAINT_RUBRIC:
                    for pos, new_pairs in new_constraint_rubrics.items():
                        if not new_pairs:
                            continue
                        if is_reset_mode:
                            # Reset mode: drop the previously persisted constraint
                            # rubrics in lockstep with dropping ``c_prev`` from the
                            # prompt. Only fires on samples where a new constraint
                            # was actually generated this epoch (decision=no
                            # samples skip this loop and keep their carry-forward).
                            constraint_rubrics[pos] = list(new_pairs)
                        else:
                            constraint_rubrics.setdefault(pos, []).extend(new_pairs)
        finally:
            self._sleep_rm()

        # No Step 4 rollout in any kept mode — every adaptive-using mode
        # ``adaptive_conditional_rubric`` runs Stage 5 on rollouts_A.
        rollouts_B: list[list[str]] = list(rollouts_A)

        # ── GEN wake #2: adaptive rubric (runs for every mode in
        #    _MODES_WITH_ADAPTIVE).
        if self._mode in _MODES_WITH_ADAPTIVE:
            target_prompts = prompts
            eligible = [pos for pos in range(N) if len(rollouts_B[pos]) >= 2]
            self._wake_rm()
            try:
                logger.info("[Step 5] adaptive rubric for %d eligible samples", len(eligible))
                adaptive_rubrics = self._gen_adaptive_rubric(
                    target_prompts,
                    rollouts_B,
                    eligible,
                    general_rubrics=general_rubrics,
                    constraint_rubrics=constraint_rubrics,
                    adaptive_rubrics_persisted=adaptive_rubrics_persisted,
                )
            finally:
                self._sleep_rm()

            if self._mode in _MODES_WITH_ADAPTIVE_APPEND:
                # Accumulate across epochs: prepend previously persisted adaptive
                # pairs, then append freshly generated ones. Positions whose
                # fresh generation failed still retain the persisted rubric.
                merged: dict[int, list[dict]] = {}
                for pos in range(N):
                    old = adaptive_rubrics_persisted.get(pos, [])
                    new = adaptive_rubrics.get(pos, [])
                    if old or new:
                        merged[pos] = list(old) + list(new)
                adaptive_rubrics = merged

        # ── Step 7: Override + optional in-place mutation ──────────────────
        modified = 0
        for pos in range(N):
            real_idx = dataset_idxs[pos]
            g = general_rubrics.get(pos, [])
            c = constraint_rubrics.get(pos, []) if self._mode in _MODES_WITH_CONSTRAINT_RUBRIC else []
            a = adaptive_rubrics.get(pos, []) if self._mode in _MODES_WITH_ADAPTIVE else []
            has_new_constraint = (
                self._mode in _MODES_WITH_CONSTRAINT and decisions.get(pos, False) and bool(constraints.get(pos))
            )

            if has_new_constraint:
                # Reset mode re-bases on the original prompt so persisted
                # output is ``X + c_new`` instead of ``X + c_prev + c_new``.
                # Decision=no samples fall through to the else branch and keep
                # ``prompts[pos]`` (= current override = ``X + c_prev``)
                # untouched, in lockstep with the constraint-rubric carry-
                # forward at Step 3.
                join_base = original_prompts[pos] if is_reset_mode else prompts[pos]
                new_prompt_str = join_prompt_constraint(join_base, constraints[pos])
            else:
                new_prompt_str = prompts[pos]

            self._apply_override(real_idx, new_prompt_str, g, c, a)
            if g or c or a:
                modified += 1

            if batch is not None:
                self._mutate_inplace(
                    pos=pos,
                    new_prompt_str=new_prompt_str,
                    general=g,
                    constraint=c,
                    adaptive=a,
                    batch=batch,
                )

        logger.info("modify_batch summary: %d/%d samples with non-empty rubric", modified, N)
        return modified

    def _modify_batch_eva(
        self,
        indices: list[int],
        batch: DataProto | None,
        prompts: list[str],
        dataset_idxs: list[int],
        rollouts_A: list[list[str]],
        step1_failed: set[int],
        general_rubrics: dict[int, list[dict]],
    ) -> int:
        """EVA baseline pipeline.

        Implements the canonical EVA paper algorithm:
          1.5  Score every (prompt, rollout) pair with the scalar RM
               (ArmoRM-8B).
          1.6  info(x) = max - min over each prompt's RM scores.
          1.7  WEIGHTED-SAMPLE K = round(N * eva_top_k_ratio) high-info
               prompts (no replacement, weights ∝ info; uniform fallback
               when all infos are -inf or zero).
          1.8  Sleep RM, wake judge.
          1.9  For each of K selected positions, run M evolutions; each
               evolution uniformly samples one of 5 distilabel methods.
               Evolved pool size up to K*M (parse failures dropped).
          1.10 80/20 mix: round(N * eva_evolved_mix_ratio) batch slots get
               their prompt replaced by a random sample (with replacement
               when pool < n_evolved) from the evolved pool. The remaining
               slots keep their original prompt + persisted rubric.
          2    Regenerate general rubric ONLY for the evolved-mix slots.
        """
        N = len(indices)

        # Step 1.5
        logger.info("[EVA 1.5] waking scalar RM for rollout scoring")
        self._rm_manager.wake_up("scalar_rm")
        try:
            scores = self._score_rollouts_with_rm(prompts, rollouts_A, step1_failed)
        finally:
            self._rm_manager.sleep("scalar_rm")

        # Step 1.6: info-scores + argmax/argmin tracking for the audit log.
        infos: list[float] = []
        y_plus: list[int] = []
        y_minus: list[int] = []
        for pos, s in enumerate(scores):
            if pos in step1_failed or len(s) < 2:
                infos.append(float("-inf"))
                y_plus.append(-1)
                y_minus.append(-1)
            else:
                infos.append(float(max(s) - min(s)))
                # argmax / argmin with stable tie-break (first occurrence).
                y_plus.append(int(max(range(len(s)), key=lambda i: s[i])))
                y_minus.append(int(min(range(len(s)), key=lambda i: s[i])))

        # Step 1.7: weighted high-info sample.
        K = max(1, int(round(N * self._eva_top_k_ratio)))
        selected = self._weighted_sample_high_info(infos, K)
        selected_set = set(selected)
        logger.info(
            "[EVA 1.7] weighted-sampled K=%d / N=%d high-info positions (ratio=%.2f)",
            len(selected),
            N,
            self._eva_top_k_ratio,
        )

        # Step 1.8 + 1.9: sleep RM, wake judge, run M evolutions per
        # selected position. The pool is a flat list of (source_pos,
        # method, evolved_text); we materialise it as parallel lists for
        # downstream sampling.
        logger.info("[EVA 1.8] waking judge for Evol-Instruct")
        self._rm_manager.wake_up("judge")
        pool_text: list[str] = []
        pool_source_pos: list[int] = []
        pool_method: list[str] = []
        per_pos_parse_ok: dict[int, bool] = {}  # any successful evolution for a row
        try:
            M = self._eva_evolutions_per_prompt
            # Flatten K*M (prompt, method) pairs into a single judge batch
            # so the call goes through ``_call_generator`` once.
            method_choices = list(self._eva_rng.choice(self._eva_methods, size=len(selected) * M, replace=True))
            rendered: list[str] = []
            req_source: list[int] = []
            req_method: list[str] = []
            for k_idx, pos in enumerate(selected):
                base_prompt = prompts[pos]
                for j in range(M):
                    method = str(method_choices[k_idx * M + j])
                    template = self._evol_instruct_templates[method]
                    rendered.append(template.format(prompt=base_prompt))
                    req_source.append(pos)
                    req_method.append(method)

            evolved_texts, parse_ok_flags = self._run_evol_instruct(rendered, req_method)
            n_pool_attempts = len(rendered)
            for k_idx, pos in enumerate(selected):
                per_pos_parse_ok[pos] = False
            for i, (text, ok) in enumerate(zip(evolved_texts, parse_ok_flags, strict=True)):
                src = req_source[i]
                if ok and text:
                    pool_text.append(text)
                    pool_source_pos.append(src)
                    pool_method.append(req_method[i])
                    per_pos_parse_ok[src] = True

            n_pool = len(pool_text)
            logger.info(
                "[EVA 1.9] evolved pool: %d / %d successful (M=%d per prompt)",
                n_pool,
                n_pool_attempts,
                M,
            )

            # Step 1.10: 80/20 mix. If the entire pool is empty, the run is
            # unsalvageable for this batch — we fall back to passthrough so
            # training still progresses.
            assigned_prompt: dict[int, str] = {}
            assigned_source: dict[int, int] = {}
            assigned_method: dict[int, str] = {}
            if n_pool == 0:
                logger.warning("[EVA 1.10] evolved pool is empty after parse — leaving batch unchanged")
            else:
                n_evolved = int(round(N * self._eva_evolved_mix_ratio))
                # Cap at N because round() can technically exceed it for
                # ratios just under 1 with floating point.
                n_evolved = min(n_evolved, N)
                # Sample which N batch slots become evolved (no replacement
                # over slots — each slot gets at most one assignment).
                slots = list(self._eva_rng.choice(N, size=n_evolved, replace=False))
                # Pool sample with replacement when pool < n_evolved (per
                # spec: "Pool smaller than n_evolved → sample with
                # replacement"); without replacement otherwise.
                pool_replace = n_pool < n_evolved
                pool_pick = self._eva_rng.choice(n_pool, size=n_evolved, replace=pool_replace)
                for slot_idx, pi in zip(slots, pool_pick, strict=True):
                    pos = int(slot_idx)
                    pi = int(pi)
                    assigned_prompt[pos] = pool_text[pi]
                    assigned_source[pos] = pool_source_pos[pi]
                    assigned_method[pos] = pool_method[pi]
                logger.info(
                    "[EVA 1.10] 80/20 mix: %d/%d slots assigned evolved prompts (pool=%d, replace=%s)",
                    len(assigned_prompt),
                    N,
                    n_pool,
                    pool_replace,
                )

            # Step 2: regenerate general rubric ONLY for slots whose prompt
            # was replaced by an evolved one. Non-evolved slots keep their
            # persisted rubric verbatim.
            evolved_positions = sorted(assigned_prompt.keys())
            if evolved_positions:
                logger.info(
                    "[EVA 2] regenerating general rubric for %d evolved-mix slot(s)",
                    len(evolved_positions),
                )
                eva_prompts = list(prompts)
                for pos in evolved_positions:
                    eva_prompts[pos] = assigned_prompt[pos]
                fresh_general = self._gen_general_rubric(eva_prompts, evolved_positions)
                for pos, pairs in fresh_general.items():
                    general_rubrics[pos] = pairs
        finally:
            self._rm_manager.sleep("judge")

        # Audit log: one record per row (selected or not) to make selection
        # behaviour and Evol-Instruct quality auditable post-hoc.
        self._dump_eva_audit(
            dataset_idxs=dataset_idxs,
            prompts=prompts,
            scores=scores,
            infos=infos,
            y_plus=y_plus,
            y_minus=y_minus,
            selected_set=selected_set,
            per_pos_parse_ok=per_pos_parse_ok,
            assigned_prompt=assigned_prompt,
            assigned_source=assigned_source,
            assigned_method=assigned_method,
        )

        # Step 7: persist overrides + in-place mutate the live batch.
        modified = 0
        for pos in range(N):
            real_idx = dataset_idxs[pos]
            new_prompt_str = assigned_prompt.get(pos, prompts[pos])
            g = general_rubrics.get(pos, [])
            # EVA mode does not produce constraint / adaptive criteria.
            self._apply_override(real_idx, new_prompt_str, g, [], [])
            if g:
                modified += 1
            if batch is not None:
                self._mutate_inplace(
                    pos=pos,
                    new_prompt_str=new_prompt_str,
                    general=g,
                    constraint=[],
                    adaptive=[],
                    batch=batch,
                )
        logger.info(
            "[EVA] modify_batch summary: evolved_slots=%d, kept=%d, with_rubric=%d/%d",
            len(assigned_prompt),
            N - len(assigned_prompt),
            modified,
            N,
        )
        return modified

    def _weighted_sample_high_info(self, infos: list[float], k: int) -> list[int]:
        """Weighted no-replacement sample of K positions by info-score.

        Edge cases (per EVA spec):
          * ``-inf`` entries (rollout failures) are unselectable — dropped
            up-front. K is capped at the number of valid rows.
          * If every valid info is zero (all rollouts tied at the RM),
            weight-proportional sampling is undefined → fall back to a
            uniform sample over the valid rows. Same for any case where
            the post-mask weight sum is non-positive.
        """
        valid = [i for i, v in enumerate(infos) if v != float("-inf")]
        if not valid:
            return []
        k = min(k, len(valid))

        weights = np.array([infos[i] for i in valid], dtype=np.float64)
        # Negative infos are not part of the spec but defend anyway —
        # weighted sampling requires non-negative weights. Clip + check.
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= 0.0:
            picks = self._eva_rng.choice(len(valid), size=k, replace=False)
        else:
            probs = weights / total
            picks = self._eva_rng.choice(len(valid), size=k, replace=False, p=probs)
        return [valid[int(i)] for i in picks]

    def _score_rollouts_with_rm(
        self,
        prompts: list[str],
        rollouts: list[list[str]],
        step1_failed: set[int],
    ) -> list[list[float]]:
        """Send every (prompt, response) pair to the scalar RM in one batch.

        Returns a list shaped [N][R] where R = number of valid rollouts for
        that prompt (may be 0 for step1_failed). The flat batch is then
        de-flattened back into per-prompt lists in the same order as
        ``rollouts``.
        """
        flat_pairs: list[tuple[str, str]] = []
        offsets: list[int] = []  # start index of each prompt's responses in flat_pairs
        for pos in range(len(prompts)):
            offsets.append(len(flat_pairs))
            if pos in step1_failed:
                continue
            for resp in rollouts[pos]:
                flat_pairs.append((prompts[pos], resp))
        offsets.append(len(flat_pairs))

        if not flat_pairs:
            return [[] for _ in prompts]

        flat_scores = self._rm_manager.score_pairs(flat_pairs)
        scores: list[list[float]] = []
        for pos in range(len(prompts)):
            scores.append(flat_scores[offsets[pos] : offsets[pos + 1]])
        return scores

    def _run_evol_instruct(
        self,
        rendered_prompts: list[str],
        methods: list[str],
    ) -> tuple[list[str | None], list[bool]]:
        """Send a batch of distilabel-rendered prompts to the judge and
        parse out the rewritten prompts.

        ``methods[i]`` selects which marker to look for ("breadth" → the
        in-breadth template emits ``#Created Prompt#:``; the four in-depth
        methods emit ``#Rewritten Prompt#:``). We take everything after the
        LAST occurrence of the marker, strip whitespace, and treat empty as
        a parse failure.
        """
        if not rendered_prompts:
            return [], []
        outputs = asyncio.run(self._call_generator(rendered_prompts))
        thinking_pat = self._parse_patterns.get("thinking")
        evolved: list[str | None] = []
        parse_ok: list[bool] = []
        for raw, method in zip(outputs, methods, strict=True):
            _, body = split_thinking(raw, thinking_pat)
            marker = "#Created Prompt#:" if method == "breadth" else "#Rewritten Prompt#:"
            idx = body.rfind(marker)
            # Strict: no marker → parse failure. Models that ignore the
            # template often emit chat-style preambles, so passing the
            # whole body through would silently corrupt the prompt.
            text = body[idx + len(marker) :].strip() if idx >= 0 else ""
            if text:
                evolved.append(text)
                parse_ok.append(True)
            else:
                evolved.append(None)
                parse_ok.append(False)
        return evolved, parse_ok

    def _dump_eva_audit(
        self,
        dataset_idxs: list[int],
        prompts: list[str],
        scores: list[list[float]],
        infos: list[float],
        y_plus: list[int],
        y_minus: list[int],
        selected_set: set[int],
        per_pos_parse_ok: dict[int, bool],
        assigned_prompt: dict[int, str],
        assigned_source: dict[int, int],
        assigned_method: dict[int, str],
    ) -> None:
        # Reuse _dump_debug so records land in
        # ``{debug_dir}/eva_evolution.jsonl`` next to the existing per-stage
        # JSONLs (general_rubric.jsonl, etc.).
        records = [
            {
                "label": "eva_evolution",
                "epoch": self._current_epoch,
                "sample_idx": real_idx,
                "info_score": None if infos[pos] == float("-inf") else infos[pos],
                "rollouts_rm_scores": list(scores[pos]),
                "y_plus_idx": y_plus[pos],
                "y_minus_idx": y_minus[pos],
                "selected_for_evolution": pos in selected_set,
                # Whether THIS row produced any successful evolution (only
                # meaningful when the row was selected).
                "parse_ok": per_pos_parse_ok.get(pos, False) if pos in selected_set else False,
                "original_prompt": prompts[pos],
                # ``evolved_prompt``: the prompt this row's training will
                # actually use after the 80/20 mix. Identical to
                # ``assigned_evolved_prompt`` when set; otherwise None
                # (meaning the slot kept its original prompt).
                "evolved_prompt": assigned_prompt.get(pos),
                "assigned_evolved_prompt": assigned_prompt.get(pos),
                "evolved_pool_source_pos": assigned_source.get(pos),
                "evolution_method": assigned_method.get(pos),
            }
            for pos, real_idx in enumerate(dataset_idxs)
        ]
        self._dump_debug(records)

    def _tokenized_prompt_len(self, prompt_str: str) -> int:
        """Token count of ``prompt_str`` under the policy chat template.

        Mirrors how ``AgentLoop`` turns a raw user prompt into ``prompt_ids``
        (`tokenizer.apply_chat_template(..., add_generation_prompt=True)`),
        so the count matches what downstream padding compares against
        ``data.max_prompt_length``.
        """
        return len(
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_str}],
                add_generation_prompt=True,
                **self._apply_chat_template_kwargs,
            )
        )

    # ── Per-step generation helpers ────────────────────────────────────────

    def _gen_general_rubric(
        self,
        prompts: list[str],
        eligible: list[int],
    ) -> dict[int, list[dict]]:
        template = self._load_template("general_rubric")
        inputs: dict[int, str] = {i: template.format(instruction=prompts[i]) for i in eligible}

        def parse_fn(raw: str):
            _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
            return parse_rubric_output(resp, self._parse_patterns)

        def validate_fn(parsed) -> bool:
            return parsed is not None and is_valid_rubric(parsed[1], require_importance=True)

        def format_parsed(parsed):
            if parsed is None:
                return {"rubric": []}
            _meta, pairs = parsed
            return {"rubric": list(pairs)}

        def fallback_parse_fn(raw: str):
            _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
            return parse_rubric_output_lenient(resp, self._parse_patterns)

        def fallback_validate_fn(parsed) -> bool:
            return parsed is not None and bool(parsed[1])

        raw_results = self._generate_with_retry_keyed(
            inputs,
            parse_fn=parse_fn,
            validate_fn=validate_fn,
            format_parsed=format_parsed,
            label="general_rubric",
            fallback_parse_fn=fallback_parse_fn,
            fallback_validate_fn=fallback_validate_fn,
        )
        return {i: list(parsed[1]) for i, parsed in raw_results.items()}

    def _gen_constraint_judgment(
        self,
        prompts: list[str],
        rollouts_A: list[list[str]],
        eligible: list[int],
        turn_idx: int | None = None,
    ) -> tuple[dict[int, bool], dict[int, str], dict[int, list[dict]]]:
        template = self._load_template("constraint_judgment")
        inputs: dict[int, str] = {}
        for i in eligible:
            if len(rollouts_A[i]) < 2:
                continue
            inputs[i] = template.format(
                instruction=prompts[i],
                response_1=rollouts_A[i][0],
                response_2=rollouts_A[i][1],
            )

        def parse_fn(raw: str):
            _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
            return parse_constraint_judgment(resp, self._parse_patterns)

        def validate_fn(parsed) -> bool:
            # Retry when the parse is unusable for downstream:
            #   - missing/ambiguous decision (parsed is None or decision is None)
            #   - decision=YES with an empty/invalid rubric — otherwise Step 7
            #     would join the new constraint into raw_prompt while
            #     requirements_constraint stays empty, yielding a sample whose
            #     reward never scores the added constraint (silent corruption).
            if parsed is None or parsed[0] is None:
                return False
            need_c, _c_text, _analysis, c_rubric = parsed
            if need_c and not is_valid_rubric(c_rubric, require_importance=True):
                return False
            return True

        def format_parsed(parsed):
            if parsed is None:
                return {"decision": None, "constraint": "", "analysis": "", "constraint_rubric": []}
            need_c, c_text, analysis, c_rubric = parsed
            return {
                "decision": need_c,
                "constraint": c_text,
                "analysis": analysis,
                "constraint_rubric": list(c_rubric),
            }

        extra_fields = {i: {"base_responses": [rollouts_A[i][0], rollouts_A[i][1]]} for i in inputs}
        if turn_idx is not None:
            for i in extra_fields:
                extra_fields[i]["turn"] = turn_idx

        raw_results = self._generate_with_retry_keyed(
            inputs,
            parse_fn=parse_fn,
            validate_fn=validate_fn,
            format_parsed=format_parsed,
            extra_fields=extra_fields,
            label="constraint_judgment",
        )

        decisions: dict[int, bool] = {}
        constraints: dict[int, str] = {}
        constraint_rubrics: dict[int, list[dict]] = {}
        for i, t in raw_results.items():
            need_c, c_text, _analysis, c_rubric = t
            decisions[i] = bool(need_c)
            if need_c:
                constraints[i] = c_text
                constraint_rubrics[i] = list(c_rubric)
        return decisions, constraints, constraint_rubrics

    def _gen_constraint_always(
        self,
        prompts: list[str],
        eligible: list[int],
    ) -> tuple[dict[int, bool], dict[int, str], dict[int, list[dict]]]:
        """Always-on constraint generator (no rollouts, no decision gate).

        The judge sees only the instruction and is expected to emit a
        ``<constraint>`` plus a ``<rubric>``. There is no ``<decision>`` tag
        — every successful parse is treated as ``decision=yes`` so the
        constraint and its rubric are always accumulated. Used by modes in
        ``_MODES_WITHOUT_BASE_ROLLOUT``.
        """
        template = self._load_template("constraint_judgment")
        inputs: dict[int, str] = {i: template.format(instruction=prompts[i]) for i in eligible}

        def parse_fn(raw: str):
            _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
            constraint_text = parse_constraint(resp, self._parse_patterns).strip()
            if not constraint_text:
                return None
            _, rubric_pairs = parse_rubric_output(resp, self._parse_patterns)
            if not is_valid_rubric(rubric_pairs, require_importance=True):
                return None
            analysis = parse_analysis(resp, self._parse_patterns)
            return constraint_text, analysis, list(rubric_pairs)

        def validate_fn(parsed) -> bool:
            return parsed is not None

        def format_parsed(parsed):
            # decision=None on parse failure (matches ``_gen_constraint_judgment``
            # convention so ``constraint_judgment.jsonl`` readers can use the
            # same failure detector across modes); decision=True on success
            # since this mode never gates.
            if parsed is None:
                return {"decision": None, "constraint": "", "analysis": "", "constraint_rubric": []}
            c_text, analysis, c_rubric = parsed
            return {
                "decision": True,
                "constraint": c_text,
                "analysis": analysis,
                "constraint_rubric": list(c_rubric),
            }

        raw_results = self._generate_with_retry_keyed(
            inputs,
            parse_fn=parse_fn,
            validate_fn=validate_fn,
            format_parsed=format_parsed,
            label="constraint_judgment",
        )

        decisions: dict[int, bool] = {}
        constraints: dict[int, str] = {}
        constraint_rubrics: dict[int, list[dict]] = {}
        for i, t in raw_results.items():
            c_text, _analysis, c_rubric = t  # type: ignore[misc]
            decisions[i] = True
            constraints[i] = c_text
            constraint_rubrics[i] = list(c_rubric)
        return decisions, constraints, constraint_rubrics

    def _metric_saturation_gate(
        self,
        prompts: list[str],
        rollouts: list[list[str]],
        general_rubrics: dict[int, list[dict]],
        eligible: list[int],
        indices: list[int],
    ) -> list[int]:
        """Return positions whose rollouts are saturated under the general rubric.

        Scores each eligible prompt's Step 1 rollouts against its general rubric
        with the per-criterion ``judge`` protocol (reused via the generator/
        judge backend, generator == judge), computes the cross-rollout std of the
        per-rollout weighted-mean score, and flags ``std <
        self._saturation_std_threshold`` as saturated. Per-prompt metrics are
        dumped to ``constraint_metric_summary.jsonl`` for calibration. Heavy
        lifting (input build / parse / aggregate) lives in
        ``metric_saturation_gate.py`` to keep this module thin.
        """
        template = self._load_template("metric_judge")
        score_inputs, meta = build_score_inputs(
            prompts,
            rollouts,
            general_rubrics,
            eligible,
            template,
            min_valid_rollouts=self._metric_min_valid_rollouts,
        )
        if not score_inputs:
            logger.info("[metric-gate] no scorable samples (eligible=%d)", len(eligible))
            return []
        raw_scores = self._generate_with_retry_keyed(
            score_inputs,
            parse_fn=parse_score_strict,
            validate_fn=lambda s: s is not None,
            format_parsed=lambda s: s,
            fallback_parse_fn=parse_score_fallback,
            fallback_validate_fn=lambda s: True,
            label="metric_score",
        )
        saturated, metrics = aggregate_saturation(raw_scores, meta, self._saturation_std_threshold)
        logger.info(
            "[metric-gate] saturated %d/%d scored (std < %.2f)",
            len(saturated),
            len(meta),
            self._saturation_std_threshold,
        )
        self._dump_debug(
            [
                {
                    "label": "constraint_metric_summary",
                    "epoch": self._current_epoch,
                    "dataset_idx": indices[pos],
                    "pos": pos,
                    "threshold": self._saturation_std_threshold,
                    **m,
                }
                for pos, m in metrics.items()
            ]
        )
        return sorted(saturated)

    def _gen_constraint_rewrite_judgment(
        self,
        prompts: list[str],
        rollouts_A: list[list[str]],
        eligible: list[int],
    ) -> tuple[dict[int, bool], dict[int, str]]:
        """Decide whether each instruction needs a constraint and, on yes,
        return a fully rewritten instruction that folds the new constraint
        into the original prompt (instead of appending it).

        The judge sees the same inputs as ``_gen_constraint_judgment`` and is
        expected to emit ``<decision>yes|no</decision>`` plus a
        ``<rewrite>`` body when decision=yes. Decision=no falls through with
        an empty rewrite. A yes-without-rewrite is treated as a parse failure
        and retried.
        """
        template = self._load_template("constraint_rewrite")
        inputs: dict[int, str] = {}
        for i in eligible:
            if len(rollouts_A[i]) < 2:
                continue
            inputs[i] = template.format(
                instruction=prompts[i],
                response_1=rollouts_A[i][0],
                response_2=rollouts_A[i][1],
            )

        def parse_fn(raw: str):
            _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
            return parse_constraint_rewrite_judgment(resp, self._parse_patterns)

        def validate_fn(parsed) -> bool:
            return parsed is not None and parsed[0] is not None

        def format_parsed(parsed):
            if parsed is None:
                return {"decision": None, "rewrite": "", "analysis": ""}
            decision, rewrite_text, analysis = parsed
            return {
                "decision": decision,
                "rewrite": rewrite_text,
                "analysis": analysis,
            }

        extra_fields = {
            i: {
                "original_instruction": prompts[i],
                "base_responses": [rollouts_A[i][0], rollouts_A[i][1]],
            }
            for i in inputs
        }

        raw_results = self._generate_with_retry_keyed(
            inputs,
            parse_fn=parse_fn,
            validate_fn=validate_fn,
            format_parsed=format_parsed,
            extra_fields=extra_fields,
            label="constraint_rewrite",
        )

        decisions: dict[int, bool] = {}
        rewrites: dict[int, str] = {}
        for i, t in raw_results.items():
            decision, rewrite_text, _analysis = t  # type: ignore[misc]
            decisions[i] = bool(decision)
            if decision:
                rewrites[i] = rewrite_text
        return decisions, rewrites

    @staticmethod
    def _format_existing_rubric(*groups: list[dict]) -> str:
        """Render accumulated rubric groups for the v2 adaptive template's
        ``{existing_rubric}`` slot. Groups are concatenated in the given order
        (general + constraint + previously persisted adaptive), so the judge
        sees everything that's been accumulated so far and can skip on coverage
        instead of redundantly re-emitting the same class of criterion each epoch.
        """
        pairs: list[dict] = []
        for g in groups:
            pairs.extend(g)
        if not pairs:
            return "(No prior criteria.)"
        lines = []
        for i, p in enumerate(pairs, 1):
            rub = str(p.get("rubric", "")).strip()
            imp = int(p.get("importance", 50))
            lines.append(f"{i}. (importance={imp}) {rub}")
        return "\n".join(lines)

    def _gen_adaptive_rubric(
        self,
        target_prompts: list[str],
        rollouts: list[list[str]],
        eligible: list[int],
        general_rubrics: dict[int, list[dict]] | None = None,
        constraint_rubrics: dict[int, list[dict]] | None = None,
        adaptive_rubrics_persisted: dict[int, list[dict]] | None = None,
    ) -> dict[int, list[dict]]:
        """Generate adaptive rubric criteria.

        For modes in ``_MODES_WITH_ADAPTIVE_DECISION``, the judge uses a v2-style
        template that renders ``<existing_rubric>`` from the currently-persisted
        general + constraint + previously-accumulated adaptive rubric (all three
        must be supplied) and emits its own ``<decision>yes|no</decision>``:

          * decision=no  → ``(False, [])`` — no criteria returned for that sample.
          * decision=yes → ``(True, pairs)`` after the usual rubric validation.
          * decision missing, or yes + malformed rubric → retry.

        Showing the previously-accumulated adaptive rubric lets the judge see
        its own prior output and skip on coverage, so cross-epoch append
        (``_MODES_WITH_ADAPTIVE_APPEND``) doesn't redundantly re-emit the same
        class of criterion each epoch.

        Other modes use the legacy v1 parse path (no decision tag, lenient
        fallback on exhaustion).
        """
        template = self._load_template("adaptive_rubric")
        use_decision = self._mode in _MODES_WITH_ADAPTIVE_DECISION
        general_rubrics = general_rubrics or {}
        constraint_rubrics = constraint_rubrics or {}
        adaptive_rubrics_persisted = adaptive_rubrics_persisted or {}

        inputs: dict[int, str] = {}
        for i in eligible:
            if len(rollouts[i]) < 2:
                continue
            fmt_kwargs = {
                "instruction": target_prompts[i],
                "response_1": rollouts[i][0],
                "response_2": rollouts[i][1],
            }
            if use_decision:
                fmt_kwargs["existing_rubric"] = self._format_existing_rubric(
                    general_rubrics.get(i, []),
                    constraint_rubrics.get(i, []),
                    adaptive_rubrics_persisted.get(i, []),
                )
            inputs[i] = template.format(**fmt_kwargs)

        if use_decision:

            def parse_fn(raw: str):
                _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
                decision = parse_decision(resp, self._parse_patterns)
                if decision is None:
                    return None
                if not decision:
                    return (False, [])
                _, pairs = parse_rubric_output(resp, self._parse_patterns)
                if not is_valid_rubric(pairs, require_importance=True):
                    return None
                return (True, list(pairs))

            def validate_fn(parsed) -> bool:
                return parsed is not None

            fallback_parse_fn = None
            fallback_validate_fn = None
        else:

            def parse_fn(raw: str):
                _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
                return parse_rubric_output(resp, self._parse_patterns)

            def validate_fn(parsed) -> bool:
                return parsed is not None and is_valid_rubric(parsed[1], require_importance=True)

            def fallback_parse_fn(raw: str):
                _, resp = split_thinking(raw, self._parse_patterns.get("thinking"))
                return parse_rubric_output_lenient(resp, self._parse_patterns)

            def fallback_validate_fn(parsed) -> bool:
                return parsed is not None and bool(parsed[1])

        def format_parsed(parsed):
            if parsed is None:
                return {"decision": None, "rubric": []} if use_decision else {"rubric": []}
            first, pairs = parsed
            if use_decision:
                return {"decision": bool(first), "rubric": list(pairs)}
            return {"rubric": list(pairs)}

        extra_fields = {i: {"base_responses": [rollouts[i][0], rollouts[i][1]]} for i in inputs}

        raw_results = self._generate_with_retry_keyed(
            inputs,
            parse_fn=parse_fn,
            validate_fn=validate_fn,
            format_parsed=format_parsed,
            extra_fields=extra_fields,
            label="adaptive_rubric",
            fallback_parse_fn=fallback_parse_fn,
            fallback_validate_fn=fallback_validate_fn,
        )
        return {i: list(parsed[1]) for i, parsed in raw_results.items()}

    # ── Retry wrappers ─────────────────────────────────────────────────────

    def _generate_with_retry_keyed(
        self,
        inputs: dict[Hashable, str],
        parse_fn: Callable,
        validate_fn: Callable,
        format_parsed: Callable | None = None,
        extra_fields: dict[Hashable, dict] | None = None,
        label: str = "",
        fallback_parse_fn: Callable | None = None,
        fallback_validate_fn: Callable | None = None,
        retry_oversample: float | None = None,
        max_retry_override: int | None = None,
    ) -> dict[Hashable, object]:
        """Retry up to ``self._generator_max_retry`` for a strict parse.

        After retries are exhausted, if ``fallback_parse_fn`` is provided, each
        remaining sample's last raw output is re-parsed leniently. The salvaged
        result is kept when ``fallback_validate_fn`` returns truthy (or when it
        is not provided). This lets us accept a partial rubric instead of
        dropping the sample entirely.

        ``retry_oversample`` (>=1.0) duplicates each pending prompt K times
        in retry rounds (attempt >= 1) and accepts the first parse-success
        per prompt, same heavy-tail compaction as ``_gen_rollouts``. Attempt
        0 always sends 1 sample/prompt for back-compat. When ``None``,
        defaults to ``self._generator_retry_oversample``.

        ``max_retry_override`` (int >= 0, optional) caps retry rounds for
        this call only; defaults to ``self._generator_max_retry``.
        """
        if not inputs:
            return {}

        if retry_oversample is None:
            retry_oversample = self._generator_retry_oversample
        if retry_oversample < 1.0:
            raise ValueError(f"retry_oversample must be >= 1.0; got {retry_oversample!r}")
        max_retry = self._generator_max_retry if max_retry_override is None else int(max_retry_override)
        if max_retry < 0:
            raise ValueError(f"max_retry_override must be >= 0; got {max_retry_override!r}")

        remaining: dict[Hashable, str] = dict(inputs)
        results: dict[Hashable, object] = {}
        debug_records: list[dict] = []
        last_raw: dict[Hashable, str] = {}

        for attempt in range(max_retry + 1):
            if not remaining:
                break
            idxs = list(remaining.keys())
            prompts = [remaining[i] for i in idxs]
            # Oversample on retries only. Attempt 0 keeps 1 sample/prompt
            # so success-on-first-try is unchanged; retry rounds batch K
            # duplicates per pending prompt and pick the first parse-success.
            if attempt == 0 or retry_oversample <= 1.0:
                n_per = 1
            else:
                n_per = max(1, int(round(retry_oversample)))

            if n_per > 1:
                expanded = [p for p in prompts for _ in range(n_per)]
            else:
                expanded = prompts
            outputs = asyncio.run(self._call_generator(expanded))

            still_remaining: dict[Hashable, str] = {}
            for k, i in enumerate(idxs):
                slice_outputs = outputs[k * n_per : (k + 1) * n_per]
                chosen_raw = slice_outputs[-1] if slice_outputs else ""
                chosen_parsed = None
                chosen_ok = False
                for raw in slice_outputs:
                    parsed = parse_fn(raw)
                    if validate_fn(parsed):
                        chosen_raw = raw
                        chosen_parsed = parsed
                        chosen_ok = True
                        break
                if not chosen_ok:
                    chosen_parsed = parse_fn(chosen_raw)

                last_raw[i] = chosen_raw
                if chosen_ok:
                    results[i] = chosen_parsed
                else:
                    still_remaining[i] = remaining[i]
                if self._debug_dump_dir is not None:
                    record = {
                        "epoch": self._current_epoch,
                        "sample_idx": i,
                        "label": label,
                        "attempt": attempt,
                        "status": "ok" if chosen_ok else "parse_fail",
                        "raw_output": chosen_raw,
                    }
                    if n_per > 1:
                        record["oversample"] = n_per
                    if format_parsed is not None:
                        record["parsed"] = format_parsed(chosen_parsed)
                    if extra_fields is not None and i in extra_fields:
                        record.update(extra_fields[i])
                    debug_records.append(record)

            remaining = still_remaining
            if remaining and attempt < max_retry:
                next_n = max(1, int(round(retry_oversample))) if retry_oversample > 1.0 else 1
                if next_n > 1:
                    logger.info(
                        "%s retry %d/%d: %d samples remaining (next attempt n=%d)",
                        label,
                        attempt + 1,
                        max_retry,
                        len(remaining),
                        next_n,
                    )
                else:
                    logger.info(
                        "%s retry %d/%d: %d samples remaining",
                        label,
                        attempt + 1,
                        max_retry,
                        len(remaining),
                    )

        salvaged = 0
        if remaining and fallback_parse_fn is not None:
            for i in list(remaining.keys()):
                raw = last_raw.get(i)
                if raw is None:
                    continue
                fb_parsed = fallback_parse_fn(raw)
                fb_ok = fb_parsed is not None and (
                    fallback_validate_fn is None or bool(fallback_validate_fn(fb_parsed))
                )
                if fb_ok:
                    results[i] = fb_parsed
                    del remaining[i]
                    salvaged += 1
                if self._debug_dump_dir is not None:
                    record = {
                        "epoch": self._current_epoch,
                        "sample_idx": i,
                        "label": label,
                        "attempt": max_retry + 1,
                        "status": "fallback_salvaged" if fb_ok else "fallback_fail",
                        "raw_output": raw,
                    }
                    if format_parsed is not None:
                        record["parsed"] = format_parsed(fb_parsed)
                    if extra_fields is not None and i in extra_fields:
                        record.update(extra_fields[i])
                    debug_records.append(record)

        if salvaged:
            logger.info(
                "%s: salvaged %d sample(s) via lenient fallback after retries",
                label,
                salvaged,
            )
        if remaining:
            logger.warning(
                "%s: %d samples failed all %d retries%s",
                label,
                len(remaining),
                max_retry,
                " (fallback also failed)" if fallback_parse_fn is not None else "",
            )

        self._dump_debug(debug_records)
        return results

    # ── Override persistence + in-place mutation ───────────────────────────

    def _apply_override(
        self,
        real_idx: int,
        prompt_str: str,
        general: list[dict],
        constraint: list[dict],
        adaptive: list[dict],
    ) -> None:
        """Persist prompt + the three rubric groups as a per-row override.

        Always writes all three groups — empty lists for ones not applicable to
        the current mode — so the dataset's view is self-consistent and a mode
        switch (or reload) won't leave stale rubrics from a previous group.
        """
        override = {
            "raw_prompt": [{"role": "user", "content": prompt_str}],
        }
        for group, pairs in (
            ("general", general),
            ("constraint", constraint),
            ("adaptive", adaptive),
        ):
            override[f"requirements_{group}"] = [p["rubric"] for p in pairs]
            override[f"weights_{group}"] = [int(p["importance"]) for p in pairs]
        self._dataset.set_override(real_idx, override)

    def _mutate_inplace(
        self,
        pos: int,
        new_prompt_str: str,
        general: list[dict],
        constraint: list[dict],
        adaptive: list[dict],
        batch: DataProto,
    ) -> None:
        """Mutate batch columns at position ``pos`` to reflect the modification.

        Called only in the step schedule, after ``set_override`` has persisted
        the same change. The trainer invokes this BEFORE ``_get_gen_batch``
        splits the batch, so every relevant column (``raw_prompt`` /
        ``requirements`` / ``weights`` / ``prompt``) is still on the single
        combined batch and one write per column is enough.
        """
        combined = general + constraint + adaptive
        reqs = [p["rubric"] for p in combined]
        ws = [int(p["importance"]) for p in combined]
        new_messages = [{"role": "user", "content": new_prompt_str}]

        ntb = batch.non_tensor_batch
        if "raw_prompt" in ntb:
            ntb["raw_prompt"][pos] = new_messages
        if "requirements" in ntb:
            ntb["requirements"][pos] = reqs
        if "weights" in ntb:
            ntb["weights"][pos] = ws
        if "prompt" in ntb:
            ntb["prompt"][pos] = new_prompt_str

    # ── DataProto + rollout helpers ────────────────────────────────────────

    def _build_gen_batch_from_indices(
        self,
        indices: list[int],
        prompt_overrides: dict[int, str] | None = None,
    ) -> DataProto:
        """Build a DataProto from dataset rows at the given logical indices.

        ``prompt_overrides`` maps logical index -> replacement prompt string. When
        present, the sample's ``raw_prompt`` becomes ``[{"role":"user","content":p}]``.
        """
        from verl.utils.dataset.rl_dataset import collate_fn

        data_list = []
        for idx in indices:
            row_dict = self._dataset[idx]
            if prompt_overrides is not None and idx in prompt_overrides:
                row_dict = copy.copy(row_dict)
                row_dict["raw_prompt"] = [
                    {"role": "user", "content": prompt_overrides[idx]},
                ]
            data_list.append(row_dict)
        batch_dict = collate_fn(data_list)
        return DataProto.from_single_dict(batch_dict)

    def _gen_rollouts(
        self,
        indices: list[int],
        n: int = 2,
        min_valid: int = 2,
        prompt_overrides: dict[int, str] | None = None,
    ) -> list[list[str]]:
        """Run rollouts per sample with retry on truncation/empty response.

        First attempt requests ``n`` samples per prompt. Each retry round
        requests ``round(n * data_generator.rollout_retry_oversample)``
        samples per prompt (clamped to ``>= n``); the additional samples
        cover the heavy-tail truncation cases where a stuck prompt would
        otherwise bounce through every retry round at the original ``n``.
        Cap on retry rounds is ``data_generator.rollout_max_retry`` (defaults
        to ``max_retry``). Failures → empty list (caller settles).
        """
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        assert self._rollout_mgr is not None, "rollout manager not set"
        N = len(indices)
        rollouts: list[list[str] | None] = [None] * N
        pending_local: list[int] = list(range(N))  # local position in `indices`

        num_workers = len(self._rollout_mgr.agent_loop_workers)

        for attempt in range(self._rollout_max_retry + 1):
            if not pending_local:
                break

            # Oversample on retry rounds only. attempt=0 stays at n for
            # back-compat; attempt>=1 scales by rollout_retry_oversample.
            if attempt == 0:
                n_attempt = n
            else:
                n_attempt = max(n, int(round(n * self._rollout_retry_oversample)))

            sub_indices = [indices[p] for p in pending_local]
            sub_overrides = None
            if prompt_overrides is not None:
                sub_overrides = {
                    indices[p]: prompt_overrides[indices[p]] for p in pending_local if indices[p] in prompt_overrides
                }

            sub_batch = self._build_gen_batch_from_indices(sub_indices, sub_overrides)
            preview_batch = sub_batch.repeat(repeat_times=n_attempt, interleave=True)
            preview_batch, pad_size = pad_dataproto_to_divisor(preview_batch, num_workers)

            output = self._rollout_mgr.generate_sequences(preview_batch)
            output = unpad_dataproto(output, pad_size)

            still_pending: list[int] = []
            for k, p in enumerate(pending_local):
                texts = [self._decode_response(output, k * n_attempt + j) for j in range(n_attempt)]
                valid = [t for t in texts if t]
                if len(valid) >= min_valid:
                    rollouts[p] = valid[:n]
                else:
                    still_pending.append(p)

            pending_local = still_pending
            if pending_local and attempt < self._rollout_max_retry:
                logger.info(
                    "rollout retry %d/%d: %d samples remaining (next attempt n=%d)",
                    attempt + 1,
                    self._rollout_max_retry,
                    len(pending_local),
                    max(n, int(round(n * self._rollout_retry_oversample))),
                )

        if pending_local:
            logger.warning(
                "rollout: %d samples failed all %d retries",
                len(pending_local),
                self._rollout_max_retry,
            )

        for p in range(N):
            if rollouts[p] is None:
                rollouts[p] = []
        return rollouts  # type: ignore[return-value]

    def _gen_generator_base_responses(
        self,
        prompts: list[str],
        n: int = 2,
    ) -> list[list[str]]:
        """Generate ``n`` base responses per prompt using the 8B generator.

        Used by modes in ``_MODES_WITH_GENERATOR_BASE_ROLLOUT`` where the
        constraint judge should see responses from the generator model
        rather than the policy. Each prompt is sent ``n`` times to get
        ``n`` independent samples; positions that come back empty are
        retried up to ``self._max_retry``. Caller must hold the GEN wake
        — these calls go through ``_call_generator``. Reuses the
        configured generator sampling knobs (temperature / top_p / top_k /
        enable_thinking) and strips the ``<think>`` block the same way
        ``_decode_response`` does for policy rollouts, so downstream
        stages see a consistent "response only" string.
        """
        N = len(prompts)
        rollouts: list[list[str]] = [[] for _ in range(N)]
        thinking_pat = self._parse_patterns.get("thinking")

        for attempt in range(self._max_retry + 1):
            flat_prompts: list[str] = []
            flat_pos: list[int] = []
            for pos in range(N):
                need = n - len(rollouts[pos])
                for _ in range(need):
                    flat_prompts.append(prompts[pos])
                    flat_pos.append(pos)
            if not flat_prompts:
                break

            outputs = asyncio.run(self._call_generator(flat_prompts))

            for k, out in enumerate(outputs):
                _, resp = split_thinking(out, thinking_pat)
                resp = resp.strip()
                if resp:
                    rollouts[flat_pos[k]].append(resp)

            still_pending = sum(1 for r in rollouts if len(r) < n)
            if still_pending == 0:
                break
            if attempt < self._max_retry:
                logger.info(
                    "generator base response retry %d/%d: %d samples remaining",
                    attempt + 1,
                    self._max_retry,
                    still_pending,
                )

        incomplete = sum(1 for r in rollouts if len(r) < n)
        if incomplete:
            logger.warning(
                "generator base response: %d samples failed all %d retries",
                incomplete,
                self._max_retry,
            )
        return rollouts

    def _decode_response(self, output: DataProto, idx: int) -> str:
        response_ids = output.batch["responses"][idx]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        _, response = split_thinking(text, self._parse_patterns.get("thinking"))
        return response

    def _extract_instruction(self, messages) -> str:
        if isinstance(messages, str):
            return messages
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                )
        return ""

    # ── Template loading ───────────────────────────────────────────────────

    def _load_template(self, key: str) -> str:
        if key in self._template_cache:
            return self._template_cache[key]
        path = self._template_paths.get(key)
        if not path:
            raise ValueError(f"No template path configured for key {key!r}")
        text = Path(path).read_text(encoding="utf-8")
        self._template_cache[key] = text
        return text

    # ── Generator HTTP call ────────────────────────────────────────────────

    async def _call_local(self, session: aiohttp.ClientSession, prompt: str) -> str:
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        if self._model_path:
            payload["model"] = self._model_path
        if self._top_k is not None and int(self._top_k) >= 0:
            payload["top_k"] = int(self._top_k)
        if self._enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}
        url = f"http://{self._router_addr}/v1/chat/completions"
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Empty response from {url}")
        return choices[0].get("message", {}).get("content", "") or ""

    async def _call_generator(self, prompts: list[str]) -> list[str]:
        # In-cluster vLLM router. Keep total=None — under load generations can
        # legitimately exceed any fixed cap.
        connector = aiohttp.TCPConnector(limit=max(0, self._concurrency_limit))
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            return await asyncio.gather(*[self._call_local(session, p) for p in prompts])

    # ── Debug dump ─────────────────────────────────────────────────────────

    def _dump_debug(self, records: list[dict]) -> None:
        if not self._debug_dump_dir or not records:
            return
        self._debug_dump_dir.mkdir(parents=True, exist_ok=True)
        # Group by label → separate file per pipeline stage.
        by_label: dict[str, list[dict]] = {}
        for rec in records:
            by_label.setdefault(rec["label"], []).append(rec)
        for label, group in by_label.items():
            path = self._debug_dump_dir / f"{label}.jsonl"
            with path.open("a") as f:
                for rec in group:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
