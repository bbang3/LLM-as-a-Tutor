"""
AdaptiveDataset: extends RLHFDataset with per-row override support for online data modification.

Each training step, OnlineDataGenerator runs a preview rollout on the current batch,
calls an 8B generator to produce constraint/rubric modifications, and calls
set_override(idx, {...}) to persist the change. On the next access of that dataset
index (either same epoch cross-batch or any future epoch), __getitem__ returns the
modified row instead of the original.

Usage in yaml config:
    data:
      custom_cls:
        path: verl/llm_tutor/adaptive_dataset.py
        name: AdaptiveDataset
      shuffle: false   # required: _dataset_idx must always map to the same logical sample
      online_data_generator_mode: rubric  # "constraint" | "rubric" | "constraint_rubric"
"""

import logging
from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset import RLHFDataset

logger = logging.getLogger(__name__)


# Three persistent rubric groups. Each carries `requirements_<group>` (list[str])
# and `weights_<group>` (list[int]) in the override dict. At read time they are
# concatenated into the legacy `requirements` / `weights` columns so downstream
# reward/keep_keys paths don't need to change.
_RUBRIC_GROUPS = ("general", "constraint", "adaptive")


class AdaptiveDataset(RLHFDataset):
    """Extends RLHFDataset with _row_overrides for per-step online modification.

    __getitem__ injects ``_dataset_idx`` into each returned dict so that
    OnlineDataGenerator can call set_override(idx, ...) with the correct index.
    Overrides are applied on the next (and all subsequent) accesses of that index,
    persisting across epoch boundaries without touching the underlying HF dataframe.

    Supported override keys:
        "raw_prompt"              – replaces the messages list returned as raw_prompt
        "requirements_general"    – general rubric texts (instruction-only, generated once)
        "weights_general"         – matching importance weights
        "requirements_constraint" – accumulated constraint rubric texts (appended each
                                    epoch with a fresh constraint; persists even when a
                                    later epoch's constraint judgment returns no)
        "weights_constraint"      – matching importance weights
        "requirements_adaptive"   – adaptive rubric texts from the latest epoch
                                    (replaced each epoch)
        "weights_adaptive"        – matching importance weights

    The three rubric groups are concatenated into ``requirements`` / ``weights``
    at __getitem__ time so downstream reward functions still read a single list.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        self._row_overrides: dict[int, dict] = {}  # dataset_idx → {col: new_val}
        self._active_indices: list[int] = list(range(len(self.dataframe)))  # indices not dropped

        n_seeded = self._seed_general_rubrics_from_dataframe()
        logger.info(
            "AdaptiveDataset initialised: %d samples, mode=%s, pre-seeded general rubrics: %d",
            len(self.dataframe),
            config.get("online_data_generator_mode", "constraint"),
            n_seeded,
        )

    def _seed_general_rubrics_from_dataframe(self) -> int:
        """Seed ``requirements_general`` / ``weights_general`` overrides from the
        input parquet's columns of the same names.

        This lets the offline general-rubric script (``data_generation/
        offline_general_rubric.py``) pre-fill general rubrics that the online
        pipeline then treats as already-persisted, so ``modify_batch`` skips
        Step 2 for those samples. Without this seed, general generation runs on
        the first visit and a parse failure would cause the sample to drop.

        Returns the number of rows that had non-empty general seeds.
        """
        cols = set(getattr(self.dataframe, "column_names", []) or [])
        if "requirements_general" not in cols or "weights_general" not in cols:
            return 0

        n_seeded = 0
        for idx in range(len(self.dataframe)):
            row = self.dataframe[idx]
            reqs = row.get("requirements_general")
            ws = row.get("weights_general")
            if reqs is None or ws is None:
                continue
            # HF datasets return numpy arrays for list-typed columns; normalise.
            reqs_list = [str(r) for r in reqs]
            ws_list = [int(w) for w in ws]
            if not reqs_list or len(reqs_list) != len(ws_list):
                continue
            self._row_overrides[idx] = {
                "requirements_general": reqs_list,
                "weights_general": ws_list,
            }
            n_seeded += 1
        return n_seeded

    def _require_general_rubric_seed(self) -> None:
        """Fail-fast: every active row must have a non-empty general rubric.

        Catches misconfiguration (forgot to run ``offline_general_rubric.py``,
        wrong train_files path, parquet missing the columns) at trainer
        startup instead of mid-epoch when the online pipeline silently
        regenerates from scratch and may drop samples on parse failures.
        """
        missing: list[int] = []
        for idx in self._active_indices:
            override = self._row_overrides.get(idx, {})
            reqs = override.get("requirements_general") or []
            if not reqs:
                missing.append(idx)
        if not missing:
            return
        head = missing[:10]
        more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise ValueError(
            f"AdaptiveDataset: {len(missing)}/{len(self._active_indices)} rows are "
            f"missing a general rubric (idxs: {head}{more}). "
            f"Pre-generate it with `python -m data_generation.offline_general_rubric "
            f"--config-path <configs> --config-name <child config>` to populate the "
            f"`requirements_general` and `weights_general` columns on the train parquet."
        )

    def set_override(self, idx: int, override: dict) -> None:
        """Persist a modification for dataset row ``idx``.

        Called by OnlineDataGenerator after each per-step modification.
        Merges with any existing override for the same index so that constraint
        and rubric overrides can co-exist when mode="constraint_rubric".

        Enforces that each rubric group's ``requirements_<g>`` and ``weights_<g>``
        are written together and have matching lengths once the merge result
        contains both, since the reward function validates the pair and would
        otherwise raise mid-training.
        """
        if idx in self._row_overrides:
            merged = dict(self._row_overrides[idx])
            merged.update(override)
        else:
            merged = dict(override)

        for group in _RUBRIC_GROUPS:
            req_key = f"requirements_{group}"
            w_key = f"weights_{group}"
            has_req = req_key in merged
            has_w = w_key in merged
            if has_req != has_w:
                raise ValueError(f"override for idx={idx}: {req_key} and {w_key} must be set together")
            if has_req and has_w and len(merged[req_key]) != len(merged[w_key]):
                raise ValueError(
                    f"override for idx={idx}: {req_key} length ({len(merged[req_key])}) "
                    f"!= {w_key} length ({len(merged[w_key])})"
                )

        self._row_overrides[idx] = merged

    def get_original_prompt(self, idx: int) -> str:
        """Return the immutable original prompt text for ``idx``.

        Reads the underlying dataframe's ``prompt`` column and bypasses any
        ``_row_overrides`` accumulation. Used by reset modes that need the
        un-modified base prompt as the join target each epoch.
        """
        return self._extract_prompt_text(self.dataframe[idx].get("prompt"))

    def get_accumulated_rubrics(self, idx: int) -> dict[str, list[dict]]:
        """Return the currently persisted rubric pairs for ``idx`` grouped by source.

        Returns ``{"general": [...], "constraint": [...], "adaptive": [...]}`` where
        each list contains ``{"rubric": str, "importance": int}`` pairs. Missing
        groups yield empty lists — callers can treat "not yet generated" and
        "generated empty" uniformly.
        """
        override = self._row_overrides.get(idx, {})
        result: dict[str, list[dict]] = {}
        for group in _RUBRIC_GROUPS:
            reqs = override.get(f"requirements_{group}", [])
            ws = override.get(f"weights_{group}", [])
            result[group] = [{"rubric": str(r), "importance": int(w)} for r, w in zip(reqs, ws, strict=True)]
        return result

    def drop_samples(self, indices_to_drop: list[int]) -> None:
        """Remove samples by their original dataframe indices.

        Dropped samples are excluded from __len__ / __getitem__ so the
        dataloader never sees them.  Overrides for dropped indices are also
        cleaned up.
        """
        drop_set = set(indices_to_drop)
        self._active_indices = [i for i in self._active_indices if i not in drop_set]
        for i in indices_to_drop:
            self._row_overrides.pop(i, None)
        logger.info("Dropped %d samples, %d active remain", len(drop_set), len(self._active_indices))

    def __len__(self):
        return len(self._active_indices)

    def get_adaptive_state(self) -> dict:
        """Return the full mutable state for checkpointing.

        Includes per-row overrides and the active index list so that a
        checkpoint can be resumed into the exact dataset view the trainer was
        iterating when the checkpoint was taken.
        """
        return {
            "row_overrides": dict(self._row_overrides),
            "active_indices": list(self._active_indices),
        }

    def load_adaptive_state(self, state: dict) -> None:
        """Restore state produced by ``get_adaptive_state``.

        ``active_indices`` is optional for backward compatibility with older
        snapshots that did not save it. Any stale ``general_rubric_cache`` field
        from a pre-hotfix snapshot is silently ignored — general rubrics now
        live inside ``row_overrides`` as ``requirements_general``.
        """
        self._row_overrides = dict(state.get("row_overrides", {}))
        active = state.get("active_indices")
        if active is not None:
            self._active_indices = list(active)

    @staticmethod
    def _extract_prompt_text(prompt) -> str:
        """Extract the last user message as plain text from various prompt formats."""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)):
            msgs = prompt
        elif hasattr(prompt, "tolist"):
            msgs = prompt.tolist()
        else:
            return str(prompt)
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def export_snapshot(self, path: str) -> None:
        """Export the current dataset state with per-group rubric breakdown.

        Schema aligns with the offline generation output
        (``generate_constraint_rubric.py``) so epoch snapshots can be
        analysed / compared with the same tooling.
        """
        import pandas as pd

        records = []
        for idx in self._active_indices:
            raw_row = dict(self.dataframe[idx])
            override = self._row_overrides.get(idx, {})

            original_prompt = self._extract_prompt_text(raw_row.get("prompt"))

            if "raw_prompt" in override:
                prompt = self._extract_prompt_text(override["raw_prompt"])
            else:
                prompt = original_prompt

            gen_reqs = list(override.get("requirements_general", []))
            gen_ws = list(override.get("weights_general", []))
            con_reqs = list(override.get("requirements_constraint", []))
            con_ws = list(override.get("weights_constraint", []))
            adp_reqs = list(override.get("requirements_adaptive", []))
            adp_ws = list(override.get("weights_adaptive", []))

            reqs = gen_reqs + con_reqs + adp_reqs
            ws = gen_ws + con_ws + adp_ws

            sources = ["general"] * len(gen_reqs) + ["constraint"] * len(con_reqs) + ["adaptive"] * len(adp_reqs)

            constraint_added = bool(con_reqs)
            adaptive_added = bool(adp_reqs)

            records.append(
                {
                    "prompt": prompt,
                    "original_prompt": original_prompt,
                    "constraint_added": constraint_added,
                    "adaptive_added": adaptive_added,
                    "requirements": reqs,
                    "general_requirements": gen_reqs,
                    "adaptive_requirements": adp_reqs,
                    "constraint_requirements": con_reqs,
                    "criterion_sources": sources,
                    "weights": ws,
                    "general_weights": gen_ws,
                    "adaptive_weights": adp_ws,
                    "constraint_weights": con_ws,
                    "data_source": raw_row.get("data_source", ""),
                }
            )

        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)
        logger.info("Exported dataset snapshot (%d rows) to %s", len(df), path)

    @staticmethod
    def _concat_rubric_groups(override: dict) -> tuple[list | None, list | None]:
        """Concatenate ``requirements_<g>``/``weights_<g>`` across groups.

        Returns (None, None) if no group key is present in ``override`` so
        callers can distinguish "override doesn't touch rubrics" from
        "override sets rubrics to empty".
        """
        if not any(f"requirements_{g}" in override for g in _RUBRIC_GROUPS):
            return None, None
        reqs: list = []
        ws: list = []
        for g in _RUBRIC_GROUPS:
            reqs.extend(override.get(f"requirements_{g}", []))
            ws.extend(override.get(f"weights_{g}", []))
        return reqs, ws

    def __getitem__(self, item):
        # Map logical index to actual dataframe index via active_indices.
        real_idx = self._active_indices[item]
        row_dict: dict = self.dataframe[real_idx]
        row_dict["raw_prompt"] = self._build_messages(row_dict)

        # Expose the real dataframe index so OnlineDataGenerator can call set_override.
        row_dict["_dataset_idx"] = real_idx

        # Apply any pending override for this row (constraint and/or rubric).
        if real_idx in self._row_overrides:
            override = self._row_overrides[real_idx]
            if "raw_prompt" in override:
                row_dict["raw_prompt"] = override["raw_prompt"]
            reqs, ws = self._concat_rubric_groups(override)
            if reqs is not None:
                row_dict["requirements"] = reqs
                row_dict["weights"] = ws

        # Dummy tensor required so DataProto.batch is never empty.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s", index)
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
