"""Offline general-rubric pre-generation, driven by the training config.

Reads the same Hydra config the trainer uses, then for every row in
``cfg.data.train_files``:

  1. Build a general-rubric prompt from ``cfg.data_generator.general_rubric_prompt``.
  2. Call the generator (local vLLM) using the same model + sampling params the
     online pipeline would use (``cfg.data_generator.{model_path, max_tokens,
     temperature, top_p, top_k, enable_thinking}``).
  3. Parse with ``cfg.data_generator.parse_patterns`` and validate; retry up to
     ``cfg.data_generator.max_retry`` times.

Writes the result back to the same parquet (overwriting), adding two columns:

  * ``requirements_general`` (list[str])
  * ``weights_general``      (list[int])

``AdaptiveDataset`` auto-seeds these into per-row overrides on init, so the
online pipeline's Step 2 (general rubric) becomes a no-op for every seeded row
and ``modify_dataset_epoch`` can never drop a sample on general-rubric failure.

Usage (mirrors the trainer's CLI):

    uv run python -m data_generation.offline_general_rubric \\
        --config-path /home/sangminhwang/prompting-llm-verl/configs \\
        --config-name <child config that has hydra.searchpath set>

Hydra overrides are accepted on the command line, e.g.
``data.train_files=./other.parquet`` or
``reward_model.model.path=Qwen/Qwen3-8B``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from data_generation.parse_utils import (
    is_valid_rubric,
    parse_rubric_output,
    split_thinking,
)


def _resolve_path(p: str, root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return root / path


def _build_prompt_local(tokenizer, template: str, instruction: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": template.format(instruction=instruction)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _instruction_text(prompt) -> str:
    """Pull the user instruction out of the parquet's ``prompt`` column.

    Wildchecklists stores ``prompt`` as a chat-template list of {role, content}
    dicts (a numpy array of dicts after parquet round-trip). We pick the first
    user turn's content. Multi-turn conversations aren't concatenated — the
    rubric template expects a single instruction string. Strings pass through.
    """
    if isinstance(prompt, str):
        return prompt
    try:
        turns = list(prompt)
    except TypeError as e:
        raise TypeError(f"Unsupported 'prompt' type: {type(prompt).__name__}") from e
    for turn in turns:
        if isinstance(turn, dict) and turn.get("role") == "user":
            content = turn.get("content")
            if isinstance(content, str):
                return content
    raise ValueError(f"No user-role turn with string content in prompt: {prompt!r}")


def _train_file(cfg: DictConfig) -> str:
    """Return the (single) training file path or HF dataset ID as a string.
    Lists are rejected — the offline script writes back to the same path so
    a list would be ambiguous.
    """
    files = cfg.data.train_files
    if isinstance(files, str):
        return files
    if isinstance(files, (list, tuple)) or OmegaConf.is_list(files):
        as_list = list(files)
        if len(as_list) != 1:
            raise ValueError(
                f"data.train_files must be a single path for offline rubric pre-gen, "
                f"got {len(as_list)} entries: {as_list!r}"
            )
        return as_list[0]
    raise TypeError(f"Unsupported data.train_files type: {type(files).__name__}")


def _is_hf_dataset_id(s: str) -> bool:
    """True when s looks like a HuggingFace Hub dataset ID (owner/name)."""
    return not s.endswith((".parquet", ".json")) and not Path(s).is_absolute() and "/" in s and not Path(s).exists()


def _check_hf_dataset(dataset_id: str) -> None:
    """Download an HF dataset and verify every row already has a general rubric.

    * If all rows are seeded → prints a summary and exits 0 (skip Stage 1).
    * If any rows are missing rubrics → exits with an error asking the user to
      generate rubrics locally first and push the seeded version to the Hub.
    """
    import datasets as hf_datasets

    print(f"[hf] Detected HuggingFace dataset: {dataset_id!r}")
    print(f"[hf] Downloading to local cache ...")
    ds = hf_datasets.load_dataset(dataset_id)["train"]
    print(f"[hf] {len(ds)} rows loaded.")

    if "requirements_general" not in ds.column_names:
        sys.exit(
            f"[hf] Column 'requirements_general' not found in {dataset_id!r}.\n"
            "      Download the dataset to a local parquet, run offline rubric "
            "pre-gen on it, then push the seeded version to the Hub."
        )

    missing = sum(1 for row in ds if not row.get("requirements_general"))
    if missing:
        sys.exit(
            f"[hf] {missing}/{len(ds)} rows in {dataset_id!r} are missing "
            "'requirements_general'.\n"
            "      Download the dataset to a local parquet, run offline rubric "
            "pre-gen on it, then push the seeded version to the Hub."
        )

    print(f"[hf] All {len(ds)} rows already have general rubrics — skipping Stage 1.")
    sys.exit(0)


def _model_name(cfg: DictConfig) -> str:
    name = cfg.data_generator.get("model_path", None)
    if name:
        return str(name)
    name = cfg.reward_model.model.get("path", None)
    if name:
        return str(name)
    raise ValueError(
        "Could not resolve generator model name from "
        "data_generator.model_path or reward_model.model.path"
    )


def _gpu_mem_util(cfg: DictConfig) -> float:
    util = cfg.data_generator.get("gpu_memory_utilization", None)
    if util is not None:
        return float(util)
    util = cfg.reward_model.rollout.get("gpu_memory_utilization", None)
    if util is not None:
        return float(util)
    return 0.9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline general-rubric pre-gen using the training config.",
        # Keep argparse out of Hydra's way: forward unknown args as overrides.
        allow_abbrev=False,
    )
    parser.add_argument("--config-path", required=True, help="Directory containing the Hydra config.")
    parser.add_argument("--config-name", required=True, help="Config name (without .yaml).")
    args, overrides = parser.parse_known_args()

    config_dir = os.path.abspath(args.config_path)
    if not os.path.isdir(config_dir):
        sys.exit(f"--config-path is not a directory: {config_dir}")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    root = Path.cwd()
    train_file = _train_file(cfg)
    if _is_hf_dataset_id(train_file):
        _check_hf_dataset(train_file)  # exits 0 on success, non-zero on error

    parquet_path = _resolve_path(train_file, root)
    if not parquet_path.exists():
        sys.exit(f"Training parquet not found: {parquet_path}")

    dg = cfg.data_generator
    parse_patterns = OmegaConf.to_container(dg.get("parse_patterns") or {}, resolve=True)
    max_retry = int(dg.get("max_retry", 5))
    overwrite_existing = bool(dg.get("offline_overwrite_existing", False))

    template_path = _resolve_path(str(dg.general_rubric_prompt), root)
    template = template_path.read_text(encoding="utf-8")

    sampling = {
        "temperature": float(dg.get("temperature", 0.6)),
        "top_p": float(dg.get("top_p", 0.95)),
        "max_tokens": int(dg.get("max_tokens", 8192)),
        "n": 1,
    }
    top_k = dg.get("top_k", None)
    if top_k is not None and int(top_k) >= 0:
        sampling["top_k"] = int(top_k)
    enable_thinking = bool(dg.get("enable_thinking", True))

    display_model = _model_name(cfg)

    print(f"[config]   train_files:        {parquet_path}")
    print(f"[config]   model:              {display_model}")
    print(f"[config]   sampling:           {sampling}  enable_thinking={enable_thinking}")
    print(f"[config]   max_retry:          {max_retry}")
    print(f"[config]   overwrite_existing: {overwrite_existing}")

    df = pd.read_parquet(parquet_path)
    N = len(df)
    print(f"[load]     {N} rows, columns={list(df.columns)}")

    if "prompt" not in df.columns:
        sys.exit("Input parquet missing required 'prompt' column.")

    # Decide which rows need fresh generation. With overwrite=False, rows that
    # already have a non-empty general rubric are kept as-is; with True, every
    # row is regenerated.
    has_existing = "requirements_general" in df.columns and "weights_general" in df.columns

    def _row_has_seed(i: int) -> bool:
        if not has_existing:
            return False
        reqs = df.iloc[i]["requirements_general"]
        ws = df.iloc[i]["weights_general"]
        try:
            return reqs is not None and ws is not None and len(reqs) > 0 and len(ws) > 0
        except TypeError:
            return False

    if overwrite_existing:
        todo = list(range(N))
    else:
        todo = [i for i in range(N) if not _row_has_seed(i)]
    n_skip = N - len(todo)
    print(f"[plan]     generating for {len(todo)}/{N} rows (skipping {n_skip} already seeded)")

    if not todo:
        print("[done]     nothing to do — every row already has a general rubric.")
        return

    from data_generation.vllm_dp import build_llm

    print(f"[load]     loading vLLM ({display_model})")
    model = build_llm(
        display_model,
        _gpu_mem_util(cfg),
        cfg.get("data_parallel_size", "auto"),
    )
    tokenizer = model.get_tokenizer()

    results: dict[int, list[dict]] = {}
    remaining = list(todo)
    for attempt in range(max_retry + 1):
        if not remaining:
            break
        prompts = [
            _build_prompt_local(tokenizer, template, _instruction_text(df.iloc[i]["prompt"]), enable_thinking)
            for i in remaining
        ]
        outputs = model.generate(prompts, sampling)
        raw_outputs = [out[0] if out else "" for out in outputs]
        still: list[int] = []
        for k, idx in enumerate(remaining):
            raw = raw_outputs[k]
            _, resp = split_thinking(raw, parse_patterns.get("thinking"))
            _, pairs = parse_rubric_output(resp, parse_patterns)
            if is_valid_rubric(pairs, require_importance=True):
                results[idx] = pairs
            else:
                still.append(idx)
        remaining = still
        if remaining and attempt < max_retry:
            print(f"  retry {attempt + 1}/{max_retry}: {len(remaining)} samples remaining")

    model.shutdown()

    failed_set = set(remaining)
    new_reqs: list[list[str]] = []
    new_ws: list[list[int]] = []
    for i in range(N):
        if i in results:
            pairs = results[i]
            new_reqs.append([str(p["rubric"]) for p in pairs])
            new_ws.append([int(p["importance"]) for p in pairs])
        elif i in failed_set:
            # All retries exhausted. Write an empty rubric so the parquet is
            # still complete and usable; surface a warning at the end so the
            # caller (and Stage 2) can decide whether to proceed.
            new_reqs.append([])
            new_ws.append([])
        else:
            # Pre-seeded row that wasn't regenerated (overwrite=False path).
            row = df.iloc[i]
            new_reqs.append([str(r) for r in row["requirements_general"]])
            new_ws.append([int(w) for w in row["weights_general"]])

    df = df.copy()
    df["requirements_general"] = new_reqs
    df["weights_general"] = new_ws

    df.to_parquet(parquet_path, index=False)
    n_regen = len(results)
    n_failed = len(failed_set)
    n_preserved = N - n_regen - n_failed
    print(
        f"[write]    overwrote {parquet_path} — "
        f"{n_regen} regenerated, {n_preserved} preserved, {n_failed} failed (empty rubric)"
    )
    if failed_set:
        head = sorted(failed_set)[:50]
        more = f" ... and {n_failed - 50} more" if n_failed > 50 else ""
        print(
            f"[warn]     {n_failed}/{N} rows have an EMPTY general rubric after {max_retry} retries. "
            f"Downstream will treat them as zero-criterion (no reward signal). "
            f"Failed indices: {head}{more}"
        )


if __name__ == "__main__":
    main()
