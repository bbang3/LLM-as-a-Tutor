from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

from verl.utils.import_utils import load_extern_object

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
for path in (CURRENT_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_SOURCE_FN_CACHE: dict[tuple[str, str], Any] = {}


def _normalize_score_result(result: Any, source_name: str) -> dict[str, Any]:
    if isinstance(result, dict):
        if "score" not in result:
            raise ValueError(f"Reward scorer for data_source={source_name!r} must return a 'score' field.")
        normalized = dict(result)
    else:
        normalized = {"score": result}

    normalized["score"] = float(normalized["score"])
    return normalized


def _load_source_fn(source_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    module_path = source_cfg.get("path")
    fn_name = source_cfg.get("name")
    if not module_path or not fn_name:
        raise ValueError("Each source_dispatch entry must define path and name.")

    cache_key = (module_path, fn_name)
    if cache_key not in _SOURCE_FN_CACHE:
        _SOURCE_FN_CACHE[cache_key] = load_extern_object(module_path=module_path, object_name=fn_name)

    return _SOURCE_FN_CACHE[cache_key], dict(source_cfg.get("reward_kwargs", {}))


def _get_source_cfg(source_dispatch: dict[str, Any] | None, data_source: str) -> dict[str, Any]:
    dispatch_table = dict(source_dispatch or {})
    dispatch_cfg = dispatch_table.get(data_source, dispatch_table.get("__default__"))
    if dispatch_cfg is None:
        raise ValueError(f"No source_dispatch entry found for data_source={data_source!r}.")
    return dispatch_cfg


async def _call_source_fn(raw_fn: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(raw_fn)
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if not has_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    if inspect.iscoroutinefunction(raw_fn):
        return await raw_fn(**kwargs)
    return raw_fn(**kwargs)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    reward_router_address: str | None = None,
    reward_model_tokenizer=None,
    wildchecklists_kwargs: dict[str, Any] | None = None,
    source_dispatch: dict[str, Any] | None = None,
    **legacy_kwargs,
) -> dict[str, Any]:
    rubric_kwargs = {**legacy_kwargs, **(wildchecklists_kwargs or {})}
    dispatch_table = dict(source_dispatch or {})
    dispatch_table.setdefault(
        "__default__",
        {
            "path": "verl/llm_tutor/reward_wildchecklists.py",
            "name": "compute_score_rubric_list",
            "reward_kwargs": rubric_kwargs,
        },
    )
    dispatch_cfg = _get_source_cfg(dispatch_table, data_source)

    raw_fn, reward_kwargs = _load_source_fn(dispatch_cfg)
    call_kwargs = {
        "data_source": data_source,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        "reward_router_address": reward_router_address,
        "reward_model_tokenizer": reward_model_tokenizer,
        **reward_kwargs,
    }
    return _normalize_score_result(await _call_source_fn(raw_fn, call_kwargs), source_name=data_source)


compute_score_dispatch = compute_score
