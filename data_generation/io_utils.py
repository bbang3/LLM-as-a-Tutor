"""Configuration loading, dataset I/O, and record saving utilities."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from datasets import Dataset, load_dataset


def load_config(config_path: Path) -> dict:
    """Load a YAML config file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, root: Path) -> Path:
    """Resolve *path_str* relative to *root* (unless already absolute)."""
    p = Path(path_str)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_samples(
    ds_cfg: dict,
    root: Path | None = None,
    strip_rubrics: bool = False,
) -> list[dict]:
    """Load samples from a local file or HuggingFace dataset.

    Supports ``path`` key (local JSON/JSONL) or ``name``+``split`` (HF Hub).
    If *strip_rubrics* is True, only the ``prompt`` field is kept and
    ``requirements``/``weights`` are initialised to empty lists (useful for
    pipelines that generate their own rubrics from scratch).
    """
    num_samples = ds_cfg.get("num_samples")

    if "path" in ds_cfg:
        if root is None:
            root = Path.cwd()
        file_path = resolve_path(ds_cfg["path"], root)
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
        else:
            with open(file_path, encoding="utf-8") as f:
                samples = json.load(f)
        if num_samples is not None:
            samples = samples[:num_samples]
    else:
        dataset = load_dataset(ds_cfg["name"], split=ds_cfg["split"])
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        samples = list(dataset)

    if strip_rubrics:
        samples = [{"prompt": s["prompt"], "requirements": [], "weights": []} for s in samples]

    return samples


def save_records(records: list[dict], out_cfg: dict, root: Path) -> None:
    """Save records to JSON/JSONL and optionally push to HuggingFace Hub."""
    out_dir = resolve_path(out_cfg["dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "filename" not in out_cfg:
        raise ValueError("output.filename is required in the config but was not set.")
    filename_stem = str(out_cfg["filename"])
    output_format = str(out_cfg.get("format", "json")).lower().strip()
    if output_format not in {"jsonl", "json"}:
        raise ValueError(f"Unsupported output.format: {output_format!r}")

    if output_format == "json":
        out_path = out_dir / f"{filename_stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to {out_path}")
    else:
        out_path = out_dir / f"{filename_stem}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL to {out_path}")

    hub_repo = out_cfg.get("hub_repo")
    if hub_repo:
        Dataset.from_list(records).push_to_hub(hub_repo)
        print(f"Pushed to Hub: {hub_repo}")
    else:
        print("hub_repo not set; skipping upload.")
