# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
- Preprocess the wildchat dataset to parquet format
- Supports HuggingFace datasets, JSON, and JSONL files as input
"""

import argparse
import json
import os

import datasets
from datasets import load_dataset
from templates.utils import load_template


def _is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


def generate_rl_dataset_from_json(
    dataset_path: str,
    dataset_name: str,
    prompt_column_name: str,
    is_conversation: bool,
    save_dir: str = "~/data",
    num_samples: int = None,
    meta_prompt_template_name: str = None,
):
    """
    Preprocess a JSON/JSONL file to parquet format, preserving all original keys.
    """
    if meta_prompt_template_name is not None:
        meta_prompt_template = load_template(meta_prompt_template_name)
    else:
        meta_prompt_template = None

    with open(dataset_path, "r", encoding="utf-8") as f:
        if dataset_path.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            raw = json.load(f)
            records = raw if isinstance(raw, list) else [raw]

    if num_samples is not None:
        records = records[:num_samples]

    before = len(records)
    records = [
        r
        for r in records
        if "requirements" not in r or (isinstance(r.get("requirements"), list) and len(r["requirements"]) > 0)
    ]
    if len(records) < before:
        print(f"Filtered {before - len(records)}/{before} records with empty requirements")

    processed = []
    for idx, record in enumerate(records):
        if is_conversation:
            prompt = record[prompt_column_name][0]["content"]
        else:
            prompt = record[prompt_column_name]

        if meta_prompt_template is not None:
            prompt = meta_prompt_template.format(seed_prompt=prompt)

        row = {**record}
        row["data_source"] = dataset_name
        row["prompt"] = [{"role": "user", "content": prompt}]
        row["ability"] = "alignment"
        row["reward_model"] = {"style": "model"}
        row["extra_info"] = {"split": "train", "index": idx}
        processed.append(row)

    local_dir = os.path.expanduser(save_dir)
    os.makedirs(os.path.join(local_dir, dataset_name), exist_ok=True)

    ds = datasets.Dataset.from_list(processed)
    local_path = os.path.join(local_dir, dataset_name, "train.parquet")
    ds.to_parquet(local_path)
    print(f"Saved train ({len(processed)} rows) -> {local_path}")
    print(f"Preserved keys: {ds.column_names}")


def generate_rl_dataset(
    dataset_path: str,
    dataset_name: str,
    prompt_column_name: str,
    is_conversation: bool,
    save_dir: str = "~/data",
    subset_name: str = None,
    splits: list[str] = None,
    num_samples: int = None,
    meta_prompt_template_name: str = None,
):
    """
    Preprocess the dataset to parquet format.
    Args:
        dataset_path: The path to the raw dataset.
        dataset_name: The name of the dataset.
        prompt_column_name: The name of the column that contains the prompt.
        is_conversation: Whether the dataset is in a conversation format.
        save_dir: The directory to save the preprocessed dataset.
        subset_name: The name of the subset to preprocess.
        splits: List of splits to process. Defaults to ["train"]. Processes all available
                splits when set to None.
        num_samples: Number of samples to use (applied per split).
        meta_prompt_template_name: The meta prompt template name.
    """
    if _is_json_file(dataset_path):
        return generate_rl_dataset_from_json(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            prompt_column_name=prompt_column_name,
            is_conversation=is_conversation,
            save_dir=save_dir,
            num_samples=num_samples,
            meta_prompt_template_name=meta_prompt_template_name,
        )

    if subset_name is not None:
        full_dataset = load_dataset(dataset_path, subset_name)
    else:
        full_dataset = load_dataset(dataset_path)

    if meta_prompt_template_name is not None:
        meta_prompt_template = load_template(meta_prompt_template_name)
    else:
        meta_prompt_template = None

    target_splits = splits if splits is not None else list(full_dataset.keys())

    local_dir = os.path.expanduser(save_dir)
    os.makedirs(os.path.join(local_dir, dataset_name), exist_ok=True)

    def make_map_fn(split):
        def process_fn(example, idx):
            if is_conversation:
                prompt = example.pop(prompt_column_name)[0]["content"]
            else:
                prompt = example.pop(prompt_column_name)

            if meta_prompt_template is not None:
                prompt = meta_prompt_template.format(seed_prompt=prompt)

            data = {
                "data_source": dataset_name,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "alignment",
                "reward_model": {
                    "style": "model",
                },
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    for split in target_splits:
        if split not in full_dataset:
            print(f"Warning: split '{split}' not found in dataset, skipping.")
            continue

        split_data = full_dataset[split]

        if num_samples is not None:
            split_data = split_data.select(range(min(num_samples, len(split_data))))

        split_data = split_data.map(function=make_map_fn(split), with_indices=True)

        local_path = os.path.join(local_dir, dataset_name, f"{split}.parquet")
        split_data.to_parquet(local_path)
        print(f"Saved {split} ({len(split_data)} rows) -> {local_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the raw dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset.")
    parser.add_argument(
        "--prompt_column_name",
        type=str,
        required=True,
        help="The name of the column that contains the prompt.",
    )
    parser.add_argument(
        "--is_conversation",
        action="store_true",
        default=False,
        help="Whether the dataset is a conversation dataset.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./datasets",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="The name of the subset to preprocess. Some datasets require a subset name to be specified.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Splits to process (e.g. --splits train validation). Defaults to all available splits.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use per split.",
    )
    parser.add_argument(
        "--meta_prompt_template_name",
        type=str,
        default=None,
        help="The meta prompt to train prompter model.",
    )

    args = parser.parse_args()

    generate_rl_dataset(
        args.dataset_path,
        args.dataset_name,
        args.prompt_column_name,
        args.is_conversation,
        args.save_dir,
        args.subset_name,
        args.splits,
        args.num_samples,
        args.meta_prompt_template_name,
    )
