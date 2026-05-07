# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import ListConfig, OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import (
    find_latest_ckpt_path,
    should_save_ckpt_esi,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn, load_extern_object
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__name__)


def _freq_triggered(freq, global_steps, is_last_step, is_last_batch_in_epoch):
    """Return True if a frequency-gated action (save/test) should fire this step.

    `freq` may be:
    - the string "epoch" (fire at every epoch boundary),
    - a positive int (fire every N steps),
    - a list combining the above (e.g. [25, "epoch"]).
    Non-positive ints / None disable the action.
    """
    if isinstance(freq, (list, ListConfig)):
        return any(_freq_triggered(f, global_steps, is_last_step, is_last_batch_in_epoch) for f in freq)
    if isinstance(freq, str):
        if freq == "epoch":
            return is_last_batch_in_epoch or is_last_step
        raise ValueError(f"Unsupported freq value: {freq!r} (expected 'epoch' or int)")
    if freq is None or freq <= 0:
        return False
    return is_last_step or global_steps % freq == 0


class PolicyRayPPOTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning for policy training."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self._reward_keep_keys = self._collect_reward_keep_keys()

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        # ``val_files`` may be null when in-training validation is disabled;
        # skip val dataloader construction in that case. ``test_freq`` and
        # ``val_before_train`` should also be off so ``_validate()`` is never
        # called.
        val_files = self.config.data.get("val_files", None)
        if val_dataset is None and val_files:
            from omegaconf import OmegaConf

            val_data_config = OmegaConf.merge(self.config.data, OmegaConf.create({"filter_overlong_prompts": False}))
            val_dataset = create_rl_dataset(
                val_files,
                val_data_config,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        if self.val_dataset is not None:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
            val_loader_size: int | str = len(self.val_dataloader)
        else:
            self.val_dataloader = None
            val_loader_size = 0

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        logger.info(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{val_loader_size}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.debug(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, filename_suffix=None):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(
            dump_path,
            f"{self.global_steps}{'.' + filename_suffix if filename_suffix else ''}.jsonl",
        )

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.debug(f"Dumped generations to {filename}")

    @staticmethod
    def _normalize_reward_extra_info_batch(reward_extra_info: dict[str, Any], batch_size: int) -> dict[str, list[Any]]:
        normalized: dict[str, list[Any]] = {}
        for key, values in reward_extra_info.items():
            if isinstance(values, np.ndarray):
                batch_values = values.tolist()
            elif isinstance(values, list):
                batch_values = values
            else:
                batch_values = [values]

            if len(batch_values) == 1 and batch_size > 1:
                batch_values = batch_values * batch_size

            if len(batch_values) != batch_size:
                raise ValueError(
                    f"Validation metric {key!r} has batch length {len(batch_values)}, expected {batch_size}."
                )
            normalized[key] = batch_values
        return normalized

    def _log_rollout_data(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _collect_reward_keep_keys(self) -> set[str]:
        custom_reward_cfg = self.config.get("custom_reward_function", {})
        reward_kwargs = custom_reward_cfg.get("reward_kwargs", {}) if custom_reward_cfg else {}
        source_dispatch = reward_kwargs.get("source_dispatch", {}) if reward_kwargs else {}

        keep_keys: set[str] = set({"data_source", "reward_model", "extra_info", "uid"})
        if not hasattr(source_dispatch, "items"):
            return keep_keys

        for _, source_cfg in source_dispatch.items():
            if not hasattr(source_cfg, "get"):
                continue
            route_keep_keys = source_cfg.get("keep_keys", [])
            if isinstance(route_keep_keys, str):
                route_keep_keys = [route_keep_keys]
            for key in route_keep_keys or []:
                if isinstance(key, str) and key:
                    keep_keys.add(key)

        return keep_keys

    def _aggregate_validation_metrics_for_source(
        self,
        data_source: str,
        reward_extra_infos_dict: dict[str, list[Any]],
    ) -> dict[str, float]:
        custom_reward_cfg = self.config.get("custom_reward_function", {})
        reward_kwargs = custom_reward_cfg.get("reward_kwargs", {})
        source_dispatch = dict(reward_kwargs.get("source_dispatch", {}) or {})
        source_cfg = source_dispatch.get(data_source, source_dispatch.get("__default__"))
        if not source_cfg or not hasattr(source_cfg, "get"):
            return {}

        aggregate_name = source_cfg.get("aggregate_name")
        if not aggregate_name:
            return {}

        aggregate_fn = load_extern_object(module_path=source_cfg["path"], object_name=aggregate_name)
        result = aggregate_fn(reward_extra_infos_dict)
        return {str(key): float(value) for key, value in dict(result).items()}

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            reward_extra_info = result["reward_extra_info"]

            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info.update(
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                reward_extra_infos_dict.update(reward_extra_info)
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # Keep keys needed for reward. Additional source-specific keys can be
        # declared in custom_reward_function.reward_kwargs.source_dispatch.*.keep_keys.
        reward_model_keys = self._reward_keep_keys & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        """Run rollouts on val_dataloader and (optionally) dump them to disk.

        Validation produces raw per-source rollout dumps under
        ``validation_data_dir`` and returns an empty metric dict — no
        per-source aggregate metric computation is performed. Callers gate
        this on ``trainer.test_freq`` and ``trainer.val_before_train``.
        """
        if self.val_dataloader is None:
            logger.info("[validation] no val_dataloader configured; skipping")
            return {}

        source_samples: dict[str, list[dict]] = defaultdict(list)
        sample_inputs: list[str] = []
        sample_outputs: list[str] = []
        sample_scores: list[float] = []
        sample_turns: list[np.ndarray] = []

        fallback_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)

        n_val_batches = len(self.val_dataloader) if hasattr(self.val_dataloader, "__len__") else None
        logger.info(
            "[validation] step=%d: rollout over %s batches (raw-dump only; adapter dispatch disabled)",
            self.global_steps,
            n_val_batches if n_val_batches is not None else "?",
        )

        for batch_idx, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            orig_sources = np.array([str(s) for s in test_batch.non_tensor_batch["data_source"]])
            global_n = fallback_n

            test_batch = test_batch.repeat(repeat_times=global_n, interleave=True)

            # Snapshot per-sample row dicts before ``_get_gen_batch`` pops
            # non_tensor fields off ``test_batch``.
            pre_gen_rows = [
                {k: v[i] for k, v in test_batch.non_tensor_batch.items()} for i in range(len(test_batch.batch))
            ]

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            _uniq_srcs, _src_counts_arr = np.unique(orig_sources, return_counts=True)
            _src_counts = dict(sorted(zip(_uniq_srcs.tolist(), _src_counts_arr.tolist(), strict=True)))
            logger.info(
                "[validation] batch %s/%s: rollout start — %d samples (%s) x%d runs",
                batch_idx + 1,
                n_val_batches if n_val_batches is not None else "?",
                len(orig_sources),
                ", ".join(f"{s}={c}" for s, c in _src_counts.items()),
                global_n,
            )
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            logger.info(
                "[validation] batch %s/%s: rollout done",
                batch_idx + 1,
                n_val_batches if n_val_batches is not None else "?",
            )

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            input_ids = test_output_gen_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            if "__num_turns__" in test_output_gen_batch.non_tensor_batch:
                sample_turns.append(test_output_gen_batch.non_tensor_batch["__num_turns__"])

            for i_expanded, row in enumerate(pre_gen_rows):
                src = str(row.get("data_source", "unknown"))
                run_idx = i_expanded % global_n
                source_samples[src].append(
                    {
                        "row": row,
                        "response": output_texts[i_expanded],
                        "input_text": input_texts[i_expanded],
                        "run_idx": run_idx,
                    }
                )
                sample_inputs.append(input_texts[i_expanded])
                sample_outputs.append(output_texts[i_expanded])
                sample_scores.append(0.0)

        logger.info(
            "[validation] rollout phase done; per-source counts: %s",
            {src: len(samples) for src, samples in source_samples.items()},
        )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            for src, samples in source_samples.items():
                self._dump_generations(
                    inputs=[s["input_text"] for s in samples],
                    outputs=[s["response"] for s in samples],
                    gts=[None] * len(samples),
                    scores=[0.0] * len(samples),
                    reward_extra_infos_dict={},
                    dump_path=val_data_dir,
                    filename_suffix=src,
                )

        metric_dict: dict[str, float] = {}
        if len(sample_turns) > 0:
            sample_turns_cat = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns_cat.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns_cat.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns_cat.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel],
                    config=self.config.reward_model,
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

        # Online data generator: active only when train_dataset is AdaptiveDataset.
        self.data_generator = None

        if self.config.data.get("custom_cls", {}).get("name") == "AdaptiveDataset":
            from llm_tutor.online_data_generator import (
                OnlineDataGenerator,
            )

            dg_cfg = self.config.get("data_generator", {}) or {}
            mode = dg_cfg.get("mode", "constraint_rubric")
            schedule = dg_cfg.get("schedule", "epoch")
            self.data_generator = OnlineDataGenerator(
                config=self.config,
                tokenizer=self.tokenizer,
                mode=mode,
                schedule=schedule,
            )
            self.data_generator.set_managers(self.async_rollout_manager, self.reward_loop_manager)
            self.data_generator.set_dataset(self.train_dataset)
            # Fail fast if the TRAIN dataset is missing general rubrics.
            if hasattr(self.train_dataset, "_require_general_rubric_seed"):
                self.train_dataset._require_general_rubric_seed()
            logger.info(
                "AdaptiveDataset detected — OnlineDataGenerator initialized (mode=%s, schedule=%s)",
                mode,
                schedule,
            )
        else:
            logger.info("AdaptiveDataset not in use — reading dataset as-is")

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        logger.info(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.debug(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    str(Role.Critic),
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # Persist AdaptiveDataset mutable state (overrides + active indices)
        # so a resumed run sees the same dataset view the checkpoint was
        # taken against. Without this the resumed run would re-run the full
        # per-epoch generator pipeline and produce different overrides/drops.
        if hasattr(self.train_dataset, "get_adaptive_state"):
            adaptive_state_path = os.path.join(local_global_step_folder, "adaptive_dataset.pt")
            torch.save(self.train_dataset.get_adaptive_state(), adaptive_state_path)

        # Also dump a human-readable parquet snapshot for inspection. The
        # state above is the source of truth for resume; this parquet is only
        # for debugging (per-step view of the dataset at each save point).
        if hasattr(self.train_dataset, "export_snapshot"):
            snapshot_path = os.path.join(local_global_step_folder, "dataset_snapshot.parquet")
            try:
                self.train_dataset.export_snapshot(snapshot_path)
            except Exception as e:
                logger.warning("Failed to export dataset snapshot at %s: %s", snapshot_path, e)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            logger.debug("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                logger.info("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_from_path
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        logger.info(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        logger.info(f"Setting global step to {self.global_steps}")
        logger.info(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # AdaptiveDataset state (overrides + active indices) must be restored
        # BEFORE the dataloader state, because the saved dataloader position
        # was recorded against the post-drop dataset length.
        if hasattr(self.train_dataset, "load_adaptive_state"):
            adaptive_state_path = os.path.join(global_step_folder, "adaptive_dataset.pt")
            if os.path.exists(adaptive_state_path):
                adaptive_state = torch.load(adaptive_state_path, weights_only=False)
                self.train_dataset.load_adaptive_state(adaptive_state)
                logger.info(
                    "AdaptiveDataset restored: %d overrides, %d active samples",
                    len(adaptive_state.get("row_overrides", {})),
                    len(adaptive_state.get("active_indices", [])),
                )
            else:
                logger.warning(
                    "AdaptiveDataset state not found at %s; resumed run will "
                    "re-generate overrides from scratch (non-deterministic)",
                    adaptive_state_path,
                )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            # A checkpoint saved at a clean epoch boundary persists a
            # StatefulDataLoader state whose iterator is already exhausted.
            # Loading it would make the resumed run's first `enumerate()` yield
            # zero batches, silently skipping the entire next epoch. Only load
            # the saved state for mid-epoch resumes; clean-boundary resumes
            # start the next epoch with a fresh iterator.
            if self.global_steps % len(self.train_dataloader) == 0:
                logger.info(
                    "Skipping dataloader state restore at clean epoch boundary "
                    "(global_steps=%d, len(train_dataloader)=%d): saved state is "
                    "exhausted and would skip the next epoch",
                    self.global_steps,
                    len(self.train_dataloader),
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(
        self,
        batch: DataProto,
        metrics,
        logging_prefix="global_seqlen",
        keep_minibatch=False,
    ):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            entity_name=self.config.trainer.entity_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            logger.info(f"Initial validation metrics: {val_metrics}")
            tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # Last epoch's data-generation timings, re-injected into every step's
        # timing_raw so wandb gets a continuous value across the epoch.
        gen_timing_raw: dict[str, float] = {}
        # When resuming from a mid-epoch checkpoint, the AdaptiveDataset state for
        # the current epoch was already produced by the original session and
        # restored from the checkpoint — skip pre-epoch generation for the first
        # outer iteration only. (global_steps was incremented by 1 above, so a
        # clean epoch boundary leaves global_steps - 1 divisible by dataloader len.)
        skip_first_pre_epoch_gen = (self.global_steps - 1) % len(self.train_dataloader) != 0
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            # ── Pre-epoch online data generation (epoch schedule only) ──────
            # Per-epoch parquet snapshots are intentionally NOT written here:
            # save_freq triggers (incl. "epoch") already dump dataset_snapshot.parquet
            # into each global_step_N/ checkpoint folder, and the contents are
            # identical when save_freq fires at an epoch boundary.
            if skip_first_pre_epoch_gen:
                logger.info(
                    "Epoch %d — skipping pre-epoch dataset generation (mid-epoch resume)",
                    epoch,
                )
                skip_first_pre_epoch_gen = False
            elif self.data_generator is not None and self.data_generator.schedule == "epoch":
                logger.info("Epoch %d — running pre-epoch dataset generation", epoch)
                # Reset so marked_timer's accumulate-into-dict semantics don't
                # sum across epochs.
                gen_timing_raw = {}
                with marked_timer("data_generation", gen_timing_raw, color="magenta"):
                    n_modified = self.data_generator.modify_dataset_epoch(epoch=epoch)
                logger.info(
                    "Pre-epoch generation complete: %d samples modified",
                    n_modified,
                )

                logger.info(f" TIMES: pre-epoch data generation (epoch={epoch}) ".center(80, "="))
                logger.info(f"{'stage':<32}{'elapsed (s)':>12}")
                logger.info("-" * 44)
                for key, val in gen_timing_raw.items():
                    logger.info(f"{key:<32}{val:>12.3f}")
                logger.info("=" * 80)

            epoch_num_batches = len(self.train_dataloader)
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                is_last_batch_in_epoch = batch_idx == epoch_num_batches - 1
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                if gen_timing_raw:
                    timing_raw.update(gen_timing_raw)

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                if "prompts" in batch.batch:
                    n_prompts = len(batch.batch["prompts"])
                elif len(batch.batch) > 0:
                    first_batch_key = next(iter(batch.batch.keys()))
                    n_prompts = len(batch.batch[first_batch_key])
                else:
                    n_prompts = 0
                logger.info(
                    f"[Step {self.global_steps}] "
                    f"prompts={n_prompts}  "
                    f"rollout.n={self.config.actor_rollout_ref.rollout.n}  "
                    "start policy training step"
                )

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # ── Per-step online data generation (step schedule only) ───
                # Runs BEFORE _get_gen_batch so that mutating raw_prompt (and
                # requirements/weights/prompt) on the combined batch propagates
                # naturally into both sides after the split.
                #
                # ``modify_batch`` takes LOGICAL indices (positions in the
                # active set), but the batch carries REAL dataframe indices in
                # ``_dataset_idx``. Convert via ``_active_indices`` so drops
                # done elsewhere (not currently done in step mode, but possible
                # if modes are mixed) don't silently read the wrong rows.
                if self.data_generator is not None and self.data_generator.schedule == "step":
                    real_idxs = [int(i) for i in batch.non_tensor_batch["_dataset_idx"]]
                    active = getattr(self.train_dataset, "_active_indices", None)
                    if active is not None:
                        real_to_logical = {r: i for i, r in enumerate(active)}
                        step_indices = [real_to_logical[r] for r in real_idxs]
                    else:
                        step_indices = real_idxs
                    with marked_timer("data_generation", timing_raw, color="magenta"):
                        self.data_generator.set_epoch(epoch)
                        self.data_generator.set_step(self.global_steps)
                        n_modified = self.data_generator.modify_batch(
                            step_indices,
                            batch=batch,
                        )
                    logger.info(
                        "[Step %d] per-step modify_batch: %d/%d modified",
                        self.global_steps,
                        n_modified,
                        len(step_indices),
                    )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    logger.info("1. Generate policy completions".center(80, "-"))
                    # if self.async_rollout_mode:
                    #     raw_prompt = gen_batch_output.non_tensor_batch["raw_prompt"][0]
                    #     policy_rollout_input_prompt = self.tokenizer.apply_chat_template(
                    #         raw_prompt,
                    #         add_generation_prompt=True,
                    #         tokenize=False,
                    #     )
                    # else:
                    #     policy_rollout_input_prompt = self.tokenizer.batch_decode(
                    #         gen_batch_output.batch["input_ids"][:1], skip_special_tokens=True
                    #     )[0]

                    # logger.debug(
                    #     f"[Step {self.global_steps}] prompt:\n"
                    #     f"{[policy_rollout_input_prompt]}"
                    # )

                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    logger.info("2. Assign policy rewards".center(80, "-"))
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    logger.info("3. Compute log_probs/values/advantages".center(80, "-"))
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import (
                            apply_bypass_mode,
                        )

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import (
                                    calculate_debug_metrics,
                                )

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        logger.info("4. Update critic".center(80, "-"))
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    else:
                        logger.info("4. Skip critic update".center(80, "-"))

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        logger.info("5. Update actor".center(80, "-"))
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.val_reward_fn is not None and _freq_triggered(
                    self.config.trainer.test_freq,
                    self.global_steps,
                    is_last_step,
                    is_last_batch_in_epoch,
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                save_freq = self.config.trainer.save_freq
                save_enabled = isinstance(save_freq, (str, list, ListConfig)) or (
                    isinstance(save_freq, (int, float)) and save_freq > 0
                )
                if save_enabled and (
                    esi_close_to_expiration
                    or _freq_triggered(save_freq, self.global_steps, is_last_step, is_last_batch_in_epoch)
                ):
                    if esi_close_to_expiration:
                        logger.warning("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Print per-step timing summary
                step_keys_order = [
                    "gen",
                    "reward",
                    "old_log_prob",
                    str(Role.RefPolicy),
                    "values",
                    "adv",
                    "update_critic",
                    "update_actor",
                    "testing",
                    "save_checkpoint",
                ]
                logger.info(f" TIMES: step {self.global_steps} ".center(80, "="))
                logger.info(f"{'step':<32}{'elapsed (s)':>12}")
                logger.info("-" * 44)
                for key in step_keys_order:
                    if key in timing_raw:
                        logger.info(f"{key:<32}{timing_raw[key]:>12.3f}")
                other_keys = [k for k in timing_raw if k not in step_keys_order and k != "step"]
                for key in other_keys:
                    logger.info(f"{key:<32}{timing_raw[key]:>12.3f}")
                logger.info("-" * 44)
                logger.info(f"{'TOTAL (step)':<32}{steps_duration:>12.3f}")
                max_response_length = batch.batch["responses"].shape[-1]
                response_lengths = batch.batch["response_mask"].bool().sum(dim=-1)
                prompt_lengths = batch.batch["attention_mask"][:, :-max_response_length].bool().sum(dim=-1)
                logger.info(f"{'max prompt len':<32}{int(prompt_lengths.max()):>12}")
                logger.info(f"{'max response len':<32}{int(response_lengths.max()):>12}")
                logger.info(f"{'mean response len':<32}{response_lengths.float().mean().item():>12.1f}")
                if "rubric_judgement_completions" in batch.non_tensor_batch:
                    all_judge_lens = []
                    for outputs_json in batch.non_tensor_batch["rubric_judgement_completions"]:
                        try:
                            for judge_out in json.loads(outputs_json):
                                all_judge_lens.append(len(judge_out))
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if all_judge_lens:
                        logger.info(f"{'max judge output len':<32}{max(all_judge_lens):>12}")
                        logger.info(f"{'mean judge output len':<32}{sum(all_judge_lens) / len(all_judge_lens):>12.1f}")
                logger.info("=" * 80)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                tracker.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}",
                        sub_dir=f"step{self.global_steps}",
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    logger.info(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # If the outer loop exited naturally without `is_last_step` ever firing
        # (e.g. AdaptiveDataset.drop_samples made the real step count lower
        # than the init-time total_training_steps), still guarantee a final
        # save + validation on the last trained weights.
        last_completed_step = self.global_steps - 1
        if last_completed_step >= 1:
            # We're here because is_last_step never fired. Replicate what the in-loop
            # gating *would* have decided at last_completed_step, treating it as an
            # epoch boundary (the loop's last batch always has is_last_batch_in_epoch=True).
            already_validated = _freq_triggered(
                self.config.trainer.test_freq,
                last_completed_step,
                is_last_step=False,
                is_last_batch_in_epoch=True,
            )
            if self.val_reward_fn is not None and (last_val_metrics is None or not already_validated):
                logger.info(
                    "Post-loop fallback: running final validation at step %d "
                    "(is_last_step never fired — likely because AdaptiveDataset drops "
                    "reduced the actual step count below total_training_steps=%d)",
                    last_completed_step,
                    self.total_training_steps,
                )
                val_metrics = self._validate()
                if val_metrics:
                    last_val_metrics = val_metrics
                    # wandb's internal step counter was advanced to
                    # self.global_steps (= last_completed_step + 1) by the
                    # final per-step tracker.log inside the loop. Logging at
                    # last_completed_step here would be rejected as "less
                    # than current step"; use self.global_steps so wandb
                    # plots the final eval at the post-training position.
                    tracker.log(data=val_metrics, step=self.global_steps)
            save_freq = self.config.trainer.save_freq
            save_enabled = isinstance(save_freq, (str, list, ListConfig)) or (
                isinstance(save_freq, (int, float)) and save_freq > 0
            )
            already_saved = _freq_triggered(
                save_freq,
                last_completed_step,
                is_last_step=False,
                is_last_batch_in_epoch=True,
            )
            if save_enabled and not already_saved:
                logger.info(
                    "Post-loop fallback: saving final checkpoint at step %d",
                    last_completed_step,
                )
                # Inside the step loop global_steps is incremented after
                # _save_checkpoint, so the checkpoint dir always matches the
                # step number that was just completed. Mirror that here:
                # global_steps was incremented after the last completed step,
                # so roll it back for the save call.
                self.global_steps -= 1
                try:
                    self._save_checkpoint()
                finally:
                    self.global_steps += 1
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
        logger.info(f"Final validation metrics: {last_val_metrics}")
        progress_bar.close()
