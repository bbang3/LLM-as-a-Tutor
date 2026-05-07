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

import logging
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@register("weighted_dapo")
class WeightedDAPORewardManager(AbstractRewardManager):
    """Reward Manager based on Weighted DAPO"""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        num_rollouts=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_rollouts = num_rollouts

        if self.overlong_buffer_cfg is not None:
            assert (
                self.max_resp_len is not None
            ), f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            assert (
                self.max_resp_len >= self.overlong_buffer_cfg.len
            ), "max_resp_len must be larger than overlong_buffer.len"

    def _collect_values_from_reward_tensor(self, data: DataProto):
        # Get device from rm_scores to ensure tensor is on the same device
        rm_scores = data[0].batch["rm_scores"]
        reward_values = torch.zeros(len(data), dtype=torch.float32, device=rm_scores.device)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            reward = data_item.batch["rm_scores"][valid_response_length - 1]
            reward_values[i] = reward

        return reward_values.reshape(-1, self.num_rollouts)

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        print("\nAdding overlong buffer penalty to reward tensor\n")

        if not self.overlong_buffer_cfg.enable:
            return self._extract_reward_from_rm_scores(data, return_dict)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        reward_values = self._collect_values_from_reward_tensor(data)
        if self.overlong_buffer_cfg.penalty_weight == "std":
            reward_values_std = reward_values.std(dim=1)
            penalty_coeffs = reward_values_std.repeat_interleave(self.num_rollouts)
        elif self.overlong_buffer_cfg.penalty_weight == "gap":
            reward_values_gap = reward_values.max(dim=1).values - reward_values.min(dim=1).values
            penalty_coeffs = reward_values_gap.repeat_interleave(self.num_rollouts)
        else:
            raise ValueError(f"Invalid penalty weight: {self.overlong_buffer_cfg.penalty_weight}")

        assert len(penalty_coeffs) == len(
            data
        ), f"len(penalty_coeffs) != len(data): {len(penalty_coeffs)} != {len(data)}"

        already_print_data_sources = {}
        logger.debug(f"Processing {len(data)} data items")

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().item()

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            reward = data_item.batch["rm_scores"][valid_response_length - 1]

            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)

            reward += overlong_reward * penalty_coeffs[i]
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"].append(overlong_reward)
                reward_extra_info["overlong_reward_weighted"].append(overlong_reward * penalty_coeffs[i].item())
                reward_extra_info["overlong"].append(overlong_reward < 0)
                reward_extra_info["overlong_coeff"].append(penalty_coeffs[i].item())

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
