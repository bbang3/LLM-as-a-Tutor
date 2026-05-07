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

import inspect
import logging
import os
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register as register_manager
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register as register_manager_legacy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register_manager("rubric")
@register_manager_legacy("rubric")
class RubricRewardManager(RewardManagerBase):
    """Reward manager tailored for rubric-based (checklist) rewards.

    Inherits directly from RewardManagerBase (not RateLimitedRewardManager) to avoid
    asyncio.wait_for timeouts that can cause orphaned in-flight requests on the reward
    model vLLM server, eventually leading to CUDA illegal memory access errors.
    Rate limiting is unnecessary for an internal vLLM server.
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        tokenizer: AutoTokenizer | None = None,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        **kwargs,
    ):
        if config is None:
            config = DictConfig({"reward_model": {}})
        if tokenizer is None:
            raise TypeError("RubricRewardManager requires `tokenizer`.")
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def _compute_reward(
        self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict
    ) -> dict | float:
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            return await self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            return await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

    async def _decode_prompt(self, data_item) -> str:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
        valid_prompt_length = max(valid_prompt_length, 0)
        valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
        )

    @staticmethod
    def _extract_ground_truth(data_item) -> Any:
        reward_model_info = data_item.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_model_info, Mapping):
            return reward_model_info.get("ground_truth", None)
        return None

    def _build_extra_info(self, data_item, prompt_str: str) -> dict[str, Any]:
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        if not isinstance(extra_info, Mapping):
            extra_info = {"dataset_extra_info": extra_info}
        else:
            extra_info = dict(extra_info)

        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if isinstance(tool_extra_fields, Mapping):
            extra_info.update(tool_extra_fields.items())

        if "prompt_str" not in extra_info:
            extra_info["prompt_str"] = prompt_str
        if "enable_thinking" not in extra_info:
            extra_info["enable_thinking"] = bool(
                self.config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking", False)
            )

        ignored_keys = {
            "extra_info",
            "tool_extra_fields",
            "reward_model",
            "data_source",
            "uid",
        }
        for key, value in data_item.non_tensor_batch.items():
            if key in ignored_keys:
                continue
            if key not in extra_info:
                extra_info[key] = value

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores
        return extra_info

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch.get("data_source", "unknown")
        ground_truth = self._extract_ground_truth(data_item)
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        prompt_str = await self._decode_prompt(data_item)
        extra_info = self._build_extra_info(data_item, prompt_str=prompt_str)

        result = await self._compute_reward(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        reward_extra_info: dict[str, Any] = {}
        if isinstance(result, dict):
            reward = float(result.get("score", 0.0))
            reward_extra_info.update(result)
        else:
            reward = float(result)

        reward_extra_info.setdefault("score", reward)

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Make the manager callable for compatibility with the existing reward manager interface."""
        import asyncio
        from collections import defaultdict

        import torch

        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        async def process_batch():
            tasks = [self.run_single(data[i : i + 1]) for i in range(len(data))]
            return await asyncio.gather(*tasks)

        results = self.loop.run_until_complete(process_batch())

        for i, result in enumerate(results):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            reward_tensor[i, valid_response_length - 1] = result["reward_score"]
            for key, value in result.get("reward_extra_info", {}).items():
                reward_extra_info[key].append(value)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
