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

from verl.trainer.ppo.policy_ray_trainer import PolicyRayPPOTrainer

logger = logging.getLogger(__name__)


class RubricPolicyRayPPOTrainer(PolicyRayPPOTrainer):
    """Policy trainer configured for rubric-based GRPO reward computation."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is not None:
            self._validate_rubric_setup(config)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validate_rubric_setup(config) -> None:
        if not config.reward_model.use_reward_loop:
            raise ValueError(
                "RubricPolicyRayPPOTrainer requires `reward_model.use_reward_loop=True` "
                "so rubric scores are computed by reward loop workers."
            )

        custom_reward_path = config.custom_reward_function.get("path", None)
        if custom_reward_path is None:
            raise ValueError(
                "RubricPolicyRayPPOTrainer requires `custom_reward_function.path` "
                "(e.g. `verl/llm_tutor/reward_function.py`)."
            )

        if config.reward_model.get("reward_manager", "naive") != "rubric":
            logger.warning(
                "For rubric training, set `reward_model.reward_manager=rubric` so prompt/requirement fields "
                "are forwarded to the custom rubric reward function."
            )

    def init_workers(self):
        super().init_workers()

        if not self.use_reward_loop or not hasattr(self, "reward_loop_manager"):
            return

        original_compute_rm_score = self.reward_loop_manager.compute_rm_score

        def _compute_rm_score_with_meta_fix(batch):
            rm_scores = original_compute_rm_score(batch)
            # Remove potentially stale reward_extra_keys from batch.meta_info so that
            # rm_scores' authoritative version is used after union (avoids assertion failure
            # when the agent loop stored an empty or different reward_extra_keys list).
            if rm_scores.meta_info.get("reward_extra_keys") is not None:
                batch.meta_info.pop("reward_extra_keys", None)
            return rm_scores

        self.reward_loop_manager.compute_rm_score = _compute_rm_score_with_meta_fix
