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
import os

import torch


def set_basic_config(level):
    """
    This function sets the global logging format and level. It will be called when import verl.

    The ``VERL_LOG_LEVEL`` environment variable (e.g. ``DEBUG``, ``INFO``, ``WARNING``)
    overrides the default level so a run can be made more/less verbose without
    touching code. Ray workers inherit the driver's environment, so setting this
    on the driver also takes effect in every worker process.
    """
    env_level = os.environ.get("VERL_LOG_LEVEL")
    if env_level:
        resolved = getattr(logging, env_level.upper(), None)
        if isinstance(resolved, int):
            level = resolved
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)
    # basicConfig is a no-op if the root logger already has handlers (common
    # when Ray workers reimport verl). Force the level regardless so the env
    # var always wins.
    logging.getLogger().setLevel(level)


def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")
