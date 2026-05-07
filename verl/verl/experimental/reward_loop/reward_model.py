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

import asyncio
import logging
import os
from copy import deepcopy

import aiohttp

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Identifier for the default (8B generative judge) engine. Existing call sites
# (``RewardLoopManager``, the ``RewardLoopWorker``) call ``wake_up()`` /
# ``sleep()`` without a ``name`` argument; the default below preserves their
# behaviour.
DEFAULT_ENGINE = "judge"
SCALAR_RM_ENGINE = "scalar_rm"


class RewardModelManager:
    """Reward model manager.

    Holds one or more vLLM rollout-engine pools sharing the same GPU placement
    group via name-based wake/sleep rotation:

      * ``"judge"`` — the 8B generative judge (always present).
      * ``"scalar_rm"`` — optional scalar reward model used by the EVA baseline
        (instantiated when ``scalar_rm_config`` is supplied).

    Both engines colocate on the existing reward-model resource pool and never
    hold weights at the same time. Callers are responsible for invoking
    ``wake_up(name)`` / ``sleep(name)`` so only one engine is resident before
    a batch of inference calls.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        resource_pool: RayResourcePool = None,
        scalar_rm_config: dict | None = None,
    ):
        """
        Args:
            config: Judge reward-model configuration (existing semantics).
            resource_pool: Ray placement group. When supplied, both the judge
                and (optional) scalar RM colocate inside it.
            scalar_rm_config: Optional dict with the scalar-RM knobs:
                ``{"path": str, "max_model_len": int}``. The scalar RM
                inherits the judge's ``gpu_memory_utilization`` (only one
                engine is awake at a time, so a separate util knob is moot).
                The replica count is auto-derived to match the judge
                (``world_size // tp``) so the GPU pool is always fully
                utilised. When ``None`` the manager behaves exactly as
                before.
        """
        self.config = config
        self.resource_pool = resource_pool
        self._engines: dict[str, dict] = {}

        self._initialize_judge()
        if scalar_rm_config is not None:
            self._initialize_scalar_rm(scalar_rm_config)

        assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."
        # Free both engines on startup; trainer wakes them on demand.
        if self.config.rollout.free_cache_engine:
            for name in self._engines:
                self.sleep(name)

    # ── Engine bring-up ────────────────────────────────────────────────────

    def _initialize_judge(self) -> None:
        """Bring up the existing 8B generative-judge engine pool + router."""
        rollout_world_size = self.config.rollout.tensor_model_parallel_size
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size

        rollout_replica_class = get_rollout_replica_class(self.config.rollout.name)
        rollout_config = self.config.rollout
        model_config = HFModelConfig(
            path=self.config.model.path,
            external_lib=self.config.model.external_lib,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        self._launch_replicas(replicas, rollout_world_size)
        router_addr = self._launch_router([f"http://{r._server_address}" for r in replicas])

        # Backward-compat aliases used by other call sites.
        self.tokenizer = model_config.get_processor()
        self.rollout_replicas = replicas
        self.server_handles = [r._server_handle for r in replicas]
        self.server_addresses = [r._server_address for r in replicas]
        self.router_address = router_addr

        self._engines[DEFAULT_ENGINE] = {
            "replicas": replicas,
            "router_address": router_addr,
            "model_path": self.config.model.path,
        }

    def _initialize_scalar_rm(self, scalar_rm_config: dict) -> None:
        """Bring up a SECOND vLLM engine pool for the scalar RM.

        Reuses the judge's RolloutConfig but overrides path / max_model_len.
        ``gpu_memory_utilization`` is inherited from the judge: only one
        engine is awake at a time (judge ↔ scalar RM via wake/sleep), so
        the two pools should share the same util budget. The replica count
        is auto-derived from the GPU pool (``world_size // tp``) — same as
        the judge — so the colocated pool is always fully utilised. The
        pool is placed in the same resource_pool as the judge so the two
        never need to migrate GPUs.
        """
        path = scalar_rm_config["path"]
        max_model_len = int(scalar_rm_config.get("max_model_len", 4096))

        # Clone the judge's rollout config so vLLM CLI args (engine_kwargs,
        # enable_sleep_mode, gpu_memory_utilization, etc.) are inherited
        # verbatim. Mutate only the fields that differ for the scalar RM.
        rollout_config = deepcopy(self.config.rollout)
        rollout_config.max_model_len = max_model_len

        rollout_world_size = rollout_config.tensor_model_parallel_size

        world_size = (
            self.resource_pool.world_size
            if self.resource_pool
            else self.config.n_gpus_per_node * self.config.nnodes
        )
        n_replicas = world_size // rollout_world_size

        rollout_replica_class = get_rollout_replica_class(rollout_config.name)
        model_config = HFModelConfig(
            path=path,
            external_lib=self.config.model.external_lib,
            trust_remote_code=True,
        )
        replicas = [
            rollout_replica_class(
                replica_rank=100 + r,  # Offset to keep ray actor names disjoint from the judge.
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for r in range(n_replicas)
        ]
        self._launch_replicas(replicas, rollout_world_size)

        # No router for the scalar RM: ``score_pairs`` shards the batch
        # across replicas itself and POSTs directly to each replica's
        # ``_server_address``. A router would just add a hop with no benefit.
        self._engines[SCALAR_RM_ENGINE] = {
            "replicas": replicas,
            "model_path": path,
            "tokenizer": model_config.get_processor(),
        }
        logger.info(
            "scalar RM engine ready: path=%s, n_replicas=%d, gpu_mem=%.2f, max_model_len=%d",
            path,
            n_replicas,
            rollout_config.gpu_memory_utilization,
            max_model_len,
        )

    def _launch_replicas(self, replicas: list, rollout_world_size: int) -> None:
        """Init replicas (colocated or standalone) — extracted from the
        original ``_initialize_llm_servers`` logic to be reusable across the
        judge and scalar-RM pools.
        """
        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=rollout_world_size)
            assert len(split_resource_pools) >= len(replicas), (
                f"resource_pool has {len(split_resource_pools)} slots of width {rollout_world_size}"
                f" but {len(replicas)} replica(s) requested"
            )
            self._run_all(
                [
                    server.init_colocated(rp)
                    for server, rp in zip(replicas, split_resource_pools[: len(replicas)], strict=True)
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in replicas])

    @staticmethod
    def _launch_router(worker_urls: list[str]) -> str:
        from .router.naive_router import launch_router_process

        addr, _ = launch_router_process(worker_urls=worker_urls)
        return addr

    # ── Public API ─────────────────────────────────────────────────────────

    def get_router_address(self, name: str = DEFAULT_ENGINE) -> str:
        # Only the judge engine has a router; the scalar RM is addressed
        # per-replica from score_pairs.
        return self._engines[name]["router_address"]

    def wake_up(self, name: str = DEFAULT_ENGINE) -> None:
        """Wake up all rollout replicas of the named engine.

        Default (``"judge"``) preserves existing call-site behaviour so the
        trainer / reward-loop code keeps working unchanged.
        """
        self._run_all([replica.wake_up() for replica in self._engines[name]["replicas"]])

    def sleep(self, name: str = DEFAULT_ENGINE) -> None:
        """Sleep all rollout replicas of the named engine."""
        self._run_all([replica.sleep() for replica in self._engines[name]["replicas"]])

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (prompt, response) pairs with the scalar RM.

        Caller is expected to have woken the scalar RM (and slept the judge)
        first. Pairs are tokenised with the scalar RM's chat template,
        sharded round-robin across the engine's replicas to use the spare
        replicas in parallel, and POSTed to vLLM's ``/classify`` endpoint;
        the per-pair score is ``probs[-1]`` (last logit), matching the
        existing disrm pattern in ``reward_loop.compute_score_disrm``.

        Replica-level distribution: replica i serves indices [i, i+R, i+2R,
        ...]. Each replica's HTTP requests run concurrently within an
        aiohttp session, so total throughput scales with ``n_replicas``.
        Order of the returned list matches the input order.
        """
        if SCALAR_RM_ENGINE not in self._engines:
            raise RuntimeError("score_pairs called but scalar RM is not configured")
        if not pairs:
            return []

        engine = self._engines[SCALAR_RM_ENGINE]
        tokenizer = engine["tokenizer"]
        model_path = engine["model_path"]
        replicas = engine["replicas"]

        # Render each pair as the scalar RM's chat template. Reward models
        # like Skywork return a single classification head over the rendered
        # conversation; we don't add a generation prompt.
        rendered: list[str] = []
        for prompt, response in pairs:
            chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
                # Match reward_loop preprocessing: vLLM <0.11.2 also adds bos
                # so we strip the duplicate up-front.
                text = text[len(tokenizer.bos_token):]
            rendered.append(text)

        # Round-robin shard: pair index i goes to replica i % n_replicas.
        # All requests share one aiohttp session and run concurrently;
        # throughput scales with n_replicas because vLLM serves them in
        # parallel.
        n_replicas = len(replicas)
        urls = [f"http://{r._server_address}/classify" for r in replicas]

        async def _score_all() -> list[float]:
            timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:

                async def _one(i: int) -> float:
                    payload = {"model": model_path, "input": rendered[i], "activation": False}
                    async with session.post(urls[i % n_replicas], json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    # vLLM /classify returns {"data": [{"probs": [...]}, ...]}.
                    # Reward models expose a single class so probs[-1] is the
                    # scalar score (mirrors reward_loop.compute_score_disrm).
                    return float(data["data"][-1]["probs"][-1])

                return await asyncio.gather(*[_one(i) for i in range(len(rendered))])

        return asyncio.run(_score_all())

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
