"""Local vLLM helpers: single-GPU and data-parallel ``LLM`` wrappers with optional engine kwargs.

Used by data-generation scripts so they do not embed vLLM construction and worker logic.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import torch
from vllm import LLM, SamplingParams


def _llm_worker(
    rank: int,
    model_name: str,
    gpu_memory_utilization: float,
    task_queue: "mp.Queue[tuple | None]",
    result_queue: "mp.Queue",
    engine_kwargs: dict | None = None,
) -> None:
    """Worker process: owns one GPU, serves generation tasks until shutdown."""
    existing = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if existing:
        gpus = [g.strip() for g in existing.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[rank] if rank < len(gpus) else str(rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    from vllm import LLM, SamplingParams as SP  # import after setting CUDA_VISIBLE_DEVICES

    ek = dict(engine_kwargs or {})
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_sleep_mode=True,
        **ek,
    )
    tokenizer = llm.get_tokenizer()
    result_queue.put(("ready", rank, tokenizer))

    while True:
        msg = task_queue.get()
        if msg is None:  # poison pill
            llm.sleep(level=2)
            del llm
            break
        if msg[0] == "sleep":
            llm.sleep(level=1)
            result_queue.put(("sleep_done", rank))
            continue
        if msg[0] == "wake_up":
            llm.wake_up()
            result_queue.put(("wake_up_done", rank))
            continue
        task_id, prompts, sp_kwargs = msg
        raw_outputs = llm.generate(prompts, SP(**sp_kwargs))
        texts = [[o.text for o in out.outputs] for out in raw_outputs]
        result_queue.put(("result", task_id, texts))


class LLMBase:
    """Abstract local LLM with batched ``generate`` returning text lists per prompt."""

    def generate(self, prompts: list[str], sp_kwargs: dict) -> list[list[str]]:
        raise NotImplementedError

    def get_tokenizer(self):
        raise NotImplementedError

    def sleep(self) -> None:
        """Optional: release GPU memory (single-GPU or coordinated DP)."""

    def wake_up(self) -> None:
        """Optional: restore weights after :meth:`sleep`."""

    def shutdown(self) -> None:
        """Release all resources."""


class SingleGPULLM(LLMBase):
    """One process, one ``LLM`` instance."""

    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float,
        engine_kwargs: dict | None = None,
    ) -> None:
        ek = dict(engine_kwargs or {})
        self._llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_sleep_mode=True,
            **ek,
        )

    def generate(self, prompts: list[str], sp_kwargs: dict) -> list[list[str]]:
        if not prompts:
            return []
        outputs = self._llm.generate(prompts, SamplingParams(**sp_kwargs))
        return [[o.text for o in out.outputs] for out in outputs]

    def get_tokenizer(self):
        return self._llm.get_tokenizer()

    def sleep(self) -> None:
        self._llm.sleep(level=1)

    def wake_up(self) -> None:
        self._llm.wake_up()

    def shutdown(self) -> None:
        self._llm.sleep(level=2)
        del self._llm


class DataParallelLLM(LLMBase):
    """Spawn one vLLM worker process per GPU and distribute prompts across them."""

    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float,
        dp_size: int,
        engine_kwargs: dict | None = None,
    ) -> None:
        self.dp_size = dp_size
        ctx = mp.get_context("spawn")
        self._task_queues = [ctx.Queue() for _ in range(dp_size)]
        self._result_queue: mp.Queue = ctx.Queue()
        ek = dict(engine_kwargs or {})
        self._workers = [
            ctx.Process(
                target=_llm_worker,
                args=(rank, model_name, gpu_memory_utilization,
                      self._task_queues[rank], self._result_queue, ek),
            )
            for rank in range(dp_size)
        ]
        for p in self._workers:
            p.start()
        self._tokenizer = None
        ready = 0
        while ready < dp_size:
            msg = self._result_queue.get()
            assert msg[0] == "ready", f"Unexpected worker message: {msg}"
            if self._tokenizer is None:
                self._tokenizer = msg[2]
            ready += 1
        self._task_counter = 0
        print(f"DataParallelLLM: {dp_size} workers ready ({model_name})")

    def generate(self, prompts: list[str], sp_kwargs: dict) -> list[list[str]]:
        if not prompts:
            return []
        shards = [prompts[i :: self.dp_size] for i in range(self.dp_size)]
        shard_indices = [list(range(i, len(prompts), self.dp_size)) for i in range(self.dp_size)]
        pending: dict[int, list[int]] = {}
        for rank in range(self.dp_size):
            if shards[rank]:
                tid = self._task_counter
                self._task_counter += 1
                pending[tid] = shard_indices[rank]
                self._task_queues[rank].put((tid, shards[rank], sp_kwargs))
        results: dict[int, list[str]] = {}
        for _ in pending:
            _, tid, texts = self._result_queue.get()
            for orig_idx, text_list in zip(pending[tid], texts):
                results[orig_idx] = text_list
        return [results[i] for i in range(len(prompts))]

    def get_tokenizer(self):
        return self._tokenizer

    def sleep(self) -> None:
        for q in self._task_queues:
            q.put(("sleep",))
        for _ in self._task_queues:
            self._result_queue.get()  # wait for sleep_done

    def wake_up(self) -> None:
        for q in self._task_queues:
            q.put(("wake_up",))
        for _ in self._task_queues:
            self._result_queue.get()  # wait for wake_up_done

    def shutdown(self) -> None:
        for q in self._task_queues:
            q.put(None)
        for p in self._workers:
            p.join()
        print(f"DataParallelLLM: {self.dp_size} workers shut down.")


def resolve_dp_size(dp_size_cfg) -> int:
    """Resolve ``data_parallel_size`` from config or ``CUDA_VISIBLE_DEVICES``."""
    if str(dp_size_cfg).strip().lower() != "auto":
        return max(1, int(dp_size_cfg))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        return len([g for g in cuda_visible.split(",") if g.strip()])
    return torch.cuda.device_count() or 1


def build_llm(
    model_name: str,
    gpu_memory_utilization: float,
    dp_size_cfg=1,
    engine_kwargs: dict | None = None,
) -> LLMBase:
    """Construct a single-GPU or data-parallel local LLM."""
    dp_size = resolve_dp_size(dp_size_cfg)
    print(f"  data_parallel_size resolved to {dp_size} (config={dp_size_cfg!r})")
    if engine_kwargs:
        print(f"  vllm_engine_kwargs: {engine_kwargs}")
    if dp_size <= 1:
        return SingleGPULLM(model_name, gpu_memory_utilization, engine_kwargs=engine_kwargs)
    return DataParallelLLM(
        model_name, gpu_memory_utilization, dp_size, engine_kwargs=engine_kwargs
    )


__all__ = [
    "LLMBase",
    "SingleGPULLM",
    "DataParallelLLM",
    "build_llm",
    "resolve_dp_size",
]
