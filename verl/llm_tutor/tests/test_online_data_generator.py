"""Unit tests for OnlineDataGenerator + related helpers.

These tests exercise pure-logic paths of the online data generation pipeline
without requiring GPU / Ray / vLLM / any real rollout or reward server. All
external boundaries (rollout manager, HTTP generator, reward-model manager)
are stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from llm_tutor import _parse
from llm_tutor._parse import (
    is_valid_rubric,
    join_prompt_constraint,
    parse_constraint_judgment,
    parse_rubric_output,
    split_thinking,
)
from llm_tutor.online_data_generator import OnlineDataGenerator

# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #


def _make_generator(mode: str = "constraint", debug_dir: Path | None = None) -> OnlineDataGenerator:
    """Instantiate OnlineDataGenerator with a minimal dict-style config."""
    cfg = {
        "data_generator": {
            "mode": mode,
            "n_rollouts": 2,
            "max_tokens": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "enable_thinking": True,
            "parse_patterns": {},
            "general_rubric_prompt": None,
            "constraint_judgment_prompt": None,
            "adaptive_rubric_prompt": None,
            "debug_dir": str(debug_dir) if debug_dir is not None else None,
        },
    }
    # Tokenizer only used by rollout decode path which we don't call in these tests.
    tok = MagicMock()
    return OnlineDataGenerator(config=cfg, tokenizer=tok, mode=mode)


def _good_rubric(criterion: str = "Must be clear.", importance: int = 70) -> str:
    return "<rubric>\n" f"<criterion>{criterion}</criterion>\n" f"<importance>{importance}</importance>\n" "</rubric>\n"


def _good_judgment_yes() -> str:
    return (
        "<analysis>Needs constraint.</analysis>\n"
        "<decision>yes</decision>\n"
        "<constraint>Write in exactly 3 sentences.</constraint>\n" + _good_rubric("Exactly 3 sentences", 80)
    )


def _good_judgment_no() -> str:
    return "<analysis>Fine.</analysis>\n<decision>no</decision>\n"


# --------------------------------------------------------------------------- #
# _parse.py
# --------------------------------------------------------------------------- #


class TestParse:
    def test_split_thinking_complete(self):
        t, r = split_thinking("<think>reasoning</think>\nanswer body")
        assert t == "reasoning"
        assert r == "answer body"

    def test_split_thinking_truncated_open_only(self):
        t, r = split_thinking("pre text <think> the rest was truncated")
        assert t == "the rest was truncated"
        assert r == "pre text"

    def test_split_thinking_no_tags(self):
        t, r = split_thinking("  hello world  ")
        assert t == ""
        assert r == "hello world"

    def test_parse_rubric_valid(self):
        meta, pairs = parse_rubric_output(_good_rubric("Be kind", 60))
        assert pairs == [{"rubric": "Be kind", "importance": 60}]
        assert isinstance(meta, dict)

    def test_parse_rubric_two_blocks_invalid(self):
        text = _good_rubric() + "\n" + _good_rubric()
        meta, pairs = parse_rubric_output(text)
        assert pairs == []

    def test_parse_rubric_count_mismatch(self):
        text = (
            "<rubric>\n"
            "<criterion>a</criterion><criterion>b</criterion>\n"
            "<importance>50</importance>\n"
            "</rubric>"
        )
        _, pairs = parse_rubric_output(text)
        assert pairs == []

    def test_parse_rubric_bad_importance_falls_back_to_50(self):
        text = "<rubric>\n" "<criterion>ok</criterion>\n" "<importance>not_a_number</importance>\n" "</rubric>"
        _, pairs = parse_rubric_output(text)
        assert pairs == [{"rubric": "ok", "importance": 50}]

    def test_is_valid_rubric(self):
        assert is_valid_rubric([{"rubric": "x", "importance": 50}], require_importance=True)
        assert not is_valid_rubric([], require_importance=True)
        assert not is_valid_rubric([{"rubric": "", "importance": 50}], require_importance=True)
        assert not is_valid_rubric([{"rubric": "x", "importance": 200}], require_importance=True)
        assert not is_valid_rubric([{"rubric": "x", "importance": None}], require_importance=True)

    def test_parse_constraint_judgment_yes(self):
        need_c, c_text, analysis, pairs = parse_constraint_judgment(_good_judgment_yes())
        assert need_c is True
        assert "3 sentences" in c_text
        assert analysis
        assert pairs and pairs[0]["rubric"]

    def test_parse_constraint_judgment_no(self):
        need_c, c_text, analysis, pairs = parse_constraint_judgment(_good_judgment_no())
        assert need_c is False
        assert c_text == ""
        assert pairs == []

    def test_parse_constraint_judgment_yes_but_empty_constraint_is_retry(self):
        text = "<decision>yes</decision>\n<constraint>   </constraint>\n"
        need_c, *_ = parse_constraint_judgment(text)
        assert need_c is None  # retryable

    def test_parse_constraint_judgment_no_decision(self):
        text = "<analysis>blah</analysis>\n"
        need_c, *_ = parse_constraint_judgment(text)
        assert need_c is None

    def test_join_prompt_constraint_variants(self):
        assert join_prompt_constraint("", "c") == "c"
        assert join_prompt_constraint("Do x.", "c") == "Do x. c"
        assert join_prompt_constraint("first line\nsecond", "c") == "first line\nsecond\nc"
        assert join_prompt_constraint("no punct", "c") == "no punct\nc"


# --------------------------------------------------------------------------- #
# OnlineDataGenerator — pure-logic methods
# --------------------------------------------------------------------------- #


class TestExtractInstruction:
    def setup_method(self):
        self.gen = _make_generator()

    def test_string_input(self):
        assert self.gen._extract_instruction("hello") == "hello"

    def test_messages_user_string(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
        ]
        assert self.gen._extract_instruction(msgs) == "question"

    def test_messages_multimodal_parts(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "x"},
                    {"type": "text", "text": "describe"},
                    {"type": "text", "text": "this"},
                ],
            },
        ]
        assert self.gen._extract_instruction(msgs) == "describe this"

    def test_uses_last_user_message(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"},
        ]
        assert self.gen._extract_instruction(msgs) == "second"

    def test_no_user_returns_empty(self):
        assert self.gen._extract_instruction([{"role": "system", "content": "x"}]) == ""


class TestLoadTemplate:
    def test_caches_after_first_read(self, tmp_path: Path):
        tmpl_path = tmp_path / "t.txt"
        tmpl_path.write_text("Hello {instruction}")
        gen = _make_generator()
        gen._template_paths["general_rubric"] = str(tmpl_path)
        t1 = gen._load_template("general_rubric")
        # mutate the file; cached copy should win
        tmpl_path.write_text("SOMETHING ELSE")
        t2 = gen._load_template("general_rubric")
        assert t1 == "Hello {instruction}" == t2

    def test_missing_path_raises(self):
        gen = _make_generator()
        gen._template_paths["general_rubric"] = None
        with pytest.raises(ValueError):
            gen._load_template("general_rubric")


class TestApplyOverride:
    def test_writes_expected_override(self):
        gen = _make_generator()
        ds = MagicMock()
        gen._dataset = ds
        gen._apply_override(
            real_idx=42,
            prompt_str="new prompt",
            general=[{"rubric": "g1", "importance": 70}],
            constraint=[{"rubric": "c1", "importance": 80}],
            adaptive=[{"rubric": "a1", "importance": 40}, {"rubric": "a2", "importance": 55}],
        )
        ds.set_override.assert_called_once()
        args, _ = ds.set_override.call_args
        assert args[0] == 42
        override = args[1]
        assert override["raw_prompt"] == [{"role": "user", "content": "new prompt"}]
        assert override["requirements_general"] == ["g1"]
        assert override["weights_general"] == [70]
        assert override["requirements_constraint"] == ["c1"]
        assert override["weights_constraint"] == [80]
        assert override["requirements_adaptive"] == ["a1", "a2"]
        assert override["weights_adaptive"] == [40, 55]

    def test_missing_importance_raises(self):
        """No silent 50 default — missing importance must fail loudly
        so upstream parsing bugs surface instead of being masked."""
        gen = _make_generator()
        ds = MagicMock()
        gen._dataset = ds
        with pytest.raises(KeyError):
            gen._apply_override(
                real_idx=1,
                prompt_str="p",
                general=[{"rubric": "r"}],
                constraint=[],
                adaptive=[],
            )

    def test_empty_groups_still_written(self):
        """All three groups are always written (possibly empty) so a mode
        switch or reload can't leave stale rubrics from a previous group.
        """
        gen = _make_generator()
        ds = MagicMock()
        gen._dataset = ds
        gen._apply_override(
            real_idx=0,
            prompt_str="p",
            general=[{"rubric": "r", "importance": 50}],
            constraint=[],
            adaptive=[],
        )
        ov = ds.set_override.call_args[0][1]
        for key in (
            "requirements_general",
            "weights_general",
            "requirements_constraint",
            "weights_constraint",
            "requirements_adaptive",
            "weights_adaptive",
        ):
            assert key in ov
        assert ov["requirements_constraint"] == []
        assert ov["weights_adaptive"] == []


# --------------------------------------------------------------------------- #
# _generate_with_retry_keyed
# --------------------------------------------------------------------------- #


class TestGenerateWithRetryKeyed:
    def test_all_succeed_first_attempt(self):
        gen = _make_generator()

        calls: list[list[str]] = []

        async def fake_call(prompts):
            calls.append(list(prompts))
            return [f"OK:{p}" for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={1: "a", 2: "b"},
            parse_fn=lambda raw: raw if raw.startswith("OK:") else None,
            validate_fn=lambda parsed: parsed is not None,
            label="t",
        )
        assert results == {1: "OK:a", 2: "OK:b"}
        assert len(calls) == 1
        assert sorted(calls[0]) == ["a", "b"]

    def test_retry_only_failed_samples(self):
        gen = _make_generator()

        attempt_state = {"n": 0}

        async def fake_call(prompts):
            attempt_state["n"] += 1
            if attempt_state["n"] == 1:
                # first attempt: sample 2 fails
                return ["OK:" + prompts[0], "FAIL"]
            # second attempt: only one prompt, succeed
            assert len(prompts) == 1
            return ["OK:" + prompts[0]]

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={1: "a", 2: "b"},
            parse_fn=lambda raw: raw if raw.startswith("OK:") else None,
            validate_fn=lambda parsed: parsed is not None,
            label="t",
        )
        assert set(results.keys()) == {1, 2}
        assert attempt_state["n"] == 2

    def test_permanent_failure_absent_from_results(self):
        gen = _make_generator()

        async def fake_call(prompts):
            return ["FAIL"] * len(prompts)

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={7: "x"},
            parse_fn=lambda raw: None,
            validate_fn=lambda parsed: parsed is not None,
            label="lbl",
        )
        assert results == {}

    def test_empty_inputs_returns_empty(self):
        gen = _make_generator()
        gen._call_generator = MagicMock()  # should not be called
        out = gen._generate_with_retry_keyed(
            inputs={},
            parse_fn=lambda r: r,
            validate_fn=lambda p: True,
            label="t",
        )
        assert out == {}

    def test_debug_dump_records(self, tmp_path: Path):
        gen = _make_generator(debug_dir=tmp_path)

        state = {"n": 0}

        async def fake_call(prompts):
            state["n"] += 1
            # first: one ok, one bad; second: ok
            if state["n"] == 1:
                return ["OK:" + prompts[0], "BAD"]
            return ["OK:" + prompts[0]]

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={10: "a", 20: "b"},
            parse_fn=lambda r: r if r.startswith("OK:") else None,
            validate_fn=lambda p: p is not None,
            format_parsed=lambda p: {"val": p},
            extra_fields={10: {"extra": "X"}, 20: {"extra": "Y"}},
            label="dbg",
        )
        assert set(results) == {10, 20}

        out_path = tmp_path / "dbg.jsonl"
        assert out_path.exists()
        records = [json.loads(line) for line in out_path.read_text().splitlines()]
        # Two attempts recorded: three rows total (2 + 1)
        assert len(records) == 3
        statuses = sorted((r["sample_idx"], r["attempt"], r["status"]) for r in records)
        assert statuses == [
            (10, 0, "ok"),
            (20, 0, "parse_fail"),
            (20, 1, "ok"),
        ]
        # extra_fields present and parsed present
        by_sample = {(r["sample_idx"], r["attempt"]): r for r in records}
        assert by_sample[(20, 0)]["extra"] == "Y"
        assert "parsed" in by_sample[(10, 0)]

    def test_permanent_failure_logged_once_per_attempt(self, tmp_path: Path):
        gen = _make_generator(debug_dir=tmp_path)

        async def fake_call(prompts):
            return ["BAD"] * len(prompts)

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={99: "x"},
            parse_fn=lambda r: None,
            validate_fn=lambda p: p is not None,
            format_parsed=lambda p: {},
            label="lost",
        )
        assert results == {}

        out_path = tmp_path / "lost.jsonl"
        records = [json.loads(line) for line in out_path.read_text().splitlines()]
        # _max_retry + 1 attempts, each emits one parse_fail record for sample 99
        assert len(records) == gen._max_retry + 1
        assert all(r["sample_idx"] == 99 and r["status"] == "parse_fail" for r in records)
        attempts = sorted(r["attempt"] for r in records)
        assert attempts == list(range(gen._max_retry + 1))


# --------------------------------------------------------------------------- #
# _gen_general_rubric / _gen_constraint_judgment / _gen_adaptive_rubric
# --------------------------------------------------------------------------- #


class TestGenStages:
    def setup_method(self):
        self.gen = _make_generator()
        self.gen._template_cache = {
            "general_rubric": "Instr: {instruction}",
            "constraint_judgment": "I: {instruction} R1: {response_1} R2: {response_2}",
            "adaptive_rubric": "I: {instruction} R1: {response_1} R2: {response_2}",
        }

    def _stub_generator(self, mapping: dict[str, str]):
        async def fake_call(prompts):
            return [mapping[p] for p in prompts]

        self.gen._call_generator = fake_call  # type: ignore[assignment]

    def test_general_rubric(self):
        prompts = ["who are you?", "what time is it?"]
        mapping = {
            "Instr: who are you?": _good_rubric("Mention identity", 70),
            "Instr: what time is it?": _good_rubric("Give a time", 80),
        }
        self._stub_generator(mapping)
        out = self.gen._gen_general_rubric(prompts, eligible=[0, 1])
        assert set(out) == {0, 1}
        assert out[0][0]["rubric"] == "Mention identity"
        assert out[1][0]["importance"] == 80

    def test_constraint_judgment_yes_and_no(self):
        prompts = ["p0", "p1"]
        rollouts = [["r0a", "r0b"], ["r1a", "r1b"]]
        mapping = {
            "I: p0 R1: r0a R2: r0b": _good_judgment_yes(),
            "I: p1 R1: r1a R2: r1b": _good_judgment_no(),
        }
        self._stub_generator(mapping)
        decisions, constraints, constraint_rubrics = self.gen._gen_constraint_judgment(
            prompts, rollouts, eligible=[0, 1]
        )
        assert decisions == {0: True, 1: False}
        assert 0 in constraints and constraints[0]
        assert 1 not in constraints
        assert 0 in constraint_rubrics and constraint_rubrics[0]
        assert 1 not in constraint_rubrics

    def test_constraint_judgment_skips_short_rollouts(self):
        prompts = ["p0", "p1"]
        rollouts = [["only_one"], ["r1a", "r1b"]]  # sample 0 has <2 rollouts

        calls: list[list[str]] = []

        async def fake_call(ps):
            calls.append(list(ps))
            return [_good_judgment_no() for _ in ps]

        self.gen._call_generator = fake_call  # type: ignore[assignment]
        decisions, _, _ = self.gen._gen_constraint_judgment(prompts, rollouts, eligible=[0, 1])
        # sample 0 should not be in the first (and only) call at all
        assert all("r1a" in p for p in calls[0])
        assert 0 not in decisions
        assert decisions.get(1) is False

    def test_adaptive_rubric(self):
        prompts = ["p0", "p1"]
        rollouts = [["a", "b"], ["c", "d"]]
        mapping = {
            "I: p0 R1: a R2: b": _good_rubric("crit0", 55),
            "I: p1 R1: c R2: d": _good_rubric("crit1", 65),
        }
        self._stub_generator(mapping)
        out = self.gen._gen_adaptive_rubric(prompts, rollouts, eligible=[0, 1])
        assert out[0][0]["rubric"] == "crit0"
        assert out[1][0]["importance"] == 65


# --------------------------------------------------------------------------- #
# modify_dataset_epoch_end end-to-end with mocks
# --------------------------------------------------------------------------- #


_RUBRIC_GROUPS = ("general", "constraint", "adaptive")


def _ov_requirements(override: dict) -> list[str]:
    """Concatenate ``requirements_<group>`` across the three groups, in order."""
    out: list[str] = []
    for g in _RUBRIC_GROUPS:
        out.extend(override.get(f"requirements_{g}", []))
    return out


def _ov_weights(override: dict) -> list[int]:
    out: list[int] = []
    for g in _RUBRIC_GROUPS:
        out.extend(override.get(f"weights_{g}", []))
    return out


class _FakeDataset:
    """Stand-in AdaptiveDataset that records set_override / drop_samples calls.

    Tracks per-row persisted rubric groups so ``get_accumulated_rubrics`` reads
    back whatever ``set_override`` wrote, matching the real dataset's
    cross-epoch behavior.
    """

    def __init__(self, n: int = 3, persisted: dict[int, dict[str, list[dict]]] | None = None):
        # Build synthetic rows with _dataset_idx matching logical idx (no drops initially).
        self._rows = [
            {
                "raw_prompt": [{"role": "user", "content": f"prompt {i}"}],
                "_dataset_idx": i,
                "requirements": [],
                "weights": [],
            }
            for i in range(n)
        ]
        # Snapshot the immutable original prompts up-front so reset modes can
        # bypass the override-honoring ``raw_prompt`` view via
        # ``get_original_prompt`` even after ``set_override`` mutates the row.
        self._original_prompts: dict[int, str] = {i: f"prompt {i}" for i in range(n)}
        self.overrides: dict[int, dict] = {}
        self.dropped: list[int] = []
        self._persisted: dict[int, dict[str, list[dict]]] = {
            i: {g: list(groups.get(g, [])) for g in _RUBRIC_GROUPS} for i, groups in (persisted or {}).items()
        }

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    def set_override(self, real_idx, override):
        # Merge with any prior override, matching AdaptiveDataset semantics.
        if real_idx in self.overrides:
            merged = dict(self.overrides[real_idx])
            merged.update(override)
        else:
            merged = dict(override)
        self.overrides[real_idx] = merged

        # Reflect raw_prompt back into the row so the next epoch reads the
        # constraint-augmented prompt, matching AdaptiveDataset.__getitem__.
        if "raw_prompt" in override:
            self._rows[real_idx]["raw_prompt"] = override["raw_prompt"]

        # Track persisted rubric groups for next-epoch get_accumulated_rubrics.
        for g in _RUBRIC_GROUPS:
            req_key = f"requirements_{g}"
            w_key = f"weights_{g}"
            if req_key in override and w_key in override:
                pairs = [{"rubric": r, "importance": int(w)} for r, w in zip(override[req_key], override[w_key])]
                self._persisted.setdefault(real_idx, {g_: [] for g_ in _RUBRIC_GROUPS})[g] = pairs

    def drop_samples(self, indices):
        self.dropped.extend(indices)
        for i in indices:
            self._persisted.pop(i, None)

    def get_accumulated_rubrics(self, idx):
        groups = self._persisted.get(idx, {})
        return {g: list(groups.get(g, [])) for g in _RUBRIC_GROUPS}

    def get_original_prompt(self, idx):
        return self._original_prompts[idx]


def _install_common_mocks(gen: OnlineDataGenerator, rollouts_A, rollouts_B=None):
    """Patch rollout + rm managers and monkey-patch _gen_rollouts to return fixed arrays."""
    gen._rollout_mgr = MagicMock()
    gen._rollout_mgr.agent_loop_workers = [MagicMock()]
    rm_manager = MagicMock()
    gen._rm_manager = rm_manager
    gen._router_addr = "127.0.0.1:0"

    call_state = {"n": 0}

    def fake_gen_rollouts(indices, n=2, min_valid=2, prompt_overrides=None):
        call_state["n"] += 1
        if call_state["n"] == 1:
            return [list(r) for r in rollouts_A]
        return [list(r) for r in (rollouts_B or rollouts_A)]

    gen._gen_rollouts = fake_gen_rollouts  # type: ignore[assignment]
    return call_state


class TestModifyDatasetEpochEnd:
    def _setup(self, mode: str, n: int = 2):
        gen = _make_generator(mode=mode)
        ds = _FakeDataset(n=n)
        gen.set_dataset(ds)
        gen._template_cache = {
            "general_rubric": "G:{instruction}",
            "constraint_judgment": "C:{instruction}|{response_1}|{response_2}",
            "adaptive_rubric": "A:{instruction}|{response_1}|{response_2}",
        }
        return gen, ds

    # ----- constraint mode ----------------------------------------------------

    def test_constraint_mode_happy_path(self):
        gen, ds = self._setup("constraint", n=2)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"], ["b0", "b1"]])

        # Generator: general_rubric → ok; constraint_judgment → yes (sample 0), no (sample 1)
        responses: dict[str, str] = {
            "G:prompt 0": _good_rubric("g0", 70),
            "G:prompt 1": _good_rubric("g1", 60),
            "C:prompt 0|a0|a1": _good_judgment_yes(),
            "C:prompt 1|b0|b1": _good_judgment_no(),
        }

        async def fake_call(prompts):
            return [responses[p] for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        n_mod = gen.modify_dataset_epoch(epoch=0)
        assert n_mod == 2
        assert ds.dropped == []
        # Sample 0 override must carry the constraint-added prompt and combined rubric
        ov0 = ds.overrides[0]
        assert "3 sentences" in ov0["raw_prompt"][0]["content"]
        # per-group view
        assert ov0["requirements_general"] == ["g0"]
        assert ov0["requirements_constraint"] == ["Exactly 3 sentences"]
        # concat = general + constraint (constraint mode → adaptive empty)
        assert _ov_requirements(ov0) == ["g0", "Exactly 3 sentences"]
        assert _ov_weights(ov0) == [70, 80]
        # Sample 1 has no constraint, so raw_prompt stays as original instruction
        ov1 = ds.overrides[1]
        assert ov1["raw_prompt"][0]["content"] == "prompt 1"
        assert ov1["requirements_general"] == ["g1"]
        assert ov1["requirements_constraint"] == []
        assert _ov_requirements(ov1) == ["g1"]
        assert _ov_weights(ov1) == [60]

    def test_constraint_mode_drops_when_both_rubrics_empty(self):
        gen, ds = self._setup("constraint", n=1)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

        # general_rubric never parses (decision=no for constraint_judgment so no c_rubric)
        async def fake_call(prompts):
            out = []
            for p in prompts:
                if p.startswith("G:"):
                    out.append("GARBAGE")  # parse fails forever
                else:
                    out.append(_good_judgment_no())
            return out

        gen._call_generator = fake_call  # type: ignore[assignment]

        n_mod = gen.modify_dataset_epoch(epoch=0)
        assert n_mod == 0
        assert ds.dropped == [0]

    def test_constraint_mode_decision_yes_with_empty_rubric_is_retried_then_dropped(self):
        """Regression: judge emits ``<decision>yes</decision>`` plus a valid
        ``<constraint>`` body but no parseable ``<rubric>``. Without the
        ``is_valid_rubric`` guard in ``_gen_constraint_judgment.validate_fn``
        the constraint would be joined into the prompt while
        ``requirements_constraint`` stays empty, yielding a sample whose reward
        signal can never score the new constraint (silent corruption observed
        in v1.5_constraint training: 3 rows landed in this state)."""
        gen, ds = self._setup("constraint", n=1)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

        yes_no_rubric = (
            "<analysis>looks fine</analysis>\n"
            "<decision>yes</decision>\n"
            "<constraint>Limit your response to exactly 3 sentences.</constraint>\n"
            # ← no <rubric> block
        )

        async def fake_call(prompts):
            out = []
            for p in prompts:
                if p.startswith("G:"):
                    out.append(_good_rubric("g0", 70))
                else:  # constraint stage — never produces a rubric
                    out.append(yes_no_rubric)
            return out

        gen._call_generator = fake_call  # type: ignore[assignment]
        n_mod = gen.modify_dataset_epoch(epoch=0)
        # general rubric still persists, so the sample is still "modified"
        assert n_mod == 1
        assert ds.dropped == []
        ov = ds.overrides[0]
        # Constraint never accepted → no prompt mutation, no constraint rubric
        assert ov["raw_prompt"][0]["content"] == "prompt 0"
        assert ov["requirements_constraint"] == []
        assert ov["requirements_general"] == ["g0"]

    def test_constraint_reset_mode_drops_previous_constraint_each_epoch(self):
        """``constraint_reset`` mode replaces (not appends) the prior
        constraint+rubric every epoch a new constraint is accepted, while
        decision=no epochs leave both prompt and rubric untouched.

        Three-epoch tape:
          * epoch 0: judge sees the original prompt → decision=yes (c0).
            Override becomes ``X + c0``; constraint rubric = [r0].
          * epoch 1: judge sees ``X + c0`` (Step 1 still rolls out on the
            override, matching saturation-detection semantics) → decision=yes
            (c1). Override becomes ``X + c1`` (NOT ``X + c0 + c1``);
            constraint rubric = [r1] (NOT [r0, r1]).
          * epoch 2: judge sees ``X + c1`` → decision=no. Override stays
            ``X + c1`` and constraint rubric stays [r1] verbatim.
        """
        gen, ds = self._setup("constraint_reset", n=1)

        c0 = "Use bullet points only."
        r0 = "Format is bullet-list"
        c1 = "Limit to 100 words."
        r1 = "Length under 100 words"

        judgment_yes_c0 = (
            "<analysis>fix0</analysis>\n"
            "<decision>yes</decision>\n"
            f"<constraint>{c0}</constraint>\n" + _good_rubric(r0, 80)
        )
        judgment_yes_c1 = (
            "<analysis>fix1</analysis>\n"
            "<decision>yes</decision>\n"
            f"<constraint>{c1}</constraint>\n" + _good_rubric(r1, 70)
        )
        judgment_no = _good_judgment_no()

        prompt_x = "prompt 0"
        prompt_xc0 = join_prompt_constraint(prompt_x, c0)
        prompt_xc1 = join_prompt_constraint(prompt_x, c1)

        # Each epoch has its own (instruction, judgment) mapping; rollouts
        # stay fixed since we don't care about their content here, only
        # about which prompt the judge saw.
        scripts = [
            {  # epoch 0 — judge sees X
                f"G:{prompt_x}": _good_rubric("g_x", 60),
                f"C:{prompt_x}|a0|a1": judgment_yes_c0,
            },
            {  # epoch 1 — judge sees X + c0 (override carried by _FakeDataset)
                f"G:{prompt_xc0}": _good_rubric("g_xc0", 60),
                f"C:{prompt_xc0}|a0|a1": judgment_yes_c1,
            },
            {  # epoch 2 — judge sees X + c1 (reset took effect last epoch)
                f"G:{prompt_xc1}": _good_rubric("g_xc1", 60),
                f"C:{prompt_xc1}|a0|a1": judgment_no,
            },
        ]

        for epoch, responses in enumerate(scripts):
            _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

            async def fake_call(prompts, _r=responses):
                return [_r[p] for p in prompts]

            gen._call_generator = fake_call  # type: ignore[assignment]
            gen.modify_dataset_epoch(epoch=epoch)

            ov = ds.overrides[0]
            persisted_prompt = ov["raw_prompt"][0]["content"]
            if epoch == 0:
                assert persisted_prompt == prompt_xc0, f"epoch 0 → {persisted_prompt!r}"
                assert ov["requirements_constraint"] == [r0]
            elif epoch == 1:
                # Reset: NOT prompt_xc0 + c1; just X + c1.
                assert persisted_prompt == prompt_xc1, f"epoch 1 → {persisted_prompt!r}"
                assert c0 not in persisted_prompt
                assert ov["requirements_constraint"] == [r1]  # NOT [r0, r1]
            else:  # epoch 2: decision=no → carry-forward
                assert persisted_prompt == prompt_xc1, f"epoch 2 → {persisted_prompt!r}"
                assert ov["requirements_constraint"] == [r1]

    def test_constraint_mode_skips_samples_with_failed_rollout(self):
        gen, ds = self._setup("constraint", n=2)
        # Sample 1 has only 1 rollout → ineligible for constraint
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"], ["just_one"]])

        responses: dict[str, str] = {
            "G:prompt 0": _good_rubric("g0", 70),
            "G:prompt 1": _good_rubric("g1", 60),
            "C:prompt 0|a0|a1": _good_judgment_yes(),
        }

        async def fake_call(prompts):
            return [responses[p] for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        n_mod = gen.modify_dataset_epoch(epoch=0)
        # Both samples still modified via general_rubric; sample 1 has no constraint step
        assert n_mod == 2
        assert ds.dropped == []
        ov1 = ds.overrides[1]
        # sample 1 never got constraint evaluated → keeps original prompt
        assert ov1["raw_prompt"][0]["content"] == "prompt 1"
        assert _ov_requirements(ov1) == ["g1"]

    def test_general_rubric_not_regenerated_when_persisted(self):
        """When a general rubric is already persisted for a sample, later calls
        must reuse it — no G: prompt should hit the generator."""
        gen, ds = self._setup("constraint", n=1)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

        responses_e0 = {
            "G:prompt 0": _good_rubric("g0", 70),
            "C:prompt 0|a0|a1": _good_judgment_no(),
        }

        async def fake_call_e0(prompts):
            return [responses_e0[p] for p in prompts]

        gen._call_generator = fake_call_e0  # type: ignore[assignment]
        gen.modify_dataset_epoch(epoch=0)

        # Epoch 1 must NOT issue G: because general is already persisted.
        seen: list[str] = []

        async def fake_call_e1(prompts):
            seen.extend(prompts)
            return [_good_judgment_no() for _ in prompts]

        gen._call_generator = fake_call_e1  # type: ignore[assignment]
        gen.modify_dataset_epoch(epoch=1)

        assert [p for p in seen if p.startswith("G:")] == []

    def test_missing_general_is_regenerated_on_later_call(self):
        """Any call (regardless of epoch number) that finds a sample without a
        persisted general rubric regenerates it on the fly, rather than
        raising. This is the behaviour that makes the step schedule viable:
        samples first visited mid-run still get a general rubric.
        """
        gen, ds = self._setup("constraint", n=1)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

        responses = {
            "G:prompt 0": _good_rubric("g0", 70),
            "C:prompt 0|a0|a1": _good_judgment_no(),
        }

        async def fake_call(prompts):
            return [responses[p] for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        # Sample has no persisted general; this should generate it, not raise.
        gen.modify_dataset_epoch(epoch=7)

        assert ds.overrides[0]["requirements_general"] == ["g0"]

    def test_drops_only_when_combined_rubric_empty(self):
        """Drop decision is on combined rubric (general + constraint + adaptive).
        A sample with failed general but a valid constraint rubric survives,
        because the combined rubric is still non-empty — the reward function
        still has something to score against.
        """
        gen, ds = self._setup("constraint", n=2)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"], ["b0", "b1"]])

        # Sample 0 general parses fine; sample 1 general never parses.
        responses: dict[str, str] = {
            "G:prompt 0": _good_rubric("g0", 70),
            "G:prompt 1": "JUNK",
            "C:prompt 0|a0|a1": _good_judgment_no(),
            "C:prompt 1|b0|b1": _good_judgment_yes(),
        }

        async def fake_call(prompts):
            return [responses[p] for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]
        n_mod = gen.modify_dataset_epoch(epoch=0)

        # Sample 1 survives: constraint rubric populated even though general empty.
        assert n_mod == 2
        assert ds.dropped == []
        assert 0 in ds.overrides
        assert 1 in ds.overrides
        assert ds.overrides[1]["requirements_general"] == []
        assert ds.overrides[1]["requirements_constraint"] == ["Exactly 3 sentences"]

    def test_general_rubric_persisted_skips_generation_on_second_epoch(self):
        """Second epoch reuses the override-persisted general rubric (no extra G: calls)."""
        gen, ds = self._setup("constraint", n=2)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"], ["b0", "b1"]])

        # Track every prompt the generator sees so we can assert on G:* counts.
        seen: list[str] = []
        responses: dict[str, str] = {
            "G:prompt 0": _good_rubric("g0", 70),
            "G:prompt 1": _good_rubric("g1", 60),
            "C:prompt 0|a0|a1": _good_judgment_no(),
            "C:prompt 1|b0|b1": _good_judgment_no(),
        }

        async def fake_call(prompts):
            seen.extend(prompts)
            return [responses[p] for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        # Epoch 0: nothing persisted → G: for both samples fires.
        gen.modify_dataset_epoch(epoch=0)
        epoch0_G = [p for p in seen if p.startswith("G:")]
        assert sorted(epoch0_G) == ["G:prompt 0", "G:prompt 1"]
        assert ds.get_accumulated_rubrics(0)["general"] != []
        assert ds.get_accumulated_rubrics(1)["general"] != []

        # Epoch 1: general persisted via override → no new G: prompts.
        seen.clear()
        gen.modify_dataset_epoch(epoch=1)
        epoch1_G = [p for p in seen if p.startswith("G:")]
        assert epoch1_G == []

        # And the produced override still carries the previously-generated general rubric.
        assert ds.overrides[0]["requirements_general"] == ["g0"]
        assert ds.overrides[1]["requirements_general"] == ["g1"]

    def test_constraint_rubric_persists_when_later_epoch_decides_no(self):
        """Regression: if a prior epoch added a constraint (decision=yes) and
        a later epoch's judgment returns decision=no, the prior epoch's
        constraint rubric must NOT be dropped — the constraint sentence still
        sits in the prompt, so its rubric must travel with it.
        """
        gen, ds = self._setup("constraint", n=1)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

        responses_e0: dict[str, str] = {
            "G:prompt 0": _good_rubric("g0", 70),
            "C:prompt 0|a0|a1": _good_judgment_yes(),
        }

        async def fake_call_e0(prompts):
            return [responses_e0[p] for p in prompts]

        gen._call_generator = fake_call_e0  # type: ignore[assignment]
        gen.modify_dataset_epoch(epoch=0)

        ov0 = ds.overrides[0]
        assert ov0["requirements_constraint"] == ["Exactly 3 sentences"]
        # Prompt now has the constraint appended.
        post_e0_prompt = ov0["raw_prompt"][0]["content"]
        assert "3 sentences" in post_e0_prompt

        # Epoch 1: judgment returns "no". Constraint rubric must survive.
        responses_e1: dict[str, str] = {
            f"C:{post_e0_prompt}|a0|a1": _good_judgment_no(),
        }

        async def fake_call_e1(prompts):
            return [responses_e1[p] for p in prompts]

        gen._call_generator = fake_call_e1  # type: ignore[assignment]
        gen.modify_dataset_epoch(epoch=1)

        ov1 = ds.overrides[0]
        # Prompt stays the same (no new constraint added).
        assert ov1["raw_prompt"][0]["content"] == post_e0_prompt
        # Constraint rubric still present — the fix.
        assert ov1["requirements_constraint"] == ["Exactly 3 sentences"]
        # General also still present.
        assert ov1["requirements_general"] == ["g0"]

    def test_constraint_rubric_accumulates_across_yes_decisions(self):
        """Two consecutive decision=yes epochs should stack constraint rubrics."""
        gen, ds = self._setup("constraint", n=1)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

        yes_alt = (
            "<analysis>Also needs concision.</analysis>\n"
            "<decision>yes</decision>\n"
            "<constraint>Avoid passive voice.</constraint>\n" + _good_rubric("No passive voice", 65)
        )

        responses_e0: dict[str, str] = {
            "G:prompt 0": _good_rubric("g0", 70),
            "C:prompt 0|a0|a1": _good_judgment_yes(),
        }

        async def fake_call_e0(prompts):
            return [responses_e0[p] for p in prompts]

        gen._call_generator = fake_call_e0  # type: ignore[assignment]
        gen.modify_dataset_epoch(epoch=0)
        post_e0_prompt = ds.overrides[0]["raw_prompt"][0]["content"]

        responses_e1: dict[str, str] = {
            f"C:{post_e0_prompt}|a0|a1": yes_alt,
        }

        async def fake_call_e1(prompts):
            return [responses_e1[p] for p in prompts]

        gen._call_generator = fake_call_e1  # type: ignore[assignment]
        gen.modify_dataset_epoch(epoch=1)

        ov = ds.overrides[0]
        # Both constraint rubrics should be present, in order.
        assert ov["requirements_constraint"] == ["Exactly 3 sentences", "No passive voice"]
        assert ov["weights_constraint"] == [80, 65]


# --------------------------------------------------------------------------- #
# modify_batch: step-schedule in-place mutation path
# --------------------------------------------------------------------------- #


class _FakeDataProto:
    """Minimal stand-in for DataProto used to observe in-place mutation.

    ``modify_batch`` only touches ``non_tensor_batch[col][pos]`` entries — no
    tensor ops — so a dict-backed stub is sufficient.
    """

    def __init__(self, non_tensor_batch: dict):
        self.non_tensor_batch = non_tensor_batch


class TestModifyBatchInPlace:
    def _setup(self, mode: str = "constraint", n: int = 2):
        gen = _make_generator(mode=mode)
        ds = _FakeDataset(n=n)
        gen.set_dataset(ds)
        gen._template_cache = {
            "general_rubric": "G:{instruction}",
            "constraint_judgment": "C:{instruction}|{response_1}|{response_2}",
            "adaptive_rubric": "A:{instruction}|{response_1}|{response_2}",
        }
        return gen, ds

    def test_inplace_mutates_raw_prompt_and_requirements(self):
        gen, ds = self._setup("constraint", n=2)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"], ["b0", "b1"]])

        responses = {
            "G:prompt 0": _good_rubric("g0", 70),
            "G:prompt 1": _good_rubric("g1", 60),
            # Sample 0: constraint yes → new prompt + constraint rubric added
            "C:prompt 0|a0|a1": _good_judgment_yes(),
            # Sample 1: constraint no → prompt unchanged, only general
            "C:prompt 1|b0|b1": _good_judgment_no(),
        }

        async def fake_call(prompts):
            return [responses[p] for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        # Simulate the trainer's batch-with-everything state (pre _get_gen_batch).
        batch = _FakeDataProto(
            {
                "raw_prompt": [
                    [{"role": "user", "content": "prompt 0"}],
                    [{"role": "user", "content": "prompt 1"}],
                ],
                "requirements": [[], []],
                "weights": [[], []],
                "prompt": ["prompt 0", "prompt 1"],
            }
        )

        gen.modify_batch(indices=[0, 1], batch=batch)

        # Sample 0 got constraint added to prompt text and rubric list.
        new_p0 = batch.non_tensor_batch["raw_prompt"][0][0]["content"]
        assert new_p0.startswith("prompt 0") and "3 sentences" in new_p0
        assert batch.non_tensor_batch["prompt"][0] == new_p0
        assert batch.non_tensor_batch["requirements"][0] == ["g0", "Exactly 3 sentences"]
        assert batch.non_tensor_batch["weights"][0] == [70, 80]
        # Sample 1 prompt unchanged, only general rubric.
        assert batch.non_tensor_batch["raw_prompt"][1][0]["content"] == "prompt 1"
        assert batch.non_tensor_batch["requirements"][1] == ["g1"]
        assert batch.non_tensor_batch["weights"][1] == [60]

        # Override was also persisted on the dataset — source of truth for resume.
        assert "3 sentences" in ds.overrides[0]["raw_prompt"][0]["content"]
        assert ds.overrides[1]["raw_prompt"][0]["content"] == "prompt 1"

    def test_inplace_step_mode_never_drops(self):
        """Even when a sample's rubric comes out empty, the step-schedule caller
        must not see a drop — dropping is only done by ``modify_dataset_epoch``.
        """
        gen, ds = self._setup("constraint", n=1)
        _install_common_mocks(gen, rollouts_A=[["a0", "a1"]])

        async def fake_call(prompts):
            # general parses as junk forever → empty rubric
            return ["GARBAGE" for _ in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        batch = _FakeDataProto(
            {
                "raw_prompt": [[{"role": "user", "content": "prompt 0"}]],
                "requirements": [[]],
                "weights": [[]],
                "prompt": ["prompt 0"],
            }
        )
        gen.modify_batch(indices=[0], batch=batch)

        # No drop. Override still written (with empty groups).
        assert ds.dropped == []
        assert 0 in ds.overrides
        assert ds.overrides[0]["requirements_general"] == []


# --------------------------------------------------------------------------- #
# AdaptiveDataset — override / drop / active index interactions
# --------------------------------------------------------------------------- #


class _StubAdaptiveDataset:
    """Bypass RLHFDataset.__init__ and test the override/drop logic directly.

    We replicate the methods we want to exercise without pulling the full
    RLHFDataset base (which needs a tokenizer + parquet + datasets lib init).
    """

    def __init__(self, rows):
        self.dataframe = rows
        self._row_overrides: dict[int, dict] = {}
        self._active_indices: list[int] = list(range(len(rows)))

    # lift the two methods from AdaptiveDataset as-is
    set_override = None  # filled below
    drop_samples = None


def _make_stub_dataset(n: int = 4):
    from llm_tutor.adaptive_dataset import AdaptiveDataset

    rows = [{"id": i} for i in range(n)]
    ds = _StubAdaptiveDataset(rows)
    # bind unbound methods
    ds.set_override = AdaptiveDataset.set_override.__get__(ds)
    ds.drop_samples = AdaptiveDataset.drop_samples.__get__(ds)
    ds.get_adaptive_state = AdaptiveDataset.get_adaptive_state.__get__(ds)
    ds.load_adaptive_state = AdaptiveDataset.load_adaptive_state.__get__(ds)
    ds.get_accumulated_rubrics = AdaptiveDataset.get_accumulated_rubrics.__get__(ds)
    ds._require_general_rubric_seed = AdaptiveDataset._require_general_rubric_seed.__get__(ds)
    ds.__len__ = lambda self_=ds: len(self_._active_indices)
    return ds


class TestRequireGeneralRubricSeed:
    def test_passes_when_all_active_rows_have_general(self):
        ds = _make_stub_dataset(n=3)
        for i in range(3):
            ds.set_override(i, {"requirements_general": [f"r{i}"], "weights_general": [50]})
        ds._require_general_rubric_seed()  # no raise

    def test_raises_when_any_row_missing_general(self):
        ds = _make_stub_dataset(n=3)
        ds.set_override(0, {"requirements_general": ["r0"], "weights_general": [50]})
        # idx 1 + 2 missing
        with pytest.raises(ValueError, match=r"missing a general rubric"):
            ds._require_general_rubric_seed()

    def test_dropped_rows_do_not_count(self):
        """Validation looks at _active_indices, so drops let the check pass."""
        ds = _make_stub_dataset(n=3)
        ds.set_override(0, {"requirements_general": ["r0"], "weights_general": [50]})
        ds.drop_samples([1, 2])
        ds._require_general_rubric_seed()  # no raise: only idx 0 is active

    def test_empty_general_counts_as_missing(self):
        ds = _make_stub_dataset(n=2)
        ds.set_override(0, {"requirements_general": [], "weights_general": []})
        ds.set_override(1, {"requirements_general": ["r1"], "weights_general": [50]})
        with pytest.raises(ValueError, match=r"missing a general rubric"):
            ds._require_general_rubric_seed()


class TestAdaptiveDatasetOverrides:
    def test_set_override_merges(self):
        ds = _make_stub_dataset(n=3)
        ds.set_override(1, {"requirements_general": ["a"], "weights_general": [50]})
        ds.set_override(1, {"raw_prompt": [{"role": "user", "content": "x"}]})
        assert ds._row_overrides[1]["requirements_general"] == ["a"]
        assert ds._row_overrides[1]["weights_general"] == [50]
        assert ds._row_overrides[1]["raw_prompt"][0]["content"] == "x"

    def test_set_override_weights_and_requirements_stay_in_sync(self):
        ds = _make_stub_dataset(n=3)
        ds.set_override(
            2,
            {"requirements_general": ["r1", "r2"], "weights_general": [10, 90]},
        )
        ov = ds._row_overrides[2]
        assert len(ov["requirements_general"]) == len(ov["weights_general"]) == 2

    def test_drop_samples_cleans_active_and_overrides(self):
        ds = _make_stub_dataset(n=4)
        ds.set_override(2, {"requirements_general": ["x"], "weights_general": [50]})
        ds.drop_samples([1, 2])
        assert ds._active_indices == [0, 3]
        assert 2 not in ds._row_overrides

    def test_drop_samples_preserves_surviving_overrides(self):
        ds = _make_stub_dataset(n=4)
        ds.set_override(0, {"requirements_general": ["keep"], "weights_general": [80]})
        ds.set_override(3, {"requirements_general": ["also_keep"], "weights_general": [30]})
        ds.drop_samples([1, 2])
        assert ds._row_overrides[0]["requirements_general"] == ["keep"]
        assert ds._row_overrides[3]["weights_general"] == [30]
        assert ds._active_indices == [0, 3]

    def test_set_override_rejects_length_mismatch(self):
        ds = _make_stub_dataset(n=3)
        with pytest.raises(ValueError, match="length"):
            ds.set_override(0, {"requirements_general": ["a", "b"], "weights_general": [50]})

    def test_set_override_rejects_orphan_weights(self):
        """weights_<g> without matching requirements_<g> (or vice versa) must fail."""
        ds = _make_stub_dataset(n=3)
        with pytest.raises(ValueError, match="must be set together"):
            ds.set_override(0, {"weights_general": [50, 60]})

    def test_set_override_rejects_mismatch_from_merge(self):
        # First write sets requirements; second write updates weights to a
        # different length — the merged state must fail validation.
        ds = _make_stub_dataset(n=3)
        ds.set_override(
            1,
            {"requirements_general": ["a", "b", "c"], "weights_general": [10, 20, 30]},
        )
        with pytest.raises(ValueError, match="length"):
            ds.set_override(1, {"weights_general": [99]})
        # Original override should remain intact after the failed update.
        assert ds._row_overrides[1]["requirements_general"] == ["a", "b", "c"]
        assert ds._row_overrides[1]["weights_general"] == [10, 20, 30]

    def test_set_override_accepts_all_three_groups(self):
        ds = _make_stub_dataset(n=1)
        ds.set_override(
            0,
            {
                "requirements_general": ["g1", "g2"],
                "weights_general": [50, 60],
                "requirements_constraint": ["c1"],
                "weights_constraint": [70],
                "requirements_adaptive": ["a1", "a2", "a3"],
                "weights_adaptive": [30, 40, 55],
            },
        )
        state = ds.get_accumulated_rubrics(0)
        assert [p["rubric"] for p in state["general"]] == ["g1", "g2"]
        assert [p["importance"] for p in state["constraint"]] == [70]
        assert len(state["adaptive"]) == 3

    def test_get_accumulated_rubrics_returns_empty_for_unseen(self):
        ds = _make_stub_dataset(n=2)
        state = ds.get_accumulated_rubrics(0)
        assert state == {"general": [], "constraint": [], "adaptive": []}

    def test_get_adaptive_state_round_trip(self):
        """Persisting + restoring the state must reproduce overrides AND drops.

        Regression for the checkpoint-resume bug where overrides and the
        active index list lived only in memory, so a resumed training run
        re-generated everything from scratch with different random rollouts.
        """
        src = _make_stub_dataset(n=5)
        src.set_override(0, {"requirements_general": ["r0"], "weights_general": [50]})
        src.set_override(4, {"requirements_general": ["r4"], "weights_general": [70]})
        src.drop_samples([1, 3])  # active: [0, 2, 4]

        state = src.get_adaptive_state()
        assert state["active_indices"] == [0, 2, 4]
        assert set(state["row_overrides"].keys()) == {0, 4}

        dst = _make_stub_dataset(n=5)
        dst.load_adaptive_state(state)

        assert dst._active_indices == [0, 2, 4]
        assert dst._row_overrides[0]["requirements_general"] == ["r0"]
        assert dst._row_overrides[4]["weights_general"] == [70]
        # The dst dataset must behave identically to src from here on out.
        assert len(dst._active_indices) == 3

    def test_load_adaptive_state_back_compat_without_active_indices(self):
        """Older checkpoint format (overrides-only) must still load."""
        ds = _make_stub_dataset(n=4)
        ds.set_override(2, {"requirements_general": ["x"], "weights_general": [50]})
        before_active = list(ds._active_indices)

        ds.load_adaptive_state({"row_overrides": {3: {"requirements_general": ["y"], "weights_general": [60]}}})

        assert ds._active_indices == before_active  # untouched
        assert 2 not in ds._row_overrides  # replaced
        assert ds._row_overrides[3]["requirements_general"] == ["y"]

    def test_load_adaptive_state_ignores_legacy_cache_field(self):
        """Pre-hotfix checkpoints wrote a general_rubric_cache key; loading must not error."""
        ds = _make_stub_dataset(n=3)
        ds.load_adaptive_state(
            {
                "row_overrides": {},
                "active_indices": [0, 1, 2],
                "general_rubric_cache": {0: [{"rubric": "legacy", "importance": 50}]},
            }
        )
        # The legacy cache is silently dropped (general rubrics now live inside overrides).
        assert ds._row_overrides == {}
        assert ds._active_indices == [0, 1, 2]


class TestRolloutRetryOversample:
    """Coverage for ``_gen_rollouts``' retry-oversample knobs.

    Stubs out ``generate_sequences`` + ``_decode_response`` + the dataset, and
    inspects the per-attempt ``repeat_times`` to verify oversample behaviour.
    """

    def _make(self, oversample: float = 1.0, max_retry: int | None = None) -> OnlineDataGenerator:
        cfg: dict = {
            "data_generator": {
                "mode": "constraint",
                "n_rollouts": 2,
                "max_tokens": 16,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "enable_thinking": True,
                "parse_patterns": {},
                "general_rubric_prompt": None,
                "constraint_judgment_prompt": None,
                "adaptive_rubric_prompt": None,
                "max_retry": 3,
                "rollout_retry_oversample": oversample,
            },
        }
        if max_retry is not None:
            cfg["data_generator"]["rollout_max_retry"] = max_retry
        gen = OnlineDataGenerator(config=cfg, tokenizer=MagicMock(), mode="constraint")
        gen._rollout_mgr = MagicMock()
        gen._rollout_mgr.agent_loop_workers = [MagicMock()]
        return gen

    def _stub_pipeline(
        self,
        gen: OnlineDataGenerator,
        decode_per_attempt: list[list[str]],
    ) -> dict:
        """Stub _build_gen_batch_from_indices, generate_sequences, _decode_response.

        ``decode_per_attempt`` is a list of decode-output sequences, one per
        attempt. Each inner list is what ``_decode_response`` returns for the
        rollout positions in that attempt's flattened batch (length n_attempt
        × pending samples).
        """
        attempt_state = {"i": 0, "repeats": []}

        def fake_build(indices, prompt_overrides=None):
            return MagicMock(repeat=MagicMock(side_effect=self._fake_repeat(attempt_state)))

        gen._build_gen_batch_from_indices = fake_build  # type: ignore[assignment]

        def fake_generate(_):
            return MagicMock()

        gen._rollout_mgr.generate_sequences = fake_generate

        decode_state = {"i": 0}

        def fake_decode(_output, idx):
            seq = decode_per_attempt[attempt_state["i"] - 1]
            return seq[idx]

        gen._decode_response = fake_decode  # type: ignore[assignment]
        return attempt_state

    @staticmethod
    def _fake_repeat(attempt_state):
        def _impl(repeat_times, interleave=True):
            attempt_state["i"] += 1
            attempt_state["repeats"].append(repeat_times)
            return MagicMock()

        return _impl

    def test_legacy_default_preserves_n(self, monkeypatch):
        """oversample=1.0 → every attempt requests n=2 (back-compat)."""
        gen = self._make(oversample=1.0)

        # Stub padding helpers so we never touch real DataProto math.
        from verl import protocol  # noqa: F401  (sanity)

        monkeypatch.setattr(
            "verl.protocol.pad_dataproto_to_divisor",
            lambda b, _: (b, 0),
        )
        monkeypatch.setattr("verl.protocol.unpad_dataproto", lambda b, _: b)

        # Two attempts: first all-empty (forces retry), second clears.
        decode_per_attempt = [
            ["", ""],  # attempt 0: pending=[0], n=2 → 2 decodes, all empty
            ["ok-a", "ok-b"],  # attempt 1: pending=[0], n=2 → both valid
        ]
        state = self._stub_pipeline(gen, decode_per_attempt)
        result = gen._gen_rollouts([0], n=2, min_valid=2)
        assert state["repeats"] == [2, 2]
        assert result == [["ok-a", "ok-b"]]

    def test_oversample_4x_on_retry(self, monkeypatch):
        """oversample=4 → retry attempts request 8 per prompt (one round clears)."""
        gen = self._make(oversample=4.0)

        monkeypatch.setattr(
            "verl.protocol.pad_dataproto_to_divisor",
            lambda b, _: (b, 0),
        )
        monkeypatch.setattr("verl.protocol.unpad_dataproto", lambda b, _: b)

        # Attempt 0: 2 decodes both empty. Attempt 1: 8 decodes, 2 valid + 6 empty.
        decode_per_attempt = [
            ["", ""],
            ["", "ok-1", "", "", "", "ok-2", "", ""],
        ]
        state = self._stub_pipeline(gen, decode_per_attempt)
        result = gen._gen_rollouts([0], n=2, min_valid=2)
        assert state["repeats"] == [2, 8]
        assert result == [["ok-1", "ok-2"]]

    def test_rollout_max_retry_caps_loop(self, monkeypatch):
        """rollout_max_retry=1 → exactly one retry round, then drop."""
        gen = self._make(oversample=4.0, max_retry=1)

        monkeypatch.setattr(
            "verl.protocol.pad_dataproto_to_divisor",
            lambda b, _: (b, 0),
        )
        monkeypatch.setattr("verl.protocol.unpad_dataproto", lambda b, _: b)

        decode_per_attempt = [
            ["", ""],  # attempt 0
            ["", "", "", "", "", "", "", ""],  # attempt 1 also all empty
        ]
        state = self._stub_pipeline(gen, decode_per_attempt)
        result = gen._gen_rollouts([0], n=2, min_valid=2)
        # Two attempts total (initial + 1 retry), then settle as empty.
        assert state["repeats"] == [2, 8]
        assert result == [[]]

    def test_init_rejects_oversample_below_one(self):
        with pytest.raises(ValueError, match=r"rollout_retry_oversample"):
            OnlineDataGenerator(
                config={"data_generator": {"mode": "constraint", "rollout_retry_oversample": 0.5}},
                tokenizer=MagicMock(),
                mode="constraint",
            )

    def test_init_rejects_negative_max_retry(self):
        with pytest.raises(ValueError, match=r"rollout_max_retry"):
            OnlineDataGenerator(
                config={"data_generator": {"mode": "constraint", "rollout_max_retry": -1}},
                tokenizer=MagicMock(),
                mode="constraint",
            )

    def test_rollout_max_retry_defaults_to_max_retry(self):
        gen = self._make(oversample=1.0)
        assert gen._rollout_max_retry == gen._max_retry == 3


class TestGeneratorRetryOversample:
    """Coverage for ``_generate_with_retry_keyed``'s per-call oversample +
    max_retry override, plus the ``generator_retry_oversample`` /
    ``generator_max_retry`` config knobs that every parse-validated
    generator call (decision, rewrite, general_rubric, constraint_judgment,
    adaptive_*, judge_*) inherits as its default.
    """

    def test_init_generator_max_retry_defaults_to_max_retry(self):
        gen = _make_generator()
        assert gen._generator_max_retry == gen._max_retry

    def test_init_rejects_generator_oversample_below_one(self):
        with pytest.raises(ValueError, match=r"generator_retry_oversample"):
            OnlineDataGenerator(
                config={"data_generator": {"mode": "constraint", "generator_retry_oversample": 0.4}},
                tokenizer=MagicMock(),
                mode="constraint",
            )

    def test_init_rejects_negative_generator_max_retry(self):
        with pytest.raises(ValueError, match=r"generator_max_retry"):
            OnlineDataGenerator(
                config={"data_generator": {"mode": "constraint", "generator_max_retry": -2}},
                tokenizer=MagicMock(),
                mode="constraint",
            )

    def test_init_reads_generator_knobs(self):
        gen = OnlineDataGenerator(
            config={
                "data_generator": {
                    "mode": "constraint",
                    "generator_retry_oversample": 5,
                    "generator_max_retry": 1,
                }
            },
            tokenizer=MagicMock(),
            mode="constraint",
        )
        assert gen._generator_retry_oversample == 5.0
        assert gen._generator_max_retry == 1

    def test_oversample_attempt0_unchanged(self):
        """Attempt 0 always sends 1 sample/prompt, even with oversample > 1."""
        gen = _make_generator()

        calls: list[list[str]] = []

        async def fake_call(prompts):
            calls.append(list(prompts))
            return [f"OK:{p}" for p in prompts]

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={1: "a", 2: "b"},
            parse_fn=lambda raw: raw if raw.startswith("OK:") else None,
            validate_fn=lambda p: p is not None,
            label="t",
            retry_oversample=5.0,
            max_retry_override=1,
        )
        assert results == {1: "OK:a", 2: "OK:b"}
        # First (and only) attempt: 1 sample per prompt, no oversample.
        assert len(calls) == 1
        assert sorted(calls[0]) == ["a", "b"]

    def test_oversample_5x_on_retry_first_success_chosen(self):
        """Retry attempts duplicate each pending prompt K times and pick
        the first parse-success per prompt."""
        gen = _make_generator()

        attempt_state = {"n": 0, "calls": []}

        async def fake_call(prompts):
            attempt_state["n"] += 1
            attempt_state["calls"].append(list(prompts))
            if attempt_state["n"] == 1:
                # Attempt 0: both fail.
                return ["BAD", "BAD"]
            # Attempt 1: 5 dupes per prompt, mixed FAIL/OK; first OK should win.
            #   prompt "a": ["BAD", "BAD", "OK:a-third", "OK:a-fourth", "BAD"]
            #   prompt "b": ["BAD", "OK:b-second", "BAD", "OK:b-fourth", "OK:b-fifth"]
            return [
                "BAD",
                "BAD",
                "OK:a-third",
                "OK:a-fourth",
                "BAD",
                "BAD",
                "OK:b-second",
                "BAD",
                "OK:b-fourth",
                "OK:b-fifth",
            ]

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={1: "a", 2: "b"},
            parse_fn=lambda raw: raw if raw.startswith("OK:") else None,
            validate_fn=lambda p: p is not None,
            label="t",
            retry_oversample=5.0,
            max_retry_override=1,
        )
        # Attempt 0: 2 prompts × 1 sample. Attempt 1: 2 prompts × 5 dupes.
        assert [len(c) for c in attempt_state["calls"]] == [2, 10]
        # First parse-success per prompt should be returned.
        assert results == {1: "OK:a-third", 2: "OK:b-second"}

    def test_max_retry_override_caps_loop(self):
        """max_retry_override=1 → exactly one retry, then drop."""
        gen = _make_generator()

        attempt_state = {"n": 0}

        async def fake_call(prompts):
            attempt_state["n"] += 1
            return ["BAD"] * len(prompts)

        gen._call_generator = fake_call  # type: ignore[assignment]

        results = gen._generate_with_retry_keyed(
            inputs={1: "a"},
            parse_fn=lambda raw: raw if raw.startswith("OK:") else None,
            validate_fn=lambda p: p is not None,
            label="t",
            retry_oversample=5.0,
            max_retry_override=1,
        )
        assert results == {}
        # Initial + 1 retry = 2 attempts only (regardless of self._max_retry).
        assert attempt_state["n"] == 2

    def test_max_retry_override_default_uses_self_max_retry(self):
        """No override → falls back to self._max_retry attempts."""
        gen = _make_generator()

        attempt_state = {"n": 0}

        async def fake_call(prompts):
            attempt_state["n"] += 1
            return ["BAD"] * len(prompts)

        gen._call_generator = fake_call  # type: ignore[assignment]

        gen._generate_with_retry_keyed(
            inputs={1: "a"},
            parse_fn=lambda raw: None,
            validate_fn=lambda p: p is not None,
            label="t",
        )
        # _max_retry + 1 attempts in default _make_generator (max_retry=2 → 3).
        assert attempt_state["n"] == gen._max_retry + 1

    def test_oversample_below_one_rejected_per_call(self):
        gen = _make_generator()
        gen._call_generator = MagicMock()  # should not be called
        with pytest.raises(ValueError, match=r"retry_oversample"):
            gen._generate_with_retry_keyed(
                inputs={1: "a"},
                parse_fn=lambda r: r,
                validate_fn=lambda p: True,
                label="t",
                retry_oversample=0.5,
            )

    def test_negative_max_retry_override_rejected(self):
        gen = _make_generator()
        gen._call_generator = MagicMock()
        with pytest.raises(ValueError, match=r"max_retry_override"):
            gen._generate_with_retry_keyed(
                inputs={1: "a"},
                parse_fn=lambda r: r,
                validate_fn=lambda p: True,
                label="t",
                max_retry_override=-1,
            )
