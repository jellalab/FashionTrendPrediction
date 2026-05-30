"""Tests for the end-to-end orchestrator (src/run_all.py).

The real steps load weights / hit data/, so we substitute lightweight
fakes via the ``steps`` parameter and assert the orchestration contract
(order, abort-on-failure, timing dict).
"""

from __future__ import annotations

import pytest

from src.run_all import PIPELINE_STEPS, PipelineStep, run_all


def test_pipeline_steps_intended_order():
    """Step ordering encodes the data-dependency chain — pin it explicitly."""
    names = [step.name for step in PIPELINE_STEPS]
    # detect must come before any per-detection extractor
    assert names.index("Pipeline 1 / Step 1 — YOLO garment detection") == 0
    # join must follow all three extractors
    extractor_indices = [
        names.index("Pipeline 1 / Step 2A — Dominant colour extraction"),
        names.index("Pipeline 1 / Step 2B — Pattern complexity scoring"),
        names.index("Pipeline 1 / Step 2C — CLIP zero-shot refinement"),
    ]
    join_index = names.index("Pipeline 1 / Step 3 — Join per-garment attributes")
    assert max(extractor_indices) < join_index
    # visualisations consume the joined CSV
    viz_index = names.index("Pipeline 1 — Descriptive visualisations")
    assert viz_index > join_index


def test_run_all_calls_every_step_in_order():
    calls: list[str] = []

    def make_runner(name: str):
        return lambda: calls.append(name)

    steps = (
        PipelineStep("alpha", make_runner("alpha")),
        PipelineStep("beta", make_runner("beta")),
        PipelineStep("gamma", make_runner("gamma")),
    )
    timings = run_all(steps)

    assert calls == ["alpha", "beta", "gamma"]
    assert set(timings) == {"alpha", "beta", "gamma"}
    assert all(secs >= 0 for secs in timings.values())


def test_run_all_aborts_on_failure():
    calls: list[str] = []

    def fail() -> None:
        raise RuntimeError("step 2 broke")

    steps = (
        PipelineStep("ok", lambda: calls.append("ok")),
        PipelineStep("boom", fail),
        PipelineStep("never", lambda: calls.append("never")),
    )

    with pytest.raises(RuntimeError, match="step 2 broke"):
        run_all(steps)

    assert calls == ["ok"]  # downstream step never ran
