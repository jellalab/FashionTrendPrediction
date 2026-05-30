"""End-to-end orchestrator for every production pipeline.

Runs the modules in the order they're designed to feed each other:

1. **Pipeline 1 / Step 1** — :mod:`src.detect` writes ``detections.csv``.
2. **Pipeline 1 / Step 2A** — :mod:`src.color` writes ``color_attributes.csv``.
3. **Pipeline 1 / Step 2B** — :mod:`src.pattern` writes ``pattern_attributes.csv``.
4. **Pipeline 1 / Step 2C** — :mod:`src.clip_refine` writes ``clip_refinement.csv``.
5. **Pipeline 1 / Step 3** — :mod:`src.join` merges all four into
   ``yolo_fashion_attributes.csv``.
6. **Pipeline 1 visualisations** — :mod:`src.pipeline1_viz` writes the
   six descriptive PNGs.
7. **Pipeline 2** — :mod:`src.popularity` runs the consumer-behaviour
   ablation.

The interactive validation app (:mod:`src.validate`) is deliberately
excluded; it needs a human at the keyboard. Re-running is safe: every
downstream step rewrites its output cleanly.

Run from the project root::

    uv run python -m src.run_all
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.clip_refine import run_clip_refinement
from src.color import run_color_extraction
from src.detect import run_detection
from src.join import run_join
from src.pattern import run_pattern_detection
from src.pipeline1_viz import run_pipeline1_viz
from src.popularity import run_pipeline as run_popularity_pipeline
from src.utils import (
    load_clip_refine_config,
    load_color_config,
    load_detection_config,
    load_join_config,
    load_pattern_config,
    load_pipeline1_viz_config,
    load_popularity_config,
)

logger = logging.getLogger(__name__)


__all__ = ["PIPELINE_STEPS", "PipelineStep", "main", "run_all"]


@dataclass(frozen=True)
class PipelineStep:
    """One named step in the end-to-end orchestration."""

    name: str
    runner: Callable[[], Any]


# Ordered so each step's inputs exist by the time it runs:
#   detect → (color, pattern, clip_refine) → join → pipeline1_viz → popularity
# Pipeline 2 is independent of Pipeline 1 but lives at the end so the
# whole thesis stack regenerates from a single command.
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        "Pipeline 1 / Step 1 — YOLO garment detection",
        lambda: run_detection(load_detection_config()),
    ),
    PipelineStep(
        "Pipeline 1 / Step 2A — Dominant colour extraction",
        lambda: run_color_extraction(load_color_config()),
    ),
    PipelineStep(
        "Pipeline 1 / Step 2B — Pattern complexity scoring",
        lambda: run_pattern_detection(load_pattern_config()),
    ),
    PipelineStep(
        "Pipeline 1 / Step 2C — CLIP zero-shot refinement",
        lambda: run_clip_refinement(load_clip_refine_config()),
    ),
    PipelineStep(
        "Pipeline 1 / Step 3 — Join per-garment attributes",
        lambda: run_join(load_join_config()),
    ),
    PipelineStep(
        "Pipeline 1 — Descriptive visualisations",
        lambda: run_pipeline1_viz(load_pipeline1_viz_config()),
    ),
    PipelineStep(
        "Pipeline 2 — Consumer-behaviour classification",
        lambda: run_popularity_pipeline(load_popularity_config()),
    ),
)


def _banner(label: str) -> None:
    bar = "=" * (len(label) + 8)
    print()
    print(bar)
    print(f"=== {label} ===")
    print(bar)


def run_all(steps: tuple[PipelineStep, ...] = PIPELINE_STEPS) -> dict[str, float]:
    """Run every step in *steps* sequentially. Returns per-step wall-clock seconds.

    A failure in any step aborts the run — downstream stages depend on
    the upstream CSVs and would otherwise compound the error.
    """
    timings: dict[str, float] = {}
    total_started = time.time()

    for step in steps:
        _banner(step.name)
        started = time.time()
        try:
            step.runner()
        except Exception:
            elapsed = time.time() - started
            logger.exception(
                "%s failed after %.1fs — aborting orchestrator",
                step.name,
                elapsed,
            )
            raise
        elapsed = time.time() - started
        timings[step.name] = elapsed
        logger.info("%s finished in %.1fs", step.name, elapsed)

    print()
    print("=" * 60)
    print("All pipelines completed.")
    for name, secs in timings.items():
        print(f"  {secs:7.1f}s  {name}")
    print(f"  {'-' * 7}")
    print(f"  {time.time() - total_started:7.1f}s  total")
    print("=" * 60)
    return timings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_all()


if __name__ == "__main__":
    main()
