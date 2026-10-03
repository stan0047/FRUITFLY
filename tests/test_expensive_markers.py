"""Pin down the expensive/fast marker separation for the benchmark suite.

The default ``python -m pytest -q`` must never start a benchmark, yet the
expensive slice must stay collectable on demand. No benchmark test is deleted or
weakened; it is excluded only by marker.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _collect(expression: str | None, *paths: str) -> str:
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if expression is not None:
        cmd += ["-m", expression]
    cmd += list(paths)
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    return result.stdout + result.stderr


class TestMarkerSeparation:
    def test_default_collection_excludes_the_declared_4c_benchmark(self) -> None:
        out = _collect(None, "tests/test_motion_benchmark.py")
        # Classes composed entirely of benchmark-driven tests vanish from the
        # default collection ...
        for name in (
            "TestPhase4CConfiguration",
            "TestPhase4CControls",
            "TestPhase4CClaims",
            "TestScript::test_quick_mode_runs_end_to_end",
        ):
            assert name not in out, name
        # ... and within the classes that also contain fast unit tests, it is the
        # expensive methods, not the class, that are deselected.
        assert "test_two_runs_of_the_declared_configuration_agree" not in out
        assert "test_the_run_carries_a_content_fingerprint" not in out
        assert "test_the_power_limitation_is_stated_and_computed" not in out
        assert "test_the_primary_quantity_is_recurrent_minus_shuffled" not in out
        # Liveness carries the cheap, pure-function tests that stay collected.
        assert "test_a_silent_condition_is_marked_uninformative" in out
        assert "test_the_margin_pairs_shared_folds" in out
        assert "deselected" in out

    def test_the_expensive_set_is_still_collectable(self) -> None:
        out = _collect("expensive", "tests/test_motion_benchmark.py")
        for name in (
            "TestPhase4CConfiguration",
            "TestPhase4CDeterminism",
            "TestPhase4CLiveness",
            "TestPhase4CPrimaryComparison",
            "TestPhase4CControls",
            "TestPhase4CClaims",
            "TestScript::test_quick_mode_runs_end_to_end",
            "TestConstantFeatureReporting::test_the_power_limitation_is_stated_and_computed",
        ):
            assert name in out, name
        assert "57/198" in out

    def test_the_estimate_path_and_determinism_reruns_are_marked_expensive(self) -> None:
        out = _collect("expensive", "tests/test_motion_benchmark.py")
        assert "test_two_runs_of_the_declared_configuration_agree" in out
        assert "test_the_estimate_is_written_and_declares_what_it_reduced" in out

    def test_phase_5a_is_independent_and_never_expensive(self) -> None:
        fast = _collect("not expensive", "tests/test_phase5a_pilot.py")
        # Every Phase 5A test is still collected in the fast slice...
        assert fast.count("::") >= 73
        # ...and the expensive slice contains no Phase 5A item at all.
        expensive = _collect("expensive", "tests/test_phase5a_pilot.py")
        assert "::" not in expensive
