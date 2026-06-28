"""Unit tests for fatigue/ear.py — pure logic only (no video, no MediaPipe model)."""
import math

from fatigue.ear import _closure_runs, _ear_one_eye, ear_summary


class _LM:
    """Minimal stand-in for a MediaPipe landmark (just needs .x / .y)."""
    def __init__(self, x, y):
        self.x, self.y = x, y


# ---- _closure_runs: the streak counter ----
def test_closure_runs_basic():
    assert _closure_runs([False, False, True, True, True, False, True, False, False, True, True]) == [3, 1, 2]


def test_closure_runs_edges():
    assert _closure_runs([True, True, True]) == [3]      # all closed
    assert _closure_runs([False, False, False]) == []    # none closed
    assert _closure_runs([]) == []                       # empty
    assert _closure_runs([False, True, True]) == [2]     # ends while still closed


# ---- _ear_one_eye: the EAR formula ----
def _eye(height):
    """6 eye points with a given vertical opening; width fixed at 1.0."""
    top, bot = 0.5 + height / 2, 0.5 - height / 2
    return {0: _LM(0.0, 0.5), 1: _LM(0.33, top), 2: _LM(0.66, top),
            3: _LM(1.0, 0.5), 4: _LM(0.66, bot), 5: _LM(0.33, bot)}


def test_ear_open_vs_closed():
    idx = (0, 1, 2, 3, 4, 5)
    assert _ear_one_eye(_eye(0.6), idx) > 0.3       # open eye: healthy ratio
    assert _ear_one_eye(_eye(0.02), idx) < 0.1      # shut eye: near zero
    assert _ear_one_eye(_eye(0.6), idx) > _ear_one_eye(_eye(0.02), idx)


def test_ear_exact_value():
    # height 0.6, width 1.0 -> EAR = (0.6+0.6)/(2*1.0) = 0.6
    assert math.isclose(_ear_one_eye(_eye(0.6), (0, 1, 2, 3, 4, 5)), 0.6, abs_tol=1e-6)


# ---- ear_summary: the drowsiness features ----
def test_summary_detects_microsleep():
    fps = 10.0
    ears = [0.30] * 20 + [0.05] * 10 + [0.30] * 10   # a 10-frame (=1.0s) closure
    f = ear_summary(ears, fps, n_sampled=len(ears))
    assert f["ear_microsleep_count"] == 1.0                       # 1.0s >= 0.5s
    assert math.isclose(f["ear_closure_max_s"], 1.0, abs_tol=1e-6)
    assert math.isclose(f["ear_perclos"], 0.25, abs_tol=1e-9)     # 10 of 40 frames
    assert f["ear_face_detect_rate"] == 1.0


def test_summary_no_face():
    assert ear_summary([], 10.0, n_sampled=0) == {"ear_face_detect_rate": 0.0}
