"""Eye Aspect Ratio (EAR) drowsiness features from raw MediaPipe Face Mesh landmarks.

The Whissle gateway's visual_timeline only exposes a coarse boolean ``blink``
(MediaPipe's eyeBlink blendshape thresholded at 0.5). For drowsiness the *degree*
and *duration* of eye closure matter — slow lid closures and microsleeps, not just
blink counts. So here we compute the classic Eye Aspect Ratio (Soukupova & Cech,
2016) directly from the 468-point Face Mesh landmarks: a continuous eye-openness
signal per frame.

Runs locally / on-device (MediaPipe only) — no gateway or token. This is the
improvement flagged in fatigue/RESULTS.md ("eye-aspect-ratio from raw MediaPipe
landmarks would likely lift the 0.665").

    python -m fatigue.ear <video.mp4>     # demo: print EAR series + features

Install (this lane's extra deps):
    pip install -r fatigue/requirements-ear.txt

Integrate into the fatigue feature matrix (fatigue/train.py build_matrix), per clip
(the windowed clip lives at C.CLIPS / f"{cid}.mp4"):
    from fatigue.ear import ear_features
    clip = C.CLIPS / f"{cid}.mp4"
    if clip.exists():
        feat.update({f"ear_{k}": v for k, v in ear_features(clip).items()})
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# --- MediaPipe Face Mesh eye landmarks, in EAR order (p1..p6) ----------------
#   p1, p4        = horizontal eye corners (eye width)
#   p2-p6, p3-p5  = the two vertical lid pairs (eye height)
_RIGHT_EYE = (33, 160, 158, 133, 153, 144)
_LEFT_EYE = (362, 385, 387, 263, 373, 380)

EAR_CLOSED_THR = 0.20    # EAR below this = eye treated as closed (fixed; adaptive is a later step)
MICROSLEEP_MIN_S = 0.5   # a single closure lasting >= this many seconds = a microsleep
SAMPLE_FPS = 10.0        # frames per second we pull from the video

# FaceLandmarker model bundle (same model family the gateway uses).
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def ensure_model() -> Path:
    """Download the FaceLandmarker model once (~3.8 MB) if it isn't here yet."""
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _ear_one_eye(landmarks, idx) -> float:
    """EAR for one eye = (sum of vertical lid gaps) / (2 x horizontal width)."""
    p1, p2, p3, p4, p5, p6 = (np.array([landmarks[i].x, landmarks[i].y]) for i in idx)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return float(vertical / (2.0 * horizontal)) if horizontal > 1e-6 else 0.0


def ear_per_frame(video_path, sample_fps: float = SAMPLE_FPS) -> dict:
    """Run Face Mesh over the video; return the per-sampled-frame EAR series."""
    ensure_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / sample_fps))     # e.g. 30fps video, want 10 -> every 3rd frame
    eff_fps = src_fps / step

    ears, n_sampled, n_faceless, fidx = [], 0, 0, 0
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fidx % step == 0:                    # only look at every Nth frame
                n_sampled += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    ears.append(0.5 * (_ear_one_eye(lms, _RIGHT_EYE) + _ear_one_eye(lms, _LEFT_EYE)))
                else:
                    n_faceless += 1
            fidx += 1
    cap.release()
    return {"ear": ears, "n_sampled": n_sampled, "n_faceless": n_faceless, "fps": eff_fps}


def _closure_runs(closed) -> list[int]:
    """Lengths (in frames) of each consecutive run of eyes-closed."""
    runs, cur = [], 0
    for c in closed:
        if c:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


def ear_summary(ears, fps, n_sampled, closed_thr: float = EAR_CLOSED_THR) -> dict[str, float]:
    """Compute the ``ear_*`` features from a ready-made per-frame EAR series.

    Split out of ``ear_features`` so the drowsiness math can be unit-tested
    without needing a video or the MediaPipe model.
    """
    n_face, total = len(ears), (n_sampled or 1)
    if n_face == 0:
        return {"ear_face_detect_rate": 0.0}

    arr = np.array(ears)
    closed = [e < closed_thr for e in ears]
    runs = _closure_runs(closed)
    longs = [r for r in runs if (r / fps) >= MICROSLEEP_MIN_S]

    return {
        "ear_mean": float(arr.mean()),
        "ear_std": float(arr.std()),
        "ear_min": float(arr.min()),
        "ear_perclos": float(np.mean(closed)),                       # fraction of frames eyes closed
        "ear_blink_rate_pm": len(runs) / (n_face / fps / 60.0),      # eye-closures per minute
        "ear_closure_max_s": (max(runs) / fps) if runs else 0.0,     # longest single closure (microsleep)
        "ear_closure_mean_s": (float(np.mean(runs)) / fps) if runs else 0.0,
        "ear_microsleep_count": float(len(longs)),
        "ear_face_detect_rate": n_face / total,
    }


def ear_features(video_path, sample_fps: float = SAMPLE_FPS,
                 closed_thr: float = EAR_CLOSED_THR) -> dict[str, float]:
    """Per-clip EAR drowsiness features (all prefixed ``ear_``)."""
    s = ear_per_frame(video_path, sample_fps)
    return ear_summary(s["ear"], s["fps"], s["n_sampled"], closed_thr)


# ---------------------------------------------------------------------------
# Demo: run on one clip and visualise the EAR signal so blinks are visible.
# ---------------------------------------------------------------------------
def _sparkline(values, lo=0.0, hi=0.35) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    out = []
    for v in values:
        t = (v - lo) / (hi - lo) if hi > lo else 0.0
        t = min(1.0, max(0.0, t))
        out.append(blocks[round(t * (len(blocks) - 1))])
    return "".join(out)


if __name__ == "__main__":
    import sys

    video = sys.argv[1]
    print(f"Running MediaPipe EAR on: {video}\n")
    s = ear_per_frame(video)
    ears = s["ear"]
    print(f"frames sampled : {s['n_sampled']}  (effective {s['fps']:.1f} fps)")
    print(f"face found in  : {len(ears)}   faceless: {s['n_faceless']}")
    if ears:
        print(f"\nEAR over time (low = eyes closed; threshold {EAR_CLOSED_THR}):")
        print("  " + _sparkline(ears))
        print(f"  min {min(ears):.3f}   mean {np.mean(ears):.3f}   max {max(ears):.3f}")
    print("\nFeatures:")
    for k, v in ear_features(video).items():
        print(f"  {k:24} {v:.3f}")
