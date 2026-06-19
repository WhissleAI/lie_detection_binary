# Handoff & next steps (for the student taking this forward)

This doc is the "where things stand and what to do next" guide. Read the
top-level `README.md` first (pipeline + how to run), then `docs/GATEWAY.md` (the
external Whissle gateway dependency).

## Where things stand

**Done and working:**
- Full pipeline scaffolding, 5 steps + `run_all.py`, all resumable & cached.
- Dataset manifest with **speaker groups** parsed from the README (33 speakers).
- Whissle gateway extraction (`/video/analyze`) wired and documented — gives the
  transcript + per-segment metadata **and** the per-frame visual timeline.
- Audio prosody lane (librosa) — runs locally, validated on all 121 clips.
- Feature assembly → 121 features (54 text + 45 visual + 22 audio).
- **Leave-One-Speaker-Out** evaluation, 4 models × 7 modality subsets, ablations,
  permutation importance, model persistence.
- Smoke tests (`tests/test_smoke.py`), all green.

**Baseline numbers (offline `--bootstrap`: bundled transcripts + prosody, no
gateway visual yet):** best LOSO model ≈ RandomForest on text, **acc ≈ 0.63,
AUC ≈ 0.67**. These are the text+audio floor; the visual lane is empty in
bootstrap.

**The immediate to-do:** run the **real** gateway pass to populate the visual
lane and the metadata-rich transcripts:

```bash
# 1) gateway running + token in .env (see docs/GATEWAY.md)
python scripts/02_extract_av.py --overwrite     # real STT + visual, all 121 clips
python scripts/04_build_features.py
python scripts/05_train.py                       # the real tri-modal LOSO numbers
```

Then compare the per-modality ablation rows in `data/reports/cv_results.csv` to
see how much the visual + STT-metadata lanes add over text alone.

## Ideas to push the research further

Roughly in order of effort/payoff:

1. **Sanity-check the visual lane.** Eyeball a few `data/av/<clip>.json`
   `visual_timeline`s. `face_detect_rate` is low on some clips (small/oblique
   faces) — raising `VISUAL_SAMPLE_FPS`, or asking the gateway team for a
   detector tuned to low-res courtroom footage, would help coverage.
2. **Per-segment / temporal modelling.** We currently average frames to one
   vector per clip. Deception cues are *dynamic* (a gaze break right after a
   question). Try sequence models (LSTM/Transformer) over the frame/segment
   stream, or simple change-point / burst features.
3. **Better text features.** Swap the hand lexicons for embeddings
   (sentence-transformers) or an LLM-judge feature ("rate evasiveness 0–1").
   Keep them speaker-independent.
4. **Calibration + thresholds.** Report calibrated probabilities
   (`CalibratedClassifierCV`) and pick an operating point; accuracy alone hides
   the precision/recall trade-off on a 50/50 set.
5. **Fusion strategy.** We do early fusion (concatenate features). Try late
   fusion (per-modality models + a stacker) — often more robust when one
   modality is missing/noisy.
6. **Robustness / leakage audits.** Confirm no speaker leaks across folds; add a
   *shuffled-label* run as a null baseline (should collapse to ~0.5 AUC). Watch
   for trivial shortcuts (clip length, audio channel artefacts).
7. **External validity.** This dataset is tiny and US-trial-specific. Try a
   second deception corpus (e.g. Bag-of-Lies, MU3D) and test cross-dataset
   transfer — the honest measure of whether any signal generalises.

## Gotchas to remember

- **Never** switch to a random `train_test_split` — it leaks speaker identity and
  reports fake accuracy. Always group by `speaker` (`modeling/train.py` already
  does). 18 speakers have multiple clips; 7 appear in both classes.
- The bundled `Annotation/All_Gestures_*.csv` is the *original paper's* human
  annotation — a reference baseline, not our features. Don't train on it.
- `data/` is gitignored and fully regenerable; commit code + docs, not artefacts.
- This is a research probe, not a deployable lie detector. See the ethics note in
  the README.
