# Binary Lie Detector — Real-life Trial Deception

A self-contained, reproducible pipeline that predicts **deceptive vs. truthful**
from short courtroom video clips, fusing three modalities:

| Lane | Source | Signals |
|---|---|---|
| **Text** | Whissle STT (gateway `/video/analyze`) | transcript + per-segment **emotion / intent / age / gender** metadata, entities, diarization, word timing |
| **Visual** | Audio-visual hybrid intelligence (same gateway call) | per-frame **emotion, head pose, gaze, blink, mouth, attention** + hand gestures |
| **Audio** | local prosody (librosa) | pitch (F0), jitter/shimmer, pauses, voice quality |

The text + visual features come from a **single Whissle gateway call** —
`POST /video/analyze` runs Whissle ASR (with metadata tags) and the audio-visual
lane, then fuses them. Prosody is a complementary local lane. Everything else —
feature engineering, speaker-independent evaluation, and the classifier — lives
in this repo.

> ⚠️ **External dependency — the Whissle gateway is NOT in this repo.**
> The STT (transcript + metadata) and visual feature extraction run on the
> **Whissle gateway Docker** (`whissleasr/whissle-gateway`, port 9000). This repo
> only *calls* it over HTTP and parses the result. You must have the gateway
> running and a `wh_` token to do the real extraction. See **[docs/GATEWAY.md](docs/GATEWAY.md)**
> for how to run it, the full request/response contract, and troubleshooting.
> Only the audio-prosody lane runs locally here (needs `ffmpeg`).

> **Dataset:** *Real-life Trial Deception Detection* (Pérez-Rosas et al., 2015,
> Univ. of Michigan): 121 clips (61 deceptive / 60 truthful) from real trials.

> 🎓 **Taking this forward?** Start with **[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)** —
> current status, the immediate to-do (real gateway pass), and research ideas.

---

## Why this is harder than it looks (and how we handle it)

The 121 clips come from only **~33 unique speakers** — one defendant (Jodi Arias)
accounts for **32 clips**, and **7 speakers appear in both classes**. A random
train/test split lets a model memorise *who is speaking* instead of *whether they
are lying*, producing inflated, meaningless accuracy.

**We evaluate with Leave-One-Speaker-Out (LOSO) cross-validation**: every clip
from a given person is held out together. Speaker identity is parsed from the
dataset README and used as the CV grouping key. This is the only honest estimate
of generalisation to an unseen person — and it is the headline methodology of
this project.

---

## Architecture

```
                    Real-life trial clip (.mp4)
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                       ▼
 gateway /asr/transcribe  gateway /video/analyze  ffmpeg → 16k wav
 (wav: transcript +       (mp4: visual_timeline)      │
  metadata + pauses +          │                      ▼
  word conf + probs)           │               prosody (librosa)
        │                      │                      │
        ▼                      ▼                      ▼
   text_features          visual_features        audio_features
   (lexical + STT          (gaze/pose/emotion/    (F0/jitter/pauses/
    metadata probs)         blink/gestures)        voice quality)
        └──────────┬───────────┴───────────┬─────────┘
                   ▼                        ▼
             multimodal feature matrix (one row / clip)
                   │
                   ▼
   Leave-One-Speaker-Out CV  →  LogReg / SVM / RandomForest / HistGBM
                   │
                   ▼
   metrics (acc / balanced-acc / AUC / F1) + per-modality ablations
   + permutation feature importance  →  best_model.joblib
```

> Step 02 makes **two** gateway calls per clip: `/asr/transcribe` for the rich
> text + metadata lane and `/video/analyze` for the visual timeline. (The video
> endpoint also runs ASR internally, but its fuser only forwards a `segments`
> field this model doesn't emit, so the metadata would be lost — hence the
> dedicated `/asr/transcribe` call.) See [docs/GATEWAY.md](docs/GATEWAY.md).

---

## Setup

Prerequisites: **Python 3.10+**, **ffmpeg** on PATH (`brew install ffmpeg` /
`apt install ffmpeg`), and access to a **Whissle gateway** (the local docker
`whissle-gateway` on `:9000`, or `https://api.whissle.ai`).

```bash
cd lie_detection_binary
./setup.sh                      # creates .venv, installs deps, installs this package
source .venv/bin/activate

cp .env.example .env            # then edit .env:
#   WHISSLE_API_TOKEN=wh_...    (required for the gateway STT + visual step)
#   WHISSLE_GATEWAY_URL=http://localhost:9000
#   DECEPTION_DATASET_DIR=/path/to/Real-life_Deception_Detection_2016
```

The gateway requires `Authorization: Bearer wh_...`. Create a token at
<https://lulu.whissle.ai/access>.

---

## Usage

Run the whole pipeline:

```bash
python scripts/run_all.py                 # real: gateway STT + audio-visual + prosody
python scripts/run_all.py --limit 5       # quick smoke run on 5 clips
python scripts/run_all.py --bootstrap     # offline: bundled transcripts (text+audio only)
```

…or step by step:

```bash
python scripts/01_build_manifest.py       # clips → labels + speaker groups  (no token)
python scripts/02_extract_av.py           # gateway /video/analyze → STT + visual  (token)
python scripts/03_extract_audio.py        # librosa prosody                  (no token)
python scripts/04_build_features.py       # assemble feature matrix          (no token)
python scripts/05_train.py                # LOSO CV, ablations, importance    (no token)
python scripts/06_paper_comparison.py     # paper protocol vs ours + manual gestures (no token)
```

Each extraction step is **resumable** (skips clips already done; `--overwrite`
to force) and accepts `--limit N` for quick tests.

### Bootstrap mode (no token yet)

`--bootstrap` builds text-only records from the dataset's bundled transcripts so
you can exercise the text + audio pipeline immediately. Swap in your
`WHISSLE_API_TOKEN` and rerun `02_extract_av.py --overwrite` to get the real
metadata-rich transcripts **and** the visual lane.

---

## Outputs

```
data/
  manifest.csv                 clip → label, speaker, role
  wav/<clip>.wav               16 kHz mono audio (for prosody)
  av/<clip>.json               fused gateway response (transcript + segments + visual_timeline)
  audio/<clip>.json            prosody features
  features/features.parquet    the multimodal feature matrix (+ .csv)
  reports/cv_results.csv        model × modality → LOSO metrics
  reports/feature_importance.csv
  reports/paper_comparison.csv  video-out vs speaker-out, incl. manual gestures
  reports/summary.json
  models/best_model.joblib      refit best pipeline + metadata
```

`05_train.py` prints a table like (real run, 169 features, LOSO CV):

```
        model     modality  n_features  accuracy  balanced_accuracy  roc_auc    f1
      svm_rbf         text         102     0.570              0.571    0.655 0.527
      svm_rbf   text+audio         124     0.603              0.604    0.650 0.586
     hist_gbm          all         169     0.603              0.604    0.615 0.556
random_forest       visual          45     0.562              0.563    0.616 0.531
majority_baseline      —             0     0.504              0.500    0.500 0.671
```

Honest, speaker-independent numbers land around **AUC 0.62–0.66 / accuracy
~0.60** — clearly above the 0.50 base rate but far from "solved" (and lower than
papers that leak speaker identity via random splits). The **Whissle STT metadata
probability** features (behavior/age/emotion distributions) and a few
psycholinguistic rates (third-person, negation, neg-emotion) carry most of the
signal; the visual lane adds a modest independent ~0.6 AUC on its own.

> ⚠️ **Confound:** the model's audio **gender** read correlates with the label
> (`corr ≈ −0.35`) because the deceptive set is dominated by a few female
> speakers (Jodi Arias, Amanda Hayes, Crystal Mangum). So `meta_gender_*` /
> `meta_age_*` partly encode *demographics, not deception*. See
> [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) — re-run with demographics dropped to
> measure the genuine signal.

---

## Results & comparison (Pérez-Rosas et al., 2015 + LLM baselines)

All numbers below use the **whissle-large** ASR model (transcript + emotion/age/
gender/**intent** probabilities). The paper reports up to **75.2%** accuracy; our
honest, speaker-independent headline is lower — and that gap is **the CV protocol,
not a modelling flaw**. `06_paper_comparison.py` runs every feature set under both
protocols (pooled out-of-fold accuracy):

| feature set | model | leave-1-**video**-out (paper) | leave-1-**speaker**-out (honest) | leakage gap |
|---|---|---:|---:|---:|
| our_text (auto) | RandomForest | 0.752 | 0.587 | +0.165 |
| our_visual (auto) | RandomForest | 0.719 | 0.612 | +0.107 |
| our_audio (auto) | DecisionTree | 0.719 | 0.570 | +0.149 |
| our_all (text+audio+visual) | RandomForest | 0.752 | 0.529 | +0.223 |
| manual_gestures (paper's CSV) | RandomForest | 0.769 | **0.686** | +0.083 |
| **gemini_features** (LLM video scores) | RandomForest | 0.694 | **0.678** | +0.017 |
| gemini+our_all | RandomForest | **0.777** | 0.620 | +0.157 |
| *majority baseline* | — | 0.504 | 0.504 | — |

**LLM zero-shot baselines** (Gemini 2.5 Pro, no training → no CV, no leakage):

| approach | accuracy | balanced acc | AUC | deceptive-call rate |
|---|---:|---:|---:|---|
| **Gemini direct VIDEO** | **0.669** | **0.669** | **0.749** | 55% (calibrated) |
| Gemini over-features (v1 forensic prompt) | 0.554 | 0.550 | 0.631 | 93% (biased) |
| Gemini over-features (v2 neutral prompt) | 0.512 | 0.511 | 0.516 | 73% |

Our **trained** models under leave-one-speaker-out (step 05, full sweep) peak at
**AUC 0.670** (whissle-large, up from 0.655 on the small ASR model).

Takeaways:
1. **The paper's protocol leaks speaker identity.** Leave-one-*video*-out keeps 31
   of Jodi Arias's 32 clips in training when testing the 32nd, so the model learns
   the person. The "leakage gap" column is the inflation it buys (+0.02 to +0.26).
   Under the paper's own protocol we match it (our_text/our_all 0.752) and beat it
   when we add the LLM's video reads (gemini+our_all **0.777**).
2. **Best honest result = Gemini watching the raw video** (zero-shot **AUC 0.749**,
   balanced 0.669, no training, no leakage). It's well-calibrated (55% deceptive
   calls vs the 50% base rate).
3. **Reasoning over our feature *digest* fails** (AUC 0.52–0.63, chance-level) even
   though the *same features* train to AUC 0.67. Summarising the clip into a list
   of cues both loses information the video carries and primes the LLM toward
   "deceptive." A neutral, base-rate-anchored prompt (v2) cuts the bias
   (deceptive-calls 93%→73%, truthful 7→17/60) but can't manufacture signal the
   digest doesn't hold. **Watching beats reading our digest; and a trained model
   beats the LLM at reading it.**
4. **Manual gold gestures generalise best of the feature sets** (speaker-out 0.686);
   our automatic MediaPipe visual lane is noisier (0.612). Gemini's *video-derived*
   feature scores are a close second (0.678) and the most leakage-robust (gap +0.02).
5. **whissle-large helped** — better transcripts + intent lifted the trained models
   (AUC 0.655→0.670) — but did not rescue the feature-digest LLM (still chance).

> Bottom line: 75% on this dataset is a leave-one-video-out (speaker-leaky) number.
> The honest, speaker-independent ceiling here is ~0.65–0.69 accuracy / ~0.67–0.75
> AUC, and the single best honest result is **Gemini reading the raw video
> (AUC 0.749)** — not any feature-engineered pipeline.

### Best systems — with vs. without the LLM (`09_best_fusion.py`)

Concatenating all 178 features *hurts* (gemini_features alone beats gemini+our_all)
— 121 clips can't support 178 dims. **Feature selection + late fusion** fix it.
Two deployable configurations, both leave-one-speaker-out (honest):

| config | best method | accuracy | AUC |
|---|---|---:|---:|
| **A — with Gemini** | `SelectKBest(k=10)` on our+gemini features | 0.669 | **0.747** |
| | late-fusion: our model ⊕ Gemini's video prob | **0.686** | 0.707 |
| **B — self-hosted, no LLM** | weighted late-fusion (.4 txt/.4 vis/.2 aud) | **0.645** | 0.655 |
| | `hist_gbm` (best AUC; threshold-tuned acc 0.645) | 0.562→0.645 | **0.670** |

- **Config A** matches Gemini-video-alone (0.747) but as a *trained, calibratable*
  classifier. Its top features are Gemini's holistic reads — `defensiveness`,
  `overall_credibility`, `story_specificity`, `microexpression_leakage` — plus our
  head-pitch, vocal F0, and negation rate.
- **Config B** sends **no raw audio/video to any external LLM** — features are
  extracted only by the (self-hostable) Whissle gateway + local prosody, and a
  trained model predicts. Honest **AUC 0.670 / accuracy ~0.645** (the naive 0.562
  was a 0.5-threshold artifact; a weighted per-modality late-fusion is calibrated
  to 0.645 out of the box — see `10_improve_selfhosted.py`). Top cues: head-pitch
  (looking down), vocal pitch, negations, fear expression. AUC is **capped ~0.67
  by the features** — ensembling/stacking/late-fusion can't beat it; only better
  features (visual face-detection, temporal cues) would.
- Naive concat: AUC 0.640 → `SelectKBest(k=10)`: **0.747**. The lesson is
  selection, not concatenation.

> So we can show **both**: a stronger result *with* Gemini (AUC 0.747 / acc 0.686),
> and a respectable fully self-hosted result *without any LLM and without raw media
> leaving the box* (AUC 0.670).

## Feature reference

**Text (`txt_*`)** — two groups:
- *Psycholinguistic* markers (Newman & Pennebaker; Vrij): first-person-singular
  vs. plural pronoun rates, negations, tentative/certainty/cognitive/exclusive/
  motion word rates, negative−positive emotion, type-token ratio, disfluency.
- *Whissle STT metadata* from `/asr/transcribe`: speech rate (WPM, articulation
  rate, filler/pause ratios), pause statistics (count, mean/max duration,
  long-pause fraction), per-word confidence + filler rates, overall ASR
  confidence, uncertain-word rate, entity count, and the **full per-token
  probability distributions** for every metadata category (`metaprob_<cat>_<tok>`
  for emotion / age / gender / behavior / eval / role) plus each category's
  entropy and an expected-age scalar — i.e. the model's soft read, not just the
  top-1 label.

**Visual (`vis_*`)** — aggregated over sampled frames where the speaker's face is
detected: emotion fractions + intensities + entropy, **gaze aversion**, head-pose
mean/spread and frame-to-frame **motion (fidgeting)**, blink rate, attention
(engaged) fraction, mouth-openness, speaking fraction, hand-gesture presence/
diversity, and `face_detect_rate` for coverage.

**Audio (`aud_*`)** — F0 mean/std/range/voiced-fraction + jitter proxy, RMS
loudness + shimmer proxy, silence ratio / pause count / mean pause length /
pause density, ZCR and spectral centroid/bandwidth/rolloff.

---

## Project layout

```
lie_detector/
  config.py                 env-driven paths + gateway settings
  dataset.py                manifest + speaker parsing from the README
  media.py                  ffmpeg audio extract / probe
  io_utils.py               json + cache helpers
  extraction/
    gateway.py              POST /video/analyze (STT + audio-visual)  ← step 02
    audio_prosody.py        librosa prosody                            ← step 03
  features/
    text_features.py        txt_*   (transcript + STT metadata)
    visual_features.py      vis_*   (visual_timeline aggregation)
    audio_features.py       aud_*   (prosody passthrough + derived)
    assemble.py             join → multimodal matrix
  modeling/
    metrics.py              binary metrics
    train.py                LOSO CV, models, ablations, importance
scripts/                    01…05 + run_all.py
tests/                      smoke tests
docs/
  GATEWAY.md                the external Whissle gateway: how to run it + contract
  NEXT_STEPS.md             handoff: status + research ideas (read this first)
```

---

## Notes, limitations, and ethics

- **Small, biased sample.** 121 clips / ~33 speakers from US trials. Results are
  a research signal, not a courtroom tool. Expect LOSO accuracy in the ~60–75%
  range — well above the ~50% base rate, far from "proof".
- **Deception detection is not solved.** No model here infers guilt; it predicts
  a dataset label derived from verdicts/exonerations. Do **not** deploy this to
  judge real people. Treat outputs as probabilistic and contestable.
- **Demographic confound.** A handful of female defendants dominate the deceptive
  class, so age/gender metadata correlate with the label. Some apparent
  "accuracy" is demographics, not deception — audit by dropping `meta_*`/
  `metaprob_age*`/`metaprob_gender*` and re-checking (see docs/NEXT_STEPS.md).
- **Reproducibility.** Fixed seed, deterministic LOSO folds, resumable caches.
- The bundled `Annotation/All_Gestures_*.csv` (human-annotated gestures) is a
  *reference baseline* from the original paper; we extract our own features and
  do not train on those labels.

## Citation

Pérez-Rosas, Abouelenien, Mihalcea, Burzo. *Deception Detection using Real-life
Trial Data.* ICMI 2015.
