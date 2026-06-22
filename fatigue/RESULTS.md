# Fatigue detection on UTA-RLDD — first results

Same privacy-preserving, on-device pipeline as the deception work (Whissle gateway:
`whissle-large` STT metadata + `/video/analyze` visual timeline + prosody), evaluated
**leave-one-subject-out** (the honest protocol). UTA-RLDD: webcam drowsiness videos,
binary **alert vs drowsy**, one 120 s window per video.

## Headline (23 subjects, n = 46, balanced)

| lane | feats | acc | AUC |
|---|--:|--:|--:|
| **visual (PERCLOS / blink / yawn / head-droop)** | 13 | 0.587 | **0.665** |
| STT metadata (whissle-large) | 158 | 0.522 | 0.531 |
| prosody (librosa) | 21 | 0.435 | 0.465 |
| all combined | 192 | 0.609 | 0.616 |
| *majority / chance baseline* | — | 0.500 | 0.500 |

Visual AUC 0.665 is **significant** (label-permutation test, 300 shuffles:
null mean 0.443, 95th pct 0.622, **p = 0.023**).

## The story: a self-correcting artifact

At **15 subjects** the STT-metadata lane scored **0.781** and looked like the winner.
We flagged it as suspicious — the metadata heads were firing on non-speech ambient
audio (the same person's age flipped 30–45 → 45–60 between recordings), so the
"signal" was likely a per-recording confound, not fatigue.

Adding subjects settled it: at **23 subjects the metadata collapsed to 0.531 (chance).**
A real cross-subject signal would have held; an artifact washes out as the sample
grows — which is exactly what happened. This is the same lesson as the deception
project's speaker-leakage finding: only honest, adequately-powered LOSO tells you
what is real.

## What's real, what isn't
- **Visual is the valid (modest) fatigue signal** — drowsiness-specific features
  (eye-closure/PERCLOS, blink rate, yawning, head droop) generalise across people at
  AUC 0.665, p < 0.05. Consistent with the drowsiness literature.
- **Speech / STT-metadata does not generalise here** — UTA-RLDD subjects barely talk,
  so those lanes run on noise and do not survive LOSO.

## Caveats / next steps
- 23 subjects (folds 1–2). Folds 3–5 (60 total) downloading would tighten the estimate.
- One 120 s window per video; multiple windows or full-video aggregation should help.
- The gateway's per-frame blink/PERCLOS is approximate; eye-aspect-ratio from raw
  MediaPipe landmarks would likely lift the 0.665.
- For a true multimodal speech+fatigue story, pair with a talking-while-fatigued
  corpus (e.g. the INTERSPEECH SLEEP corpus) where the audio lane has real signal.
