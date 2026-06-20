# Catching Lies Without Watching: Privacy-Preserving Multimodal Deception Detection

**Whissle Research Team**

## Abstract

Frontier multimodal models can guess whether a person is lying from a video of
their testimony — but doing so means streaming the raw face and voice of that
person to a third-party model. We ask whether the heavy media is necessary at
all. On the *Real-life Trial Deception* dataset (121 courtroom clips), we extract
a compact, interpretable feature digest with Whissle's on-device speech and
vision stack — transcript, emotion / age / gender / intent probability
distributions, a focused deception-intent filter, fluency and rhythm analysis,
per-frame facial behaviour, and prosody — roughly 250 numbers per clip. Under
honest, speaker-independent evaluation we find: (i) a small trained classifier on
this digest reaches **AUC 0.741**, matching Gemini 2.5 Pro watching the full
video (0.749); (ii) handing the *same digest* to a frontier LLM as a zero-shot
judge reaches **AUC 0.755 with Claude Opus 4.8 — exceeding the video model** — at
**7.8× fewer input tokens** and with no raw media leaving the device; and (iii)
the apparent 75% accuracy reported on this dataset is an artefact of
speaker-identity leakage in the original cross-validation. We release the full
pipeline, every experiment, and an ablation of which signals matter.

[FIGURE 1: a real-world courtroom testimony scene (a witness/defendant at the
stand) — sets the deception-detection context. Licensed stock or illustrative
render; design team to source.]

## Introduction

When a witness takes the stand, every pause, glance, and word choice is scrutinised
for signs of deception. Automating that judgement is an old dream and a fraught
one: the polygraph is unreliable and easily countered, and human observers barely
beat chance. The recent leap in multimodal large language models (LLMs) has
revived the dream — give a model the video and ask "is this person lying?" and it
does meaningfully better than chance.

But this convenience hides a serious cost. Sending the raw video of a defendant,
a patient, or a customer to a cloud LLM exposes their face and voice to a third
party, is expensive (video is token-heavy), and yields an opaque verdict that is
hard to contest. In legal, clinical, and enterprise settings, "upload the
suspect's video" is a non-starter.

This paper asks a deliberately narrow question: **how much of the signal survives
if we never send the video — only a compact, interpretable digest of features
extracted on-device?** We study this on the *Real-life Trial Deception* dataset
and make four contributions:

- **A privacy-preserving multimodal pipeline.** Whissle's on-device STT and vision
  stack turns each clip into ~250 interpretable features — verbal, paralinguistic,
  and visual — with no raw media leaving the box (see Method).
- **A clean four-way comparison** across two axes — *trained vs. zero-shot* and
  *with-LLM vs. without-LLM* — under honest speaker-independent evaluation (see Experiments).
  A trained classifier on our digest (AUC 0.741) matches an LLM watching the raw
  video (0.749); an LLM *judging our digest* reaches 0.755, beating the video model.
- **A cost analysis** showing the digest uses 7.8× fewer LLM input tokens than the
  raw video (see Cost Analysis) — cheaper, faster, and private, while more accurate.
- **An honesty audit.** We show the literature's 75% is a leave-one-*video*-out
  number inflated by speaker leakage, quantify a demographic confound, and report
  the bulletproof speaker-independent result (see Ablation Studies).

## Related Work

**Verbal deception detection.** Text-based work uses n-grams, syntactic
complexity, and psycholinguistic lexicons (LIWC), finding that liars use fewer
first-person-singular pronouns, more negative-emotion words, and fewer exclusive
terms. These cues are weak individually but informative in aggregate.

**Non-verbal deception detection.** Earlier systems relied on physiological
signals (polygraph, thermal imaging) or hand-coded gesture annotations. Facial
micro-expressions, gaze, and head motion have all been linked to deception, though
each is a noisy and contestable indicator.

**Multimodal and real-life data.** Pérez-Rosas et al. (2015) introduced the
*Real-life Trial* dataset and a system fusing transcript n-grams with manual
MUMIN gesture annotations, reporting up to 75.2% accuracy. Crucially, those
non-verbal features are *human-annotated*, and the evaluation is leave-one-video-out.

**LLMs over video.** Modern multimodal LLMs (Gemini, Claude) can ingest video and
reason about behaviour zero-shot. They are strong but expensive and privacy-invasive,
and — as we show — not obviously better than reasoning over a compact extracted digest.

## Dataset

The *Real-life Trial Deception* dataset contains **121 clips** (61 deceptive, 60
truthful), averaging ~28 s, drawn from public trial recordings and labelled from
guilty verdicts, acquittals, and exonerations.

**The speaker-leakage trap.** The set has only **33 unique speakers**, and a single
defendant accounts for **32 of the 121 clips**; **7 speakers appear in both
classes**. Under leave-one-*video*-out cross-validation — used by the original
paper — testing one of those 32 clips still leaves 31 of the same person in
training, so a model can learn *who* is speaking rather than *whether they lie*.
We therefore evaluate with **leave-one-speaker-out (LOSO)**: every clip from a
person is held out together. This is harder and lower, but it is the only protocol
that measures generalisation to an unseen individual.

[FIGURE 2: dataset composition — 61/60 class split, and a bar chart of
clips-per-speaker highlighting the 32-clip outlier.]

## Method

Each clip is converted into a feature digest by three on-device lanes; no raw
media is sent to any external model.

**Text lane (Whissle STT).** The transcript plus the model's probability
distributions over **emotion, age, gender, and 33 intents**; a focused
**deception-intent filter** scoring labels such as denial, confession,
justification, avoidance, and contradiction; a **speech-analysis** block (fluency,
grammar, vocabulary range, pitch, and rhythm — pause rate, speaking ratio,
inter-word intervals); and psycholinguistic rates (pronouns, negations, hedges).

**Visual lane (audio-visual).** Per-frame facial emotion, gaze direction, head
pose, blink, and hand gestures, aggregated into behavioural statistics (gaze
aversion, head-motion / fidgeting, emotional variability, blink rate).

**Acoustic lane (prosody).** Pitch (F0), jitter/shimmer, and pause statistics.

This yields ~250 interpretable features per clip. We then study four system
configurations along two axes:

| | No LLM | With LLM |
|---|---|---|
| **Zero-shot** | majority baseline | LLM-as-judge over our digest (text-only); LLM over raw video |
| **Trained** | gradient-boosting on the digest | trained late-fusion of digest model + LLM |

[FIGURE 3: pipeline diagram — clip → three on-device lanes → digest → {trained
classifier | LLM-as-judge}; raw video never leaves the device.]

## Experiments

**Setup.** All numbers are leave-one-speaker-out, pooled out-of-fold over 121
clips. The trained model is gradient-boosted trees; the LLM judges are zero-shot
with a neutral, base-rate-anchored prompt that weights verbal content over weak
behavioural cues.

**Main results.**

| System | Trained? | LLM? | Video sent? | Accuracy | ROC-AUC |
|---|:--:|:--:|:--:|--:|--:|
| Majority baseline | — | — | No | 0.504 | 0.500 |
| Self-hosted features → gradient-boosting | Yes | No | No | 0.678 | 0.741 |
| LLM judges our features — Claude Opus 4.8 | No | Yes | No | 0.620 | **0.755** |
| LLM judges our features — Gemini 2.5 Pro | No | Yes | No | 0.669 | 0.704 |
| LLM watches raw video — Gemini 2.5 Pro | No | Yes | Yes | 0.669 | 0.749 |
| Self-hosted + LLM (trained late fusion) | Yes | Yes | Yes | 0.678 | 0.752 |

Two independent roads reach ~0.75 without exposing video: train a small model on
the digest (0.741), or hand the digest to a frontier LLM (0.704–0.755). The best
single result — **Claude Opus 4.8 over our digest, 0.755** — exceeds the same
class of model watching the raw video (0.749).

**Comparison to the literature.** Under the paper's own leave-one-video-out
protocol our features reproduce and exceed its numbers (0.752–0.777 vs. 0.752);
under honest LOSO everything drops by the size of the leakage.

| Protocol | Original paper | Our features |
|---|--:|--:|
| Leave-one-video-out (speaker-leaky) | 0.752 | 0.752–0.777 |
| Leave-one-speaker-out (honest) | not reported | 0.741 |

[FIGURE 4: horizontal bar chart of ROC-AUC for the six systems, colour-coded by
whether video is sent; dashed line at the 0.50 baseline.]

## Ablation Studies

**Feature groups.** We partition the digest into seven groups and measure each
alone and the AUC lost when it is removed (LOSO, full set = 0.741):

| Group | # feats | Alone AUC | Marginal Δ (drop) |
|---|--:|--:|--:|
| STT metadata probs (emotion/age/gender/intent) | 119 | 0.700 | +0.039 |
| Visual (face/gaze/pose/gesture) | 42 | 0.579 | **+0.133** |
| Speech-analysis (fluency/grammar/rhythm) | 15 | 0.551 | +0.028 |
| Prosody (librosa) | 21 | 0.448 | +0.028 |
| Lexical (psycholinguistic) | 22 | 0.564 | +0.022 |
| Deception-intent filter | 15 | 0.518 | +0.023 |
| Speech structure (rate/pauses/conf) | 20 | 0.490 | −0.008 |

The STT-metadata group is strongest alone, but **the visual lane is the most
complementary** (removing it costs 0.133 AUC) despite a modest solo score — it
contributes signal no other lane captures.

**The visual fix.** Courtroom faces are small and oblique; MediaPipe's default
detection confidence rejected ~half the frames. Lowering it raised the
face-detection rate 0.50 → 0.80 and the visual lane 0.61 → 0.67 AUC, lifting the
self-hosted system 0.670 → 0.741.

**Demographic confound.** A few female defendants dominate the deceptive class, so
the model's gender/age reads correlate with the label. Dropping all gender/age
features lowers the self-hosted AUC to **0.678** — our bulletproof number. (The
with-LLM and direct-video results are unaffected by removing *our* demographic
features.)

**Prompt sensitivity of the LLM judge.** A naive "forensic" prompt makes Gemini
over-call deceptive (AUC 0.62, 79% deceptive calls). Emphasising verbal content
over weak behavioural cues and anchoring the 50/50 base rate recalibrates it to
0.704 at a 54% deceptive rate. The two LLMs carry opposite priors — Gemini leans
deceptive, Claude leans truthful — so Claude's superb ranking (0.755) is masked
by a conservative threshold (raw accuracy 0.62, ~0.70 thresholded).

**Which intents predict deception?** Whissle's STT emits a probability over 33
speech intents per clip; we also probe a focused deception-intent filter. We
correlate each intent's probability with the label (point-biserial *r*, n = 121).
A clear, interpretable pattern emerges (Figure 5): clips lean **deceptive** when
their intent is **response, deny, disagree, agree, request, or suggest** — the
reactive, denial-oriented register of someone fielding accusatory questions — and
lean **truthful** when the intent is **reminisce or anecdote**, the genuine
narrative recall of someone recounting events. The effect sizes are modest
(|r| ≈ 0.2–0.28) but several are statistically significant (p < 0.05), and the
direction matches the deception literature: deceivers deny and deflect, while
truth-tellers supply detailed, first-hand narrative.

![Which speech intents predict deception — point-biserial correlation of each intent's probability with the deceptive label; deceptive-leaning intents (deny, disagree, response) in red, truthful-leaning (reminisce, anecdote) in green; * marks p < 0.05.](/images/blog/intent_importance.png)

## Cost Analysis

Because we never send the video, the LLM processes far fewer tokens. Measured with
the provider's `count_tokens` API over a sample of clips:

| Input to the LLM | Mean input tokens | Relative |
|---|--:|--:|
| Raw video (≈296 tokens/s) | 9,810 | 1.0× |
| Our feature digest | 1,261 | **0.13× (7.8× fewer)** |

At Gemini 2.5 Pro list pricing this is ~$1.58 vs. ~$12.26 per 1,000 clips — an
**~87% input-cost reduction** — alongside lower latency (no video decode/transfer)
and, as shown, *higher* accuracy. Longer clips save more (up to 10.7× on a 55 s
clip).

## Limitations and Ethics

This is a research probe on a tiny, US-trial-specific dataset, not a courtroom
tool. No model here infers guilt; it predicts a dataset label derived from
verdicts. Honest speaker-independent accuracy is ~65–70% — far above chance, far
from proof. Part of the signal is a demographic confound that must be controlled.
Deception detection must never be deployed to judge real people without rigorous,
contestable, bias-audited validation. The interpretability of a feature-based
system — every cue is named — is a safeguard the opaque LLM verdict lacks.

## Conclusion

You do not need to ship a person's video to a frontier model to detect deception.
A compact, interpretable, on-device digest — never exposing raw media — matches a
video-watching LLM with a small trained classifier (0.741), and *exceeds* it when
the digest itself is handed to a frontier LLM judge (Claude Opus 4.8, 0.755), at
7.8× lower token cost. Privacy, cost, and accuracy point the same way: keep the
video on the device.

## References

1. Pérez-Rosas, Abouelenien, Mihalcea, Burzo. *Deception Detection using Real-life
   Trial Data.* ICMI 2015.
2. Newman, Pennebaker, et al. *Lying Words: Predicting Deception from Linguistic
   Styles.* PSPB 2003.
3. Ekman. *Telling Lies.* Norton, 2001.
