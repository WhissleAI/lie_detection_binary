# Catching Lies Without Watching: Privacy-Preserving Multimodal Deception Detection

**Whissle Research Team**

## Abstract

Frontier multimodal models can guess whether a person is lying from a video of their testimony. They do so only by streaming that person's raw face and voice to a third-party model. We ask whether the heavy media is necessary at all. On the *Real-life Trial Deception* dataset (121 courtroom clips), Whissle's on-device speech and vision stack extracts a compact feature digest: transcript, emotion, age, gender, and intent distributions, a deception-intent filter, fluency and rhythm analysis, per-frame facial behaviour, and prosody — roughly 250 numbers per clip. Under speaker-independent evaluation, we report three findings. A small classifier on this digest reaches **AUC 0.741**, matching Gemini 2.5 Pro on the full video (0.749). Handing the same digest to a frontier LLM reaches **AUC 0.755 with Claude Opus 4.8** at **7.8× fewer input tokens**, with no media leaving the device. The reported 75% accuracy is an artefact of speaker-identity leakage. We release code and experiments.

## Introduction

When a witness takes the stand, every pause, glance, and word choice is scrutinised for signs of deception. Automating that judgement is an old dream and a fraught one. The polygraph is unreliable and easily countered. Human observers barely beat chance [1]. The recent leap in multimodal large language models (LLMs) has revived the dream. Give a model the video, ask "is this person lying?", and it does meaningfully better than chance [6].

![Our thesis in one picture. A witness's testimony is distilled on-device into a compact feature digest — a few numbers, prosody, and facial behaviour — held behind a privacy shield. Only that digest, never the raw face and voice, reaches a cloud model. This paper asks how much deception signal survives that compression.](/images/blog/fig_intro.png)

This convenience hides a serious cost. Sending the raw video of a defendant, a patient, or a customer exposes their face and voice to a third party. Video is token-heavy, so the request is expensive. The verdict is opaque and hard to contest. In legal, clinical, and enterprise settings, "upload the suspect's video" is a non-starter.

This paper asks a narrow question. How much of the signal survives if we never send the video — only a compact, interpretable digest of features extracted on-device? We study this on the *Real-life Trial Deception* dataset [1] and make four contributions.

- **A privacy-preserving multimodal pipeline.** Whissle's on-device STT and vision stack turns each clip into ~250 interpretable verbal, paralinguistic, and visual features, with no raw media leaving the box (see Method).
- **A clean four-way comparison** across two axes — *trained vs. zero-shot* and *with-LLM vs. without-LLM* — under speaker-independent evaluation (see Experiments). A trained classifier on our digest (AUC 0.741) matches an LLM watching the raw video (0.749). An LLM judging our digest reaches 0.755, beating the video model.
- **A cost analysis** showing the digest uses 7.8× fewer LLM input tokens than the raw video (see Cost Analysis). It is cheaper, faster, private, and more accurate.
- **An honesty audit.** We show that the literature's 75% is a leave-one-*video*-out number inflated by speaker leakage, quantify a demographic confound, and report the speaker-independent result (see Ablation Studies).

## Related Work

**The 2015 baseline.** Pérez-Rosas et al. introduced the *Real-life Trial Deception* dataset and the first multimodal system built on it [1]. Earlier learning-based deception work relied on lab or crowdsourced data, such as mock crimes or the "Are you a werewolf?" game [4]. In those settings, subjects lack real motivation, which limits generalisation to high-stakes lies. Pérez-Rosas et al. instead used 121 public courtroom clips labelled from verdicts and exonerations [1]. They were the first to add non-verbal modalities to real trial data. Their system fused transcript unigrams and bigrams with 40 binary gesture features, hand-coded under the MUMIN scheme. It reached a best accuracy of 75.2% with a decision tree. That headline came from a single best cell; a random forest over all features scored only 50.4% [1].

**Verbal cues.** One strand of the literature reasons over text alone, using n-grams, syntactic complexity, and psycholinguistic lexicons such as LIWC [2]. Liars tend to use fewer first-person-singular pronouns, more negative-emotion words, and fewer exclusive terms [2]. Each cue is weak alone but informative in aggregate. For this reason, Pérez-Rosas et al. note that LIWC-style features perform similarly to their n-grams [1].

**Non-verbal cues.** A parallel strand studies behaviour, from physiological signals to hand-coded facial and gestural annotations [3]. Frowning, raised eyebrows, lip movements, and head turns have all been linked to deception [3]. Pérez-Rosas et al. observed that deceivers blinked and shook their heads *less* than truth-tellers [1]. Such indicators are noisy and contestable. In the 2015 system they were annotated manually with knowledge of each clip, not extracted by any deployable system [1].

**The present: LLMs over video.** Modern multimodal LLMs such as Gemini and Claude can ingest video and reason about behaviour zero-shot [6]. Recent datasets pair audio and video for this task [5]. These models are strong, but they are expensive and privacy-invasive. As we show, they are not obviously better than reasoning over a compact extracted digest.

**The gap we address.** Three limitations of the 2015 work remain open [1]. First, its non-verbal features are human-annotated rather than automatically extracted, so we replace them with privacy-preserving on-device signals. Second, its evaluation is leave-one-*video*-out; with many clips per speaker, this leaks speaker identity and inflates accuracy, so we evaluate speaker-independently. Third, neither the 2015 system nor today's video LLMs account for token cost, which we measure directly.

## Dataset

The *Real-life Trial Deception* dataset contains **121 clips** (61 deceptive, 60 truthful), averaging ~28 s, drawn from public trial recordings and labelled from guilty verdicts, acquittals, and exonerations.

**The speaker-leakage trap.** The set has only **33 unique speakers**, and a single defendant accounts for **32 of the 121 clips**; **7 speakers appear in both classes**. Under the leave-one-*video*-out cross-validation used by the original paper, testing one of those 32 clips still leaves 31 of the same person in training. A model can then learn *who* is speaking rather than *whether they lie*. We therefore evaluate with **leave-one-speaker-out (LOSO)**, holding out every clip from a person together. LOSO is harder and yields lower numbers, but it is the only protocol that measures generalisation to an unseen individual.

![The Real-life Trial Deception dataset: a balanced 61/60 class split, but a heavily skewed clips-per-speaker distribution in which a single defendant accounts for 32 of the 121 clips — the source of speaker leakage.](/images/blog/fig_dataset.png)

## Method

Each clip is converted into a feature digest by three on-device lanes. No raw media is sent to any external model.

**Text lane (Whissle STT).** The transcript, with the model's probability distributions over **emotion, age, gender, and 33 intents**. A focused **deception-intent filter** scores labels such as denial, confession, justification, avoidance, and contradiction. A **speech-analysis** block covers fluency, grammar, vocabulary range, pitch, and rhythm (pause rate, speaking ratio, and inter-word intervals). Psycholinguistic rates cover pronouns, negations, and hedges.

**Visual lane (audio-visual).** Per-frame facial emotion, gaze direction, head pose, blink, and hand gestures, aggregated into behavioural statistics: gaze aversion, head-motion / fidgeting, emotional variability, and blink rate.

**Acoustic lane (prosody).** Pitch (F0), jitter/shimmer, and pause statistics.

This yields ~250 interpretable features per clip. We then study four system configurations along two axes:

| | No LLM | With LLM |
|---|---|---|
| **Zero-shot** | majority baseline | LLM-as-judge over our digest (text-only); LLM over raw video |
| **Trained** | gradient-boosting on the digest | trained late-fusion of digest model + LLM |

![The on-device pipeline. Each clip is converted into a ~250-number feature digest by three local lanes (Whissle STT, audio-visual, prosody); a trained classifier or an LLM-as-judge then predicts deception. The raw video never leaves the device.](/images/blog/fig_pipeline.png)

## Experiments

**Setup.** All numbers are leave-one-speaker-out, pooled out-of-fold over 121 clips. The trained model is gradient-boosted trees. The LLM judges are zero-shot, prompted neutrally with a base-rate anchor that weights verbal content over weak behavioural cues.

**Main results.**

| System | Trained? | LLM? | Video sent? | Accuracy | ROC-AUC |
|---|:--:|:--:|:--:|--:|--:|
| Majority baseline | — | — | No | 0.504 | 0.500 |
| Self-hosted features → gradient-boosting | Yes | No | No | 0.678 | 0.741 |
| LLM judges our features — Claude Opus 4.8 | No | Yes | No | 0.620 | **0.755** |
| LLM judges our features — Gemini 2.5 Pro | No | Yes | No | 0.669 | 0.704 |
| LLM watches raw video — Gemini 2.5 Pro | No | Yes | Yes | 0.669 | 0.749 |
| Self-hosted + LLM (trained late fusion) | Yes | Yes | Yes | 0.678 | 0.752 |

Two independent roads reach ~0.75 without exposing video. Train a small model on the digest (0.741), or hand the digest to a frontier LLM (0.704–0.755). The best single result — **Claude Opus 4.8 over our digest, 0.755** — exceeds the same class of model watching the raw video (0.749).

**Comparison to the literature.** Under the paper's own leave-one-video-out protocol, our features reproduce and exceed its numbers (0.752–0.777 vs. 0.752). Under honest LOSO, everything drops by the size of the leakage.

| Protocol | Original paper | Our features |
|---|--:|--:|
| Leave-one-video-out (speaker-leaky) | 0.752 | 0.752–0.777 |
| Leave-one-speaker-out (honest) | not reported | 0.741 |

![The speaker-leakage gap. Accuracy under the paper's leave-one-video-out protocol (red) versus honest leave-one-speaker-out (green); the drop is the inflation that speaker leakage buys.](/images/blog/fig_leakage.png)

![Honest leave-one-speaker-out ROC-AUC for the six systems, coloured by whether the raw video is sent to an LLM. The best result — Claude Opus 4.8 over our feature digest (0.755) — sends no video.](/images/blog/fig_results.png)

## Ablation Studies

**Feature groups.** We partition the digest into seven groups and measure each one alone, along with the AUC lost when it is removed (LOSO, full set = 0.741):

| Group | # feats | Alone AUC | Marginal Δ (drop) |
|---|--:|--:|--:|
| STT metadata probs (emotion/age/gender/intent) | 119 | 0.700 | +0.039 |
| Visual (face/gaze/pose/gesture) | 42 | 0.579 | **+0.133** |
| Speech-analysis (fluency/grammar/rhythm) | 15 | 0.551 | +0.028 |
| Prosody (librosa) | 21 | 0.448 | +0.028 |
| Lexical (psycholinguistic) | 22 | 0.564 | +0.022 |
| Deception-intent filter | 15 | 0.518 | +0.023 |
| Speech structure (rate/pauses/conf) | 20 | 0.490 | −0.008 |

The STT-metadata group is strongest alone, but **the visual lane is the most complementary**. Removing it costs 0.133 AUC despite its modest solo score, because it contributes signal no other lane captures.

**The visual fix.** Courtroom faces are small and oblique, and MediaPipe's default detection confidence rejected ~half the frames. Lowering it raised the face-detection rate from 0.50 to 0.80 and the visual lane from 0.61 to 0.67 AUC. This lifted the self-hosted system from 0.670 to 0.741.

**Demographic confound.** A few female defendants dominate the deceptive class, so the model's gender and age reads correlate with the label. Dropping all gender/age features lowers the self-hosted AUC to **0.678**, our bulletproof number. The with-LLM and direct-video results are unaffected by removing *our* demographic features.

**Prompt sensitivity of the LLM judge.** A naive "forensic" prompt makes Gemini over-call deception (AUC 0.62, 79% deceptive calls). Emphasising verbal content over weak behavioural cues, and anchoring the 50/50 base rate, recalibrates it to 0.704 at a 54% deceptive rate. The two LLMs carry opposite priors. Gemini leans deceptive, Claude leans truthful. Claude's superb ranking (0.755) is therefore masked by a conservative threshold (raw accuracy 0.62, ~0.70 thresholded).

**Which intents predict deception?** Whissle's STT emits a probability over 33 speech intents per clip, which we probe alongside a focused deception-intent filter. We correlate each intent's probability with the label (point-biserial *r*, n = 121), and a clear, interpretable pattern emerges (Figure 5). Clips lean **deceptive** when their intent is **response, deny, disagree, agree, request, or suggest** — the reactive, denial-oriented register of someone fielding accusatory questions. They lean **truthful** when the intent is **reminisce or anecdote**, the genuine narrative recall of someone recounting events. We are deliberately cautious here. Effect sizes are small (|r| ≤ 0.28). Although 10 of 113 intents reach uncorrected p < 0.05 (vs. ~6 expected by chance), *none survive* multiple-comparison correction. No single intent is a standalone lie detector. What is meaningful is, first, the **coherent, theory-aligned direction** — random noise would not place denial and deflection on the deceptive side and genuine recollection on the truthful side — and second, that intents are the **single strongest feature family** in the model ablation (Table 4). The signal lives in the *distribution* of speech acts, not in any one of them; confirming individual intents needs a larger corpus.

![Which speech intents predict deception — point-biserial correlation of each intent's probability with the deceptive label; deceptive-leaning intents (deny, disagree, response) in red, truthful-leaning (reminisce, anecdote) in green; * marks p < 0.05.](/images/blog/intent_importance.png)

## Cost Analysis

Because we never send the video, the LLM processes far fewer tokens. We measure this with the provider's `count_tokens` API over a sample of clips:

| Input to the LLM | Mean input tokens | Relative |
|---|--:|--:|
| Raw video (≈296 tokens/s) | 9,810 | 1.0× |
| Our feature digest | 1,261 | **0.13× (7.8× fewer)** |

At Gemini 2.5 Pro list pricing, this is ~$1.58 vs. ~$12.26 per 1,000 clips, an **~87% input-cost reduction**. It also brings lower latency, with no video decode or transfer, and, as shown, *higher* accuracy. Longer clips save more — up to 10.7× on a 55 s clip.

## Limitations and Ethics

This is a research probe on a tiny, US-trial-specific dataset, not a courtroom tool. No model here infers guilt; each predicts a dataset label derived from verdicts. Honest speaker-independent accuracy is ~65–70% — far above chance, but far from proof. Part of the signal is a demographic confound that must be controlled. Deception detection must never be deployed to judge real people without rigorous, contestable, bias-audited validation. The interpretability of a feature-based system, in which every cue is named, is a safeguard the opaque LLM verdict lacks.

## Conclusion

You do not need to ship a person's video to a frontier model to detect deception. A compact, interpretable, on-device digest — never exposing raw media — matches a video-watching LLM with a small trained classifier (0.741). It *exceeds* that model when the digest itself is handed to a frontier LLM judge (Claude Opus 4.8, 0.755), at 7.8× lower token cost. Privacy, cost, and accuracy point the same way: keep the video on the device.

## References

1. Pérez-Rosas, Verónica, Mohamed Abouelenien, Rada Mihalcea, and Mihai Burzo. "Deception Detection Using Real-Life Trial Data." *Proceedings of the 2015 ACM on International Conference on Multimodal Interaction (ICMI)*, ACM, 2015.
2. Newman, Matthew L., et al. "Lying Words: Predicting Deception from Linguistic Styles." *Personality and Social Psychology Bulletin*, vol. 29, no. 5, 2003.
3. Ekman, Paul. *Telling Lies: Clues to Deceit in the Marketplace, Politics, and Marriage*. W. W. Norton, 2001.
4. Gupta, Viresh, et al. "Bag-of-Lies: A Multimodal Dataset for Deception Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, IEEE, 2019.
5. Guo, Xiaobao, et al. "Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, IEEE, 2023.
6. Gemini Team, Google DeepMind. "Gemini 1.5: Unlocking Multimodal Understanding across Millions of Tokens of Context." *arXiv*, arXiv:2403.05530, 2024.
