# Reference: Pérez-Rosas, Abouelenien, Mihalcea, Burzo — "Deception Detection using Real-life Trial Data" (ICMI 2015)

This is the paper that introduced the Real-life Trial Deception dataset and the
first multimodal system on it. Use it to ground Related Work and the citation.

## Motivation
- High-stakes courtroom testimony makes automatic deception detection valuable but
  fraught. The polygraph is impractical and biased; humans barely beat chance.
- Most prior learning-based deception work used data collected in lab settings or
  via crowdsourcing (e.g. mock crimes, "Are you a werewolf?" game), where subjects
  lack real motivation — limiting generalization to real high-stakes deception.
- They argue prior real-trial work used text only; theirs is the first to add
  non-verbal modalities on real trial data.

## Dataset
- 121 video clips (61 deceptive, 60 truthful) from public YouTube/court trial
  recordings, labelled from guilty verdicts, not-guilty verdicts, and exonerations.
- Average length 28.0 s. 21 unique female + 35 unique male speakers (defendants/
  witnesses), ages ~16–60. Exoneree testimonies from the Innocence Project.
- Transcribed via Amazon Mechanical Turk, including fillers (um, ah, uh) and
  ellipsis for silence; 8,055 words total, ~66 words per transcript.

## Features
- Verbal: unigram and bigram bag-of-words frequencies (words with freq > 10). They
  note LIWC-style psycholinguistic features perform similarly to n-grams and do not
  use them in the main system.
- Non-verbal: 40 binary gesture features from MANUAL human annotation using the
  MUMIN coding scheme (facial displays — eyebrows, eyes, mouth, head — and hand
  gestures/trajectories). Annotated at video level "in accordance with the overall
  judgment of truthfulness/deceitfulness." Two annotators; average inter-annotator
  agreement 75.16%, Kappa 0.57.

## Method and results
- Classifiers: Decision Tree (DT) and Random Forest (RF) from Weka, default params.
- Evaluation: leave-one-out cross-validation (over the 121 videos). Baseline 50.4%.
- Key accuracies (Table 3): Unigrams DT 60.33%; Facial displays RF 76.03%; Hand
  gestures DT 61.98%; All non-verbal RF 73.55%; All features DT 75.20% (best);
  All features RF 50.41%. A feature ablation (Table 4) shows facial displays and
  unigrams contribute most.
- Most predictive gestures: frowning, eyebrows raising, lip movements, head turns.
  Observation: deceivers blinked and shook their head LESS than truth-tellers.

## Human study
- Three annotators judged deception from four modalities (text, audio, silent
  video, full video), 484 instances. Agreement is low (Kappa 0.07–0.20); humans
  are poor deception judges. The automatic system (75.2%) outperforms humans
  (best full-video human ~59.5%), a relative improvement up to 51%.

## What is dated / limited (for the present paper to address)
- Non-verbal features are HUMAN-annotated (not automatically extracted), and coded
  with knowledge of the clip — not a deployable automatic system.
- Evaluation is leave-one-VIDEO-out: with multiple clips per speaker (e.g. many
  Jodi Arias clips), this leaks speaker identity and inflates accuracy.
- No acoustic/prosodic features in the system; audio appears only in the human study.
- The 75.2% headline is a single best Decision-Tree cell; RF on all features is 50.4%.
