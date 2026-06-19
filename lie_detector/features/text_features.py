"""Linguistic / verbal features from the gateway's fused record (text lane).

Grounded in the psycholinguistics of deception (Newman & Pennebaker; Vrij):
deceivers tend to use *fewer first-person-singular* pronouns, *more negative
emotion* words, *fewer exclusive* words (but, except), *more motion* verbs, and
show different hedging / certainty / disfluency patterns. We compute compact,
self-contained lexicon counts (no external LIWC dependency), all normalised per
word so clip length doesn't dominate.

Also folds in Whissle STT *metadata* carried on each segment: the model's
audio-derived **emotion / intent / age / gender** tags, entity counts, word
confidence, diarization speaker changes, and speech rate from word timestamps.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean

# --- compact deception-relevant lexicons -----------------------------------
_LEX = {
    "i_singular": {"i", "me", "my", "mine", "myself"},
    "we_plural": {"we", "us", "our", "ours", "ourselves"},
    "you": {"you", "your", "yours", "yourself"},
    "third_person": {"he", "she", "they", "them", "his", "her", "their", "him"},
    "negation": {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "cant", "dont", "didnt", "wasnt", "wont", "isnt", "arent"},
    "certainty": {"always", "never", "absolutely", "definitely", "certainly", "sure", "certain", "clearly", "obviously", "totally", "completely", "guarantee"},
    "tentative": {"maybe", "perhaps", "probably", "possibly", "guess", "suppose", "think", "seem", "seemed", "sort", "kind", "somewhat", "apparently", "presumably", "likely"},
    "cognitive": {"because", "cause", "know", "knew", "think", "thought", "believe", "consider", "realize", "remember", "understand", "wonder"},
    "exclusive": {"but", "except", "without", "exclude", "excluding", "however", "although", "though", "unless", "besides"},
    "motion": {"go", "going", "went", "come", "came", "walk", "walked", "run", "ran", "move", "moved", "leave", "left", "drive", "drove", "arrive", "turn", "turned"},
    "neg_emotion": {"hate", "angry", "afraid", "scared", "fear", "sad", "cry", "hurt", "kill", "killed", "dead", "death", "blood", "fight", "hit", "pain", "terrible", "horrible", "awful", "worried", "upset", "mad"},
    "pos_emotion": {"love", "happy", "glad", "good", "great", "nice", "kind", "hope", "thank", "thanks", "care", "friend", "safe", "calm", "fine"},
    "disfluency": {"um", "uh", "er", "erm", "hmm", "uhh", "umm", "ah"},
    "filler_phrase_words": {"like", "well", "actually", "basically", "literally"},
}
# Audio emotions Whissle's metadata model emits (others fall into "other").
_AUDIO_EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]

_WORD_RE = re.compile(r"[a-z']+")
_SENT_RE = re.compile(r"[.!?]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def _transcript(rec: dict) -> str:
    return (rec.get("text") or rec.get("transcript") or "").strip()


def text_features(av_record: dict) -> dict[str, float]:
    text = _transcript(av_record)
    toks = _tokens(text)
    n = len(toks)
    feats: dict[str, float] = {}

    # --- surface / lexical ---
    feats["word_count"] = float(n)
    uniq = len(set(toks))
    feats["unique_words"] = float(uniq)
    feats["type_token_ratio"] = (uniq / n) if n else 0.0
    feats["avg_word_len"] = mean(len(t) for t in toks) if toks else 0.0
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    feats["sentence_count"] = float(len(sentences))
    feats["words_per_sentence"] = (n / len(sentences)) if sentences else float(n)
    feats["ellipsis_count"] = float(text.count("..."))

    # --- normalised lexicon rates (per word) ---
    counts = {name: sum(1 for t in toks if t in lex) for name, lex in _LEX.items()}
    for name, c in counts.items():
        feats[f"rate_{name}"] = (c / n) if n else 0.0
        feats[f"cnt_{name}"] = float(c)

    # composite deception-marker indices
    feats["pronoun_i_minus_we"] = feats["rate_i_singular"] - feats["rate_we_plural"]
    feats["neg_minus_pos_emotion"] = feats["rate_neg_emotion"] - feats["rate_pos_emotion"]
    feats["repetition_ratio"] = 1.0 - feats["type_token_ratio"]

    # ----------------------------------------------------------------------
    # Whissle STT structure + metadata (per-segment emotion/intent/age/gender)
    # ----------------------------------------------------------------------
    segments = av_record.get("segments", []) or []
    feats["n_segments"] = float(len(segments))

    # words + timing for speech rate / confidence
    all_words = [w for s in segments for w in (s.get("words") or [])]
    if not all_words and av_record.get("words"):
        all_words = av_record["words"]
    confs = [w.get("confidence") for w in all_words if isinstance(w, dict) and w.get("confidence") is not None]
    feats["word_conf_mean"] = float(mean(confs)) if confs else 0.0

    times = [(w.get("start"), w.get("end")) for w in all_words
             if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None]
    speech_span = (max(e for _, e in times) - min(s for s, _ in times)) if times else None
    dur = av_record.get("duration") or (float(av_record.get("video_params", {}).get("duration", "0").rstrip("s")) if av_record.get("video_params") else None)
    feats["words_per_second"] = (n / speech_span) if speech_span and speech_span > 0 else ((n / dur) if dur else 0.0)

    # audio-emotion distribution across segments (fraction of segments per label)
    seg_emotions = [(_seg_meta(s).get("emotion") or "").lower() for s in segments]
    seg_emotions = [e for e in seg_emotions if e]
    ecount = Counter(seg_emotions)
    nseg_e = len(seg_emotions)
    for e in _AUDIO_EMOTIONS:
        feats[f"audio_emo_frac_{e}"] = (ecount.get(e, 0) / nseg_e) if nseg_e else 0.0
    feats["audio_emo_other_frac"] = (
        sum(c for k, c in ecount.items() if k not in _AUDIO_EMOTIONS) / nseg_e) if nseg_e else 0.0
    feats["audio_emo_diversity"] = float(len(ecount))

    # intent / entities / speakers
    intents = {(_seg_meta(s).get("intent") or "") for s in segments} - {""}
    feats["n_distinct_intents"] = float(len(intents))
    feats["n_entities"] = float(sum(len(s.get("entities") or []) for s in segments))
    speakers = {s.get("speaker") for s in segments if s.get("speaker")}
    feats["n_speakers"] = float(len(speakers))

    return feats


def _seg_meta(seg: dict) -> dict:
    m = seg.get("metadata")
    return m if isinstance(m, dict) else {}
