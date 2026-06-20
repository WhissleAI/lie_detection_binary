"""Linguistic + Whissle-STT-metadata features (text lane).

Two groups of features come from the gateway's ``/asr/transcribe`` record:

1. **Lexical / psycholinguistic** — from the transcript. Grounded in deception
   research (Newman & Pennebaker; Vrij): first-person-singular vs. plural pronoun
   rates, negations, tentative/certainty/cognitive/exclusive/motion words,
   negative−positive emotion, type-token ratio, disfluency.

2. **Whissle STT metadata** — the model's own read of the audio:
   - ``speech_rate``: WPM, articulation rate, spoken/pause ratios, filler rate
   - ``pauses``: count, mean/max duration, long-pause fraction (hesitation)
   - ``words``: per-word confidence + filler flags → low-confidence / filler rates
   - overall ``confidence``, ``uncertain_words`` (hesitation proxy)
   - clip-level ``metadata`` (emotion / age / gender / behavior) + soft
     ``metadata_probs`` distributions (emotion uncertainty via top-prob/entropy).
"""

from __future__ import annotations

import math
import re
from statistics import mean, pstdev

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
_AGE_ORDER = {"AGE_0_18": 0.0, "AGE_18_30": 1.0, "AGE_30_45": 2.0, "AGE_45_60": 3.0, "AGE_60PLUS": 4.0}

_WORD_RE = re.compile(r"[a-z']+")
_SENT_RE = re.compile(r"[.!?]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def _transcript(rec: dict) -> str:
    return (rec.get("text") or rec.get("transcript") or "").strip()


def _strip_prefix(token: str | None) -> str:
    """'EMOTION_HAPPY' -> 'happy'; None/'' -> ''."""
    if not token:
        return ""
    return token.split("_", 1)[1].lower() if "_" in token else token.lower()


def _dist_map(dist) -> dict[str, float]:
    """[{token, probability}, ...] -> {token: probability}."""
    out: dict[str, float] = {}
    if isinstance(dist, list):
        for d in dist:
            if isinstance(d, dict) and d.get("token"):
                out[d["token"]] = float(d.get("probability", 0.0))
    return out


def _entropy(probs) -> float:
    return float(-sum(p * math.log(p + 1e-12) for p in probs if p > 0))


def collect_metadata_vocab(records) -> dict[str, list[str]]:
    """Union of metadata_probs tokens per category, across all records.

    Data-driven (not hardcoded) so the feature set tracks whatever metadata
    vocabulary the gateway's STT model emits. Sorted for stable column order.
    """
    vocab: dict[str, set] = {}
    for rec in records:
        mp = (rec or {}).get("metadata_probs") or {}
        for cat, dist in mp.items():
            if isinstance(dist, list):
                toks = vocab.setdefault(cat, set())
                for d in dist:
                    if isinstance(d, dict) and d.get("token"):
                        toks.add(d["token"])
    return {cat: sorted(toks) for cat, toks in vocab.items()}


def text_features(rec: dict, meta_vocab: dict[str, list[str]] | None = None) -> dict[str, float]:
    text = _transcript(rec)
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

    counts = {name: sum(1 for t in toks if t in lex) for name, lex in _LEX.items()}
    for name, c in counts.items():
        feats[f"rate_{name}"] = (c / n) if n else 0.0
    feats["pronoun_i_minus_we"] = feats["rate_i_singular"] - feats["rate_we_plural"]
    feats["neg_minus_pos_emotion"] = feats["rate_neg_emotion"] - feats["rate_pos_emotion"]
    feats["repetition_ratio"] = 1.0 - feats["type_token_ratio"]

    # ----------------------------------------------------------------------
    # Whissle STT metadata
    # ----------------------------------------------------------------------
    # speech_rate dict
    sr = rec.get("speech_rate") or {}
    dur = float(sr.get("duration_sec") or 0.0)
    feats["sr_wpm"] = float(sr.get("words_per_minute") or 0.0)
    feats["sr_articulation_wpm"] = float(sr.get("articulation_rate_wpm") or 0.0)
    feats["sr_filler_rate"] = float(sr.get("filler_rate") or 0.0)
    feats["sr_filler_count"] = float(sr.get("filler_count") or 0.0)
    feats["sr_spoken_ratio"] = (float(sr.get("spoken_sec") or 0.0) / dur) if dur else 0.0
    feats["sr_pause_ratio"] = (float(sr.get("total_pause_sec") or 0.0) / dur) if dur else 0.0
    feats["sr_pause_count"] = float(sr.get("pause_count") or 0.0)

    # pauses list
    pauses = rec.get("pauses") or []
    durs = [float(p.get("duration", 0.0)) for p in pauses if isinstance(p, dict)]
    feats["pause_mean_dur"] = float(mean(durs)) if durs else 0.0
    feats["pause_max_dur"] = float(max(durs)) if durs else 0.0
    feats["pause_long_frac"] = (sum(1 for d in durs if d >= 0.5) / len(durs)) if durs else 0.0
    feats["pause_rate_per_sec"] = (len(durs) / dur) if dur else 0.0

    # words: confidence + filler
    words = rec.get("words") or []
    wconf = [float(w.get("confidence")) for w in words if isinstance(w, dict) and w.get("confidence") is not None]
    feats["word_conf_mean"] = float(mean(wconf)) if wconf else 0.0
    feats["word_conf_min"] = float(min(wconf)) if wconf else 0.0
    feats["word_conf_std"] = float(pstdev(wconf)) if len(wconf) > 1 else 0.0
    feats["low_conf_word_rate"] = (sum(1 for c in wconf if c < 0.5) / len(wconf)) if wconf else 0.0
    nfiller = sum(1 for w in words if isinstance(w, dict) and w.get("filler"))
    feats["filler_word_rate"] = (nfiller / len(words)) if words else 0.0

    feats["asr_confidence"] = float(rec.get("confidence") or 0.0)
    unc = rec.get("uncertain_words") or []
    feats["uncertain_word_count"] = float(len(unc))
    feats["uncertain_word_rate"] = (len(unc) / n) if n else 0.0
    feats["n_entities"] = float(len(rec.get("entities") or []))

    # ----------------------------------------------------------------------
    # Metadata probability distributions (full per-token probs, not just top-1)
    # ----------------------------------------------------------------------
    meta = rec.get("metadata") or {}
    feats["meta_has_intent"] = 1.0 if meta.get("intent") else 0.0

    probs = rec.get("metadata_probs") or {}
    vocab = meta_vocab or collect_metadata_vocab([rec])  # fall back to this record's own tokens
    for cat, tokens in vocab.items():
        dist = _dist_map(probs.get(cat))
        for tok in tokens:
            feats[f"metaprob_{cat}_{_strip_prefix(tok)}"] = dist.get(tok, 0.0)
        feats[f"metaprob_{cat}_entropy"] = _entropy(list(dist.values()))
    # expected age (continuous) from the age distribution, when present
    age_dist = _dist_map(probs.get("age"))
    if age_dist:
        num = sum(_AGE_ORDER.get(t.upper(), 0.0) * p for t, p in age_dist.items() if t.upper() in _AGE_ORDER)
        den = sum(p for t, p in age_dist.items() if t.upper() in _AGE_ORDER)
        feats["meta_age_expected"] = (num / den) if den else 0.0

    return feats
