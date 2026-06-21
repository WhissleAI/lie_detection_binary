#!/usr/bin/env python3
"""Step 18 — generate the paper's motivating intro figure with Gemini 3 Pro image.

Produces paper/figures/fig_intro.png: a clean editorial illustration of the
paper's thesis — a courtroom testimony distilled, on-device, into a compact
feature digest, so the raw video never has to be streamed to a cloud LLM.

    python scripts/18_gen_intro_figure.py
"""

from __future__ import annotations

from lie_detector.config import CFG

MODEL = "gemini-3-pro-image"

PROMPT = (
    "A clean, professional editorial illustration for an AI research paper "
    "(flat vector style, muted palette with dark teal-green #124E3F and a single "
    "warm red accent, generous white space, wide 16:9 landscape, NO text or "
    "lettering anywhere).\n"
    "Concept: privacy-preserving deception detection. Left: a stylised courtroom "
    "witness stand with a faceless, abstract figure giving testimony (illustrative, "
    "not photorealistic, no recognisable face). Middle: their testimony flows into "
    "a small on-device chip that distils it into abstract tokens — a few numbers, "
    "a tiny waveform, simple face/gaze icons — kept inside a shield outline to "
    "signal privacy. Right: a large cloud / LLM server, with the raw video feed to "
    "it faded out and crossed, while only the small distilled tokens reach it. "
    "Calm, editorial, conceptual; suitable as Figure 1 of a serious paper."
)


def main() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=CFG.gemini_api_key)
    resp = client.models.generate_content(
        model=MODEL, contents=[PROMPT],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    out = CFG.project_root / "paper" / "figures" / "fig_intro.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = False
    for part in resp.candidates[0].content.parts:
        data = getattr(part, "inline_data", None)
        if data and data.data:
            out.write_bytes(data.data)
            saved = True
            print(f"✓ wrote {out} ({len(data.data)} bytes) via {MODEL}")
            break
    if not saved:
        raise SystemExit("No image returned. Try MODEL='gemini-3.1-flash-image' or 'imagen-4.0-generate-001'.")


if __name__ == "__main__":
    main()
