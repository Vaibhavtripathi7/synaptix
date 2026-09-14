# legacy/

The original Feb–Mar 2025 prototype, kept for reference only; nothing here is imported by the new code.

- `scripts/synap.py` – Flask "PDF Summarizer" (PyMuPDF → whole-text Gemini prompt → pyvis graph). `dup.py` is a near-identical copy.
- `scripts/{text_image_table_inference,Competitor_profile,SWOT_analysis,competitive_strategy,key_analysis,competitor_comparison}.py`
  – Vertex AI Studio exports of the competitor-analysis prompt chain, to be ported to `app/analysis/`.
- `scripts/parsing_main.py`, `scripts/main_gem.py` – the two standalone stages later merged into `synap.py`.

The Google API key that was hard-coded in these files has been redacted and must be rotated.
