# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Streamlit web app that analyzes chromosome karyogram images and produces ISCN 2020 notations (e.g. `46,XY`, `47,XX,+21`). It combines OpenCV-based chromosome detection with Vision-Language Model (VLM) interpretation. For educational/research use only — not for clinical diagnosis.

## Commands

```bash
pip install -r requirements.txt           # install deps
streamlit run app.py                      # run locally on http://localhost:8501
```

Deployment is via Railway (`railway.toml`) / Heroku-style (`Procfile`); both invoke `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`. There is no test suite, lint config, or build step — the project is a single `app.py` script.

`test_images/` holds reference karyograms (Down syndrome, Klinefelter, Triple X, Turner, etc.) used for manual verification of provider/CV behavior.

## Architecture

The entire app lives in **`app.py`** (~2400 lines). Three layers:

### 1. `ChromosomeDetector` — OpenCV pipeline (lines ~56–586)
- `detect_chromosomes()` runs multiple segmentation strategies (Otsu, adaptive threshold + morphology) and picks the best result. Area thresholds are **scaled by image area** relative to a 1MP baseline — do not hardcode pixel areas.
- `detect_karyogram_positions()` assumes an **arranged karyogram** (rows by Denver group). It uses fixed image-fraction grid regions (row 4 = bottom 25%, Group G = right 30%) to count chromosomes at position 21, 22, and the sex region. This grid is approximate and will mislabel non-standard layouts.
- `_estimate_denver_groups()` and `_detect_sex_chromosome_region()` are heuristic and feed into the two-stage / CV+VLM pipelines.

### 2. `KaryotypeAnalyzer` — Provider dispatcher (lines ~790–1690)
`analyze()` dispatches on an `APIProvider` enum. Several provider methods exist in code (`OPENAI`, `ANTHROPIC`, `GEMINI`, `CONSENSUS`, `TWO_STAGE`, `CV_VLM`, `MOCK`) but **the sidebar UI only exposes OpenAI, CV+VLM, and Demo Mode** — the others are reachable only by editing `display_sidebar_settings()`. Models used: OpenAI `gpt-4o`, Anthropic `claude-sonnet-4-20250514`, Gemini via `google-genai`.

**CV+VLM (`_analyze_with_cv_vlm`) is the most important path** and has non-obvious fallback logic:
1. CV detects total count and position counts.
2. If total is outside `[44, 50]` → fall back to `_analyze_with_cv_hints()` (visual VLM with CV position hints), not pure VLM-only.
3. Otherwise VLM interprets CV counts (no visual counting) via `CV_VLM_INTERPRETATION_PROMPT`.
4. If returned `confidence < 50` → re-run with `_analyze_with_cv_hints()` for visual verification.
5. `_analyze_with_cv_hints` contains a **Triple X misclassification guard**: when CV reports `pos21 >= 3` but `sex_region_count <= 1`, the prompt warns the VLM to distinguish three small acrocentric chromosomes (true +21) from three medium submetacentrics (47,XXX). This is load-bearing — `test_images/triple_x_47XXX*` exists specifically to validate it (see commit `22b7ff0`).

`_parse_response()` tries **four extraction strategies in order**: raw JSON → fenced code block → brace-balanced substring → trailing-comma cleanup. Any new prompt should still produce a JSON object; provider responses are inconsistent about wrapping it in markdown.

### 3. Streamlit UI (lines ~1692–end)
- All state goes through `st.session_state` (`analysis_result`, `uploaded_image`, `raw_response`, `saved_api_key`, `cv_detection`).
- `inject_local_storage_script()` writes/reads the OpenAI key from browser `localStorage` via injected `<script>` tags. The save path uses `json.dumps()` to escape the key — this is the XSS fix from commit `1904652`; **do not interpolate user input directly into JS strings.** Key format is validated (`sk-` prefix, ≥20 chars, alphanumeric/`-`/`_`) before storage.
- Result rendering branches on `result.get('is_consensus')` and `result.get('pipeline') == 'two_stage'` — three separate display functions (`display_results`, `display_consensus_results`, `display_two_stage_results`).

## Prompts

Two module-level prompt templates (lines ~633–788):
- `KARYOTYPE_ANALYSIS_PROMPT` — direct visual analysis (used by OpenAI/Anthropic/Gemini providers). Chain-of-thought walking the model through "count at each labeled position → distinguish +21 vs XXY by position 21 count".
- `CV_VLM_INTERPRETATION_PROMPT` — pure interpretation of CV counts; explicitly tells the VLM **not** to count visually. Uses Python `.format()` with `{cv_results}`, `{total_count}`, `{pos21_count}` placeholders — keep other `{` `}` doubled.

When changing prompts, preserve the JSON schema in the output section: downstream `_finalize_result()` only sets defaults for `notation`, `chromosome_count`, `sex_chromosomes`, `abnormalities`, `confidence`, `interpretation`, `detailed_findings` — missing fields silently become defaults.

## Disclaimer

This is medical-adjacent software. Per `user-manual.txt` §10, results require validation by a qualified cytogeneticist. Don't remove the disclaimer banner or weaken the "educational/research only" framing.
