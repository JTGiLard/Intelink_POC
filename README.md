# Intelink

**Intelink** is a Streamlit-based analyst workstation for demo and hackathon use: semantic search over a local document corpus, entity-centric retrieval, relationship graphing, and tiered evidence presentation with explicit trust controls.

## Overview

Intelink ingests structured and unstructured operational text (reports, emails, chat logs, etc.), builds a FAISS-backed embedding index, and surfaces results through a dark-themed **mission board** UI: relationship graph, AI assessment, key identifiers, and tabbed deep dives (evidence, identity clusters, entity profile, link analysis, activity timeline).

## Key capabilities

- **Hybrid semantic + keyword retrieval** with tunable keyword boost  
- **Entity-focused gating** for short subject-style queries (reduces unrelated semantic hits)  
- **Evidence tiers:** primary, related context, and hidden/ignored (with configurable score floors)  
- **Relationship graph** with normalized edge trust filtering and direct / indirect / weak styling  
- **Identity clustering** for full-name style queries  
- **Admin / debug diagnostics** (retrieval gate + classification; visible only to `admin` or when `DEBUG_MODE` is enabled in code)  
- **Demo authentication** with role-based sidebar controls  

## Screenshots

_Add screenshots here after your demo run (mission board, evidence tab, relationship graph)._

## Architecture (high level)

```text
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  data/      │────▶│ chunking +       │────▶│ FAISS index     │
│  (txt/doc)  │     │ embeddings       │     │ (.cache_index/) │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
┌────────────────────────────────────────────────────▼────────────────────┐
│  app.py (Streamlit) — login, search, aggregate, graph, tabs, admin UI    │
└──────────────────────────────────────────────────────────────────────────┘
         │ uses
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  intelligence/ — loaders, entities, index_store, link_graph, timeline   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tech stack

| Layer | Choice |
|--------|--------|
| UI | [Streamlit](https://streamlit.io/) ≥ 1.32 |
| Embeddings | OpenAI `text-embedding-3-small` (via `openai` Python SDK) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| NLP | [spaCy](https://spacy.io/) (optional warm-up; pipeline as configured in project) |
| Tables / charts | pandas, Plotly |

## Repository layout

| Path | Purpose |
|------|---------|
| `app.py` | Main Streamlit entry point |
| `intelligence/` | Indexing, chunking, entities, graph, timeline helpers |
| `data/` | Sample corpus (text sources for demo) |
| `.cache_index/` | Generated index cache (gitignored) |
| `assets/` | Placeholder for static assets (logos, images) |
| `pages/` | Reserved for optional multipage Streamlit apps |
| `docs/` | Extra documentation |
| `.streamlit/` | Notes on secrets (do not commit `secrets.toml`) |

## How to run locally

### 1. Python environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Environment variables

Copy `.env.example` to `.env` and set:

```env
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4o-mini
```

`MODEL_NAME` is reserved for future LLM features; embeddings use the embedding model defined in code.

Alternatively, use `.streamlit/secrets.toml` locally (file is gitignored — see `.streamlit/README.md`).

### 3. Launch Streamlit

From the repository root:

```bash
streamlit run app.py
```

The app opens in your browser (default Streamlit port **8501**).

## Streamlit Cloud

1. Connect the GitHub repository.  
2. Set **Secrets** with `OPENAI_API_KEY` (and optionally other keys later).  
3. Main file: **`app.py`**.  
4. Ensure `packages.txt` or build steps include **spaCy English model** if the hosted build does not run `spacy download` automatically (add a `packages.txt` or startup script if needed for your platform).

## Demo login (intended for prototype only)

| Username | Password | Notes |
|----------|----------|--------|
| `intel` | `admin123` | Analyst view (simplified messaging) |
| `admin` | `admin123` | Operator view + sidebar tuning + debug expander |

**Security:** These credentials are for demos only. Replace with a real auth system before any production use.

## Known limitations

- **OpenAI dependency:** Without `OPENAI_API_KEY`, the app stops after login with a configuration message (no stack trace for missing key).  
- **Synthetic / sample data:** Bundled `data/` is illustrative; conclusions are only as good as the corpus.  
- **Trust heuristics:** Edge thresholds and evidence tiers are **tunable constants** in `app.py`; they are not legal or investigative guarantees.  
- **spaCy / optional deps:** Some paths degrade gracefully if optional packages are missing; full quality assumes a working spaCy model.  

## Future enhancements

- Pluggable auth (OAuth / SSO)  
- Persistent case folders and export  
- Optional LLM narrative layer driven by `MODEL_NAME`  
- Automated tests for retrieval gating and graph edge filters  

## License / hackathon

Built for demonstration and hackathon judging. Adapt attribution and license as needed for your event.

---

### Git: suggested commands before push

```bash
git status
git add .
git commit -m "Refine Intelink analyst experience and retrieval trust controls"
git push origin main
```

Review `git status` for unintended files (local `.env`, caches, `venv/`) before committing.
