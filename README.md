# Nexa | AI Support Studio

**Project:** Multi-Agent AI Helpdesk · **Student:** Amr Semary (24P0066)

A full visual redesign and backend refresh of a four-specialist course-platform helpdesk. Streamlit provides the chat UI; a PyTorch bidirectional **LSTM** predicts the intent; the **billing / technical / booking / general** specialist agents run grounded tools; a local **RAG** knowledge base supplies cited policies. Optional **LangChain Groq / Ollama** adapters enhance the replies. **The entire demo works offline and without API keys** with verifiable rule-backed answers.

> **Academic/AI disclosure:** The original code and this refactor were produced with AI assistance. Review the implementation and metrics before submitting it in an assessed context. The sample invoices, schedule, and policies are **fictional demo data**, not real account information or commitments.

## Fast start (Windows / macOS / Linux)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit (normally http://localhost:8501). The model checkpoint is **already bundled**, so training or API setup is **not required** for the first launch.

### Optional cloud-generated replies

```bash
pip install -r requirements-llm.txt
# copy .env.example to .env, fill in GROQ_API_KEY
streamlit run app.py
```

For a local language model, install Ollama, run `ollama pull llama3.2:3b` and set `USE_OLLAMA=true` in `.env` (requires `requirements-llm.txt`). Both providers revert automatically to the offline grounded response if they are unreachable. No secret should be committed to git.

For experimental semantic FAISS retrieval: `pip install -r requirements-semantic.txt` and set `RAG_BACKEND=semantic`. Default answers use the fast, dependency-free local knowledge retriever. The main multi-agent assistant always cites the local verified source sections.

## Architecture

```text
Streamlit chat + prompt shortcuts + agent trace + export
            |
            v
PyTorch LSTM classifier (fallback: transparent keyword rules)
            |
        Intent router
            |
    +-------+----------+----------+---------+
    | Billing          | Booking  | General |
    |       Technical  |          |         |
    +-------+----------+----------+---------+
            |
      Verified demo tools
      - check_invoice (mock database)
      - check_availability (illustrative slots only)
      - search_docs (local Markdown section retrieval)
            |
  Offline answer OR optional LangChain Groq/Ollama reply
            |
      Response + sources + trace
```

**Why offline mode?** No chat should crash because an API key is missing, the embedding model wasn't downloaded, or an LLM rate limit was reached. Agents remain separate specialist workflows and are not being presented as systems with real billing or booking integrations. The optional LLM is given read-only tool evidence; the agent **does not** take real-world actions.

## Retraining and evaluation

```bash
python -m src.data_gen       # regenerate UNIQUE texts with template-grouped splits
python -m src.train          # LSTM: metrics + checkpoint, early stopping
python -m src.evaluate       # test set: macro-F1, per-class metrics and confusion matrix
python smoke_test.py        # offline pipeline
python -m pytest -q         # regression tests (pip install pytest)
```

The training data is synthetic. Unlike the original dataset, the current generation uses an **approximately 70/15/15 holdout by source template** (14/3/3 of each intent's 20 templates), then removes duplicate messages. Dataset counts are not perfectly balanced after de-duplication. This is a more difficult but more credible measure of generalization than splitting near-identical template variants randomly. Evaluation numbers and mistakes are saved under `reports/`; do not describe them as production accuracy.

For GRU experiments: `python -m src.train --rnn_type gru --out models/classifier_gru.pt`. Use `--epochs`, `--patience`, `--batch_size`, and `--seed` for reproducible experiments. The application uses `models/classifier.pt` by default. Even if that file is absent, an explicit rules-based fallback keeps routing available, and the sidebar shows which backend is active.

## File map

| File | Purpose |
| --- | --- |
| `app.py` | Completely refreshed responsive chat workspace, 4 quick actions, traces, export/reset |
| `src/model.py`, `dataset.py`, `train.py`, `evaluate.py` | LSTM/GRU training + validation + held-out test |
| `src/classify.py` | Model inference + safe rule fallback |
| `src/agents.py` | Four specialist workflows with auditable tool calls |
| `src/mock_backends.py` | Consistent demo invoices/calendar: no real actions |
| `src/knowledge.py` | Fast local document-section search with explicit source metadata |
| `src/rag.py` | Retriever adapter; optional FAISS semantic mode |
| `src/llm.py` | Optional LangChain provider adapters; graceful offline degradation |
| `rag_docs/*.md` | Fictional demo platform policies and support knowledge |
| `tests/`, `smoke_test.py` | Offline regression and integration checks |
| `reports/evaluation_report.md` | **Measured** test results, not placeholder metrics |

## Improvements shipped

- Entire UI redesigned and renamed **Nexa**; former name and graduation-cap icon removed.
- Sample-prompt launch cards, chat history, export, clear-chat, visible agent/route/tool telemetry.
- Working offline-first routing, retrieval and support replies; no eager import of optional cloud packages.
- Trained LSTM checkpoint included; deterministic dataset generator and automatic evaluation artifacts.
- No repeated messages across splits; harder template-level holdout, explicit dataset limitations.
- Deterministic mock backends, no fabricated account or live reservation claims.
- Cloud request failures fall back to useful grounded answers instead of exposing raw API errors.
- Configurations separated into minimal core and optional LLM/semantic install profiles.
