<h1 align="center">Aria</h1>
<p align="center">A grounded AI parts agent for HVAC </p>

<p align="center">
  <img src="https://img.shields.io/badge/Agent-tool--calling-orange" alt="Tool-calling agent">
  <img src="https://img.shields.io/badge/GPT--4o-vision-412991?logo=openai&logoColor=white" alt="GPT-4o Vision">
  <img src="https://img.shields.io/badge/RAG-grounded-4CAF50" alt="RAG">
  <img src="https://img.shields.io/badge/ChromaDB-vector%20store-6E56CF" alt="ChromaDB">
  <img src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/TTS-voice-9C27B0" alt="Voice">
</p>

---

Aria helps HVAC field technicians find the right part, confirm it actually fits, and get the guide for what to do next — by symptom, by part number, or by a photo of the part itself.

## What it actually does

- **Finds parts by symptom or description** — semantic search over a parts catalog, not just exact-match lookup
- **Verifies compatibility deterministically** — a straight data lookup against a part/model compatibility table, with zero LLM guessing involved in the yes/no
- **Surfaces guides by intent** — installation, troubleshooting, or policy, filtered before the semantic search even runs
- **Reads a part number straight from a photo** — image input goes to GPT-4o as part of the same chat turn, no separate OCR step
- **Talks back** — replies can be played as speech, not just read
- **Always ends with next-step suggestions** — every response closes with a short set of follow-up options instead of a dead end

## The grounding rule this is built around

The system prompt is explicit: every part number, compatibility claim, and price has to come from a tool result — never invented. Compatibility specifically requires a `check_compatibility` tool call; the agent isn't allowed to reason its way to a compatibility answer on its own. That's the same trust-first design as the other two agents in this portfolio, applied here to a technician diagnosing equipment in the field instead of a customer shopping online.

## Design decisions and trade-offs

- **Deterministic compatibility, not an LLM guess** — a technician acting on a wrong compatibility call wastes a truck roll. The check is a plain data lookup on purpose.
- **Voice replies via server-side TTS**, not the browser's built-in speech API — a deliberate 1–2 second delay traded for a voice that doesn't sound robotic, for a technician who may be listening hands-free mid-repair.
- **Single-file vanilla frontend, no build step** — same reasoning as the other agents: a drop-in widget shouldn't require a framework toolchain to run.

## Known limitations

- **No seed data yet.** `products.json`, `guides.json`, and `compatibility.json` currently ship empty. The architecture is fully wired — RAG indexing, tool-calling, grounding — but there's nothing to search or verify against until real (or realistic sample) HVAC parts data is added.
- **API base URL is hardcoded** to `http://localhost:8000` in the frontend — not yet configurable for a non-local deployment.

## My role

Built the full system solo: the grounded agent loop and its four tools, the RAG indexing pipeline, the vision and voice integration, the FastAPI backend, and the frontend widget.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS, no framework |
| Backend | Python FastAPI |
| Vector store | ChromaDB `EphemeralClient` |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o — chat, vision, and tool-calling |
| Voice | OpenAI TTS, `nova` voice |
| Agent tools | `search_parts`, `search_guides`, `check_compatibility`, `get_part_by_number` |

## Setup

**Backend**
```bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload       # http://localhost:8000
```

**Frontend**

Open `frontend/index.html` directly in a browser, or serve it:
```bash
cd frontend
python3 -m http.server 8080     # http://localhost:8080
```

## Roadmap

- Seed `products.json`, `guides.json`, and `compatibility.json` with real or realistic sample HVAC data
- Make the frontend's API base URL configurable instead of hardcoded to localhost

> Demo and screenshots coming soon.
