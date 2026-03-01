---
title: ResearchMind Backend
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ResearchMind Backend

Multi-Agent AI Research Assistant API built with FastAPI, LangGraph, and FAISS.

Live API: [https://baibhavbaidya-researchmind-backend.hf.space](https://baibhavbaidya-researchmind-backend.hf.space)  
Frontend: [https://researchmind-bb.vercel.app](https://researchmind-bb.vercel.app)

---

## Overview

ResearchMind is a production-grade multi-agent research pipeline where five specialized AI agents collaborate in real-time to search, summarize, critique, fact-check, and synthesize answers from web sources and uploaded PDF documents.

Each query passes through a LangGraph-orchestrated pipeline:

```
User Query → Searcher → Summarizer → Critic → Fact Checker → Synthesizer → Answer
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Agent Orchestration | LangGraph |
| LLM | Groq (LLaMA 3.3 70B) |
| Vector Store | FAISS |
| Hybrid Search | FAISS + BM25 (rank-bm25) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Web Search | Tavily API |
| Authentication | Firebase Admin SDK |
| Database | SQLite via SQLAlchemy |
| Real-time Streaming | WebSockets |
| Rate Limiting | SlowAPI |
| PDF Parsing | PyMuPDF |
| Deployment | Docker on HuggingFace Spaces |

---

## Agents

### 1. Searcher Agent
Decides where to search based on context. If documents are uploaded, searches the hybrid vector store (FAISS semantic + BM25 keyword). Always augments with web search via Tavily. Combines and ranks results by score.

### 2. Summarizer Agent
Takes raw search results and generates concise, structured summaries for each source using the LLM. Preserves source attribution.

### 3. Critic Agent
Reviews each summary for quality, relevance, and reliability. Filters out low-quality or irrelevant summaries before fact-checking.

### 4. Fact Checker Agent
Cross-checks claims across multiple sources. Identifies verified claims (agreed upon by multiple sources) and disputed claims (contradicted across sources).

### 5. Synthesizer Agent
Generates the final comprehensive answer by synthesizing verified information, citing sources inline, and structuring the response in clean markdown.

---

## Features

- **Real-time streaming** — Agent progress streamed over WebSocket so users see live updates
- **Hybrid RAG** — Combines FAISS semantic search + BM25 keyword search for better document retrieval
- **Per-user isolation** — Each user gets their own vector store, uploaded documents, and chat history
- **Follow-up chat** — Fast conversational follow-ups without re-running the full pipeline
- **Firebase authentication** — Supports Google OAuth and Email/Password sign-in
- **Rate limiting** — Per-endpoint rate limits to prevent abuse
- **Content filtering** — Input validation and sanitization on all user inputs
- **File security** — PDF-only uploads, 20MB size limit, filename sanitization

---

## Project Structure

```
backend/
├── main.py                  # FastAPI app, startup, CORS, user retriever cache
├── api/
│   ├── routes.py            # REST API endpoints
│   └── websocket.py         # WebSocket endpoint for real-time streaming
├── agents/
│   ├── searcher.py          # Searcher Agent
│   ├── summarizer.py        # Summarizer Agent
│   ├── critic.py            # Critic Agent
│   ├── factchecker.py       # Fact Checker Agent
│   └── synthesizer.py       # Synthesizer Agent
├── core/
│   ├── graph.py             # LangGraph pipeline definition
│   ├── vectorstore.py       # FAISS vector store per user
│   ├── retriever.py         # Hybrid BM25 + FAISS retriever
│   ├── embeddings.py        # Sentence Transformer embeddings
│   ├── websearch.py         # Tavily web search integration
│   ├── firebase_auth.py     # Firebase token verification
│   ├── database.py          # SQLAlchemy models and DB init
│   └── security.py          # Rate limiting, input validation
└── utils/
    └── pdf_parser.py        # PDF text extraction and chunking
```

---

## API Reference

### Authentication
All endpoints (except `/health`) require a Firebase ID token in the Authorization header:
```
Authorization: Bearer <firebase_id_token>
```

---

### Health Check
```
GET /health
```
Returns server status and active user count. No authentication required.

**Response:**
```json
{
  "status": "healthy",
  "active_users": 3
}
```

---

### Upload Document
```
POST /upload
```
Upload a PDF document to the user's personal vector store.

**Body:** `multipart/form-data`
- `file` — PDF file (max 20MB)

**Response:**
```json
{
  "message": "Document uploaded successfully",
  "filename": "research-paper.pdf",
  "chunks_created": 42,
  "page_count": 8,
  "document_id": null
}
```

---

### Delete Single Document
```
DELETE /documents/{filename}
```
Remove a specific document and rebuild the vector store from remaining files.

---

### Clear All Documents
```
DELETE /documents
```
Clear all uploaded documents and reset the user's vector store.

---

### Research Query (REST)
```
POST /query
```
Run the full 5-agent research pipeline synchronously.

**Body:**
```json
{
  "query": "What are the benefits of intermittent fasting?",
  "use_documents": true
}
```

**Response:**
```json
{
  "query": "...",
  "answer": "...",
  "sources": [{"index": 1, "source": "...", "url": "..."}],
  "agent_logs": ["Searcher: found 7 results", "..."],
  "status": "complete"
}
```

---

### Research Query (WebSocket)
```
WS /ws/query?token=<firebase_id_token>
```
Run the pipeline with real-time agent progress streaming.

**Send:**
```json
{
  "query": "Your research question",
  "use_documents": true
}
```

**Receives (multiple messages):**
```json
{"agent": "Searcher", "status": "running", "message": "Searching web..."}
{"agent": "Synthesizer", "status": "complete", "data": {"answer": "...", "sources": [...]}}
```

---

### Follow-up Question
```
POST /followup
```
Ask a follow-up question based on a previous research result. Uses a single LLM call instead of the full pipeline — fast and conversational.

**Body:**
```json
{
  "original_query": "What is quantum computing?",
  "original_answer": "...",
  "followup_question": "Can you explain qubits in simpler terms?"
}
```

**Response:**
```json
{
  "answer": "..."
}
```

---

### Get Chat History
```
GET /history?limit=20
```
Retrieve the user's past research queries and answers (max 50).

---

### Delete Chat History
```
DELETE /history
```
Clear all chat history for the current user.

---

### Delete Account
```
DELETE /account
```
Permanently delete the user's account data including chat history, uploaded documents, and vector store.

---

## Local Development

### Prerequisites
- Python 3.11+
- A `service-account.json` from Firebase Console
- API keys for Groq and Tavily

### Setup

```bash
git clone https://github.com/baibhavbaidya/researchmind-backend
cd researchmind-backend

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
ALLOWED_ORIGINS=http://localhost:5173
```

Place your `service-account.json` from Firebase Console in the root directory.

Run:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000`

---

## Docker

```bash
docker build -t researchmind-backend .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  -e FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}' \
  researchmind-backend
```

---

## Deployment

Deployed on **HuggingFace Spaces** using Docker.

Environment variables set in Space settings:
- `GROQ_API_KEY`
- `TAVILY_API_KEY`
- `ALLOWED_ORIGINS`
- `FIREBASE_SERVICE_ACCOUNT_JSON`

---

## Rate Limits

| Endpoint | Limit |
|---|---|
| `/upload` | 10/minute |
| `/query` | 20/minute |
| `/followup` | 30/minute |
| `/history` GET | 30/minute |
| `/history` DELETE | 5/minute |
| `/account` DELETE | 3/minute |

---

