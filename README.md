# Voice AI Customer Service Assistant

A real-time, enterprise-grade, multilingual (Egyptian Arabic & English) Voice AI customer service assistant designed for retail banking customer support (Banque Misr domain). The system uses FastAPI and WebSockets to enable persistent, bidirectional streaming audio sessions powered by Silero VAD, Groq Whisper (with local Faster-Whisper fallback), a stateful Conversation Layer, two-pass RAG retrieval using BAAI/bge-m3 dense embeddings and Qdrant vector database, Groq Llama 3.3 70B (with local Ollama support), Silma Neural Arabic TTS, Redis, and PostgreSQL.

## Features

- **Voice Activity Detection (Silero VAD):** Real-time 32ms frame-level speech activity detection via ONNX Runtime with a 200ms pre-speech ring buffer and 1000ms silence cutoff.
- **Speech-to-Text (STT):** High-speed cloud STT using Groq Whisper (`whisper-large-v3`) with zero-downtime fallback to local Faster-Whisper (CTranslate2).
- **Stateful Conversation Layer:** Redis-backed multi-turn dialogue management, turn-by-turn script/language detection, hybrid intent routing, active product entity tracking, and coreference resolution.
- **RAG & Quality Gate:** Dense vector search using `BAAI/bge-m3` embeddings and Qdrant HNSW vector database with a dynamic Cosine similarity quality gate ($Score \ge 0.58$) and Pass-2 LLM keyword expansion recovery to eliminate hallucinations.
- **Phonetic Speech Formatting & Sub-Clause TTS:** Normalizes Arabic digits (`150` $\rightarrow$ `مائة وخمسون`), expands currency (`EGP` $\rightarrow$ `جنيه مصري`), splices text into sub-180-character clauses, synthesizes speech via Silma Neural TTS, and merges raw WAV headers for smooth browser playback.
- **Persistent WebSocket Voice Sessions:** Sub-2-second total voice turnaround time (TTFT < 1.9s) over persistent `/ws/audio` binary WebSockets.
- **Browser-Based Developer Test Client:** Single-page Web Audio API interface with dynamic visualizer and live transcript DOM cards.
- **Automated Evaluation Suite:** Benchmarking suite evaluating retrieval precision, recall, faithfulness, and latency, exporting reports to Markdown, JSON, and Excel.
- **PostgreSQL:** Infrastructure database persistence driver.
- **Redis 7:** Asynchronous session state memory with 3600-second TTL.
- **Qdrant v1.15.3:** High-performance vector database hosting domain knowledge base vectors.

## Project Structure

```text
Voice AI Assistance/
├── app/                        # Application core (FastAPI, Conversation Layer, RAG, Speech Formatting, TTS)
├── client/                     # Single-page Web Audio API browser client (HTML, CSS, JS)
├── data/                       # Banque Misr knowledge base seed JSON files (cards, loans, policies)
├── evaluation/                 # Quality evaluation suite and metric export scripts
├── examples/                   # Standalone RAG execution and profiler scripts
├── input/                      # Audio ingestion, frame adapter, Silero VAD, speech buffer, and STT providers
├── llm/                        # Language model abstractions (Groq Llama 3.3 70B & Ollama) and prompts
├── orchestration/              # High-level pipeline orchestrator
├── reports/                    # Generated benchmark evaluation reports (JSON, Markdown, Excel)
├── scripts/                    # Maintenance, diagnostic, and dataset enrichment utilities
├── tests/                      # Comprehensive test suite covering conversation phases and voice pipeline
├── docker-compose.yml          # Container configuration for Redis, Qdrant, and PostgreSQL
├── initialize_knowledge_base.py# CLI seed script for chunking and indexing knowledge vectors into Qdrant
├── requirements.txt            # Python dependency manifest
├── SYSTEM_DOCUMENTATION.md     # In-depth architectural blueprint and sequence diagrams
└── README.md                   # Project overview and quickstart guide
```

## Prerequisites & Requirements

### System Requirements
- **Python:** Version `3.11` or higher
- **Container Runtime:** Docker Desktop / Docker Compose
- **Version Control:** Git
- **Hardware:** Microphone and web browser (Google Chrome or Microsoft Edge recommended for Web Audio API support)

### Required API Credentials & Environment Variables
- **Groq API Key:** Required for cloud STT (`whisper-large-v3`) and LLM (`llama-3.3-70b-versatile`) inference. Get a key at [console.groq.com](https://console.groq.com).
- **Environment File:** Copy `.env.example` to `.env` and insert your credentials.

---

## Installation & How to Run

Follow these step-by-step instructions to set up and start the application:

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Voice-AI-Assistance
```

### Step 2: Set Up Python Virtual Environment
```bash
python -m venv .venv
```
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```
Open `.env` in your editor and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### Step 5: Start Docker Infrastructure Services
Start Redis 7, Qdrant v1.15.3, and PostgreSQL 15 containers in the background:
```bash
docker compose up -d
```
*Verify containers are running:*
```bash
docker ps
```

### Step 6: Initialize Knowledge Base Vectors
Seed the Qdrant vector database with Banque Misr knowledge base document chunks:
```bash
python initialize_knowledge_base.py
```

### Step 7: Launch the Application Server
Run the FastAPI application with Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Accessing the Browser Client

Once Uvicorn is running:
1. Open your web browser (Chrome / Edge) and navigate to:
   ```text
   http://localhost:8000
   ```
2. Click the **Microphone** button to allow audio capture.
3. Speak your query in Egyptian Arabic or English (e.g., *"عايز اعرف مصاريف بطاقة فيزا جولد"* or *"What are the fees for Platinum card?"*).
4. The system will stream speech frames over WebSocket `/ws/audio`, transcribe text in real-time, retrieve vector context from Qdrant, generate grounded responses via Groq LLM, and stream synthesized audio back for playback.

---

## Current Pipeline

```text
Microphone (Web Audio API)
      │
      ▼
WebSocket (/ws/audio)
      │
      ▼
Silero VAD (ONNX) & SpeechBuffer
      │
      ▼
Groq Whisper STT (Fallback: Faster-Whisper)
      │
      ▼
Conversation Layer & Intent Router (Redis Session Memory)
      │
      ▼
RAG Engine (bge-m3 Embeddings + Qdrant Vector DB + Quality Gate)
      │
      ▼
Groq LLM Llama 3.3 70B (Fallback: Ollama)
      │
      ▼
Speech Response Formatter & Clause Chunker (<=180 chars)
      │
      ▼
Silma Neural TTS & Audio WAV Header Merger
      │
      ▼
Browser Audio Playback & JSON Response
```

## Notes

- **Cloud Acceleration & Fallback:** Cloud-accelerated Groq Whisper and Llama 3.3 70B provide sub-2-second turnaround times, with automatic fallback support for local Faster-Whisper and Ollama models.
- **Managed Infrastructure:** PostgreSQL, Redis, and Qdrant services are fully managed with Docker Compose.
- **Complete Enterprise Voice Stack:** VAD endpointing, STT, dialogue state management, coreference resolution, vector search, quality gating, digit/currency speech normalization, and neural TTS synthesis are fully implemented and production-ready.
- **Automated Benchmarking:** Execute `python -m evaluation.run_quality_suite` to run automated RAG accuracy and latency tests.
