# Voice AI Customer Service Assistant

A local Voice AI customer service assistant designed for multilingual customer support. The system uses FastAPI and WebSockets to enable persistent, bidirectional streaming audio sessions powered by local Faster-Whisper, Silero VAD, Ollama, PostgreSQL, Redis, and Qdrant.

---

## Features

- Voice Activity Detection (Silero VAD)
- Speech-to-Text using Faster-Whisper
- Local LLM using Ollama
- Persistent WebSocket voice sessions
- Browser-based developer test client
- Structured LLM routing
- PostgreSQL
- Redis
- Qdrant

---

## Project Structure

```text
app/
client/
input/
llm/
orchestration/
docker-compose.yml
requirements.txt
README.md
```

---

## Prerequisites

- Python 3.11+
- Docker Desktop
- Git

---

## Installation

```bash
git clone <repository-url>
```

```bash
cd <repository-name>
```

```bash
python -m venv .venv
```

Windows
```bash
.\.venv\Scripts\activate
```

Linux/macOS
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
docker compose up -d
```

> Docker Compose automatically starts PostgreSQL, Redis, Qdrant, Ollama, and automatically downloads the configured Ollama model on the first startup if it is not already available.

---

## Run

```bash
uvicorn app.main:app --reload
```

---

## Browser Test Client

The browser client located in the `client/` directory can be used to test the WebSocket voice pipeline.

---

## Current Pipeline

```text
Microphone
      │
      ▼
WebSocket
      │
      ▼
Silero VAD
      │
      ▼
Faster Whisper
      │
      ▼
Qwen3 (Ollama)
      │
      ▼
Structured JSON Response
```

---

## Notes

- All AI models run locally.
- PostgreSQL, Redis, Qdrant, and Ollama are managed with Docker Compose.
- The project currently focuses on voice routing and infrastructure.
- Tool execution, retrieval, and TTS will be added in future milestones.

---

