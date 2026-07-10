# CLAUDE.md — Voice AI Customer Service Agent
# Permanent Development Guide

> This file is the single source of truth for all future coding sessions.
> Read it fully before writing any code.

---

## 1. Project Overview

An **enterprise-grade, multilingual (Arabic / English) Voice AI Customer Service Agent** designed primarily for the Egyptian market.

The system listens to customer speech, understands intent, retrieves business knowledge and customer memory, generates intelligent responses, and replies in natural speech.

**Primary languages:** Arabic, English  
**Future languages:** French, German, Spanish  
**Primary market:** Egypt

---

## 2. Vision

The goal is **not** to convert speech to text.

The goal is an **intelligent digital employee** that behaves exactly like a professional customer service representative:

- Understands customer intent
- Remembers customer history
- Accesses company knowledge
- Answers naturally in Arabic or English
- Escalates when appropriate
- Supports future integrations without architectural changes

---

## 3. Design Philosophy

### 3.1 Simplicity First (KISS)

Choose the simplest solution that solves the current problem.  
Do not build abstractions before they are necessary.  
Do not implement speculative features.

### 3.2 YAGNI

You Aren't Gonna Need It.  
Do not implement features for hypothetical future requirements.

### 3.3 Rule of Three

Duplicate once. Duplicate twice. Abstract only on the **third** occurrence.

### 3.4 Think Before Coding

Always follow this workflow:

```
Understand the problem
→ Design the solution
→ Define models
→ Define interfaces
→ Review architecture
→ Implement
→ Test
→ Refactor
```

Never start implementation immediately.

---

## 4. Architecture Overview

The system is a **linear, layered pipeline**. Each layer has exactly one responsibility. The output of one layer is the input of the next.

```
Microphone
    ↓
Source Layer         (audio capture)
    ↓
Audio Adapter        (format normalization)
    ↓
Audio Preprocessing  (quality improvement)
    ↓
VAD                  (speech detection)
    ↓
Speech Buffer        (utterance assembly)
    ↓
STT                  (speech → text)
    ↓
Transcript
    ↓
Orchestrator         (traffic controller)
    ↓
Context Builder      (data assembly)
    ↓
Agent                (reasoning)
    ↓
Response Layer       (formatting + safety)
    ↓
TTS                  (text → speech)
    ↓
Customer hears response
```

The entire lifecycle must remain **asynchronous**.

Every layer communicates through **explicit typed models**, never raw dicts or tuples.

---

## 5. Layer Responsibilities

| Layer | Responsibility | Input | Output |
|---|---|---|---|
| **Source** | Capture raw audio from any source | Physical audio | `AudioFrame` |
| **Adapter** | Normalize all sources to a unified format | `AudioFrame` (raw) | `AudioFrame` (normalized) |
| **Preprocessing** | Improve audio quality (resample, mono, volume) | `AudioFrame` | `AudioFrame` (clean) |
| **VAD** | Detect speech vs. silence | `AudioFrame` | `SpeechFrame` or silence |
| **Speech Buffer** | Collect frames until utterance complete | `SpeechFrame` stream | `SpeechSegment` |
| **STT** | Convert speech to text | `SpeechSegment` | `Transcript` |
| **Orchestrator** | Coordinate execution, create IDs, route, handle errors | `Transcript` | calls Context Builder → Agent → Response |
| **Context Builder** | Assemble all context before calling LLM | `Transcript` + session | `ConversationContext` |
| **Agent** | Reason, detect intent, call tools, generate response | `ConversationContext` | `AgentResponse` |
| **Response** | Format, validate safety, prepare output | `AgentResponse` | Plain text |
| **TTS** | Convert text to natural speech | Plain text | Audio stream |

### Critical Layer Rules

- The **Input layer** knows nothing about AI.
- The **Orchestrator** coordinates; it never makes business decisions.
- The **Context Builder** prepares data only; it contains no reasoning.
- The **Agent** never accesses databases directly. Everything it needs must be in `ConversationContext`.
- The **Response layer** never accesses Memory.
- The **Input layer** never calls the LLM.

---

## 6. Forbidden Dependencies

These cross-layer dependencies are **strictly forbidden**:

```
Agent → Database          (forbidden)
Response → Memory         (forbidden)
Input → LLM               (forbidden)
Any layer → Another layer's implementation details (forbidden)
```

Allowed direction only: `Input → Orchestrator → Context → Agent → Response`

---

## 7. Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Web framework | FastAPI |
| Configuration | Pydantic Settings |
| ORM | SQLAlchemy |
| Migrations | Alembic |
| Relational DB | PostgreSQL 17 |
| Cache | Redis 7 |
| Vector DB | Qdrant v1.15.3 |
| Containerization | Docker Compose |
| Audio capture | sounddevice |
| Audio processing | numpy |
| VAD | Silero VAD |
| STT | Faster Whisper |
| LLM | Qwen 3 Instruct |
| TTS | Kokoro TTS |
| ASGI server | Uvicorn (standard) |

**Do not introduce technologies not listed here** without explicit discussion and approval.

---

## 8. Project Folder Structure

```
voice-agent/
│
├── app/                    # Infrastructure: FastAPI, config, DB, DI, APIs
│   ├── config/             # Settings only (Pydantic Settings)
│   │   └── settings.py
│   ├── db/                 # Database connections
│   ├── api/                # REST API routes
│   └── main.py             # FastAPI app entry point
│
├── input/                  # Audio input pipeline
│   ├── models/
│   ├── sources/            # MicrophoneSource, etc.
│   ├── adapter/            # Format normalization
│   ├── preprocessing/      # Resample, mono, volume
│   ├── vad/                # Silero VAD integration
│   └── buffer/             # SpeechBuffer
│
├── orchestration/          # Pipeline coordinator (no AI logic)
│
├── context/                # ConversationContext builder
│   ├── builder/
│   ├── retrievers/
│   ├── prompt/
│   └── services/
│
├── agent/                  # Reasoning engine
│   ├── core/
│   ├── providers/          # Qwen integration
│   ├── tools/
│   └── models/
│
├── response/               # Customer output
│   ├── formatter/
│   ├── tts/                # Kokoro TTS integration
│   └── streaming/
│
├── memory/                 # Conversation memory
│   ├── session/            # Redis-backed session memory
│   ├── long_term/          # Qdrant-backed long-term memory
│   ├── retrievers/
│   └── storage/
│
├── knowledge/              # Company knowledge / RAG
│   ├── documents/
│   ├── retrievers/
│   ├── indexing/
│   └── embeddings/
│
├── models/                 # Shared domain models (framework-independent)
│
├── shared/                 # Utilities, no business logic
│   ├── logging/
│   ├── exceptions/
│   ├── utils/
│   ├── constants/
│   └── types/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── end_to_end/
│
├── storage/                # Docker volume mounts
│   ├── postgres/
│   ├── redis/
│   └── qdrant/
│
├── docs/
│   ├── planing.md
│   └── skills.md
│
├── docker-compose.yml
├── requirements.txt
├── .env
├── .env.example
└── CLAUDE.md               # This file
```

**No folder should contain logic unrelated to its declared responsibility.**

---

## 9. Current Project Status

### Milestone 1: Project Foundation — Partially Complete

What exists today:
- `docker-compose.yml` with PostgreSQL 17, Redis 7, Qdrant v1.15.3
- `app/config/settings.py` with Pydantic Settings
- `app/main.py` with bare FastAPI app
- `.env.example` with required variables
- `requirements.txt` with foundation packages

What is **not yet built** (all future milestones):
- Full folder structure (`input/`, `orchestration/`, `context/`, `agent/`, `response/`, `memory/`, `knowledge/`, `models/`, `shared/`, `tests/`)
- Database connections (`app/db/`)
- Audio pipeline (Milestones 2–6)
- Orchestrator (Milestone 7)
- Customer database (Milestone 8)
- Memory system (Milestone 9)
- Knowledge base / RAG (Milestone 10)
- Context Builder (Milestone 11)
- Agent / Qwen integration (Milestone 12)
- Response layer (Milestone 13)
- TTS / Kokoro (Milestone 14)
- Full pipeline (Milestone 15)
- Optimization (Milestone 16)
- Production readiness (Milestone 17)

> **Rule:** Complete and fully test one milestone before starting the next.
> Never implement future milestone features ahead of time.

---

## 10. Shared Domain Models

Every layer communicates using **typed models**, never raw `dict`, `list`, or `tuple`.

### Canonical Models (to be defined in `models/`)

| Model | Origin | Consumed by |
|---|---|---|
| `AudioFrame` | Source / Adapter | Preprocessing, VAD |
| `SpeechFrame` | VAD | Speech Buffer |
| `SpeechSegment` | Speech Buffer | STT |
| `Transcript` | STT | Orchestrator |
| `ConversationContext` | Context Builder | Agent |
| `AgentResponse` | Agent | Response Layer |
| `CustomerProfile` | Customer DB | Context Builder |
| `Memory` | Memory system | Context Builder |
| `KnowledgeChunk` | Knowledge base | Context Builder |
| `ToolCall` | Agent | Tool executor |

Models must be **framework-independent** (no FastAPI, no SQLAlchemy inside domain models).

---

## 11. SOLID Principles

Apply only when they reduce complexity. Do not force SOLID where a simple solution is enough.

| Principle | Rule |
|---|---|
| **Single Responsibility** | Every class does one thing. Split if a class has multiple unrelated jobs. |
| **Open / Closed** | Extend via new classes, not by modifying existing ones. |
| **Liskov Substitution** | Implementations must be substitutable for their abstractions. |
| **Interface Segregation** | Create narrow interfaces. Do not force clients to depend on methods they do not use. |
| **Dependency Inversion** | Higher layers define interfaces. Lower layers implement them. |

---

## 12. Clean Architecture Rules

- Business logic must **never** depend on frameworks, databases, external APIs, AI providers, or infrastructure.
- Infrastructure depends on abstractions. Never the opposite.
- The conversation/orchestration logic must never depend on a specific STT engine, TTS engine, or LLM provider. These sit behind interfaces (e.g. `SpeechRecognizer`, `SpeechSynthesizer`, `ConversationModel`).

**Dependency Rule Example:**

```
AudioSource          ← abstract interface (higher layer owns it)
    ↑
MicrophoneSource     ← concrete implementation
WebRTCSource         ← concrete implementation
TwilioSource         ← concrete implementation
```

Business logic depends on `AudioSource`, never on `MicrophoneSource`.

---

## 13. Dependency Injection Guidelines

Never instantiate services inside business logic.

**Bad:**
```python
class ContextBuilder:
    def __init__(self):
        self.repo = CustomerRepository()   # tight coupling
```

**Good:**
```python
class ContextBuilder:
    def __init__(
        self,
        customer_repository: CustomerRepository,
        memory_repository: MemoryRepository,
        knowledge_repository: KnowledgeRepository,
    ):
        ...                                # injected
```

- Use **FastAPI Dependency Injection** for HTTP-layer dependencies.
- Use **Constructor Injection** for service-layer dependencies.
- All dependencies must be **explicit**. No hidden globals.

---

## 14. Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Folders | `snake_case` | `speech_buffer/` |
| Files | `snake_case` | `audio_frame.py` |
| Classes | `PascalCase` | `SpeechBuffer` |
| Functions / methods | `snake_case` | `detect_speech()` |
| Variables | `snake_case` | `audio_frame` |
| Constants | `UPPER_CASE` | `MAX_SILENCE_FRAMES` |
| Interfaces / ABCs | Descriptive, no `I` prefix | `AudioSource`, `MemoryRepository` |

**Names that explain intent — always:**
- Good: `SpeechRecognizer`, `AudioBuffer`, `ConversationMemory`, `VoiceActivityDetector`
- Bad: `Manager`, `Helper`, `Util`, `Common`, `Data`, `Processor`

---

## 15. Configuration Rules

Configuration lives **exclusively** in `app/config/settings.py`.

**Forbidden outside `Settings`:**
```python
os.getenv("POSTGRES_DB")   # never use directly
```

**Correct:**
```python
from app.config.settings import settings

settings.POSTGRES_DB
settings.REDIS_PORT
settings.QDRANT_HTTP_PORT
```

Never hardcode: API keys, file paths, port numbers, model names.

### Adding New Config Variables

1. Add the variable to `.env.example` (without a value).
2. Add it to `app/config/settings.py` under the appropriate section comment.
3. Set the actual value in `.env` locally.
4. Never commit `.env`.

---

## 16. Error Handling Strategy

**Never** silently ignore exceptions:
```python
except:
    pass   # forbidden
```

### Pipeline Stage Classification

| Failure Type | Response |
|---|---|
| **Recoverable** (transient timeout, temporary provider error) | Retry with backoff; degrade gracefully |
| **Unrecoverable** (malformed audio stream, fatal config error) | Fail fast; log with context (session ID, pipeline stage, timestamp); surface clear error signal upstream |

The Orchestrator must handle errors and prevent the conversation state from getting stuck (e.g., user waiting indefinitely because STT crashed without emitting an error event).

---

## 17. Logging Strategy

Log:
- Application startup and shutdown
- Errors
- Important state changes

For real-time pipelines, log **state transitions** with timestamps:
- `listening → processing → speaking`
- Barge-in triggered
- Endpoint detected
- Per-stage latency milestones

This is essential for debugging latency issues after the fact, since voice interactions cannot be paused for inspection.

**Avoid excessive logging.** Logs must be signal, not noise.

---

## 18. Async and Streaming Rules

All IO must be async. Avoid blocking calls inside async pipeline stages.

### Real-Time Pipeline Specific Rules

1. **Latency budget awareness.** Every stage (VAD → STT → LLM → TTS) must have a known latency target. Design so per-stage time is measurable.

2. **Use async generators / streams.** Do not buffer-then-process a stream. Audio chunks, partial transcripts, and token-by-token LLM output must flow as generators.

3. **Cancellation is first-class.** Barge-in must cancel in-flight LLM calls and TTS playback cleanly. Handle `asyncio.CancelledError` explicitly in every component. Never let cancellation leak tasks or leave conversation state inconsistent.

4. **Backpressure.** Define explicit behavior when a downstream stage is slower than upstream production (e.g., TTS slower than LLM tokens): buffer with a size cap, drop, or block. Never allow unbounded queues.

5. **Idempotent state transitions.** The Orchestrator's state machine (`listening / thinking / speaking / interrupted`) must transition only through well-defined events. Protect against race conditions between late STT results and new barge-in events.

---

## 19. Database Responsibilities

| Database | Responsibility |
|---|---|
| **PostgreSQL** | Customer profiles, conversation history, session metadata, structured data |
| **Redis** | Session memory (active conversations), short-lived cache |
| **Qdrant** | Long-term memory vectors, knowledge base embeddings, semantic search |

The Agent layer **never** accesses any database directly.
Database access belongs to repositories, which are injected into services.

---

## 20. AI Model Responsibilities

| Model | Role | Replaceability |
|---|---|---|
| **Silero VAD** | Speech detection only. Answers: "Does this frame contain speech?" | Replaceable with WebRTC VAD or any VAD |
| **Faster Whisper** | Speech-to-text only. Produces `Transcript`. | Replaceable with any STT model |
| **Qwen 3 Instruct** | Reasoning, intent detection, tool calling, response generation | Replaceable with GPT, Llama, GLM, etc. |
| **Kokoro TTS** | Text-to-speech only. Produces streaming audio. | Replaceable with any TTS engine |

Every AI model sits **behind an interface**. Business logic depends on the interface, not the model.

---

## 21. Interfaces — When to Create Them

Create an abstraction **only when**:
- Multiple implementations are expected.
- An external provider may change.
- The dependency belongs to infrastructure.

Do **not** create interfaces just for the sake of architecture.

---

## 22. Overengineering Avoidance Rules

- Do not build abstractions before they are necessary.
- Do not introduce design patterns for the sake of patterns.
- Do not create interfaces for classes that will only ever have one implementation (unless that implementation is an external provider).
- Do not add layers to the pipeline that do not correspond to a real responsibility.
- Prefer a simple function over a class hierarchy when the function is enough.
- Apply SOLID only when it reduces complexity.

---

## 23. Coding Standards

### Functions

- Do one thing only.
- Have descriptive names.
- Stay short (~40 lines max unless complexity genuinely requires more).
- Have minimal side effects.

### Classes

- Single responsibility.
- No God Objects.
- Prefer composition over inheritance.
- Only use inheritance for genuine "is-a" relationships.

### Comments

- Comments explain **WHY**.
- Code explains **HOW**.
- Never leave commented-out dead code.

### Avoid

- God Classes
- Utility dumping grounds (`utils.py` with unrelated functions)
- Static methods everywhere
- Deep inheritance hierarchies
- Circular dependencies
- Tight coupling
- Hidden side effects
- Global mutable state
- Premature optimization
- Blocking calls inside async pipeline stages
- Unbounded queues/buffers in streaming paths

---

## 24. Testing Strategy

Every milestone must include passing tests before moving to the next.

| Type | Coverage |
|---|---|
| **Unit tests** | Each layer, each service, each model |
| **Integration tests** | Layer-to-layer communication |
| **End-to-end tests** | Complete pipeline from audio in to audio out |
| **Performance tests** | Per-stage latency under realistic load |
| **Manual voice tests** | Real Arabic/English speech verification |

### Testing Real-Time / Async Code

- Mock audio streams and provider responses (STT / LLM / TTS) as **async iterables**. Pipeline logic must be testable without real audio hardware or network calls.
- Test **barge-in and cancellation paths explicitly** — these are the highest-risk areas.
- Test **partial/streaming inputs** (incomplete transcripts, chunked audio) in addition to complete inputs.

---

## 25. Definition of Done

A milestone is **complete** when:

- [ ] All deliverables in that milestone are implemented.
- [ ] Unit tests pass.
- [ ] Integration tests pass (where applicable).
- [ ] No silent exception handlers (`except: pass`).
- [ ] No `os.getenv()` outside `settings.py`.
- [ ] No hardcoded credentials, ports, or model names.
- [ ] All public APIs use typed models (no raw `dict` / `tuple` crossing layer boundaries).
- [ ] All new config variables are in `.env.example` and `settings.py`.
- [ ] Code review checklist passed (see Section 27).

---

## 26. Development Roadmap Summary

| # | Milestone | Status |
|---|---|---|
| 1 | Project Foundation (FastAPI, Docker, Config, DB connections) | Partial |
| 2 | Audio Input Layer (AudioSource, MicrophoneSource, AudioAdapter, AudioFrame) | Not started |
| 3 | Audio Preprocessing (Resampler, MonoConverter, VolumeNormalizer) | Not started |
| 4 | Voice Activity Detection (Silero VAD, SpeechDetector, SpeechFrame) | Not started |
| 5 | Speech Buffer (SpeechBuffer, SpeechSegment) | Not started |
| 6 | Speech To Text (WhisperService, Transcript) | Not started |
| 7 | Orchestration (Orchestrator, ConversationSession) | Not started |
| 8 | Customer Database (Model, Repository, Service) | Not started |
| 9 | Memory System (Session Memory, Long-term Memory) | Not started |
| 10 | Knowledge Base (RAG, Embeddings, Retrieval) | Not started |
| 11 | Context Builder (ConversationContext assembly) | Not started |
| 12 | Agent (Qwen, Tool Calling, AgentResponse) | Not started |
| 13 | Response Layer (Formatter, Streaming, Response Models) | Not started |
| 14 | Text To Speech (Kokoro, Streaming Audio) | Not started |
| 15 | Full Conversation Pipeline (end-to-end) | Not started |
| 16 | Optimization (parallel retrieval, streaming, caching) | Not started |
| 17 | Production Readiness (Docker, logging, monitoring, health checks) | Not started |

---

## 27. Code Review Checklist

Before considering any piece of work done, verify:

- [ ] Is it readable by another engineer in under 5 minutes?
- [ ] Is it as simple as it can be?
- [ ] Does it follow SOLID where applicable?
- [ ] Does it follow Clean Architecture (dependencies point inward)?
- [ ] Can naming be improved to better express intent?
- [ ] Is any duplication acceptable given the Rule of Three?
- [ ] Are responsibilities clearly separated?
- [ ] Is there unnecessary abstraction that should be removed?
- [ ] Is there unnecessary inheritance that should become composition?
- [ ] Can this be made simpler without losing correctness?
- [ ] (Real-time systems) Is cancellation handled explicitly?
- [ ] (Real-time systems) Is backpressure defined?
- [ ] (Real-time systems) Are all latency-critical paths free of blocking calls?

If any answer reveals complexity — **simplify**.

---

## 28. Instructions for Future Coding Sessions

1. **Read this file first.** Always.
2. **Identify the current milestone** from Section 26 and work only within it.
3. **Never skip a milestone.** A milestone that is not fully tested is not complete.
4. **Define models before implementation.** No code touches a layer boundary without a typed model.
5. **Define the interface before the implementation** when a component is an external provider or has multiple expected implementations.
6. **Inject, never instantiate.** Services must be injected, not created inside business logic.
7. **Config only from `settings`.** Never call `os.getenv()` outside `app/config/settings.py`.
8. **Async first.** All IO is async. No blocking calls in pipeline stages.
9. **Keep it simple.** If you feel the urge to add a pattern, ask whether it is solving an existing problem or a hypothetical one.
10. **Test before moving on.** No milestone is complete without passing tests.
11. **Do not modify other layers** when working in one layer. Each layer is a self-contained unit.
12. **Do not invent architecture.** Follow what is described in `docs/planing.md`. If you need to deviate, document and discuss first.

---

## 29. Success Metrics

The system will be measured against:

| Metric | Notes |
|---|---|
| Startup time | FastAPI + all services ready |
| Audio capture latency | Frames produced without delay |
| STT accuracy | Arabic and English transcription quality |
| LLM response time | Time from Transcript to AgentResponse |
| TTS latency | Time from text to first audio chunk |
| Overall conversation latency | Customer speaks to customer hears response |
| Memory retrieval accuracy | Correct memories surfaced per turn |
| Knowledge retrieval accuracy | Correct knowledge chunks per query |
| Error rate | Pipeline failures per N conversations |
| Resource consumption | CPU / RAM / GPU under load |

---

*This document was generated from `docs/planing.md`, `docs/skills.md`, `docker-compose.yml`, `requirements.txt`, `.env.example`, and the current source code.*
*Last updated: Milestone 1 (partial).*
