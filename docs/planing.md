# Voice AI Customer Service Agent

> Enterprise-grade multilingual (Arabic / English) Voice AI Customer Service Agent.

---

# 1. Project Overview

## Description

This project is an enterprise-grade Voice AI Customer Service Agent designed primarily for the Egyptian market.

The system accepts spoken audio from customers, understands natural language, retrieves business knowledge and customer context, generates intelligent responses, and replies using natural speech.

The architecture is designed to be:

- Modular
- Extensible
- Maintainable
- Production-ready
- AI-first
- Clean Architecture based
- SOLID compliant

The project should avoid unnecessary complexity (Overengineering) while remaining scalable enough for enterprise deployment.

---

# 2. Vision

The goal is not simply converting speech into text.

The goal is building an intelligent employee that can communicate with customers exactly like a professional customer service representative.

The assistant should:

- Understand customer intent.
- Remember customer history.
- Access company knowledge.
- Answer naturally.
- Escalate when needed.
- Speak fluently in Arabic and English.
- Support future integrations without architectural changes.

---

# 3. Design Philosophy

This project follows several principles.

## Simplicity First

Choose the simplest solution that solves the current problem.

Avoid building abstractions before they become necessary.

Avoid speculative features.

---

## Modular Design

Every module should have one responsibility.

Modules should be replaceable without affecting the rest of the system.

Example:

Silero VAD can later become WebRTC VAD without changing the pipeline.

Whisper can later become another STT model.

Qwen can later become GPT, Llama, GLM, etc.

---

## Clean Architecture

Business logic must never depend on infrastructure.

Infrastructure depends on abstractions.

Never the opposite.

---

## SOLID Principles

Follow all SOLID principles.

Especially:

- Single Responsibility
- Dependency Inversion
- Interface Segregation

---

## Dependency Injection

Never instantiate services directly inside business logic.

Services should always be injected.

---

## Explicit Models

Every input/output should have explicit models.

Avoid dictionaries whenever possible.

Good:

AudioFrame
Transcript
ConversationContext
CustomerProfile

Bad:

dict
Any

---

## Composition over Inheritance

Prefer composition whenever possible.

Inheritance should be used only when the relationship is truly "is-a".

---

## Replaceability

Any AI model should be replaceable.

Any database should be replaceable.

Any provider should be replaceable.

Business logic should never know implementation details.

---

# 4. Target Users

Primary market:

Egypt.

Supported languages:

- Arabic
- English

Future:

- French
- German
- Spanish

---

# 5. Core Capabilities

The assistant should be capable of:

- Listening continuously
- Detecting speech
- Converting speech to text
- Understanding customer intent
- Retrieving business knowledge
- Retrieving customer memory
- Building conversation context
- Calling tools
- Generating responses
- Speaking naturally
- Streaming responses
- Handling interruptions
- Handling multiple sessions
- Supporting future omnichannel integrations

---

# 6. Technology Stack

Backend

- Python
- FastAPI

Configuration

- Pydantic Settings

ORM

- SQLAlchemy

Database

- PostgreSQL

Cache

- Redis

Vector Database

- Qdrant

Containerization

- Docker Compose

Voice Models

VAD

- Silero VAD

Speech To Text

- Faster Whisper

LLM

- Qwen 3 Instruct

Text To Speech

- Kokoro TTS

# 7. High Level Architecture

The system is divided into independent layers.

Each layer has a single responsibility.

The output of one layer becomes the input of the next layer.

```
Microphone

↓

Source Layer

↓

Audio Adapter

↓

Audio Preprocessing

↓

Voice Activity Detection (VAD)

↓

Speech Buffer

↓

Speech To Text (STT)

↓

Transcript

↓

Orchestrator

↓

Context Builder

↓

Agent

↓

Response

↓

Text To Speech (TTS)

↓

Audio Output
```

---

# 8. Layer Responsibilities

## 8.1 Source Layer

Responsibility:

Receive raw audio from any source.

Supported sources:

- Computer microphone
- USB microphone
- Phone call
- SIP
- Twilio
- WebRTC
- Audio file

This layer knows nothing about AI.

Its only responsibility is producing AudioFrames.

Output:

AudioFrame

---

## 8.2 Adapter Layer

Responsibility:

Normalize every audio source into a unified format.

Different devices produce audio differently.

Example:

48000 Hz

↓

16000 Hz

Stereo

↓

Mono

int16

↓

float32

Every source should produce identical AudioFrames.

Business logic should never care where audio came from.

---

## 8.3 Audio Preprocessing

Responsibility:

Improve audio quality before inference.

Current preprocessing:

- Resampling
- Mono conversion
- Volume normalization

Future preprocessing:

- Noise suppression
- Echo cancellation
- Automatic gain control

The preprocessing layer should never contain business logic.

---

## 8.4 Voice Activity Detection

Responsibility:

Detect speech.

Input:

AudioFrame

Output:

SpeechFrame

or

Silence

Silero VAD is responsible for answering one question:

"Does this frame contain speech?"

Nothing else.

---

## 8.5 Speech Buffer

Responsibility:

Collect consecutive speech frames.

The buffer starts collecting when speech begins.

The buffer stops when silence exceeds the configured threshold.

Example:

Speech

Speech

Speech

Speech

Silence

Silence

Silence

↓

Buffer Complete

↓

Send to STT

The STT model should never receive tiny fragments.

It always receives complete utterances.

---

## 8.6 Speech To Text

Responsibility:

Convert speech into text.

Model:

Faster Whisper

Output:

Transcript

Example:

Audio

↓

"Hello, I would like to know my order status."

No business logic exists here.

---

## 8.7 Orchestrator

The Orchestrator is the traffic controller.

It never makes business decisions.

Responsibilities:

- Receive transcript
- Create request id
- Create session id
- Route execution
- Call Context Builder
- Call Agent
- Call Response Layer
- Handle exceptions
- Emit events

The orchestrator coordinates.

It never thinks.

---

## 8.8 Context Builder

The Context Builder prepares everything required before calling the LLM.

Responsibilities:

- Retrieve customer profile
- Retrieve session memory
- Retrieve long-term memory
- Retrieve knowledge
- Retrieve similar conversations
- Retrieve tools
- Build prompt
- Build conversation state

Output:

ConversationContext

The Context Builder contains no reasoning.

It prepares data only.

---

## 8.9 Agent

The Agent is responsible for reasoning.

Input:

ConversationContext

Output:

AgentResponse

Responsibilities:

- Understand intent
- Decide next action
- Answer questions
- Call tools
- Generate structured response

The Agent should never access databases directly.

Everything must already exist inside ConversationContext.

---

## 8.10 Response Layer

Responsibilities:

Convert AgentResponse into customer-facing output.

Tasks:

- Post-processing
- Formatting
- Safety validation
- Streaming support

Output:

Plain text

---

## 8.11 Text To Speech

Responsibilities:

Convert response text into natural speech.

Requirements:

- Arabic
- English
- Low latency
- Streaming

Current model:

Kokoro TTS

Future models should be replaceable.

---

# 9. Complete Request Lifecycle

Customer speaks

↓

Microphone

↓

Audio Source

↓

Audio Adapter

↓

Audio Preprocessing

↓

Silero VAD

↓

Speech Buffer

↓

Faster Whisper

↓

Transcript

↓

Orchestrator

↓

Context Builder

↓

Conversation Context

↓

Qwen Agent

↓

Response

↓

Kokoro

↓

Customer hears response

This entire lifecycle should remain asynchronous.

Every layer should communicate through explicit models.

No layer should depend on implementation details of another layer.

# 10. Project Folder Structure

The project is organized into independent layers.

Each folder represents a single responsibility.

No folder should contain unrelated logic.

```
voice-agent/

│
├── app/
│
├── input/
│
├── orchestration/
│
├── context/
│
├── agent/
│
├── response/
│
├── memory/
│
├── knowledge/
│
├── models/
│
├── shared/
│
├── tests/
│
├── storage/
│
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

---

# 11. Folder Responsibilities

## app/

Application infrastructure.

Contains everything related to FastAPI and configuration.

```
app/

config/

db/

api/

main.py
```

Responsibilities

- FastAPI
- Configuration
- Database connections
- Dependency Injection
- Startup / Shutdown
- REST APIs

---

## input/

Responsible for receiving audio.

Contains:

```
input/

models/

sources/

adapter/

preprocessing/

vad/

buffer/
```

Responsibilities

Receive audio.

Normalize audio.

Detect speech.

Prepare complete utterances.

Never performs reasoning.

---

## orchestration/

Contains the application coordinator.

Responsibilities

Receive transcript.

Create execution pipeline.

Call Context Builder.

Call Agent.

Call Response.

Emit events.

Handle failures.

No AI logic exists here.

---

## context/

Responsible for building ConversationContext.

Contains:

```
context/

builder/

retrievers/

prompt/

services/
```

Responsibilities

Retrieve customer.

Retrieve memory.

Retrieve knowledge.

Retrieve session.

Retrieve tools.

Build prompt.

Produce ConversationContext.

---

## agent/

Contains the reasoning engine.

Contains:

```
agent/

core/

providers/

tools/

models/
```

Responsibilities

Intent detection.

Reasoning.

Tool calling.

Decision making.

Response generation.

No database access.

No infrastructure code.

---

## response/

Responsible for customer output.

Contains

```
response/

formatter/

tts/

streaming/
```

Responsibilities

Prepare final response.

Convert text to speech.

Handle streaming.

Support future channels.

---

## memory/

Responsible for conversation memory.

Contains

```
memory/

session/

long_term/

retrievers/

storage/
```

Responsibilities

Store memories.

Retrieve memories.

Rank memories.

Expire session memories.

---

## knowledge/

Responsible for company knowledge.

Contains

```
knowledge/

documents/

retrievers/

indexing/

embeddings/
```

Responsibilities

RAG.

Embedding generation.

Knowledge retrieval.

Knowledge indexing.

---

## models/

Shared domain models.

Examples

```
AudioFrame

Transcript

ConversationContext

Customer

Memory

AgentResponse

KnowledgeChunk

ToolCall

SpeechSegment
```

Models should remain framework-independent.

---

## shared/

Contains reusable utilities.

Examples

```
logging/

exceptions/

utils/

constants/

types/
```

No business logic belongs here.

---

## tests/

Contains

```
unit/

integration/

end_to_end/
```

Every layer should have its own tests.

---

# 12. Layer Dependencies

Allowed dependencies

```
Input

↓

Orchestrator

↓

Context

↓

Agent

↓

Response
```

Forbidden

```
Agent

↓

Database
```

Forbidden

```
Response

↓

Memory
```

Forbidden

```
Input

↓

LLM
```

Every layer should communicate only with its adjacent layer whenever possible.

---

# 13. Dependency Rule

Higher layers define interfaces.

Lower layers implement them.

Example

```
AudioSource

↑

MicrophoneSource

WebRTCSource

TwilioSource
```

Business logic depends on

AudioSource

Never on

MicrophoneSource

---

# 14. Shared Models

Every layer communicates using models.

Never dictionaries.

Examples

Good

```
AudioFrame

Transcript

ConversationContext

CustomerProfile

SpeechSegment

AgentResponse
```

Avoid

```
dict

list

tuple
```

unless they are internal implementation details.

---

# 15. Naming Conventions

Folders

snake_case

Classes

PascalCase

Functions

snake_case

Variables

snake_case

Constants

UPPER_CASE

Interfaces

Descriptive names.

Avoid prefixes like

IUserService

Use

UserService

AudioSource

MemoryRepository

CustomerRepository

---

# 16. Dependency Injection

Never create services manually.

Avoid

```
service = CustomerService()
```

Prefer

FastAPI Dependency Injection

or

Constructor Injection

Example

```
class ContextBuilder:

    def __init__(
        self,
        customer_repository,
        memory_repository,
        knowledge_repository,
    ):
        ...
```

Dependencies should always be explicit.

---

# 17. Configuration Rules

Configuration should exist only inside

```
app/config
```

Never call

```
os.getenv()
```

outside Settings.

Everything should be accessed through

```
settings
```

Example

```
settings.POSTGRES_DB

settings.REDIS_PORT

settings.QDRANT_HTTP_PORT
```

This guarantees a single source of truth.

# 19. Development Roadmap

The project should be developed incrementally.

Each milestone should produce a working system.

Never implement future features before finishing the current milestone.

Every milestone must be fully tested before moving to the next.

---

# Milestone 1

## Project Foundation

Goal

Build the project infrastructure.

Deliverables

- Project structure
- Docker Compose
- PostgreSQL
- Redis
- Qdrant
- FastAPI
- Configuration
- Database Connections

Definition of Done

✓ FastAPI starts successfully.

✓ Docker containers are healthy.

✓ PostgreSQL connection established.

✓ Redis connection established.

✓ Qdrant connection established.

---

# Milestone 2

## Audio Input Layer

Goal

Receive audio from microphone.

Deliverables

AudioSource

MicrophoneSource

AudioAdapter

AudioFrame

Definition of Done

✓ Audio is continuously captured.

✓ Frames have fixed duration.

✓ AudioFrames are produced correctly.

---

# Milestone 3

## Audio Preprocessing

Goal

Normalize incoming audio.

Deliverables

AudioNormalizer

Resampler

MonoConverter

VolumeNormalizer

Definition of Done

✓ Every source produces identical format.

✓ Sample rate always 16kHz.

✓ Mono audio.

✓ Float32 samples.

---

# Milestone 4

## Voice Activity Detection

Goal

Detect speech.

Deliverables

Silero VAD

SpeechDetector

SpeechFrame

Definition of Done

✓ Silence ignored.

✓ Speech detected correctly.

✓ False positives minimized.

---

# Milestone 5

## Speech Buffer

Goal

Collect complete utterances.

Deliverables

SpeechBuffer

SpeechSegment

Definition of Done

✓ Buffer starts on speech.

✓ Stops after silence threshold.

✓ Produces complete utterance.

---

# Milestone 6

## Speech To Text

Goal

Convert speech into transcript.

Deliverables

WhisperService

Transcript

Definition of Done

✓ Arabic transcription.

✓ English transcription.

✓ Streaming support planned.

---

# Milestone 7

## Orchestration

Goal

Coordinate execution.

Deliverables

Orchestrator

ConversationSession

ExecutionContext

Definition of Done

✓ Receives transcript.

✓ Calls Context Builder.

✓ Calls Agent.

✓ Handles failures.

---

# Milestone 8

## Customer Database

Goal

Create customer domain.

Deliverables

Customer Model

Repository

Service

Definition of Done

✓ Customer lookup.

✓ Customer creation.

✓ Session association.

---

# Milestone 9

## Memory System

Goal

Store conversation memory.

Deliverables

Session Memory

Long-term Memory

Memory Retrieval

Definition of Done

✓ Previous conversations stored.

✓ Similar memories retrieved.

✓ Session expiration works.

---

# Milestone 10

## Knowledge Base

Goal

Enable RAG.

Deliverables

Embedding Generator

Document Indexer

Knowledge Retriever

Definition of Done

✓ Company documents indexed.

✓ Semantic retrieval works.

---

# Milestone 11

## Context Builder

Goal

Create ConversationContext.

Deliverables

Customer

Memory

Knowledge

Conversation History

Prompt Builder

Definition of Done

✓ Context assembled correctly.

✓ Prompt contains relevant information.

---

# Milestone 12

## Agent

Goal

Reasoning.

Deliverables

Qwen Provider

Prompt Executor

Tool Calling

AgentResponse

Definition of Done

✓ Intent detection.

✓ Response generation.

✓ Tool execution.

---

# Milestone 13

## Response Layer

Goal

Prepare customer response.

Deliverables

Formatter

Streaming

Response Models

Definition of Done

✓ Response formatted.

✓ Metadata attached.

---

# Milestone 14

## Text To Speech

Goal

Generate speech.

Deliverables

Kokoro Service

Streaming Audio

Definition of Done

✓ Arabic speech.

✓ English speech.

✓ Low latency.

---

# Milestone 15

## Full Conversation Pipeline

Goal

Complete end-to-end conversation.

Pipeline

Customer

↓

Microphone

↓

VAD

↓

Speech Buffer

↓

Whisper

↓

Orchestrator

↓

Context Builder

↓

Agent

↓

TTS

↓

Customer

Definition of Done

✓ End-to-end conversation succeeds.

---

# Milestone 16

## Optimization

Goal

Reduce latency.

Tasks

Parallel retrieval

Streaming

Prompt optimization

Cache

Connection pooling

Definition of Done

Latency reduced.

---

# Milestone 17

## Production Readiness

Goal

Prepare deployment.

Tasks

Docker

Logging

Monitoring

Configuration

Health Checks

Graceful Shutdown

Definition of Done

Production deployment succeeds.

---

# Testing Strategy

Every milestone must include tests.

Unit Tests

Integration Tests

End-to-End Tests

Performance Tests

Manual Voice Tests

No milestone is considered complete until tests pass.

---

# Success Metrics

Startup Time

Audio Latency

STT Accuracy

LLM Response Time

TTS Latency

Overall Conversation Latency

Memory Retrieval Accuracy

Knowledge Retrieval Accuracy

Error Rate

Resource Consumption

---

# Final Product

An enterprise-grade multilingual Voice AI Customer Service Agent capable of serving customers naturally in Arabic and English while maintaining conversation history, customer memory, company knowledge, and scalable architecture.