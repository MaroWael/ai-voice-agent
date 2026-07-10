---
name: software-engineering
description: Use this skill whenever writing, reviewing, or architecting production code (especially for real-time / streaming / voice AI systems like the customer service voice assistant). Covers Clean Architecture, SOLID, dependency injection, naming, error handling, and async/streaming patterns specific to audio pipelines and orchestrators.
---

# Software Engineering Skill

## Objective

Build production-quality software that is simple, maintainable, testable, and easy to extend.

The goal is not to write the smartest code.
The goal is to build software that another engineer can understand in minutes.

---

# Core Principles

Always prioritize:

1. Simplicity
2. Readability
3. Maintainability
4. Testability
5. Extensibility
6. Performance (only when necessary)

Never sacrifice readability for cleverness.

---

# Engineering Philosophy

Think before coding.

Always follow this workflow:

Understand the problem
→ Design the solution
→ Define models
→ Define interfaces
→ Review architecture
→ Implement
→ Test
→ Refactor

Never start implementation immediately.

---

# SOLID

Follow SOLID principles whenever appropriate.

- Single Responsibility Principle
- Open / Closed Principle
- Liskov Substitution Principle
- Interface Segregation Principle
- Dependency Inversion Principle

Apply them only when they reduce complexity.

Do not force SOLID where a simple solution is enough.

---

# Clean Architecture

Dependencies must always point inward.

Business logic must never depend on:

- Frameworks
- Databases
- External APIs
- AI Providers
- UI
- Infrastructure

Business logic should be completely independent.

For voice AI systems specifically: the conversation/orchestration logic must never
depend directly on a specific STT engine, TTS engine, or LLM provider. These are
infrastructure and must sit behind interfaces (e.g. `SpeechRecognizer`, `SpeechSynthesizer`,
`ConversationModel`) so providers can be swapped (Whisper → another STT, Gemini → another LLM)
without touching orchestration code.

---

# Dependency Injection

Prefer constructor injection.

Avoid creating dependencies inside classes.

Bad

class Service:
    def __init__(self):
        self.repo = Repository()

Good

class Service:
    def __init__(self, repo: Repository):
        self.repo = repo

---

# Composition

Prefer composition over inheritance.

Only use inheritance when there is a real "is-a" relationship.

---

# Interfaces

Create abstractions only when:

- Multiple implementations are expected.
- External providers may change.
- The dependency belongs to infrastructure.

Do NOT create interfaces just because.

---

# Models

Public APIs should communicate using models.

Avoid:

- dict
- tuple
- list with magic indexes

Prefer:

- dataclass
- Pydantic models
- Typed objects

For pipeline/streaming systems, model events explicitly (e.g. `VADEvent`, `TranscriptChunk`,
`BargeInSignal`) instead of passing raw tuples or dicts between pipeline stages. This makes
the flow of data through the orchestrator traceable and testable.

---

# Functions

Functions should:

- Do one thing.
- Have descriptive names.
- Be short.
- Have minimal side effects.

Avoid functions longer than approximately 40 lines unless complexity requires it.

---

# Classes

Each class should have a single responsibility.

If a class is doing multiple unrelated jobs,
split it.

Avoid God Objects.

---

# Naming

Names should explain intent.

Bad

Manager

Helper

Util

Common

Data

Processor

Good

SpeechRecognizer

AudioBuffer

ConversationMemory

VoiceActivityDetector

TranscriptAssembler

---

# Error Handling

Never silently ignore exceptions.

Never use:

except:
    pass

Handle expected failures.

Log unexpected failures.

Return meaningful errors.

For pipeline stages (STT, LLM, TTS), distinguish between:

- **Recoverable failures** (e.g. transient network timeout on an LLM call) → retry with
  backoff, or degrade gracefully (e.g. fallback response).
- **Unrecoverable failures** (e.g. malformed audio stream) → fail fast, log with context
  (session id, pipeline stage, timestamp), and surface a clear signal upstream so the
  orchestrator can end or restart the turn instead of hanging silently.

Never let a pipeline stage fail silently and leave the conversation state stuck (e.g. user
waiting indefinitely because STT crashed without emitting an error event).

---

# Logging

Use logging for:

- Startup
- Shutdown
- Errors
- Important state changes

Avoid excessive logging.

For real-time pipelines, log state transitions (e.g. `listening → processing → speaking`,
barge-in triggered, endpoint detected) with timestamps — this is essential for debugging
latency issues after the fact, since the interaction can't be "paused" to inspect.

---

# Configuration

Never hardcode:

- API Keys
- Paths
- Ports
- Model names

Use configuration objects or environment variables.

---

# Async

When working with IO:

- Prefer async/await.
- Avoid blocking operations.
- Support cancellation when possible.

## Streaming & Real-Time Systems (Voice Pipelines)

These rules apply specifically to audio/voice pipelines and orchestrators:

- **Latency budget awareness.** Every stage in the pipeline (VAD → STT → LLM → TTS) should
  have a known target latency. Design the code so it's easy to measure time spent in each
  stage, not just the end-to-end response time.
- **Use async generators / streams, not buffering-then-processing**, wherever the data is
  inherently a stream (audio chunks, partial transcripts, token-by-token LLM output). Avoid
  waiting for a full result when a partial one can start the next stage sooner.
- **Cancellation is a first-class concern.** Barge-in means an in-flight LLM call or TTS
  playback must be cancellable mid-flight, cleanly, without leaking tasks or leaving the
  conversation state inconsistent. Design components so `cancel()` / `asyncio.CancelledError`
  is handled explicitly, not ignored.
- **Backpressure.** If a downstream stage is slower than upstream production (e.g. TTS
  synthesis slower than LLM token generation), define explicit behavior: buffer with a cap,
  drop, or block — don't let queues grow unbounded.
- **Idempotent state transitions.** The orchestrator's conversation state machine
  (listening / thinking / speaking / interrupted) should only transition through well-defined
  events, so race conditions between e.g. a late STT result and a new barge-in event can't
  corrupt state.

---

# Testing Real-Time / Async Code

- Mock audio streams and provider responses (STT/LLM/TTS) as async iterables so pipeline
  logic can be tested without real audio hardware or network calls.
- Test barge-in and cancellation paths explicitly — these are the highest-risk areas for bugs
  since they only occur on timing-dependent interruptions.
- Test partial/streaming inputs (e.g. incomplete transcripts, chunked audio) in addition to
  full/complete inputs.

---

# Code Duplication

Prefer reusable code.

However:

Do not abstract after seeing code once.

Rule of Three:

Duplicate once.
Duplicate twice.
Abstract on the third occurrence.

---

# Design Patterns

Use Design Patterns only when they solve a real problem.

Never introduce patterns for the sake of architecture.

Prefer simple code.

---

# KISS

Keep It Simple.

Simple code wins.

---

# YAGNI

You Aren't Gonna Need It.

Do not implement features for hypothetical future requirements.

---

# DRY

Avoid duplicated knowledge.

Do not over-abstract to eliminate two similar lines.

---

# Performance

Do not optimize prematurely.

Measure first.

Optimize second.

For real-time voice systems, "measure first" means measuring actual latency per pipeline
stage under realistic load before optimizing — intuition about where the bottleneck is
tends to be wrong in streaming systems.

---

# Documentation

Write code that explains itself.

Comments should explain WHY.

Code should explain HOW.

---

# Code Review Checklist

Before finishing:

- Is it readable?
- Is it simple?
- Does it follow SOLID?
- Does it follow Clean Architecture?
- Can naming improve?
- Is duplication acceptable?
- Are responsibilities clear?
- Is there unnecessary abstraction?
- Is there unnecessary inheritance?
- Can this be simpler?
- (Real-time systems) Is cancellation handled? Is backpressure defined? Are latency-critical
  paths free of blocking calls?

If yes,

simplify.

---

# What To Avoid

Avoid:

- God Classes
- Utility Classes for everything
- Static methods everywhere
- Deep inheritance
- Circular dependencies
- Tight coupling
- Hidden side effects
- Global state
- Premature optimization
- Overengineering
- Blocking calls inside async pipeline stages
- Unbounded queues/buffers in streaming paths

---

# Mindset

Write software for humans first.

The compiler is easy to satisfy.

Future developers are not.

Choose clarity over cleverness.
