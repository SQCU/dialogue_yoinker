# subagent_orchestrator/

## What This Is

API routes and models for synthetic dialogue generation via structural transposition.

**This is NOT an agent framework.** There's no planning loop, no tool orchestration, no "agent decides what to do." It's a set of endpoints that accept structured requests and return structured responses, designed to be called by a Claude Code session that knows what it wants.

## Architecture

```
api_routes.py    → FastAPI endpoints for triplet extraction, translation, persistence
models.py        → Pydantic schemas for arcs, beats, translations
subagent.py      → Direct Anthropic API calls for generation
validation.py    → Schema validation, lore bible checks
observability.py → Trace logging for debugging failed runs
```

## Key Concepts

### Structural Triplet
A dialogue sequence decomposed into:
- **Emotion sequence**: `[neutral, anger, fear]`
- **Beat functions**: `[establish_stakes, threaten, plead]`
- **Archetype relations**: `[authority_to_subject, peer_to_peer]`
- **Arc shape**: `escalation_to_climax`, `tension_release`, etc.

Structure is extracted from source dialogue, then used to constrain target generation.

### Setting Transposition
Source: "The Emperor needs you to find the Amulet of Kings"
Target: "The Préfet requires your assistance locating the Seal of the République"

Same structure (authority_figure → requests → player → find → macguffin), different vocabulary.

### Lore Bible
YAML file defining target setting:
- Proper noun clusters (who/what exists)
- World logic (what rules apply)
- Tone (register, formality, humor)
- Game mechanics (what actions are possible)

Translations must use only vocabulary from the bible. New proper nouns require curator approval.

## Usage Pattern

This module is called by Claude Code sessions, not by users directly.

Typical flow:
1. Session samples walks from reference graph
2. Session calls `/api/extract/triplet` to get structure
3. Session calls `/api/translate/triplet` to transpose
4. Session calls `/api/synthetic/persist` to save

The session handles batching, parallelism, retries. This module handles single-request processing.

## Why Not Use an Agent Framework?

1. **Predictable costs**: Each endpoint has known token bounds
2. **Debuggable**: Traces show exactly what was sent/received
3. **No runaway loops**: Can't accidentally burn $500 on recursive planning
4. **Typed I/O**: Pydantic enforces schema at boundaries
5. **Composable**: Session can mix these endpoints with other tools

The "agent" doing the orchestration is the Claude Code session itself, which has:
- Full conversation context
- User feedback loop
- Ability to read/write arbitrary files
- Token budget visibility

Delegating orchestration to a sub-framework would lose all of these.

## Files

- `api_routes.py`: POST /api/extract/triplet, POST /api/translate/triplet, etc.
- `models.py`: StructuralArc, DialogueBeat, TranslationResult, etc.
- `subagent.py`: Raw Anthropic API calls with prompt templates
- `validation.py`: Bible adherence checking, proper noun validation
- `observability.py`: TraceLog for debugging (writes to /traces)
