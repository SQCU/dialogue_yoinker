# Ticket Queue Decoupled Orchestration - Session Notes

**Date**: 2025-12-22
**Run ID**: `run_20251222_225616_gallia`

## What Was Built

### Ticket Queue System (`workflow/ticket_queue.py`)

A decoupled work distribution system where:
- Orchestrator creates runs and populates ticket queues
- Subagents independently claim/submit tickets via HTTP API
- Results persist to disk regardless of orchestrator state
- Downstream tickets auto-created on completion (parse→translate→curate)

Key classes:
```python
class Ticket:
    ticket_id: str
    worker_type: WorkerType  # structural_parser | translation_engine | lore_curator
    status: TicketStatus     # pending | claimed | completed | failed | needs_review
    input_data: dict
    output_data: dict
    worker_concerns: list[dict]  # {level, message, suggestion}

class TicketQueueManager:
    def create_run(target_bible, source_games, walks, config) -> RunQueue
    def claim_ticket(run_id, worker_type, worker_id) -> dict
    def submit_ticket(run_id, ticket_id, output_data, worker_concerns) -> dict
```

### API Routes (`workflow/ticket_routes.py`)

```
POST /api/runs                    - Create run with sampled walks
GET  /api/runs                    - List all runs
GET  /api/runs/{id}/status        - Queue status summary
POST /api/runs/{id}/claim         - Claim next ticket by worker type
POST /api/runs/{id}/submit        - Submit result (auto-persists, auto-creates downstream)
GET  /api/runs/{id}/concerns      - All worker concerns
GET  /api/runs/{id}/tickets/{tid} - Specific ticket details
```

### Bug Fixes

**Deadlock in nested locking**: Changed `threading.Lock()` to `threading.RLock()`
because `claim_ticket()` calls `get_queue()` which both acquire the lock.

**Sample count limit**: `/api/sample` has max 20 per request. Fixed
`sample_walks_from_api()` to batch requests.

## Test Run Results

### Run Configuration
- **Target**: Gallia (French bureaucratic-procedural)
- **Sources**: falloutnv + oblivion
- **Sample count**: 50 walks
- **Max walk length**: 5 nodes

### Final Status (from disk persistence)
```
Parse:     47/50 completed
Translate: 47/47 completed
Curate:    3/3  completed
```

3 tickets stalled in "claimed" state from a haiku agent that couldn't execute
bash commands in its sandbox - demonstrates failure isolation works.

### Curated Proper Nouns
| Noun | Verdict | Rationale |
|------|---------|-----------|
| la Légion | APPROVE | French Foreign Legion resonance, institutional weight |
| Barrage | APPROVE | Literal "dam" + metaphoric "checkpoint/obstruction" |
| Hexagone | APPROVE | French metonym for centralized government (like "the Beltway") |

## Architecture Observations

### What Works Well

1. **Persistence survives orchestrator confusion**: Even when server in-memory
   state diverged from disk (race condition with multiple workers), the disk
   file remained authoritative source of truth.

2. **Subagent API weight is light**: ~1.2M tokens for 42 translations ≈ 30k
   tokens/translation including reasoning. Suitable for subscription-limited
   usage.

3. **Auto-cascade of downstream tickets**: Parse completion auto-creates
   translate ticket. Translate completion with new nouns auto-creates curate
   tickets. No orchestrator intervention needed.

4. **Worker concerns as structured feedback**: Workers can flag issues at
   `info`, `review`, or `error` levels without blocking the pipeline.

### What's Missing

1. **Source node ID passthrough**: Translation output is `translated_texts: [...]`
   but doesn't preserve per-beat linkage to source node IDs like `0x14dfea`.

2. **Synthetic graph topology**: Results are flat `LinkedSynthetic` entries.
   No mechanism for Gallia corpus to form its OWN graph structure with topic
   transitions independent of source topology.

3. **Divergent growth specification**: Current pipeline pastiches beat-level
   dynamics but doesn't specify how synthetic corpus develops novel macro-structure.

## Token Usage

| Agent | Role | Tokens |
|-------|------|--------|
| structural-parser (haiku) | Attempted but sandbox-blocked | ~47k |
| translation-engine (sonnet) | 47 translations | ~1.5M |
| lore-curator (sonnet) | 3 curations | ~3k |

Parser work done via direct `uv run python` script (no Claude API cost).

## Next Questions

The core architectural tension identified:

| Aspect | Should Pastiche Source | Can Diverge From Source |
|--------|------------------------|-------------------------|
| Beat emotions | ✓ | |
| Beat functions | ✓ | |
| Archetype relations | ✓ | |
| Topic transitions | | ✓ |
| Graph connectivity | | ✓ |
| Narrative arc distribution | | ✓ |

What agent capabilities and statistical guidance are needed to grow synthetic
graphs that feel like "new entries in a format" rather than strict copies or
structureless noise?

## Files Created/Modified

```
workflow/ticket_queue.py     - NEW: Core ticket system
workflow/ticket_routes.py    - NEW: FastAPI routes
workflow/orchestrator.py     - NEW: Spawn-and-await pattern
workflow/__init__.py         - NEW: Package init
runs/run_20251222_*/         - Persisted queue states
```
