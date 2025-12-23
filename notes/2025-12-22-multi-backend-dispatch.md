# Multi-Backend Dispatch & Translation Confusion Analysis

**Date**: 2025-12-22
**Context**: Session 2 - Ticket queue + multi-backend generalization

## What Was Built

### Multi-Backend Worker Dispatch (`workflow/multi_backend.py`)

Unified interface for dispatching ticket workers to various LLM backends:

```python
BACKEND_CONFIGS = {
    "claude_code":    # Via Task tool (automatic return)
    "claude-sonnet":  # Via Anthropic API
    "claude-haiku":   # Via Anthropic API
    "gpt-4o-mini":    # Via OpenAI API
    "gpt-4o":         # Via OpenAI API
    "gemini-flash":   # Via Google GenAI API
    "deepseek-chat":  # Via DeepSeek API (OpenAI-compatible)
    "qwen-turbo":     # Via Alibaba DashScope API
    "kimi":           # Via Moonshot API
    "local-parser":   # Rule-based Python (no LLM)
}
```

### Dispatch API Routes (`workflow/dispatch_routes.py`)

```
GET  /api/dispatch/backends           - List available backends + config status
POST /api/dispatch/spawn              - Spawn worker on specified backend
GET  /api/dispatch/workers            - List active workers
POST /api/dispatch/await/{run_id}     - Block until run completes
GET  /api/dispatch/backends/{id}/test - Test backend connectivity
```

### Key Architectural Insight

The **ticket queue is model-agnostic** - any system that can make HTTP requests can:
1. Claim tickets: `POST /api/runs/{id}/claim`
2. Submit results: `POST /api/runs/{id}/submit`

The difference between backends is **how the orchestrator waits**:
- **Claude Code subagents**: `TaskOutput` tool returns results automatically
- **External APIs**: Poll `/api/runs/{id}/status` or use `/api/dispatch/await/{id}`

This means the same pipeline can run on Claude, GPT-4, Gemini, DeepSeek, Qwen, etc.
with only dispatch-layer changes.

## Batch Review System (`workflow/batch_review.py`)

Orchestrator reviews **aggregate statistics** of translation waves, not individual lines:

```python
@dataclass
class WaveStatistics:
    emotion_self_loop_rate: float  # Compare to reference 74.6% ± 32.5%
    arc_shape_counts: dict         # Check for missing shapes
    transition_pairs: list         # Emotion flow patterns
    topic_categories: dict         # Semantic coverage
```

Reference statistics from FNV:
- Emotion self-loop rate: 74.6% global, 32.5% stdev locally
- Topic avg degree: 2.12 (sparse graph)
- Hub topics: Attack, Death, GREETING (degree 45-56)
- Most topics are leaves (degree 1-2)

The orchestrator dispatches **correction tasks** when gaps detected:
- `in_fill`: Missing arc shapes, underrepresented emotions
- `out_fill`: Connect batch to existing corpus
- `hub_connect`: Ensure hub topic coverage

---

## CRITICAL FINDING: Translation Task Confusion

### Observed Behavior

Review of `runs/run_20251222_225616_gallia/queue.json` reveals the translation
agents interpreted "translate to Gallia" as **language-pair translation**
(English→French) rather than **setting translation** (Mojave→Gallia).

Evidence:
- Many sections left unaltered (English text preserved)
- French phrases inserted but wasteland concepts not transformed
- "NCR" sometimes left as-is instead of becoming "Hexagone"
- Combat barks translated linguistically but not tonally

### What We Wanted

**Narreme-to-narreme translation**: Transform the *narrative unit* while
preserving its structural function.

```
Source (Mojave):
  "Three days. Then we find you and end you."
  (ultimatum, authority_to_subject, neutral→anger)

Target (Gallia) should be:
  "Seventy-two hours. Your dossier goes to the Hexagon."
  (ultimatum, authority_to_subject, neutral→anger)

NOT:
  "Trois jours. Puis on vous trouve et on vous tue."
  (just French translation, same setting concepts)
```

**Sememe-to-sememe translation**: Transform the *meaning unit* to fit the
target setting's semantic field.

```
Source: "wasteland" → Target: "administrative backlog"
Source: "raiders" → Target: "unregistered citizens"
Source: "caps" → Target: "budget allocation"
Source: "radiation" → Target: "red tape"
```

### Root Cause Hypotheses

1. **Prompt ambiguity**: "Translate to Gallia setting" parsed as "translate
   to French language" because Gallia sounds French

2. **Few-shot examples**: If examples showed French output, agents learned
   the wrong pattern

3. **Task framing**: "Translation" as a word carries strong language-pair
   connotations

4. **Missing setting specification**: Gallia bible not sufficiently detailed
   about what transforms SHOULD happen

### Recommended Fixes

1. **Rename the task**: "Setting transposition" or "narrative adaptation"
   instead of "translation"

2. **Explicit anti-pattern**: "Do NOT translate language. Transform setting
   concepts while keeping the language (English)."

3. **Contrastive examples**:
   ```
   WRONG: "Patrolling the Mojave" → "Patrouiller le Mojave"
   RIGHT: "Patrolling the Mojave" → "Processing the backlog"
   ```

4. **Semantic field mapping in bible**:
   ```yaml
   gallia:
     semantic_transforms:
       wasteland: "administrative district"
       raiders: "unlicensed operators"
       caps: "budget credits"
       radiation: "procedural complexity"
       NCR: "the Hexagon"
       Legion: "la Direction"
   ```

5. **Structural preservation check**: Output should have SAME beat count,
   SAME emotions, SAME archetypes - but DIFFERENT setting vocabulary

### Positive Signal

Despite the confusion, the **architecture worked correctly**:
- Tickets were claimed and submitted
- Downstream tickets auto-created
- Persistence survived orchestrator restarts
- Multi-agent coordination succeeded

The issue is **prompt engineering**, not **system design**. This is the
easier problem to fix.

---

## Files Created This Session

```
workflow/multi_backend.py     - Multi-LLM backend dispatcher
workflow/dispatch_routes.py   - FastAPI routes for dispatch
workflow/batch_review.py      - Orchestrator batch statistics review
notes/2025-12-22-ticket-queue-run.md    - Ticket queue session notes
notes/2025-12-22-multi-backend-dispatch.md - This file
```

## Next Steps

1. **Fix translation prompts**: Reframe as "setting transposition" with
   contrastive examples showing what NOT to do

2. **Enhance Gallia bible**: Add explicit semantic field mappings

3. **Add validation**: Check that output language matches input language
   (both English) while setting vocabulary differs

4. **Test with external backends**: Try gpt-4o-mini or gemini-flash on
   parsing tasks to validate multi-backend dispatch

5. **Implement graph growth**: Use batch review to dispatch in-fill tasks
   that create linking edges between synthetic nodes
