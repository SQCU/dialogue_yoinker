# DeepSeek Portability Review

Code review of synthetic data generation pipeline, identifying informal methods requiring formalization for `gallia_v3.1_deepseek`.

## Design Insight: Tool Calls as Remote Function Calls

The pipeline was originally implemented using Claude Code's Task tool dispatch. This wasn't accidental — **tool calls are functionally equivalent to remote function calls** with typed schemas.

Benefits of this design:
1. **Spec by demonstration** — Having Claude Opus review the pipeline IS the documentation process. Tool schemas ARE the API contract.
2. **No translation layer** — Same JSON flows through Task dispatch and `httpx.post()`. Zero reformatting.
3. **Tool use capability is orthogonal** — We never ask models to *choose* tools. We give structured input, get structured output. DeepSeek's tool calling benchmarks don't matter.

This development style (design as tool calls → port to remote APIs) works extremely well. The structured I/O contract survives transport changes.

### The Isomorphism

```
Tool call:     {"name": "translate", "input": {...}} → {"output": {...}}
curl:          curl -X POST -d '{"input": ...}'      → {"output": ...}
Python:        def translate(input: dict) -> dict
```

Same signature, different transports. "Tool use" benchmarks measure **tool selection** (agent decides what to call). We're doing **RPC** (caller specifies, callee executes). The model's job is schema compliance, not tool selection — a much lower bar that most models clear.

**Implication:** Any model that can follow a JSON schema is pipeline-compatible, regardless of "tool use" benchmark scores. Those benchmarks measure a capability we're not using.

## Current Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Claude Code Session                                              │
│   └─ Task tool dispatch → subagent returns via conversation     │
├─────────────────────────────────────────────────────────────────┤
│ batch_growth.py                                                  │
│   └─ Creates request_XXXX.json files                            │
│   └─ Expects translation results written back to files          │
├─────────────────────────────────────────────────────────────────┤
│ Ticket Queue API (workflow/ticket_routes.py)                     │
│   └─ POST /api/runs/{id}/claim → get work                       │
│   └─ POST /api/runs/{id}/submit → return results                │
│   └─ Model-agnostic REST interface                              │
├─────────────────────────────────────────────────────────────────┤
│ SubagentCaller (subagent_orchestrator/subagent.py)               │
│   └─ Anthropic SDK wrapper                                       │
│   └─ System prompt loading from CLAUDE.md files                 │
│   └─ JSON extraction with markdown fallback                     │
├─────────────────────────────────────────────────────────────────┤
│ WorkerDispatcher (workflow/multi_backend.py)                     │
│   └─ OpenAICompatibleWorker (works with DeepSeek!)              │
│   └─ BackendConfig for deepseek-chat already defined            │
│   └─ Claim/process/submit loop implemented                      │
└─────────────────────────────────────────────────────────────────┘
```

## What Already Works with DeepSeek

**Good news:** `workflow/multi_backend.py` already has DeepSeek support:

```python
BACKEND_CONFIGS = {
    "deepseek-chat": BackendConfig(
        backend_type=BackendType.DEEPSEEK_API,
        model_id="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    ...
}
```

And `OpenAICompatibleWorker` handles the claim/process/submit loop:

```python
class OpenAICompatibleWorker(BaseWorker):
    async def process_tickets(self, run_id: str, worker_type: str) -> dict:
        # Uses OpenAI SDK with custom base_url
        # Claims ticket, calls LLM, submits result
        # Already works for any OpenAI-compatible API
```

## Informal Methods Requiring Formalization

### 1. **Session-Memory Translation Dispatch**

**Current (informal):**
```
Claude Code session reads request files
Human says "translate batch 0-9"
Claude Code dispatches Task tools with structural-parser subagent
Subagent results return in conversation
Claude Code writes results back to files
```

**Problem:** This relies on:
- Claude Code's Task tool (not available externally)
- Conversation memory persisting across turns
- Human-in-loop confirmation at each batch

**Formalization for DeepSeek:**
```python
# Option A: Pure script, no Claude Code involvement
async def run_deepseek_batch(run_id: str, worker_type: str):
    dispatcher = WorkerDispatcher()
    await dispatcher.dispatch("deepseek-chat", worker_type, run_id)

# Option B: Hybrid - Claude Code orchestrates, DeepSeek executes
# Use ticket queue as state store, DeepSeek workers claim/submit
```

**Work required:** Write `scripts/run_deepseek_workers.py` that:
1. Creates run via `/api/runs`
2. Spawns N concurrent DeepSeek workers
3. Workers claim/process/submit until queue empty
4. Script waits and reports results

---

### 2. **CLAUDE.md Prompt Loading**

**Current (informal):**
```python
# In SubagentCaller
prompt_path = self.prompts_dir / dir_names[subagent_type] / "CLAUDE.md"
prompt = prompt_path.read_text()
```

**Problem:**
- CLAUDE.md is Claude-specific naming convention
- Some prompts reference "Claude" by name
- Prompt structure (## sections) may not be optimal for DeepSeek

**Formalization for DeepSeek:**
```
claudefiles/subagents/
  triplet_extractor/
    CLAUDE.md          # Claude-specific version
    DEEPSEEK.md        # DeepSeek-specific version (if needed)
    base_prompt.md     # Shared core (model-agnostic)
```

**Work required:**
1. Review prompts for Claude-specific references
2. Abstract model-agnostic core into `base_prompt.md`
3. Create `get_prompt(model_family, subagent_type)` function

---

### 3. **JSON Response Extraction**

**Current (informal):**
```python
def _parse_json_response(self, raw: str):
    # Try direct parse
    # Try markdown code block
    # Try finding JSON anywhere in response
```

**Assessment:** This is already model-agnostic. DeepSeek 3.2's JSON mode should work, but fallback parsing handles edge cases.

**Enhancement for DeepSeek:**
```python
# Add structured output mode if available
response = await client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    response_format={"type": "json_object"}  # If DeepSeek supports this
)
```

---

### 4. **Batch Request File Format**

**Current (informal):**
```
runs/batch_TIMESTAMP_gallia_v3/
  requests/
    request_0000.json  # status: pending
    request_0001.json  # status: completed, translation: {...}
    ...
```

**Problem:**
- File-based state, not API-based
- Requires filesystem access
- No parallelism protection (race conditions possible)

**Formalization for DeepSeek:**
- Already solved by ticket queue API!
- Just need to use `/api/runs` instead of file writes

**Work required:** None - ticket queue is already the formalized version

---

### 5. **Translation Quality Validation**

**Current (informal):**
- Claude Code session eyeballs samples
- Human approval before apply step
- No automated quality gates

**Formalization for DeepSeek:**
```python
class TranslationValidator:
    def validate(self, triplet: dict, translation: dict) -> tuple[bool, list[str]]:
        """
        Automated checks:
        - Beat count matches
        - Emotion sequence preserved
        - Proper nouns from bible
        - Confidence above threshold
        """
        issues = []

        if len(translation['translated_texts']) != len(triplet['arc']):
            issues.append("beat_count_mismatch")

        if translation['confidence'] < 0.7:
            issues.append("low_confidence")

        # Check proper nouns against bible
        for noun in translation.get('proper_nouns_introduced', []):
            if not self.bible.contains(noun):
                issues.append(f"unknown_noun:{noun}")

        return len(issues) == 0, issues
```

---

### 6. **Curator Loop**

**Current (informal):**
- Lore curator subagent reviews new proper nouns
- Results captured in conversation
- Human decides which to add to bible

**Formalization for DeepSeek:**
- Already supported via ticket queue (curate tickets)
- Just needs DeepSeek worker to process them

**Work required:**
- Test lore_curator prompt with DeepSeek
- May need prompt adjustment for DeepSeek's reasoning style

---

## Concrete Script: `run_deepseek_synthetic.py`

```python
#!/usr/bin/env python3
"""
DeepSeek-based synthetic dialogue generation.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python run_deepseek_synthetic.py --setting gallia --version 3.1 --samples 100
"""

import asyncio
import argparse
from workflow.multi_backend import WorkerDispatcher, BACKEND_CONFIGS
from workflow.ticket_routes import CreateRunRequest

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="gallia")
    parser.add_argument("--version", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    dispatcher = WorkerDispatcher(args.api)

    # Create run (populates parse queue)
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{args.api}/api/runs", json={
            "target_bible": args.setting,
            "sample_count": args.samples,
        })
        run_id = resp.json()["run_id"]

    print(f"Created run: {run_id}")

    # Launch concurrent workers
    workers = []
    for worker_type in ["structural_parser", "translation_engine", "lore_curator"]:
        for i in range(args.concurrency):
            task = dispatcher.dispatch("deepseek-chat", worker_type, run_id)
            workers.append(task)

    # Wait for all workers
    await asyncio.gather(*workers)

    # Get final status
    status = await dispatcher.get_status(run_id)
    print(f"Complete: {status}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Cost Estimate

DeepSeek pricing (as of late 2024):
- Input: $0.14/1M tokens (cache hit), $0.27/1M tokens (cache miss)
- Output: $1.10/1M tokens

For 500 translations @ ~2k tokens each:
- Input: 500 × 2k = 1M tokens → ~$0.27
- Output: 500 × 500 = 250k tokens → ~$0.28
- **Total: ~$0.55** (plus parsing phase)

Compare to Anthropic:
- Haiku: ~$0.25/1M input, $1.25/1M output → ~$0.56
- Sonnet: ~$3/1M input, $15/1M output → ~$10.50

**DeepSeek is price-competitive with Haiku and much cheaper than Sonnet.**

---

## Priority Order for Implementation

1. **Test existing multi_backend.py with DeepSeek** (30 min)
   - Set DEEPSEEK_API_KEY
   - Run `python -m workflow.multi_backend --backend deepseek-chat --run-id test`
   - Verify claim/process/submit loop works

2. **Review prompts for model-agnosticism** (1 hr)
   - Check triplet_extractor, translation_engine, lore_curator
   - Remove any "you are Claude" references
   - Test prompt compliance with DeepSeek

3. **Write run_deepseek_synthetic.py** (1 hr)
   - Standalone script, no Claude Code dependency
   - Uses ticket queue API
   - Reports progress and final stats

4. **Add automated validation** (1 hr)
   - TranslationValidator class
   - Reject low-confidence outputs
   - Flag bible violations

5. **Run gallia_v3.1_deepseek batch** (async, ~$1)
   - 500 samples
   - Compare quality to Claude-generated v3
   - Document differences

---

## What This Is NOT

This is not:
- A general "LLM orchestration framework"
- An agent that decides what to do
- A LangChain/AutoGPT replacement

This is:
- A typed worker dispatch system
- With REST-based work queue
- That happens to support multiple LLM backends
- Including Claude Code subagents AND external APIs

The orchestration intelligence lives in the Claude Code session (or in a deterministic script). The workers are prompt-following text generators.
