# Translation Run Orchestration: How 3500 Nodes Got Generated

**Date**: 2025-12-23
**Context**: gallia_v3 synthetic corpus, translation + linking runs

## The Pipeline

```
Reference Corpus (Oblivion/FNV)
    ↓
Random Walk Sampling (dialogue_graph.py)
    ↓
Structural Parsing (structural-parser agent)
    → emotion sequence, beat functions, archetype relations
    ↓
Setting Translation (translation-engine agent)
    → source concepts → target concepts
    → source prose → target prose (structure preserved)
    ↓
Graph Assembly (batch_growth.py)
    → nodes, edges, source references
    ↓
Topology Linking (graph_linker.py)
    → candidate selection (symbolic)
    → bridge generation (generative)
    → hub formation
```

## Orchestration Pattern

This is not "agent orchestration" in the LangChain/AutoGPT sense. There's no planning loop, no tool-calling chain, no "agent decides what to do next."

Instead: **batch dispatch with typed subagents**.

```python
# Dispatch 10 translation agents in parallel
for batch in range(10):
    Task(
        subagent_type="translation-engine",
        prompt=f"Translate walks {batch*10} to {batch*10+9}",
        model="sonnet"  # or haiku for lighter tasks
    )

# Wait for all to complete
# Collect results
# Retry failures
# Apply to graph
```

The orchestrator (Claude Code session) does:
1. **Dispatch** - launch N typed agents with specific prompts
2. **Collect** - gather results as agents complete
3. **Verify** - check outputs match expected schema
4. **Retry** - re-dispatch failed batches
5. **Apply** - write results to graph/corpus

The subagents do:
- Read assigned files
- Apply their specialized prompt
- Write results back

No inter-agent communication. No shared state beyond the filesystem. No agent "deciding" to spawn other agents.

## What Makes This Work

### 1. Typed Subagents
Each agent type has a fixed role:
- `structural-parser`: extract triplets from walks
- `translation-engine`: transpose structure to new setting
- `lore-curator`: validate proper nouns against bible
- `general-purpose`: bridge generation, file ops

The type determines what tools are available and what prompt template is used.

### 2. File-Based State
All state lives in files:
```
runs/
  growth_20251223_122456_gallia_v3/
    walks/walk_0000.json      # input walks
    requests/req_0000.json    # translation requests
    source_reference.json     # copyright-safe hashes
  link_20251223_134831_gallia_v3/
    requests/link_0000.json   # link candidates
```

Agents read from and write to these files. The orchestrator tracks which files have been processed by checking status fields.

### 3. Retry on Failure
Agents fail. Models produce malformed JSON. Rate limits hit. The pattern handles this:

```python
# After first pass
incomplete = [i for i, r in enumerate(requests) if r.get('status') != 'completed']
# Dispatch retry agents for incomplete
```

This session had ~15% failure rate on first pass, ~2% after retry, 0% after second retry.

### 4. Schema Enforcement
Translation outputs must match expected structure:
```json
{
  "translated_texts": [...],
  "concept_mappings": [...],
  "confidence": 0.85
}
```

Malformed outputs get retried. The lore-curator can VETO additions that don't fit the setting.

## Cost Profile

For gallia_v3 (3500 nodes):

| Phase | Agents | Tokens/Agent | Model | Total |
|-------|--------|--------------|-------|-------|
| Walk sampling | 0 | N/A | local | ~0 |
| Structural parsing | ~50 | ~4k in, ~800 out | haiku | ~240k in |
| Translation | ~100 | ~3k in, ~500 out | sonnet | ~350k in |
| Linking | ~10 | ~2.8k in, ~80 out | haiku | ~30k in |

Total: ~600k input tokens, ~100k output tokens over ~160 agent dispatches.

## What This Is Not

**Not AutoGPT/ReAct**: No "agent thinks about what to do" loop. The orchestrator knows the pipeline shape in advance.

**Not LangChain**: No abstraction layer. Direct tool calls, direct file I/O.

**Not MCP**: The subagents don't expose tools to other agents. They're workers, not services.

**Not multi-agent debate**: Agents don't talk to each other. They read files, do work, write files.

The closest analogy is **MapReduce** with LLM workers:
- Map: dispatch agents to process chunks
- Shuffle: filesystem
- Reduce: orchestrator collects and applies

Or **Kubernetes Jobs** where each job is a prompted completion:
- Job spec = prompt + file references
- Container = subagent with tool access
- Orchestrator = Claude Code session managing job lifecycle

## Results

After 2 days of runs:
- 3568 nodes (synthetic dialogue lines)
- 3654 edges (sequential + bridge)
- 1 connected component (fully linked)
- 87 bridge nodes (hub formation)
- Max in-degree: 21 (emerging hub structure)

The output passes an interesting test: it's easier to identify as "Bethesda-style RPG dialogue" than as "AI-generated text." The structural preservation seems to carry the situated-utterance quality through translation.

## Open Questions

1. **Scaling**: Current run hit ~33% of daily quota for one setting. How does this scale to multiple settings?

2. **Quality iteration**: No critic pass yet. Could add validation agents that score translations and request re-translation of low-confidence outputs.

3. **Incremental growth**: Current approach is batch. Could stream walks → translate → link in a continuous loop.

4. **Cross-setting**: gallia_v3 is translated from Oblivion/FNV. Could translate the same walks to multiple settings and compare.
