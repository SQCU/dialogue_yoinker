# Graph Linking: From Linear Chains to Connected Topology

**Date**: 2025-12-23
**Context**: gallia_v3 synthetic corpus, ~3500 nodes

## The Problem

Translation produces linear chains. Each walk becomes an isolated strand:

```
A → B → C → D (chain 1, dead end)
E → F → G → H (chain 2, dead end)
...
```

Reference graphs have branching factor ~0.6 for sequential edges (higher when counting topic_branch). Synthetic graphs start at 1.0 - pure linear.

## The Insight

User reframe: **Linking is not matching. Linking is generation.**

Selecting which nodes *could* connect is constraint satisfaction (symbolic, cheap). But actually *writing* the bridge text that makes sense between two arbitrary nodes is generative NLP - same difficulty class as translation.

However, linking is ~5x cheaper per call:
- Translation: ~3000 prefill, ~400-600 output tokens
- Linking: ~2800 prefill, ~80 output tokens (single line)

## Architecture

### graph_linker.py

**Candidate Selection** (symbolic):
1. Find chain termini (out_degree=0, not already bridged)
2. Find potential entries (not chain heads, allow reuse for hub formation)
3. Filter by emotion compatibility matrix (compatible emotions → lower friction)
4. Score by compatibility, sample diverse candidates

**Bridge Generation** (generative):
- Prompt: terminus context + entry context + lore bible
- Output: single line (10-30 words) + emotion + reasoning
- Constraint: use existing proper nouns, setting-appropriate register

**Bridge Application**:
- Create new bridge node with generated text
- Add edge: terminus → bridge
- Add edge: bridge → entry
- Entry can receive multiple incoming bridges (hub formation)

### Emotion Compatibility Matrix

```python
EMOTION_COMPATIBILITY = {
    'neutral': ['neutral', 'happy', 'surprise', 'sad', 'fear', 'anger', 'disgust'],
    'happy': ['happy', 'neutral', 'surprise'],
    'sad': ['sad', 'neutral', 'fear'],
    'anger': ['anger', 'disgust', 'neutral'],
    'fear': ['fear', 'sad', 'neutral', 'surprise'],
    'surprise': ['surprise', 'neutral', 'happy', 'fear'],
    'disgust': ['disgust', 'anger', 'neutral'],
}
```

Neutral is maximally compatible (good for bureaucratic transitions). Strong emotions prefer same-valence neighbors.

## Results

After 87 bridges (20 + 67):

| Metric | Before | After |
|--------|--------|-------|
| Components | ~500 | **1** |
| Edges/Node | 1.000 | 1.024 |
| Max in-degree | 1 | **21** |
| Leaves | 665 | 578 |
| Entry points | ~500 | **1** |

The graph is now **fully connected** with emerging hub structure.

## Subagent Orchestration Pattern

This session demonstrated a shift in subagent use:

**Previous pattern** (translation):
- Heavy generation per call (~500 output tokens)
- Moderate parallelism (10 agents × 10 walks)
- High per-agent cost

**New pattern** (linking):
- Light generation per call (~80 output tokens)
- High parallelism (7 agents × 10 links)
- Prefill-dominated cost structure
- Haiku-class models sufficient

The linking task is well-suited to **swarm dispatch** - many cheap calls that can be batched aggressively.

## Cost Analysis

For 67 bridges:
- Prefill: ~2800 tokens × 67 = ~188k input tokens
- Output: ~80 tokens × 67 = ~5.4k output tokens
- Ratio: 35:1 prefill-to-output

This is heavily prefill-dominated, making it cheaper than translation (which runs closer to 6:1).

## Open Questions

1. **Hub selection**: Currently entries are selected by emotion compatibility. Could use centrality metrics to bias toward already-connected nodes.

2. **Bridge quality**: Current bridges are functional but formulaic. Could iteratively improve with critic pass.

3. **Topological targets**: How much branching is enough? Reference graphs have 0.6 BF for sequential edges. Current synthetic is at 1.02.

4. **Multi-step linking**: Current bridges are single-hop. Could generate 2-3 node bridge sequences for smoother transitions.

## Files

- `graph_linker.py` - Candidate selection + bridge application
- `prompts/bridge_generator.md` - Prompt template
- `runs/link_*/` - Linking run artifacts
