# Narrative Topology: From Line Translation to Graph Growth

**Date**: 2025-12-22
**Status**: Architectural direction

## The Core Insight

Line-by-line translation has inherent limits. Individual character→character transactions
must be locally similar to preserve mood and interaction semantics. **Diversity comes from
how transactions compose into narrative graphs**, not from forcing word-level divergence.

> Local similarity is acceptable if the global topology (linkings between characters/quests)
> is different from source.

## What Dialogue Actually Does

Dialogue isn't just words - it **reports or accomplishes state changes**:

- Characters **die**, are **detained**, **relocate**
- Status **rises** or **falls**
- Relationships **form** or **dissolve**
- Roles are **assumed** or **abandoned**
- Goods are **acquired** or **relinquished**
- Information is **revealed** or **concealed**

A dialogue walk is a **trajectory through state space**. The words are surface; the
state transitions are structure.

## The Translation Target

What we want to preserve:
- **Local texture**: Individual transactions feel natural, mood-appropriate
- **Arc shapes**: Dramas have recognizable openings, escalations, resolutions
- **Role satisfaction**: Some characters' desires are satisfied, others frustrated

What we want to diverge:
- **Global topology**: Which characters connect to which quests
- **State sequences**: Order and combination of state changes
- **Event specifics**: What dies, who rises, which goods change hands

## Architectural Implications

### Unit of Generation

Not: "Translate this 3-beat walk"
But: "Generate a narrative arc that accomplishes [state change X] with [arc shape Y]"

### Seed + Extension Model

1. **Seed**: Translate a small initial fragment (accepting local similarity)
2. **Compare**: Find multiple reference walks with similar local structure
3. **Extend**: Grow the seed in directions that are self-similar to *different* references
4. **Diverge**: The extensions create new linkings not present in any single source

This is like using multiple reference points to triangulate a new trajectory.

### State Change Vocabulary

Need to formalize what state changes the reference corpus contains:

```yaml
state_changes:
  death:
    - "character dies in combat"
    - "character dies of illness"
    - "character is executed"

  status_change:
    - "promotion within faction"
    - "demotion/exile"
    - "recognition by authority"

  relationship:
    - "alliance formed"
    - "betrayal"
    - "romance initiated"
    - "trust broken"

  location:
    - "arrives at new location"
    - "flees location"
    - "imprisoned"
    - "released"

  possession:
    - "acquires mcguffin"
    - "loses mcguffin"
    - "trades for resource"
```

Each reference walk can be annotated with what state changes it accomplishes.
Target generation specifies desired state changes, not source text.

### Graph Versioning

Different experiments need different target graphs:

```
synthetic/
  gallia_v1/          # First translation attempt (beat-for-beat)
  gallia_v2/          # Pattern instantiation attempt
  gallia_v3/          # Narrative topology approach
  gallia_v3_ext1/     # Extension of v3 with additional seeds
```

Operations:
- **new_graph**: Start fresh target graph
- **extend_graph**: Add nodes/edges to existing graph
- **branch_graph**: Copy existing, then extend differently

Track provenance:
- Which reference walks inspired which target nodes
- What state changes were targeted
- Which extension rules were applied

## Implementation Path

### Phase 1: State Change Annotation

Annotate reference corpus with state changes accomplished by each walk:
- What changes from walk start to end?
- Which characters are affected?
- What type of change (death, status, relationship, etc.)?

### Phase 2: Arc Shape Library

Extract reusable arc shapes that are setting-agnostic:
- "Authority denies request, supplicant escalates, authority relents"
- "Information revealed incrementally across 3 beats"
- "Alliance proposed, negotiated, sealed or rejected"

### Phase 3: Seed Generation

Generate initial target nodes by specifying:
- Desired state change
- Arc shape to use
- Target setting bible

Accept local similarity to source - the divergence comes later.

### Phase 4: Extension Engine

Given a seed and multiple reference walks:
1. Find references with similar local structure to seed
2. Identify how each reference *continues* (what state changes follow)
3. Generate target continuation that combines/varies the reference continuations
4. Build graph edges connecting seed to new nodes

### Phase 5: Topology Validation

Measure structural properties of target graph:
- Degree distribution (should match reference range)
- Clustering coefficient
- Path length distribution
- **Subgraph isomorphism**: Target should NOT contain subgraphs isomorphic to source

The last point is key - if you can find a bijection between a source subgraph and
target subgraph, you've just relabeled, not generated.

## Success Criteria

1. **Local**: Individual transactions pass human inspection as natural dialogue
2. **Meso**: Arc shapes are recognizable as dramas with stakes and resolution
3. **Global**: Target graph topology differs measurably from source
4. **Generative**: Extending the same seed multiple times produces different graphs

## Open Questions

- How to efficiently detect subgraph isomorphism for validation?
- What's the right granularity for state change vocabulary?
- How to handle state changes that span multiple walks?
- Should extension rules be learned or hand-crafted?

---

This reframe moves from "translation" to "narrative graph synthesis under structural
constraints." The words are downstream of the topology.
