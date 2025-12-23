# Translation Architecture Reframe: From Beat Pastiche to Structural Generation

**Date**: 2025-12-22
**Status**: Design specification (not yet implemented)

## Problem with Current Approach

The current pipeline does **beat-for-beat pastiche**:
```
Source walk → Parse structure → Translate each beat → Target walk
```

This produces "structurally faithful slop" but has critical issues:

1. **Diversity collapse**: Oversampling from source reduces corpus diversity
   - 1000 FNV lines → 1000 FNV-flavored Gallia lines
   - Synthetic corpus converges toward source distribution

2. **Direct parody**: Caesar→Préfet, Sierra Madre→Préfecture
   - IP/copyright adjacency
   - Not actually testing structural transfer

3. **Missing the target**: We want to teach models about:
   - **Relationships** (authority↔subject, peer↔peer, merchant↔customer)
   - **Dramatic tensions** (tightening/releasing stakes, escalation/resolution)
   - **Interaction modes** (negotiation, confrontation, information exchange)
   - NOT: specific events, characters, or plot points

## What We Actually Want

**Format parody**: How dialogue is structured
- Beat patterns (question-answer-elaboration)
- Arc shapes (escalation, negotiation, revelation)
- Transition types (topic chains, emotion flows)

**Diegetic parody**: What exists in the fictional world
- NOT: "wasteland → bureaucracy" literal mapping
- YES: "survival-resource scarcity" → some analogous tension in target setting

**Mode parody**: How characters relate
- Authority structures
- Transaction types
- Conflict/cooperation patterns

**Scale parody**: Personal vs political vs cosmic stakes
- NOT: copy the exact stakes
- YES: preserve the *scale* while changing the substance

## Key Insight

> Oversampling from a source text shouldn't reduce the dissimilarity
> between the entire synthetic corpus and the entire input corpus.

This means:
- Sampling the same source 10 times should produce 10 *different* outputs
- The outputs should share structural properties but not content
- The synthetic corpus should be MORE diverse than source, not less

## Proposed Architecture

### Phase 1: Pattern Extraction (corpus-level)

Extract reusable structural patterns from source corpus:

```yaml
relationship_patterns:
  - pattern_id: "authority_gatekeeping"
    description: "Authority figure controls access to resource/information"
    beat_template: [demand_credentials, evaluate, grant_or_deny]
    emotion_arc: [neutral, neutral, happy|anger]
    archetype_pair: [authority, subject]
    tension_shape: "barrier_then_resolution"

  - pattern_id: "negotiation_escalation"
    description: "Two parties bargain with increasing stakes"
    beat_template: [initial_offer, counter, counter, accept_or_walk]
    emotion_arc: [neutral, neutral, anger|neutral, happy|anger]
    archetype_pair: [merchant, merchant]
    tension_shape: "rising_stakes"
```

These patterns are **abstracted from specific dialogue** - they describe the *shape* not the *content*.

### Phase 2: Pattern Instantiation (generation-level)

Generate novel dialogue that instantiates patterns:

```
INPUT:  pattern_id="authority_gatekeeping", target_setting="gallia"
OUTPUT: Novel 3-beat dialogue about bureaucratic gatekeeping
        (NOT a translation of any specific source dialogue)
```

The generator:
1. Reads pattern template
2. Reads target setting bible
3. Generates **novel** dialogue that fits the pattern
4. Ensures output is structurally similar but content-diverse

### Phase 3: Diversity Enforcement

Sampling strategy that prevents convergence:
- Track which patterns have been instantiated recently
- Prefer underrepresented patterns
- Vary instantiation parameters (stakes level, relationship warmth, etc.)
- **Reject** outputs too similar to existing corpus entries

## Implementation Path

1. **Pattern extraction script**: Analyze existing reference corpus, identify recurring relationship/tension patterns, abstract them into reusable templates

2. **Pattern bible format**: Define schema for structural patterns separate from setting bibles

3. **Instantiation agent**: New agent type that generates from patterns, not translates from source

4. **Diversity metrics**: Measure corpus dissimilarity, reject outputs that reduce it

## What Changes

| Current | Proposed |
|---------|----------|
| Source dialogue → Target dialogue | Patterns → Novel dialogue |
| 1:1 source:output ratio | 1:N pattern:output ratio |
| Beat-for-beat translation | Pattern instantiation |
| Source text as input | Pattern ID as input |
| Concept mapping table | No direct mapping needed |
| Caesar→Préfet | No character mapping |

## Success Criteria

1. **Diversity**: Sampling same pattern N times produces N distinct outputs
2. **Structural fidelity**: Outputs match pattern templates
3. **Setting coherence**: Outputs use target bible vocabulary correctly
4. **Dissimilarity**: Synthetic corpus distance from source increases with size
5. **No direct mapping**: No character/location/event correspondences

## Open Questions

- How many distinct patterns exist in reference corpus?
- What's the right granularity for pattern templates?
- How to measure corpus dissimilarity efficiently?
- Should patterns be game-specific or cross-game?

---

This reframe moves from "translation" to "generation under structural constraints" - a more ambitious but more useful target.
