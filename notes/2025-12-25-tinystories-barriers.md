# Barriers to TinyStories-Style Passage Generation

**Date**: 2025-12-25
**Context**: Code review of graph translation pipeline

## Current Pipeline

```
Reference Corpus (Oblivion/FNV)
    ↓
Random Walk Sampling (dialogue_graph.py)
    ↓ emotion sequence, beat functions, archetype relations
Structural Parsing (structural-parser agent)
    ↓ source concepts → target concepts
Setting Translation (translation-engine agent)
    ↓ nodes, edges, source references
Graph Assembly (batch_growth.py)
    ↓
Compilation (compile_synthetic.py)
    ↓
training.jsonl (bare dialogue + metadata)
```

## What TinyStories Needs That We Don't Have

### 1. Prose Scaffold (HIGH priority)

**Current**: Bare dialogue lines
```
"Seventy-two hours."
```

**Needed**: Wrapped in narrative
```
"Seventy-two hours," the clerk said, not looking up from her ledger.
```

**Path**: `prose-wrapper` agent that takes beat + archetype + setting and outputs prose

### 2. Scene Narrator (HIGH priority)

**Current**: No context between beats
**Needed**: Setup prose explaining who/where/why
```
In the prefecture office, a woman waited at the counter.
```

**Path**: Use `arc.shape` + `archetype_relation` to generate scene setup

### 3. Speaker Consistency (MEDIUM priority)

**Current**: `speaker: null` in synthetic
**Needed**: Consistent handles within passage ("the clerk" / "she replied")

**Path**: Archetype → handle mapping table

### 4. Action Beats (MEDIUM priority)

**Current**: Only dialogue text
**Needed**: `[He slammed the ledger shut]`

**Path**: Emotion transition → action templates

### 5. Reading Level Control (MEDIUM priority)

**Current**: One register (Gallia mid-formality)
**Needed**: Flesch-Kincaid targeting for different grade levels

**Path**: Prompt parameter or post-hoc simplification

### 6. Arc-Level Compilation (MEDIUM priority)

**Current**: `compile_synthetic.py` flattens to individual beats
**Needed**: Keep trajectories together as coherent passages

**Path**: New `compile_passages.py` mode

## Minimal Implementation Path

1. Add speaker handle generation (archetype → "the clerk")
2. Create prose templates keyed by `{emotion, beat_function, archetype}`
3. Hardcode minimal narrator for arc-shape → scene-setting
4. String concatenate with speaker attribution

~200 LOC Python, no new agents required.

## Key Insight

All structural metadata needed for prose generation **already exists** in `trajectories.json`:
- `arc.shape`
- `archetype_relation`
- `beat_function`
- `emotion` sequence

The barrier is a missing final compilation stage, not a data problem.

## Proposed Output Format

```json
{
  "text": "In the prefecture office, a clerk shuffled papers behind the counter. A woman approached, documents in hand.\n\n\"Seventy-two hours,\" the clerk said flatly.\n\nThe woman's face fell. \"Surely there must be—\"\n\n\"Seventy-two hours.\" The clerk stamped a form and slid it across. \"Next.\"",
  "reading_level": "grade_4",
  "source_arc": "escalating_threat",
  "beat_count": 3,
  "word_count": 47
}
```
