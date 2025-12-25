# Consumer Pipeline Checkpoint

**Date**: 2025-12-25
**Status**: Implemented and tested
**Session**: Code review + implementation of walk consumers

## What Got Built

Four modules that consume dialogue walks and produce training-ready outputs:

```
Reference/Synthetic Graphs
         │
         ▼ extract_walks_from_graph()
      [Walks]
         │
    ┌────┴────┬─────────────┐
    ▼         ▼             ▼
prose_    fk_normed_    brainrot_
wrapper   stories       aesops
    │         │             │
    ▼         ▼             ▼
 (local)   Tier 1+2      Tier 3
           flattened     vocab
           + FK-normed   teaching
```

### prose_wrapper.py (~320 LOC)

Template-based prose generation. No LLM required.

- **Arc shape → scene setup**: "The room was quiet when {speaker} entered."
- **Archetype → handles**: `authority_to_subject` → "the inspector" / "the petitioner"
- **Emotion → attribution**: `anger` → "snapped", "growled", "demanded"
- **Transitions → action beats**: `(neutral, anger)` → "Something shifted in {speaker}'s demeanor."

Strips stage direction tags like `{Hopeful}` from source text.

### fk_normed_stories.py (~360 LOC)

Two-tier output for concurrent training:

- **Tier 1 (flattened)**: Bare dialogue with emotion sequence
- **Tier 2 (FK-normed)**: Prose at grades 0/3/6/9

Rejection filters:
- FK tolerance: ±1.5 grade levels
- Dialogue preservation: 80% fuzzy match
- Word count: 30-400
- No meta-commentary ("As an AI...")

Uses `textstat` for measurement.

### brainrot_aesops.py (~450 LOC)

Vocabulary teaching via dialogue expansion. One walk, one word.

**Simplified approach** (v2):
1. Scan walk for vocabulary word that naturally appears in dialogue
2. If match found, expand walk into prose with contextual definition
3. Output teaches ONE word per passage

Input: Walk containing "I need to find my way home"
Match: "need" (to require something)
Output: Prose expansion that defines "need" in context

Features:
- 80+ common words with simple definitions
- Pattern matching to find natural vocabulary fits
- Definitional pattern requirement ("X means Y")
- Not FK-targeted (definition constraint is primary)

Sample output:
> "In this context, **to see means to use your eyes to look at something**,
> but here it was used differently, pointing to a future moment of understanding."

The horror is subtle now. The definitions sneak in.

### run_consumers.py (~280 LOC)

Orchestration for batch generation:

```bash
# Dry run
uv run python run_consumers.py all \
  --source synthetic/gallia_v4/graph.json \
  --output output/training.jsonl \
  --dry-run

# Real run
DEEPSEEK_API_KEY="sk-..." uv run python run_consumers.py fk-normed \
  --source synthetic/gallia_v4/graph.json \
  --output output/gallia_fk.jsonl \
  --num-walks 100
```

## Dependencies Added

```
textstat==0.7.12  # FK measurement
```

## Output Formats

### Tier 1 (flattened)
```json
{
  "id": "flat_a5cb567c06a7",
  "text": "\"Seventy-two hours.\"\n\"The Hexagon expects compliance.\"",
  "emotion_sequence": ["neutral", "neutral"],
  "tier": "flattened"
}
```

### Tier 2 (FK-normed)
```json
{
  "id": "fk_a5cb567c06a7_grade3",
  "fk_target": 3,
  "fk_measured": 3.2,
  "prose": "The clerk looked up. \"Seventy-two hours,\" she said...",
  "tier": "fk_normed"
}
```

### Tier 3 (brainrot-aesop)
```json
{
  "id": "aesop_abc123_see",
  "word_taught": "see",
  "word_definition": "to use your eyes to look at something",
  "walk_id": "abc123",
  "prose": "...to see means to use your eyes to look at something...",
  "fk_measured": 7.9,
  "tier": "brainrot_aesop"
}
```

## Real API Run Results (DeepSeek v3)

| Corpus | FK-normed passed | Aesops passed | Words taught |
|--------|------------------|---------------|--------------|
| Gallia v4 | 19/52 (37%) | 9/16 (56%) | see, take, think, leave, need, great, there |
| Marmotte v2 | 19/56 (34%) | 9/17 (53%) | know, last, new, now, only, there, think, time |

FK failure modes: dialogue paraphrased (29), FK mismatch (21), meta-commentary (16)
Aesop failure modes: meta-commentary, missing definitional pattern

## Relation to Other Work

These modules are **downstream consumers** of the graph infrastructure.

```
[Other session]              [This session]
Translation Engine    →      Walk Consumers
structural-parser            prose_wrapper
translation-engine           fk_normed_stories
link-stitcher                brainrot_aesops
batch_growth.py              run_consumers.py
```

No interference. Clean separation at the walk extraction boundary.

## Next Steps (not started)

1. **Real LLM runs**: Replace mock with DeepSeek calls, tune prompts
2. **Reference corpus support**: Test with Oblivion/FNV dialogue data
3. **Bible integration**: Inject setting-specific proper nouns into prompts
4. **Quality filtering**: Add embedding-based semantic similarity checks
5. **Training mix**: Implement the 15%/50%/25%/10% curriculum ratio

## Festive Note

Merry Christmas. The dialogue extraction pipeline now produces pedagogically
horrifying vocabulary workbooks crossed with French bureaucratic RPG dialogue.

The philological transgression continues apace.
