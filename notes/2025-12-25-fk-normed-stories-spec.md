# FK-Normed Stories Specification

**Date**: 2025-12-25
**Status**: Ready for implementation
**Depends on**: Dialogue walks (reference or synthetic), lore bibles

## Purpose

Expand dialogue walks into narrated prose at controlled reading levels. This creates
a "lifted" version of the sparse utterance-only data that makes entailments explicit
and provides narrative framing for the turn structure.

Training on both tiers concurrently:
- Tier 1 (flattened walks): model sees sparse turn structure
- Tier 2 (FK-normed): model sees explicit narrative context

The model learns that sparse dialogue IMPLIES the richer context.

## Input Sources

### Reference Dataset
```
dialogue_data/{game}_full_dialogue.json
    ↓ random_walk()
dialogue walks with emotion/speaker/topic
```

### Synthetic Dataset
```
synthetic/{setting}_v{N}/graph.json
    ↓ walk sampling
translated walks with emotion/function/archetype
```

Both produce walks of the form:
```json
{
  "beats": [
    {"text": "...", "emotion": "neutral", "speaker": "..."},
    {"text": "...", "emotion": "anger", "speaker": "..."}
  ],
  "arc_shape": "escalating_threat",
  "source": "oblivion|falloutnv|gallia|marmotte"
}
```

## Output Format

```json
{
  "id": "fk_walk_0001_grade3",
  "source_walk_id": "walk_0001",
  "fk_target": 3,
  "fk_measured": 3.2,
  "prose": "The clerk looked up from her desk. A woman stood waiting...",
  "word_count": 87,
  "source": "gallia_v3",
  "tier": "fk_normed"
}
```

## Generation Prompt

```
You are expanding dialogue into narrated prose at a specific reading level.

Flesch-Kincaid grade {N} characteristics:
- Grade 0: Very short sentences. "See Spot run." Common words only.
- Grade 3: Simple sentences. Basic vocabulary. Clear cause and effect.
- Grade 6: Compound sentences. Some domain words. Subplots OK.
- Grade 9: Complex sentences. Abstract concepts. Nuanced emotion.

Setting context:
{bible_excerpt OR "Use the dialogue's implied setting"}

Dialogue to expand:
{walk_beats_formatted}

Write narrated prose at grade {N} that:
- Preserves all dialogue lines (may paraphrase slightly for flow)
- Adds speaker attribution appropriate to grade level
- Includes brief scene-setting and action beats
- Maintains the emotional arc ({emotion_sequence})

Prose:
```

## FK Grade Distribution

Target distribution per walk:
- Grade 0: 1 expansion
- Grade 3: 1 expansion
- Grade 6: 1 expansion
- Grade 9: 1 expansion

This gives 4× the data with controlled complexity variation.

## Rejection Filters

1. **FK Score**: `|measured - target| <= 1.5`
2. **Dialogue Preservation**: All source beats appear (fuzzy match, 80% threshold)
3. **Word Count**: `30 <= words <= 400` (scales with beat count)
4. **No Meta-Commentary**: Reject "As an AI", "I'll write", etc.

## Tier 1 Companion Format

For concurrent training, also emit the flattened walk:

```json
{
  "id": "flat_walk_0001",
  "text": "\"Seventy-two hours.\"\n\"The Hexagon expects compliance.\"\n\"The Leclerc is outside.\"",
  "source": "gallia_v3",
  "tier": "flattened",
  "emotion_sequence": ["neutral", "neutral", "anger"]
}
```

Document boundary: `\n\n` between walks, or attention mask if architecture supports.

## Efficiency Notes

- Prefix cache: bible + FK examples (~3k tokens)
- Suffix per request: walk + target level (~200 tokens)
- 4 FK levels × N walks = 4N requests at ~1.06N effective cost

## Implementation Sketch

```python
async def generate_fk_stories(walks, bible, fk_levels=[0,3,6,9]):
    prefix = build_fk_prefix(bible)

    results = []
    for walk in walks:
        # Emit tier 1 (flattened)
        results.append({
            "id": f"flat_{walk['id']}",
            "text": flatten_walk(walk),
            "tier": "flattened"
        })

        # Emit tier 2 (FK-normed) at each level
        for fk in fk_levels:
            prose = await expand_to_prose(prefix, walk, fk)
            if passes_filters(prose, fk):
                results.append({
                    "id": f"fk_{walk['id']}_grade{fk}",
                    "prose": prose,
                    "fk_target": fk,
                    "tier": "fk_normed"
                })

    return results
```

## Training Mix

Suggested ratio for concurrent training:
- 20% flattened walks (tier 1)
- 80% FK-normed prose (tier 2, mixed grades)

The sparse tier 1 data teaches turn structure; the rich tier 2 data teaches what that structure implies.
