# Consumer Pipeline Checkpoint

**Date**: 2025-12-25
**Status**: Implemented and tested with real API runs
**Session**: Code review + implementation of walk consumers

## What Got Built

Five modules that consume dialogue walks and produce training-ready outputs:

```
Reference/Synthetic Graphs
         │
         ▼ extract_walks_from_graph()
      [Walks]
         │
    ┌────┴────┬─────────────┬──────────────┐
    ▼         ▼             ▼              ▼
prose_    fk_normed_    brainrot_     vocabulary
wrapper   stories       aesops        (COCA data)
    │         │             │              │
    ▼         ▼             ▼              ▼
 (local)   Tier 1+2      Tier 3        word lists
           flattened     vocab         + definitions
           + FK-normed   teaching
```

## Quick Start

```bash
# Install dependencies
uv sync

# Dry run (mock LLM)
uv run python run_consumers.py all \
  --source synthetic/gallia_v4/graph.json \
  --output output/training.jsonl \
  --dry-run

# Real API run
DEEPSEEK_API_KEY="sk-..." uv run python run_consumers.py brainrot \
  --source synthetic/gallia_v4/graph.json \
  --output output/gallia_aesops.jsonl \
  --num-walks 30 \
  --num-aesops 10

# With LLM-based boilerplate cleaning (2x API calls, higher quality)
DEEPSEEK_API_KEY="sk-..." uv run python run_consumers.py brainrot \
  --source synthetic/marmotte_v2/graph.json \
  --output output/marmotte_aesops.jsonl \
  --num-aesops 10 \
  --use-cleaner
```

## Module Reference

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

### brainrot_aesops.py (~550 LOC)

Vocabulary teaching via dialogue expansion. v3 uses **model-driven pairing**.

**Architecture (v3):**
1. Sample 8 walks and 12 vocabulary words
2. Present both to LLM with minimum thresholds (4 walks, 5 definitions)
3. Model chooses which pairings work naturally
4. Filter output for quality

**Key insight**: Asking the model to find natural pairings is less constrained than forcing specific word-walk combinations.

Rejection filters:
- Min 4 walks incorporated
- Min 5 vocabulary words defined with patterns ("X means Y", "to X is to Y")
- Word count: 80-600
- No meta-commentary

Optional **LLM-based cleaner** (`--use-cleaner`):
- Second API call strips assistant boilerplate ("Let me write...", "Here is...")
- Recognition is easier than suppression
- Replaces regex-based meta detection with model-based extraction

### vocabulary.py (~530 LOC)

COCA frequency-based vocabulary with WordNet definitions.

```python
from vocabulary import VocabSampler

sampler = VocabSampler(max_rank=2000)  # Top 2000 words by frequency
words = sampler.sample_tuples(12)       # [(word, definition), ...]
# [('focus', 'direct one\'s attention on something'),
#  ('be', 'have the quality of being...'), ...]
```

Features:
- Downloads COCA data from GitHub on first use
- Filters to content words (nouns, verbs, adjectives, adverbs)
- WordNet definitions with fallbacks
- Inflection generation (walk → walks, walked, walking)
- Frequency-weighted sampling

### run_consumers.py (~400 LOC)

CLI orchestration:

```bash
uv run python run_consumers.py <mode> --source <path> --output <path> [options]

Modes:
  fk-normed   Generate FK-targeted prose at multiple grade levels
  brainrot    Generate vocabulary-teaching passages
  all         Both tiers

Options:
  --num-walks N       Number of walks to extract (default: 50)
  --num-aesops N      Number of aesops to generate (default: 10)
  --fk-levels S       Comma-separated FK levels (default: 0,3,6,9)
  --concurrency N     Max parallel API calls (default: 5)
  --dry-run           Use mock LLM
  --use-cleaner       LLM-based boilerplate stripping (2x API calls)
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

### Tier 3 (brainrot-aesop v3)
```json
{
  "id": "aesop_ee9ef3fbea35",
  "words_offered": ["military", "quality", "year", "sure", "make", ...],
  "words_used": ["military", "quality", "sure", "check", "in"],
  "walks_offered": 8,
  "walks_used": 7,
  "prose": "The archives of the Préfecture were a world unto themselves... **quality**—an essential and distinguishing attribute...",
  "word_count": 458,
  "fk_measured": 7.0,
  "source_corpus": "graph",
  "tier": "brainrot_aesop",
  "passed_filters": true,
  "reject_reason": null
}
```

## Real API Results (DeepSeek v3, 2025-12-25)

### Without Cleaner

| Corpus | Aesops | Pass Rate | Words Taught | Walks/Aesop | Defs/Aesop |
|--------|--------|-----------|--------------|-------------|------------|
| gallia_v4 | 10 | 50% | 29 unique | 7.6 | 6.0 |
| marmotte_v2 | 10 | 60% | 26 unique | 7.5 | 5.3 |

Failure modes: meta-commentary (40%), insufficient walks/definitions (60%)

### With Cleaner (`--use-cleaner`)

| Corpus | Aesops | Pass Rate | Words Taught | Walks/Aesop | Defs/Aesop |
|--------|--------|-----------|--------------|-------------|------------|
| gallia_v4 | 10 | 50% | 26 unique | 7.4 | 5.8 |

Failure modes: 0% meta, 100% insufficient (content-level)

**Key insight**: Cleaner eliminates false-positive meta rejections. Same pass rate, but failures are now for legitimate reasons.

## Sample Output

From `gallia_v4` with COCA vocabulary:

> The archives of the Préfecture were a world unto themselves, a labyrinth of files where truth was not a **quality**—an essential and distinguishing attribute—but a matter of proper stamps and cross-referenced dossiers.
>
> "I require the supporting documentation," he said flatly. "The file cannot proceed without it."
>
> To **check** a file is to examine so as to determine its accuracy, or its lack thereof. "This lacks the proper essence stamp."
>
> The Préfecture's methods were, in their own way, **military**—of or relating to the principles of a very specific, paperwork-driven warfare.

Definitions embedded naturally. Dialogue preserved from source walks. FK grade ~7.0.

## Dependencies

```
textstat==0.7.12  # FK measurement
nltk>=3.9        # WordNet definitions (auto-downloads data)
httpx>=0.28      # Async API calls
```

## Architecture Notes

These modules are **downstream consumers** of the graph infrastructure:

```
[Other session]              [This session]
Translation Engine    →      Walk Consumers
structural-parser            prose_wrapper
translation-engine           fk_normed_stories
link-stitcher                brainrot_aesops
batch_growth.py              run_consumers.py
                             vocabulary.py
```

Clean separation at the walk extraction boundary. No interference with graph generation.

## Iteration History

- **v1**: Attempted to use ALL 20 words with ALL walks (overconstrained)
- **v2**: Overcorrected to 1 word + 1 walk (underconstrained)
- **v3**: Model-driven pairing (8 walks, 12 words, model chooses)
- **+cleaner**: LLM-based boilerplate stripping (recognition > suppression)
- **+COCA**: Real frequency data replaces hand-coded word list

## Next Steps

1. **Curriculum mixing**: Implement 15%/50%/25%/10% tier ratios
2. **Reference corpus**: Test with raw Oblivion/FNV dialogue
3. **Bible injection**: Setting-specific proper nouns in prompts
4. **Quality metrics**: Embedding similarity, n-gram diversity
