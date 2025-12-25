# Brainrot-Aesops Specification

**Date**: 2025-12-25
**Status**: Ready for implementation
**Depends on**: Dialogue walks (reference or synthetic), word frequency lists

## Purpose

Use dialogue walks as situating context to teach common vocabulary through
contextual definition. This creates a bridge toward traditional webtext by
training the pattern: "word appears → word is defined through use → word
appears again in varied form."

Not canonical. Not naturalistic. Explicitly pedagogical in a way that makes
your skin crawl. But possibly effective for bootstrapping NLP literacy.

## The Brainrot Formula

```
INPUT:
  - 20 common words sampled from frequency list
  - 4-10 dialogue walks sampled from corpus

TASK:
  - Map at least 5 words to at least 4 walks
  - Write paragraph-length prose that:
    - Uses each selected word in defining context
    - Incorporates the dialogue naturally
    - Results in a coherent (if pedagogical) mini-narrative
```

## Word Source

Use established frequency lists:
- COCA top 5000 (Corpus of Contemporary American English)
- Or: Ogden's Basic English 850
- Or: GSL (General Service List) ~2000 words

Filter to:
- Nouns, verbs, adjectives, adverbs (skip function words)
- Words with clear definitional paraphrases
- Exclude proper nouns, slang, technical jargon

## Dialogue Source

Any walk source works:
- Reference corpus (Oblivion, FNV, Skyrim)
- Synthetic corpus (Gallia, Marmotte)
- Mixed (for maximum cursedness)

The dialogue provides:
- Situated speech acts
- Emotional grounding
- Turn structure
- Implicit narrative

## Output Format

```json
{
  "id": "aesop_0001",
  "words_targeted": ["patrol", "angry", "tired", "walk", "wish"],
  "words_used": ["patrol", "patrolling", "angry", "anger", "tired", "walk", "walked", "wish"],
  "walks_incorporated": ["walk_0042", "walk_0187", "walk_0203", "walk_0341"],
  "prose": "To patrol means to walk around watching for problems...",
  "word_count": 120,
  "fk_measured": 2.8,
  "source_corpus": "falloutnv",
  "tier": "brainrot_aesop"
}
```

## Generation Prompt

```
You are creating vocabulary-teaching passages that incorporate dialogue from games.

TARGET WORDS (define at least 5 through contextual use):
{word_list_with_simple_definitions}

DIALOGUE TO INCORPORATE (use at least 4):
{formatted_walks}

Write a paragraph-length passage that:
1. Uses each target word in a way that teaches its meaning through context
2. Incorporates the dialogue lines naturally (characters speaking)
3. Shows grammatical variations of words (walk, walked, walking)
4. Reads like educational content, not natural prose (this is intentional)
5. Keeps sentences simple (target: grade 2-4 reading level)

The result should feel like a children's vocabulary workbook that somehow
got crossed with RPG dialogue. This is correct.

Passage:
```

## Example Output

Input words: [patrol, angry, tired, walk, wish]
Input walks: ["Patrolling the Mojave almost makes you wish for a nuclear winter."]

Output:
```
To patrol means to walk around watching for trouble. The soldier was on patrol.
He walked slowly in the hot sun. "Patrolling the Mojave almost makes you wish
for a nuclear winter," he said. To wish means to want something very much. The
soldier wished for cold weather. He was tired from walking. When you are tired,
you want to rest. The soldier was also angry at the heat. To be angry means to
feel very mad. His anger made his patrol feel even longer.
```

This is ghastly. This is effective. These two facts coexist.

## Rejection Filters

1. **Word Coverage**: At least 5 target words used
2. **Dialogue Incorporation**: At least 4 walks referenced
3. **FK Score**: Grade 2-4 range (1.5 - 4.5)
4. **Definitional Pattern**: Contains "means" or "is when" or equivalent
5. **Grammatical Variation**: At least 2 words show inflection variation
6. **No Meta-Commentary**: Reject "As an AI", "I'll write", etc.

## Constraint Satisfaction Scoring

Score each output on:
- `word_coverage`: len(words_used) / len(words_targeted)
- `walk_coverage`: len(walks_incorporated) / len(walks_provided)
- `definition_density`: count("means|is when|called") / word_count
- `variation_score`: unique_inflections / unique_lemmas

Accept if: `word_coverage >= 0.5 AND walk_coverage >= 0.4 AND fk_score in [1.5, 4.5]`

## Batch Generation

```python
async def generate_brainrot_aesops(
    walks: List[Walk],
    word_list: List[str],
    n_outputs: int = 1000
):
    results = []

    for _ in range(n_outputs):
        # Sample words and walks
        words = random.sample(word_list, 20)
        sampled_walks = random.sample(walks, random.randint(4, 10))

        # Generate
        prose = await generate_aesop(words, sampled_walks)

        # Filter
        if passes_filters(prose, words, sampled_walks):
            results.append(prose)

    return results
```

## Training Integration

Tier 3 (brainrot-aesops) bridges toward webtext:

```
Curriculum order:
1. Flattened walks (sparse turn structure)
2. FK-normed stories (explicit narrative context)
3. Brainrot-aesops (definitional patterns, vocabulary grounding)
4. Webtext/wikitext (the destination)
```

Suggested mix when all tiers available:
- 15% flattened walks
- 50% FK-normed stories
- 25% brainrot-aesops
- 10% held-out webtext (to measure transfer)

## Why This Might Work

The brainrot-aesop pattern trains:
- "X means Y" → definitional structure
- "He was X. To be X means..." → anaphora + definition
- Word → inflection → word → definition → word → usage

This is exactly the pattern that appears in:
- Dictionary entries
- ESL textbooks
- Children's vocabulary workbooks
- Wikipedia intro paragraphs ("X is a Y that Z")

A model that's seen this pattern should find wikitext less surprising.

## Why This Is Horrifying

You're building a Rosetta Stone from:
1. A videogame
2. Translated through synthetic paracosms
3. Used to define basic English vocabulary
4. For training language priors in multimodal models

The philological transgression has no technical cost. It just feels wrong.
The distributional hypothesis doesn't care about provenance.

## Open Questions

- Should word sampling be weighted by frequency, or uniform?
- Should we stratify by word class (nouns vs verbs vs adjectives)?
- Is grade 2-4 the right FK target, or should we vary?
- Should walks be from one source or mixed across corpora?
- How many aesops per word to ensure coverage?
