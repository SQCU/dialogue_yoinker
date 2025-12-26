# Brainrot-Aesops v4: Many Small Outputs
 
**Date**: 2025-12-26
**Status**: Spec ready for implementation
**Supersedes**: brainrot_aesops.py (v3 "model-driven pairing")
 
## The Problem With v3
 
v3 asks the model to produce one mega-passage that incorporates 4+ walks and 5+ vocabulary definitions. This creates a compression problem:
 
```
INPUT:  8 walks, 12 words
OUTPUT: 1 passage (~500 words)
CONSTRAINTS: use 4+ walks, define 5+ words, stay coherent
```
 
Result: Dense, exhausting prose. 6 outputs from 1,349 nodes. 50% pass rate.
 
The task was mistranslated. The goal isn't "write one coherent narrative using many things." The goal is "produce many vocabulary-teaching moments, each grounded in situated dialogue."
 
## The v4 Reframe
 
```
INPUT:  n walks, m vocabulary words
OUTPUT: m' short passages (one per word or small word cluster)
        each passage uses 1-2 walks as situating context
 
CONSTRAINTS per output:
  - Define exactly 1-3 words
  - Ground in 1-2 dialogue fragments
  - 60-120 words
  - FK grade 3-6
```
 
This is a **matching problem** followed by **short generation**, not a **cramming problem**.
 
## Architecture
 
### Phase 1: Walk-Word Matching
 
Given walks with metadata and vocabulary words with definitions, find natural pairings.
 
**Matching heuristics** (no LLM needed):
1. **Lexical overlap**: word appears in walk text
2. **Emotion alignment**: "angry" pairs with anger-tagged walks
3. **Semantic field**: "patrol" pairs with guard/soldier speakers
4. **Arc shape**: "negotiate" pairs with negotiation_arc walks
 
**Matching via LLM** (optional, higher quality):
```
Given these 8 walks and 12 words, return a JSON mapping of which word(s)
pair naturally with which walk(s). Each word should map to 1-2 walks.
Do not write prose. Just return the mapping.
 
{walks}
{words}
 
Return: {"word1": ["walk_3"], "word2": ["walk_1", "walk_5"], ...}
```
 
### Phase 2: Per-Pair Generation
 
For each (word, walk) or (word_cluster, walk_cluster) pair:
 
```
VOCABULARY WORD:
{word}: {definition}
 
DIALOGUE CONTEXT:
{walk_text_with_speaker_and_emotion}
 
Write a short passage (60-120 words) that:
1. Defines "{word}" through contextual use (use "X means Y" or "to X is to Y")
2. Incorporates the dialogue naturally
3. Uses the word at least twice (once in definition, once in use)
 
Keep it simple. Grade 3-6 reading level.
 
Passage:
```
 
### Phase 3: Assembly
 
Concatenate outputs. Each is independent. No narrative coherence required between passages.
 
## Output Format
 
```json
{
  "id": "aesop_v4_0001",
  "word": "patrol",
  "definition": "to walk around an area to guard it",
  "walk_id": "walk_0042",
  "walk_text": "Patrolling the Mojave almost makes you wish for a nuclear winter.",
  "prose": "To patrol means to walk around an area watching for trouble. The soldier was on patrol in the desert. \"Patrolling the Mojave almost makes you wish for a nuclear winter,\" he said. His patrol route was long and hot.",
  "word_count": 42,
  "fk_measured": 3.2,
  "tier": "brainrot_aesop_v4"
}
```
 
## Comparison
 
| Metric | v3 (mega-passage) | v4 (many-small) |
|--------|-------------------|-----------------|
| Outputs per API batch | 1 passage | 12+ passages |
| Words per output | 5+ crammed | 1-3 focused |
| Walks per output | 4+ crammed | 1-2 focused |
| Word count | 400-600 | 60-120 |
| Coherence required | Yes (hard) | No (easy) |
| Pass rate (expected) | ~50% | ~80%+ |
| Definitions/API call | 5 | 12 |
 
## Implementation Sketch
 
```python
@dataclass
class AesopV4:
    id: str
    word: str
    definition: str
    walk_id: str
    walk_text: str
    prose: str
    word_count: int
    fk_measured: float
    tier: str = "brainrot_aesop_v4"
    passed_filters: bool = True
    reject_reason: Optional[str] = None
 
 
async def generate_aesops_v4(
    walks: List[Walk],
    n_words: int = 12,
    llm_call: callable = None,
    use_llm_matching: bool = False,
) -> List[AesopV4]:
    """
    Generate many short vocabulary-teaching passages.
 
    1. Sample n_words from COCA vocabulary
    2. Match words to walks (heuristic or LLM)
    3. Generate one short passage per (word, walk) pair
    4. Filter and return
    """
    # Sample vocabulary
    words = sample_words(n_words)
 
    # Match words to walks
    if use_llm_matching:
        matches = await llm_match_words_to_walks(words, walks, llm_call)
    else:
        matches = heuristic_match_words_to_walks(words, walks)
 
    # Generate in parallel
    results = []
    for word, definition in words:
        matched_walks = matches.get(word, [])
        if not matched_walks:
            matched_walks = [random.choice(walks)]  # fallback
 
        walk = matched_walks[0]
 
        prose = await generate_single_aesop(word, definition, walk, llm_call)
 
        # Apply filters
        passed, reason = apply_v4_filters(prose, word)
 
        results.append(AesopV4(
            id=f"aesop_v4_{hash(word + walk.id)[:8]}",
            word=word,
            definition=definition,
            walk_id=walk.id if hasattr(walk, 'id') else "unknown",
            walk_text=format_walk_text(walk),
            prose=prose if passed else "",
            word_count=len(prose.split()),
            fk_measured=measure_fk_grade(prose),
            passed_filters=passed,
            reject_reason=reason,
        ))
 
    return results
 
 
def heuristic_match_words_to_walks(
    words: List[Tuple[str, str]],
    walks: List[Walk],
) -> Dict[str, List[Walk]]:
    """Match words to walks without LLM."""
    matches = {}
 
    for word, definition in words:
        word_lower = word.lower()
        inflections = get_inflections(word)
 
        scored_walks = []
        for walk in walks:
            score = 0
            walk_text = " ".join(b.get("text", "") for b in walk.beats).lower()
 
            # Lexical overlap
            for infl in inflections:
                if infl in walk_text:
                    score += 10
 
            # Emotion alignment (crude)
            walk_emotions = [b.get("emotion", "") for b in walk.beats]
            if word in ["angry", "rage", "fury"] and "anger" in walk_emotions:
                score += 5
            if word in ["fear", "afraid", "terror"] and "fear" in walk_emotions:
                score += 5
            if word in ["happy", "joy", "pleased"] and "happy" in walk_emotions:
                score += 5
 
            # Speaker heuristics
            speakers = [b.get("speaker", "") for b in walk.beats]
            speaker_text = " ".join(s for s in speakers if s).lower()
            if word in ["guard", "patrol", "soldier"] and any(
                x in speaker_text for x in ["guard", "soldier", "agent"]
            ):
                score += 3
            if word in ["clerk", "file", "form", "document"] and any(
                x in speaker_text for x in ["clerk", "official", "bureaucrat"]
            ):
                score += 3
 
            if score > 0:
                scored_walks.append((score, walk))
 
        # Sort by score, take top 2
        scored_walks.sort(key=lambda x: -x[0])
        matches[word] = [w for _, w in scored_walks[:2]]
 
    return matches
 
 
SINGLE_AESOP_PROMPT = """VOCABULARY WORD:
{word}: {definition}
 
DIALOGUE CONTEXT:
{walk_text}
 
Write a short passage (60-120 words) that:
1. Defines "{word}" through contextual use (use "X means Y" or "to X is to Y")
2. Incorporates the dialogue naturally (quote it directly)
3. Uses the word at least twice (once in definition, once in context)
 
Keep it simple. Grade 3-6 reading level. No preamble.
 
Passage:"""
 
 
async def generate_single_aesop(
    word: str,
    definition: str,
    walk: Walk,
    llm_call: callable,
) -> str:
    """Generate one short aesop for one word grounded in one walk."""
    walk_text = format_walk_for_prompt(walk)
 
    prompt = SINGLE_AESOP_PROMPT.format(
        word=word,
        definition=definition,
        walk_text=walk_text,
    )
 
    return await llm_call(prompt)
 
 
def apply_v4_filters(
    prose: str,
    target_word: str,
    min_words: int = 40,
    max_words: int = 150,
    fk_max: float = 8.0,
) -> Tuple[bool, Optional[str]]:
    """Simpler filters for shorter outputs."""
    word_count = len(prose.split())
 
    if word_count < min_words:
        return False, "too_short"
    if word_count > max_words:
        return False, "too_long"
 
    # Check word appears
    if target_word.lower() not in prose.lower():
        return False, "word_missing"
 
    # Check definitional pattern
    if not has_definitional_pattern(prose):
        return False, "no_definition"
 
    # FK check
    fk = measure_fk_grade(prose)
    if fk > fk_max:
        return False, f"fk_too_high_{fk:.1f}"
 
    # Meta check
    prose_lower = prose.lower()
    for pattern in ["as an ai", "let me", "here is", "i'll write"]:
        if pattern in prose_lower:
            return False, "meta"
 
    return True, None
```
 
## Batch Efficiency
 
v3: 1 API call → 1 passage → 5 definitions (if pass)
v4: 1 API call → 1 passage → 1 definition, but 12 parallel calls → 12 definitions
 
With 25x concurrency on DeepSeek:
- v3: 25 calls → ~12 passed aesops → ~60 definitions
- v4: 25 calls → ~20 passed aesops → ~20 definitions
 
**But** v4 calls are simpler (shorter prompts, shorter outputs), so:
- Faster per call
- Higher pass rate
- More predictable output
- Easier to debug/iterate
 
## Optional: Clustering
 
For words that naturally cluster (e.g., "walk", "run", "move"), generate one passage defining all three using 2-3 walks. This is an optimization, not the base case.
 
```python
WORD_CLUSTERS = {
    "motion": ["walk", "run", "move", "go"],
    "emotion_anger": ["angry", "rage", "fury", "mad"],
    "bureaucracy": ["form", "file", "document", "stamp"],
}
```
 
## Migration Path
 
1. Add `brainrot_aesops_v4.py` alongside existing module
2. Add `--version v4` flag to `run_consumers.py`
3. Run both, compare output volume and quality
4. If v4 wins, deprecate v3
 
## Expected Outcomes
 
- **10x output volume**: 12 definitions per batch vs ~1
- **Higher pass rate**: 80%+ vs 50%
- **Simpler prose**: Easier for small models to learn from
- **Better coverage**: More words get definitions
- **Debuggable**: Each output is independent, easy to analyze failures
 
## Open Questions
 
1. Should matching be LLM-based or heuristic? (Start heuristic, upgrade if needed)
2. Should we cluster related words? (Defer, start with 1:1)
3. What's the optimal word count range? (Start 60-120, tune based on FK)
4. Should walks be reused across words? (Yes, if they match well)