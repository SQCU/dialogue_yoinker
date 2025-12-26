#!/usr/bin/env python3
"""
Brainrot-Aesops v4: Many Small Outputs

Instead of mega-passages cramming 4+ walks and 5+ words, v4 generates
many short passages (60-120 words) each defining 1-3 words.

This is a MATCHING problem followed by SHORT GENERATION, not a CRAMMING problem.

v3: 1 API call → 1 passage → 5 definitions (if pass)
v4: 12 parallel calls → 12 passages → 12 definitions (80%+ pass rate)
"""

import json
import random
import hashlib
import asyncio
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

from prose_wrapper import Walk, clean_text

# Import vocabulary module for COCA-based word sampling
try:
    from vocabulary import VocabSampler
    _VOCAB_SAMPLER = None  # Lazy-loaded
    HAS_VOCAB = True
except ImportError:
    HAS_VOCAB = False
    _VOCAB_SAMPLER = None


# =============================================================================
# Vocabulary Access
# =============================================================================

def get_vocab_sampler() -> 'VocabSampler':
    """Get or create the global vocabulary sampler."""
    global _VOCAB_SAMPLER
    if _VOCAB_SAMPLER is None and HAS_VOCAB:
        _VOCAB_SAMPLER = VocabSampler(max_rank=2000)
    return _VOCAB_SAMPLER


def get_inflections(word: str) -> List[str]:
    """Get known inflections of a word from vocabulary module."""
    sampler = get_vocab_sampler()
    if sampler and word in sampler.by_word:
        return sampler.by_word[word].inflections
    # Fallback: just return the word itself
    return [word]


def sample_words(n: int = 12) -> List[Tuple[str, str]]:
    """Sample vocabulary words from COCA frequency data."""
    sampler = get_vocab_sampler()
    if sampler:
        return sampler.sample_tuples(n, weighted=True)

    # Fallback if vocabulary module not available
    fallback = [
        ("walk", "to move by putting one foot in front of the other"),
        ("run", "to move fast using your legs"),
        ("help", "to make something easier for someone"),
        ("need", "to require something"),
        ("find", "to discover or locate something"),
        ("think", "to use your mind to consider something"),
        ("see", "to use your eyes to look at something"),
        ("time", "the measure of when things happen"),
        ("place", "a location or area"),
        ("good", "of high quality or pleasant"),
        ("bad", "of low quality or unpleasant"),
        ("new", "recently made or started"),
    ]
    return random.sample(fallback, min(n, len(fallback)))


# =============================================================================
# FK Measurement
# =============================================================================

def measure_fk_grade(text: str) -> float:
    """Measure Flesch-Kincaid grade level."""
    if HAS_TEXTSTAT:
        return textstat.flesch_kincaid_grade(text)
    return 3.0  # Default for testing


# =============================================================================
# Output Format
# =============================================================================

@dataclass
class AesopV4:
    """A single vocabulary-teaching passage. One word, one walk, short prose."""
    id: str
    word: str
    definition: str
    walk_id: str
    walk_text: str
    prose: str
    word_count: int
    fk_measured: float
    source_corpus: str
    tier: str = "brainrot_aesop_v4"
    passed_filters: bool = True
    reject_reason: Optional[str] = None


# =============================================================================
# Walk-Word Matching
# =============================================================================

def format_walk_text(walk: Walk) -> str:
    """Format walk for display in output."""
    lines = []
    for beat in walk.beats:
        text = clean_text(beat.get("text", ""))
        if text:
            speaker = beat.get("speaker")
            if speaker:
                lines.append(f'{speaker}: "{text}"')
            else:
                lines.append(f'"{text}"')
    return "\n".join(lines)


def format_walk_for_prompt(walk: Walk) -> str:
    """Format walk for prompt - clean dialogue only, no metadata tags."""
    lines = []
    for beat in walk.beats:
        text = clean_text(beat.get("text", ""))
        if text:
            speaker = beat.get("speaker", "Someone")
            lines.append(f'{speaker}: "{text}"')
    return "\n".join(lines)


def heuristic_match_words_to_walks(
    words: List[Tuple[str, str]],
    walks: List[Walk],
) -> Dict[str, List[Walk]]:
    """Match words to walks without LLM."""
    matches = {}

    for word, definition in words:
        word_lower = word.lower()
        inflections = get_inflections(word)
        inflections_lower = [i.lower() for i in inflections]

        scored_walks = []
        for walk in walks:
            score = 0
            walk_text = " ".join(
                clean_text(b.get("text", "")) for b in walk.beats
            ).lower()

            # Lexical overlap - word or inflection appears in walk
            for infl in inflections_lower:
                if re.search(rf'\b{re.escape(infl)}\b', walk_text):
                    score += 10

            # Emotion alignment (crude)
            walk_emotions = [b.get("emotion", "") for b in walk.beats]
            if word in ["angry", "rage", "fury", "mad"] and "anger" in walk_emotions:
                score += 5
            if word in ["fear", "afraid", "terror", "scared"] and "fear" in walk_emotions:
                score += 5
            if word in ["happy", "joy", "pleased", "glad"] and "happy" in walk_emotions:
                score += 5
            if word in ["sad", "sorrow", "grief"] and "sad" in walk_emotions:
                score += 5

            # Speaker heuristics
            speakers = [b.get("speaker", "") for b in walk.beats]
            speaker_text = " ".join(s for s in speakers if s).lower()
            if word in ["guard", "patrol", "soldier", "watch"] and any(
                x in speaker_text for x in ["guard", "soldier", "agent", "officer"]
            ):
                score += 3
            if word in ["clerk", "file", "form", "document", "paper"] and any(
                x in speaker_text for x in ["clerk", "official", "bureaucrat"]
            ):
                score += 3

            if score > 0:
                scored_walks.append((score, walk))

        # Sort by score, take top 2
        scored_walks.sort(key=lambda x: -x[0])
        matches[word] = [w for _, w in scored_walks[:2]]

    return matches


# =============================================================================
# Prompt Template
# =============================================================================

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


# =============================================================================
# Rejection Filters (simplified for short outputs)
# =============================================================================

DEFINITIONAL_PATTERNS = [
    r"\bmeans\b",
    r"\bis when\b",
    r"\bis to\b",
    r"\bis a\b",
    r"to \w+ is to",
    r"called\b",
]

META_PATTERNS = [
    "as an ai",
    "i'll write",
    "i will write",
    "here is",
    "here's the",
    "let me",
]


def has_definitional_pattern(prose: str) -> bool:
    """Check if prose contains definitional patterns."""
    prose_lower = prose.lower()
    for pattern in DEFINITIONAL_PATTERNS:
        if re.search(pattern, prose_lower):
            return True
    return False


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
    inflections = get_inflections(target_word)
    word_found = any(
        re.search(rf'\b{re.escape(infl)}\b', prose.lower())
        for infl in inflections
    )
    if not word_found:
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
    for pattern in META_PATTERNS:
        if pattern in prose_lower:
            return False, "meta"

    return True, None


# =============================================================================
# Generation
# =============================================================================

def generate_aesop_id(word: str, walk: Walk) -> str:
    """Generate stable ID for an aesop."""
    walk_content = json.dumps([b.get("text", "") for b in walk.beats])
    content = f"{word}:{walk_content}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


async def generate_single_aesop(
    word: str,
    definition: str,
    walk: Walk,
    llm_call: callable,
    source_corpus: str = "unknown",
) -> AesopV4:
    """Generate one short aesop for one word grounded in one walk."""
    aesop_id = generate_aesop_id(word, walk)
    walk_text = format_walk_for_prompt(walk)
    walk_display = format_walk_text(walk)

    prompt = SINGLE_AESOP_PROMPT.format(
        word=word,
        definition=definition,
        walk_text=walk_text,
    )

    try:
        prose = await llm_call(prompt)
    except Exception as e:
        return AesopV4(
            id=f"aesop_v4_{aesop_id}",
            word=word,
            definition=definition,
            walk_id=getattr(walk, 'id', 'unknown'),
            walk_text=walk_display,
            prose="",
            word_count=0,
            fk_measured=0.0,
            source_corpus=source_corpus,
            passed_filters=False,
            reject_reason=f"llm_error_{str(e)[:50]}",
        )

    # Apply filters
    passed, reason = apply_v4_filters(prose, word)

    return AesopV4(
        id=f"aesop_v4_{aesop_id}",
        word=word,
        definition=definition,
        walk_id=getattr(walk, 'id', 'unknown'),
        walk_text=walk_display,
        prose=prose if passed else "",
        word_count=len(prose.split()),
        fk_measured=measure_fk_grade(prose),
        source_corpus=source_corpus,
        passed_filters=passed,
        reject_reason=reason,
    )


async def generate_aesops_v4(
    walks: List[Walk],
    llm_call: callable,
    n_words: int = 12,
    source_corpus: str = "unknown",
    use_llm_matching: bool = False,
    concurrency: int = 10,
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
        # TODO: implement LLM-based matching if needed
        matches = heuristic_match_words_to_walks(words, walks)
    else:
        matches = heuristic_match_words_to_walks(words, walks)

    # Generate in parallel with semaphore
    semaphore = asyncio.Semaphore(concurrency)

    async def generate_one(word: str, definition: str, walk: Walk) -> AesopV4:
        async with semaphore:
            return await generate_single_aesop(
                word, definition, walk, llm_call, source_corpus
            )

    tasks = []
    for word, definition in words:
        matched_walks = matches.get(word, [])
        if not matched_walks:
            # Fallback: random walk
            matched_walks = [random.choice(walks)]

        # Take first matched walk
        walk = matched_walks[0]
        tasks.append(generate_one(word, definition, walk))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, AesopV4)]


async def generate_aesops_v4_batch(
    walks: List[Walk],
    llm_call: callable,
    n_batches: int = 3,
    words_per_batch: int = 12,
    source_corpus: str = "unknown",
    concurrency: int = 10,
) -> List[AesopV4]:
    """
    Generate multiple batches of aesops.

    Each batch samples fresh words and matches them to walks.
    Total outputs: n_batches * words_per_batch
    """
    all_results = []

    for batch_idx in range(n_batches):
        print(f"  Batch {batch_idx + 1}/{n_batches}...")
        batch_results = await generate_aesops_v4(
            walks,
            llm_call,
            n_words=words_per_batch,
            source_corpus=source_corpus,
            concurrency=concurrency,
        )
        all_results.extend(batch_results)

    return all_results


# =============================================================================
# CLI / Testing
# =============================================================================

async def mock_llm_call(prompt: str) -> str:
    """Mock LLM that generates short aesop-style prose."""
    # Extract word from prompt
    word_match = re.search(r'VOCABULARY WORD:\s*(\w+):', prompt)
    word = word_match.group(1) if word_match else "thing"

    # Extract dialogue from prompt
    dialogue_match = re.search(r'"([^"]+)"', prompt)
    dialogue = dialogue_match.group(1) if dialogue_match else "Something happened."

    # Generate mock short prose
    prose = f"""To {word} means to do something important. When we {word}, we take action.

The clerk said, "{dialogue}" This shows how people {word} in everyday life.

Everyone must learn to {word}. It is a basic skill that helps us get things done."""

    return prose


async def main():
    """Test the brainrot-aesops v4 generator."""

    print("=" * 60)
    print("BRAINROT-AESOPS v4 TEST (many small outputs)")
    print("=" * 60)

    # Create test walks
    walks = [
        Walk(
            beats=[
                {"text": "Patrolling the Mojave almost makes you wish for a nuclear winter.", "emotion": "neutral", "speaker": "NCR Trooper"},
                {"text": "War never changes.", "emotion": "sad", "speaker": "Narrator"},
            ],
            source="test_fnv",
        ),
        Walk(
            beats=[
                {"text": "Stop right there, criminal scum!", "emotion": "anger", "speaker": "Guard"},
                {"text": "Pay the fine or serve your sentence.", "emotion": "neutral", "speaker": "Guard"},
            ],
            source="test_oblivion",
        ),
        Walk(
            beats=[
                {"text": "I need to find my way home.", "emotion": "sad", "speaker": "Traveler"},
                {"text": "Can you help me?", "emotion": "neutral", "speaker": "Traveler"},
            ],
            source="test_custom",
        ),
        Walk(
            beats=[
                {"text": "The Hexagon expects compliance.", "emotion": "neutral", "speaker": "Official"},
                {"text": "Seventy-two hours.", "emotion": "anger", "speaker": "Official"},
            ],
            source="test_gallia",
        ),
        Walk(
            beats=[
                {"text": "Your papers are not in order.", "emotion": "disgust", "speaker": "Clerk"},
                {"text": "Come back tomorrow.", "emotion": "neutral", "speaker": "Clerk"},
            ],
            source="test_gallia",
        ),
    ]

    # Sample words
    words = sample_words(6)
    print(f"\n--- Sampled {len(words)} words ---")
    for w, d in words:
        print(f"  {w}: {d[:50]}...")

    # Test matching
    print(f"\n--- Heuristic Matching ---")
    matches = heuristic_match_words_to_walks(words, walks)
    for word, matched_walks in matches.items():
        if matched_walks:
            print(f"  {word} -> {len(matched_walks)} walk(s)")

    # Generate aesops
    print(f"\n--- Generation (6 words, 5 walks) ---")
    aesops = await generate_aesops_v4(
        walks,
        mock_llm_call,
        n_words=6,
        source_corpus="test",
        concurrency=3,
    )

    print(f"Generated {len(aesops)} aesops")

    passed = [a for a in aesops if a.passed_filters]
    failed = [a for a in aesops if not a.passed_filters]

    print(f"Passed: {len(passed)}, Failed: {len(failed)}")

    for aesop in passed[:3]:
        print(f"\n--- {aesop.word} ({aesop.id[:8]}) ---")
        print(f"FK: {aesop.fk_measured:.1f}, Words: {aesop.word_count}")
        print(f"Prose: {aesop.prose[:150]}...")

    if failed:
        print(f"\n--- Failures ---")
        for a in failed:
            print(f"  {a.word}: {a.reject_reason}")


if __name__ == "__main__":
    asyncio.run(main())
