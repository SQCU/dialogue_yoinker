#!/usr/bin/env python3
"""
Brainrot-Aesops Generator

Uses dialogue walks as situating context to teach common vocabulary through
contextual definition. Creates vocabulary-teaching passages that incorporate
game dialogue.

This is ghastly. This is effective. These two facts coexist.
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


def get_all_inflections(words: List[str]) -> Set[str]:
    """Get all inflections for a list of words."""
    result = set()
    for word in words:
        result.update(get_inflections(word))
    return result


# =============================================================================
# FK Measurement
# =============================================================================

def measure_fk_grade(text: str) -> float:
    """Measure Flesch-Kincaid grade level."""
    if HAS_TEXTSTAT:
        return textstat.flesch_kincaid_grade(text)
    return 3.0  # Default for testing


# =============================================================================
# Prompt Template
# =============================================================================

def format_walks_for_prompt(walks: List[Walk]) -> str:
    """Format multiple walks for the prompt."""
    sections = []
    for i, walk in enumerate(walks):
        lines = []
        for beat in walk.beats:
            text = clean_text(beat.get("text", ""))
            if text:
                lines.append(f'  "{text}"')
        if lines:
            sections.append(f"Walk {i+1}:\n" + "\n".join(lines))
    return "\n\n".join(sections)


def format_words_for_prompt(words: List[Tuple[str, str]]) -> str:
    """Format vocabulary words for the prompt."""
    lines = []
    for word, definition in words:
        lines.append(f"- {word}: {definition}")
    return "\n".join(lines)


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


def build_aesop_prompt(
    walks: List[Walk],
    words: List[Tuple[str, str]],
    min_walks: int = 4,
    min_words: int = 5,
) -> str:
    """Build prompt for model-driven walk/word pairing."""

    walks_str = format_walks_for_prompt(walks)
    words_str = format_words_for_prompt(words)

    prompt = f"""You have {len(walks)} dialogue sequences and {len(words)} vocabulary words.

DIALOGUE SEQUENCES:
{walks_str}

VOCABULARY WORDS:
{words_str}

Write a cohesive prose passage that:
1. Incorporates dialogue from AT LEAST {min_walks} of the walks above
2. Defines AT LEAST {min_words} vocabulary words through context (use "X means Y" or "to X is to Y" patterns)
3. Skips walks or words that don't fit naturally - quality over coverage
4. Weaves everything into a single narrative

You choose which walks and words pair well together. Not everything needs to be used.

Prose:"""

    return prompt


# =============================================================================
# Output Format
# =============================================================================

@dataclass
class BrainrotAesop:
    """A vocabulary-teaching passage. Multiple walks, multiple words, model chooses pairings."""
    id: str
    words_offered: List[str]
    words_used: List[str]
    walks_offered: int
    walks_used: int
    prose: str
    word_count: int
    fk_measured: float
    source_corpus: str
    tier: str = "brainrot_aesop"
    passed_filters: bool = True
    reject_reason: Optional[str] = None


# =============================================================================
# Rejection Filters
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


def count_words_used(prose: str, target_words: List[str]) -> Tuple[List[str], int]:
    """Count which target words (including inflections) appear in prose."""
    prose_lower = prose.lower()
    words_found = []
    inflections_found = 0

    for word in target_words:
        inflections = get_inflections(word)
        word_found = False
        for infl in inflections:
            if re.search(rf'\b{re.escape(infl)}\b', prose_lower):
                if not word_found:
                    words_found.append(word)
                    word_found = True
                inflections_found += 1

    return words_found, inflections_found


def count_walks_used(prose: str, walks: List[Walk]) -> List[str]:
    """Count which walks have lines appearing in prose."""
    prose_lower = prose.lower()
    walks_used = []

    for i, walk in enumerate(walks):
        for beat in walk.beats:
            text = clean_text(beat.get("text", "")).lower()
            if len(text) > 10:  # Skip very short lines
                # Check for fuzzy match (50% of words)
                words = text.split()
                matches = sum(1 for w in words if w in prose_lower)
                if matches / max(1, len(words)) >= 0.5:
                    walks_used.append(f"walk_{i}")
                    break

    return walks_used


def has_definitional_pattern(prose: str) -> bool:
    """Check if prose contains definitional patterns."""
    prose_lower = prose.lower()
    for pattern in DEFINITIONAL_PATTERNS:
        if re.search(pattern, prose_lower):
            return True
    return False


def count_walks_used(prose: str, walks: List[Walk]) -> int:
    """Count how many walks have dialogue appearing in the prose."""
    prose_lower = prose.lower()
    count = 0

    for walk in walks:
        for beat in walk.beats:
            text = clean_text(beat.get("text", "")).lower()
            if len(text) > 15:  # Skip very short lines
                words = text.split()
                matches = sum(1 for w in words if w in prose_lower)
                if matches / max(1, len(words)) >= 0.5:
                    count += 1
                    break  # Count each walk only once

    return count


def count_words_defined(prose: str, words: List[Tuple[str, str]]) -> List[str]:
    """Count which vocabulary words appear with definitional patterns."""
    # Strip markdown formatting for pattern matching
    prose_clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', prose)  # **bold**
    prose_clean = re.sub(r'\*([^*]+)\*', r'\1', prose_clean)  # *italic*
    prose_lower = prose_clean.lower()
    defined = []

    for word, _ in words:
        # Check if word appears
        word_present = any(
            re.search(rf'\b{re.escape(infl)}\b', prose_lower)
            for infl in get_inflections(word)
        )
        if not word_present:
            continue

        # Check for definitional pattern near the word
        # Multiple patterns to catch different definition styles
        patterns = [
            rf'\b{word}\b[^.]*\bmeans\b',           # "word means..."
            rf'\bmeans\b[^.]*\b{word}\b',           # "means to word"
            rf'\bto\s+{word}\b[^.]*\bis\b',         # "to word is..."
            rf'\b{word}\b[^.]*\bis when\b',         # "word is when..."
            rf'\ba\s+{word}\b\s+is\b',              # "a word is..."
            rf'\b{word}\b[^.]*\bis to\b',           # "word is to..."
            rf'\b{word}\b[^.—]*—[^.]*\b(to|a|the)\b',  # "word—to do something"
            rf'\bto\s+{word}\s+is\s+to\b',          # "to word is to..."
        ]
        for pattern in patterns:
            if re.search(pattern, prose_lower):
                defined.append(word)
                break

    return defined


def apply_aesop_filters(
    prose: str,
    walks: List[Walk],
    words: List[Tuple[str, str]],
    min_walks_used: int = 4,
    min_words_defined: int = 5,
    min_word_count: int = 80,
    max_word_count: int = 600,
    skip_meta_check: bool = False,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Apply rejection filters to aesop output.

    Returns: (passed, reject_reason, metrics)
    """
    metrics = {}

    # Word count
    word_count = len(prose.split())
    metrics["word_count"] = word_count
    if word_count < min_word_count:
        return False, "too_short", metrics
    if word_count > max_word_count:
        return False, "too_long", metrics

    # Meta-commentary check (skip if cleaner already handled this)
    if not skip_meta_check:
        prose_lower = prose.lower()
        for pattern in META_PATTERNS:
            if pattern in prose_lower:
                return False, f"meta_{pattern.replace(' ', '_')}", metrics

    # Count walks used
    walks_used = count_walks_used(prose, walks)
    metrics["walks_used"] = walks_used
    if walks_used < min_walks_used:
        return False, f"insufficient_walks_{walks_used}", metrics

    # Count words defined
    words_defined = count_words_defined(prose, words)
    metrics["words_defined"] = words_defined
    if len(words_defined) < min_words_defined:
        return False, f"insufficient_definitions_{len(words_defined)}", metrics

    # FK score (informational)
    fk = measure_fk_grade(prose)
    metrics["fk_measured"] = fk

    return True, None, metrics


# =============================================================================
# Prose Cleaning (LLM-based boilerplate stripping)
# =============================================================================

CLEAN_PROMPT = """Extract prose payload from this text. Return JSON only, no markdown.

Text:
{text}

Return: {{"prose": "extracted narrative here", "error": null}}

Error codes (use instead of prose if no valid narrative):
- "empty": no content
- "meta_only": only assistant commentary like "Here is..." or "Let me write..."
- "truncated": narrative cut off mid-sentence
- "malformed": unparseable

If prose exists with minor issues, include it and add "notes" field."""


async def clean_prose_payload(
    raw_text: str,
    llm_call: callable,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Use LLM to extract prose payload, stripping assistant boilerplate.

    Returns: (cleaned_prose, error_code, notes)
    """
    if not raw_text or len(raw_text) < 50:
        return None, "empty", None

    prompt = CLEAN_PROMPT.format(text=raw_text[:3000])  # Truncate if huge

    try:
        response = await llm_call(prompt)

        # Parse JSON from response
        # Handle potential markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```\w*\n?', '', response)
            response = re.sub(r'\n?```$', '', response)

        data = json.loads(response)

        prose = data.get("prose")
        error = data.get("error")
        notes = data.get("notes")

        if error:
            return None, error, notes
        if prose and len(prose) > 50:
            return prose, None, notes
        return None, "empty", notes

    except json.JSONDecodeError:
        # If LLM didn't return valid JSON, try to salvage
        # Sometimes it just returns the prose directly
        if len(raw_text) > 100 and not any(p in raw_text.lower()[:50] for p in ["let me", "here is", "i'll write"]):
            return raw_text, None, "json_parse_failed_using_raw"
        return None, "malformed", None
    except Exception as e:
        return None, f"clean_error_{str(e)[:30]}", None


# =============================================================================
# Generation Pipeline
# =============================================================================

def generate_aesop_id(walks: List[Walk], words: List[Tuple[str, str]]) -> str:
    """Generate stable ID for an aesop batch."""
    content = json.dumps({
        "walks": [[b.get("text", "") for b in w.beats] for w in walks],
        "words": [w[0] for w in words],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


async def generate_aesop(
    walks: List[Walk],
    words: List[Tuple[str, str]],
    llm_call: callable,
    source_corpus: str = "unknown",
    min_walks: int = 4,
    min_words: int = 5,
    use_cleaner: bool = False,
) -> BrainrotAesop:
    """
    Generate a brainrot-aesop from walks and words.

    The model chooses which walks and words to pair together.
    If use_cleaner=True, a second LLM call strips assistant boilerplate.
    """
    aesop_id = generate_aesop_id(walks, words)
    word_names = [w[0] for w in words]

    # Build prompt
    prompt = build_aesop_prompt(walks, words, min_walks, min_words)

    try:
        raw_prose = await llm_call(prompt)
    except Exception as e:
        return BrainrotAesop(
            id=f"aesop_{aesop_id}",
            words_offered=word_names,
            words_used=[],
            walks_offered=len(walks),
            walks_used=0,
            prose="",
            word_count=0,
            fk_measured=0.0,
            source_corpus=source_corpus,
            passed_filters=False,
            reject_reason=f"llm_error_{str(e)[:50]}",
        )

    # Optional: clean prose with second LLM call
    if use_cleaner:
        cleaned, clean_error, clean_notes = await clean_prose_payload(raw_prose, llm_call)
        if clean_error:
            return BrainrotAesop(
                id=f"aesop_{aesop_id}",
                words_offered=word_names,
                words_used=[],
                walks_offered=len(walks),
                walks_used=0,
                prose="",
                word_count=0,
                fk_measured=0.0,
                source_corpus=source_corpus,
                passed_filters=False,
                reject_reason=f"clean_{clean_error}",
            )
        prose = cleaned
    else:
        prose = raw_prose

    # Apply filters (skip meta check if cleaner was used)
    passed, reject_reason, metrics = apply_aesop_filters(
        prose, walks, words, min_walks, min_words,
        skip_meta_check=use_cleaner,  # Cleaner already handled this
    )

    return BrainrotAesop(
        id=f"aesop_{aesop_id}",
        words_offered=word_names,
        words_used=metrics.get("words_defined", []),
        walks_offered=len(walks),
        walks_used=metrics.get("walks_used", 0),
        prose=prose if passed else "",
        word_count=metrics.get("word_count", 0),
        fk_measured=metrics.get("fk_measured", 0.0),
        source_corpus=source_corpus,
        passed_filters=passed,
        reject_reason=reject_reason,
    )


async def generate_aesops_batch(
    walks: List[Walk],
    llm_call: callable,
    n_aesops: int = 10,
    walks_per_aesop: int = 8,
    words_per_aesop: int = 12,
    source_corpus: str = "unknown",
    concurrency: int = 5,
    use_cleaner: bool = False,
) -> List[BrainrotAesop]:
    """
    Generate multiple aesops with model-driven pairing.

    Each aesop gets a random sample of walks and words.
    The model decides which pairings work.

    If use_cleaner=True, each aesop uses a second LLM call to strip boilerplate.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def generate_one():
        async with semaphore:
            # Sample walks and words for this aesop
            sampled_walks = random.sample(walks, min(walks_per_aesop, len(walks)))
            sampled_words = sample_words(words_per_aesop)

            return await generate_aesop(
                sampled_walks,
                sampled_words,
                llm_call,
                source_corpus,
                use_cleaner=use_cleaner,
            )

    tasks = [generate_one() for _ in range(n_aesops)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, BrainrotAesop)]


# =============================================================================
# CLI
# =============================================================================

async def mock_llm_call(prompt: str) -> str:
    """Mock LLM that generates multi-walk, multi-word prose."""
    # Extract walks from prompt
    walk_matches = re.findall(r'Walk \d+:\n((?:  "[^"]+"\n?)+)', prompt)
    dialogues = []
    for walk_text in walk_matches:
        for line in re.findall(r'"([^"]+)"', walk_text):
            dialogues.append(line)

    # Extract vocabulary words
    word_matches = re.findall(r'- (\w+):', prompt)

    if not dialogues or not word_matches:
        return "No content found."

    # Generate mock prose with multiple walks and definitions
    passage = []
    passage.append("The bureau was busy that morning. Officials hurried between desks.")

    # Use first few dialogues
    if len(dialogues) > 0:
        passage.append(f'A clerk announced, "{dialogues[0]}"')
    if len(dialogues) > 1:
        passage.append(f'Someone replied, "{dialogues[1]}"')

    # Add definitions for several words
    for i, word in enumerate(word_matches[:6]):
        if i == 0:
            passage.append(f"To {word} means to take purposeful action. Everyone must {word} at some point.")
        elif i == 1:
            passage.append(f"A {word} is something important in daily life.")
        elif i == 2:
            passage.append(f"When you {word}, you are doing something meaningful.")

    # More dialogues
    if len(dialogues) > 2:
        passage.append(f'The supervisor added, "{dialogues[2]}"')
    if len(dialogues) > 3:
        passage.append(f'"{dialogues[3]}" came the response.')

    # More definitions
    for word in word_matches[3:5]:
        passage.append(f"To {word} is to engage in a basic human activity.")

    passage.append("Such is life in the bureau. Forms and definitions, dialogue and duty.")

    return " ".join(passage)


async def main():
    """Test the brainrot-aesops v3 generator."""

    print("=" * 60)
    print("BRAINROT-AESOPS v3 TEST (model-driven pairing)")
    print("=" * 60)

    # Create test walks
    walks = [
        Walk(
            beats=[
                {"text": "Patrolling the Mojave almost makes you wish for a nuclear winter.", "emotion": "neutral"},
                {"text": "War never changes.", "emotion": "sad"},
            ],
            source="test_fnv",
        ),
        Walk(
            beats=[
                {"text": "Stop right there, criminal scum!", "emotion": "anger"},
                {"text": "Pay the fine or serve your sentence.", "emotion": "neutral"},
            ],
            source="test_oblivion",
        ),
        Walk(
            beats=[
                {"text": "I need to find my way home.", "emotion": "sad"},
                {"text": "Can you help me?", "emotion": "neutral"},
            ],
            source="test_custom",
        ),
        Walk(
            beats=[
                {"text": "The Hexagon expects compliance.", "emotion": "neutral"},
                {"text": "Seventy-two hours.", "emotion": "anger"},
            ],
            source="test_gallia",
        ),
        Walk(
            beats=[
                {"text": "Your papers are not in order.", "emotion": "disgust"},
                {"text": "Come back tomorrow.", "emotion": "neutral"},
            ],
            source="test_gallia",
        ),
    ]

    # Sample words
    words = sample_words(10)
    print(f"\n--- Sampled {len(words)} words ---")
    print([w[0] for w in words])

    # Show prompt preview
    print(f"\n--- Prompt Preview ---")
    prompt = build_aesop_prompt(walks[:5], words[:8])
    print(prompt[:600] + "...")

    # Generate aesops
    print(f"\n--- Generation ---")
    aesops = await generate_aesops_batch(
        walks, mock_llm_call, n_aesops=2, source_corpus="test"
    )

    print(f"Generated {len(aesops)} aesops")

    for aesop in aesops:
        print(f"\n--- Aesop {aesop.id[:12]} ---")
        print(f"Passed: {aesop.passed_filters}")
        print(f"Walks: {aesop.walks_used}/{aesop.walks_offered}")
        print(f"Words defined: {aesop.words_used}")
        if aesop.passed_filters:
            print(f"Prose preview: {aesop.prose[:200]}...")
        else:
            print(f"Reject: {aesop.reject_reason}")


if __name__ == "__main__":
    asyncio.run(main())
