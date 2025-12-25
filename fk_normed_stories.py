#!/usr/bin/env python3
"""
FK-Normed Stories Generator

Expands dialogue walks into narrated prose at controlled Flesch-Kincaid reading levels.
Uses the prose_wrapper as a baseline, then applies LLM expansion for FK targeting.

Two output tiers:
- Tier 1: Flattened walks (sparse turn structure)
- Tier 2: FK-normed prose (explicit narrative context)
"""

import json
import hashlib
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("Warning: textstat not installed. FK measurement will be approximate.")

from prose_wrapper import Walk, wrap_walk, extract_walks_from_graph, clean_text


# =============================================================================
# FK Measurement
# =============================================================================

def measure_fk_grade(text: str) -> float:
    """Measure Flesch-Kincaid grade level."""
    if HAS_TEXTSTAT:
        return textstat.flesch_kincaid_grade(text)
    else:
        # Rough approximation: average sentence length + word length
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(1, sentences)
        words = len(text.split())
        syllables = sum(1 for c in text.lower() if c in 'aeiou')  # Very rough

        asl = words / sentences
        asw = syllables / max(1, words)

        # FK formula approximation
        return 0.39 * asl + 11.8 * asw - 15.59


def fk_within_tolerance(measured: float, target: int, tolerance: float = 1.5) -> bool:
    """Check if measured FK is within tolerance of target."""
    return abs(measured - target) <= tolerance


# =============================================================================
# Prompt Templates
# =============================================================================

FK_LEVEL_DESCRIPTIONS = {
    0: "Grade 0 (age 5-6): Very short sentences. Common words only. 'See Spot run.'",
    2: "Grade 2 (age 7-8): Short simple sentences. Basic vocabulary. Direct statements.",
    3: "Grade 3 (age 8-9): Simple sentences. Basic vocabulary. Clear cause-effect.",
    4: "Grade 4 (age 9-10): Simple compound sentences. Some descriptive words.",
    6: "Grade 6 (age 11-12): Compound sentences. Some domain vocabulary. Subplots OK.",
    9: "Grade 9 (age 14-15): Complex sentences. Abstract concepts. Nuanced emotion.",
}

FK_EXAMPLES = {
    0: '''Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."

Output:
The man held up three fingers. "Three days," he said.
The boy came back. The man was mad. "Time is up," he said.''',

    3: '''Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."

Output:
The soldier gave a warning. "You have three days to leave," he said.
Two days later, the soldier returned. He frowned when he saw the family
still there. "I told you three days. Time's up now."''',

    6: '''Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."

Output:
The officer's voice was flat, businesslike. "Three days. That's what
you have to settle your affairs and vacate the premises."

When he returned forty-eight hours later, his expression shifted from
neutral to something harder. "Still here? I was hoping you'd prove
smarter than the last family." He pulled out a form. "Time's up.
We do this the official way now."''',

    9: '''Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."

Output:
"Three days." The bureaucrat didn't look up from his ledger as he
spoke, the words carrying the weight of procedure rather than
personal animosity. It was simply how these things were done.

The second visit carried a different tenor. He paused at the
threshold, surveying the unchanged interior with something between
disappointment and resignation. "Still here?" The question was
almost rhetorical. They both knew the answer, and they both knew
what came next.

"Time's up." This time he met her eyes, and whatever professional
detachment he'd maintained had calcified into something colder.''',
}


def build_fk_prompt(walk: Walk, target_fk: int, bible_excerpt: Optional[str] = None) -> str:
    """Build prompt for LLM expansion at a specific FK level."""

    # Format the trajectory
    trajectory_lines = []
    for beat in walk.beats:
        emotion = beat.get("emotion", "neutral")
        text = clean_text(beat.get("text", ""))
        trajectory_lines.append(f'- [{emotion}] "{text}"')
    trajectory_str = "\n".join(trajectory_lines)

    # Get FK description and example
    fk_desc = FK_LEVEL_DESCRIPTIONS.get(target_fk, FK_LEVEL_DESCRIPTIONS[6])
    fk_example = FK_EXAMPLES.get(target_fk, FK_EXAMPLES[6])

    # Build prompt
    prompt = f"""You are expanding dialogue trajectories into narrated prose at specific reading levels.

{fk_desc}

## Example at this level:

{fk_example}

---
"""

    if bible_excerpt:
        prompt += f"""
## Setting Context:

{bible_excerpt}

---
"""

    prompt += f"""
## Your Task:

Expand the following trajectory to narrated prose at Flesch-Kincaid grade level {target_fk}.

Trajectory:
{trajectory_str}

Emotion arc: {' → '.join(b.get('emotion', 'neutral') for b in walk.beats)}

Write narrated prose that:
- Preserves all dialogue lines (may paraphrase slightly for grade-level fit)
- Adds speaker attribution appropriate to the reading level
- Includes brief scene-setting and action beats
- Maintains the emotional arc

Prose:"""

    return prompt


# =============================================================================
# Output Formats
# =============================================================================

@dataclass
class FlattenedWalk:
    """Tier 1: Sparse turn structure."""
    id: str
    text: str
    source: str
    tier: str = "flattened"
    emotion_sequence: List[str] = None

    def __post_init__(self):
        if self.emotion_sequence is None:
            self.emotion_sequence = []


@dataclass
class FKNormedStory:
    """Tier 2: FK-normed prose expansion."""
    id: str
    source_walk_id: str
    fk_target: int
    fk_measured: float
    prose: str
    word_count: int
    source: str
    tier: str = "fk_normed"
    passed_filters: bool = True
    reject_reason: Optional[str] = None


def flatten_walk(walk: Walk, walk_id: str) -> FlattenedWalk:
    """Convert walk to flattened format (Tier 1)."""
    lines = []
    for beat in walk.beats:
        text = clean_text(beat.get("text", ""))
        lines.append(f'"{text}"')

    return FlattenedWalk(
        id=f"flat_{walk_id}",
        text="\n".join(lines),
        source=walk.source,
        emotion_sequence=[b.get("emotion", "neutral") for b in walk.beats],
    )


# =============================================================================
# Rejection Filters
# =============================================================================

META_COMMENTARY_PATTERNS = [
    "as an ai",
    "i'll write",
    "i will write",
    "here is",
    "here's the",
    "let me",
    "i'd be happy",
    "i cannot",
    "i can't",
]


def check_dialogue_preservation(original_beats: List[Dict], prose: str, threshold: float = 0.8) -> Tuple[bool, float]:
    """Check if dialogue lines are preserved in the prose."""
    preserved = 0
    total = len(original_beats)

    if total == 0:
        return True, 1.0

    prose_lower = prose.lower()
    for beat in original_beats:
        text = clean_text(beat.get("text", "")).lower()
        # Check for fuzzy presence (at least 50% of words)
        words = text.split()
        if len(words) == 0:
            preserved += 1
            continue

        matches = sum(1 for w in words if w in prose_lower)
        if matches / len(words) >= 0.5:
            preserved += 1

    ratio = preserved / total
    return ratio >= threshold, ratio


def apply_rejection_filters(
    prose: str,
    original_beats: List[Dict],
    target_fk: int,
    fk_tolerance: float = 1.5,
    min_words: int = 30,
    max_words: int = 400,
) -> Tuple[bool, Optional[str], float]:
    """
    Apply rejection filters to generated prose.

    Returns: (passed, reject_reason, measured_fk)
    """
    # Word count check
    word_count = len(prose.split())
    if word_count < min_words:
        return False, f"too_short_{word_count}", 0.0
    if word_count > max_words:
        return False, f"too_long_{word_count}", 0.0

    # Meta-commentary check
    prose_lower = prose.lower()
    for pattern in META_COMMENTARY_PATTERNS:
        if pattern in prose_lower:
            return False, f"meta_commentary_{pattern}", 0.0

    # FK score check
    measured_fk = measure_fk_grade(prose)
    if not fk_within_tolerance(measured_fk, target_fk, fk_tolerance):
        return False, f"fk_mismatch_{measured_fk:.1f}_vs_{target_fk}", measured_fk

    # Dialogue preservation check
    preserved, ratio = check_dialogue_preservation(original_beats, prose)
    if not preserved:
        return False, f"dialogue_lost_{ratio:.2f}", measured_fk

    return True, None, measured_fk


# =============================================================================
# Generation Pipeline
# =============================================================================

def generate_walk_id(walk: Walk) -> str:
    """Generate a stable ID for a walk."""
    content = json.dumps([b.get("text", "") for b in walk.beats], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


async def process_walk(
    walk: Walk,
    walk_id: str,
    fk_levels: List[int],
    llm_call: callable,
    bible_excerpt: Optional[str] = None,
) -> Tuple[FlattenedWalk, List[FKNormedStory]]:
    """
    Process a single walk, producing Tier 1 (flattened) and Tier 2 (FK-normed) outputs.

    Args:
        walk: The walk to process
        walk_id: Unique identifier for this walk
        fk_levels: List of FK grade targets [0, 3, 6, 9]
        llm_call: Async function(prompt: str) -> str for LLM generation
        bible_excerpt: Optional setting context

    Returns:
        (flattened_walk, list_of_fk_stories)
    """
    # Tier 1: Flatten
    flattened = flatten_walk(walk, walk_id)

    # Tier 2: FK-normed expansions
    stories = []

    for fk in fk_levels:
        prompt = build_fk_prompt(walk, fk, bible_excerpt)

        try:
            prose = await llm_call(prompt)
        except Exception as e:
            stories.append(FKNormedStory(
                id=f"fk_{walk_id}_grade{fk}",
                source_walk_id=walk_id,
                fk_target=fk,
                fk_measured=0.0,
                prose="",
                word_count=0,
                source=walk.source,
                passed_filters=False,
                reject_reason=f"llm_error_{str(e)[:50]}",
            ))
            continue

        # Apply filters
        passed, reject_reason, measured_fk = apply_rejection_filters(
            prose, walk.beats, fk
        )

        stories.append(FKNormedStory(
            id=f"fk_{walk_id}_grade{fk}",
            source_walk_id=walk_id,
            fk_target=fk,
            fk_measured=measured_fk,
            prose=prose if passed else "",
            word_count=len(prose.split()) if passed else 0,
            source=walk.source,
            passed_filters=passed,
            reject_reason=reject_reason,
        ))

    return flattened, stories


async def process_walks_batch(
    walks: List[Walk],
    llm_call: callable,
    fk_levels: List[int] = [0, 3, 6, 9],
    bible_excerpt: Optional[str] = None,
    concurrency: int = 10,
) -> Tuple[List[FlattenedWalk], List[FKNormedStory]]:
    """
    Process multiple walks with controlled concurrency.
    """
    all_flattened = []
    all_stories = []

    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(walk: Walk, idx: int):
        async with semaphore:
            walk_id = generate_walk_id(walk)
            return await process_walk(walk, walk_id, fk_levels, llm_call, bible_excerpt)

    tasks = [process_one(walk, i) for i, walk in enumerate(walks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            print(f"Error processing walk: {result}")
            continue
        flattened, stories = result
        all_flattened.append(flattened)
        all_stories.extend(stories)

    return all_flattened, all_stories


# =============================================================================
# Output Writers
# =============================================================================

def write_training_jsonl(
    flattened: List[FlattenedWalk],
    stories: List[FKNormedStory],
    output_path: Path,
    include_failed: bool = False,
):
    """Write combined training data to JSONL."""
    with open(output_path, 'w') as f:
        # Write tier 1
        for flat in flattened:
            f.write(json.dumps(asdict(flat)) + '\n')

        # Write tier 2 (passed only unless include_failed)
        for story in stories:
            if story.passed_filters or include_failed:
                f.write(json.dumps(asdict(story)) + '\n')


def write_stats(
    flattened: List[FlattenedWalk],
    stories: List[FKNormedStory],
    output_path: Path,
):
    """Write generation statistics."""
    passed = [s for s in stories if s.passed_filters]
    failed = [s for s in stories if not s.passed_filters]

    # Group failures by reason
    failure_reasons = {}
    for s in failed:
        reason = s.reject_reason or "unknown"
        reason_type = reason.split("_")[0]
        failure_reasons[reason_type] = failure_reasons.get(reason_type, 0) + 1

    # FK distribution
    fk_dist = {}
    for s in passed:
        fk_dist[s.fk_target] = fk_dist.get(s.fk_target, 0) + 1

    stats = {
        "tier1_count": len(flattened),
        "tier2_total": len(stories),
        "tier2_passed": len(passed),
        "tier2_failed": len(failed),
        "pass_rate": len(passed) / max(1, len(stories)),
        "fk_distribution": fk_dist,
        "failure_reasons": failure_reasons,
    }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# =============================================================================
# CLI
# =============================================================================

async def mock_llm_call(prompt: str) -> str:
    """Mock LLM for testing - returns prose_wrapper output."""
    # Extract trajectory from prompt
    import re

    lines = []
    for match in re.finditer(r'- \[(\w+)\] "(.+?)"', prompt):
        emotion, text = match.groups()
        lines.append({"text": text, "emotion": emotion})

    if not lines:
        return "No trajectory found in prompt."

    walk = Walk(beats=lines, source="mock")
    output = wrap_walk(walk)
    return output.prose


async def main():
    """Test the FK-normed stories generator."""
    from pathlib import Path

    print("=" * 60)
    print("FK-NORMED STORIES GENERATOR TEST")
    print("=" * 60)

    # Create test walk
    walk = Walk(
        beats=[
            {"text": "Seventy-two hours.", "emotion": "neutral", "beat_function": "establish_stakes"},
            {"text": "The Hexagon expects compliance.", "emotion": "neutral", "beat_function": "threaten"},
            {"text": "And if I refuse?", "emotion": "anger", "beat_function": "query"},
        ],
        arc_shape="escalating_threat",
        archetype_relation="authority_to_subject",
        source="test",
    )

    walk_id = generate_walk_id(walk)
    print(f"\nWalk ID: {walk_id}")
    print(f"Beats: {len(walk.beats)}")

    # Test flattening
    flattened = flatten_walk(walk, walk_id)
    print(f"\n--- Tier 1 (Flattened) ---")
    print(flattened.text)
    print(f"Emotions: {flattened.emotion_sequence}")

    # Test FK prompt generation
    print(f"\n--- FK Prompt (Grade 3) ---")
    prompt = build_fk_prompt(walk, 3)
    print(prompt[:500] + "...")

    # Test with mock LLM
    print(f"\n--- Tier 2 (Mock LLM) ---")
    _, stories = await process_walk(
        walk, walk_id, [3, 6], mock_llm_call
    )

    for story in stories:
        print(f"\nGrade {story.fk_target}:")
        print(f"  Passed: {story.passed_filters}")
        if story.passed_filters:
            print(f"  FK measured: {story.fk_measured:.1f}")
            print(f"  Words: {story.word_count}")
            print(f"  Preview: {story.prose[:100]}...")
        else:
            print(f"  Reject: {story.reject_reason}")

    # Test stats
    print(f"\n--- Stats ---")
    stats = {
        "tier2_passed": sum(1 for s in stories if s.passed_filters),
        "tier2_failed": sum(1 for s in stories if not s.passed_filters),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
