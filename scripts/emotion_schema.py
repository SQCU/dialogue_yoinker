#!/usr/bin/env python3
"""
Emotion schema validation for synthetic dialogue graphs.

Canonical emotions from Bethesda TRDT subrecords (facial animation data):
  neutral, anger, fear, happy, sad, disgust, surprise

This module validates and maps LLM-generated emotions to canonical values.
Philosophy: treat generating models like fermentation - side effects are
our job to measure and filter, not theirs to avoid.
"""

from dataclasses import dataclass, field
from typing import Literal

# The 7 canonical emotions from Bethesda's TRDT facial animation system
# (1 baseline + 6 off-neutral sentiments)
CANONICAL_EMOTIONS = frozenset({
    "neutral",
    "anger",
    "fear",
    "happy",
    "sad",
    "disgust",
    "surprise",
})

# Mapping of common LLM-generated "creative" emotions to canonical equivalents
# When in doubt, map to neutral (least harmful default)
EMOTION_MAP: dict[str, str] = {
    # Fear variants
    "anxiety": "fear",
    "anxious": "fear",
    "unease": "fear",
    "uneasy": "fear",
    "dread": "fear",
    "bureaucratic_dread": "fear",
    "terror": "fear",
    "nervous": "fear",
    "paranoia": "fear",
    "paranoid": "fear",
    "worried": "fear",
    "worry": "fear",
    "apprehension": "fear",
    "apprehensive": "fear",
    "trepidation": "fear",
    "alarmed": "fear",
    "alarm": "fear",
    "frightened": "fear",
    "scared": "fear",
    "intimidated": "fear",
    "concern": "fear",
    "concerned": "fear",
    "dawning_horror": "fear",
    "dawning horror": "fear",
    "warning": "fear",

    # Anger variants
    "frustrated": "anger",
    "frustration": "anger",
    "irritated": "anger",
    "irritation": "anger",
    "annoyance": "anger",
    "annoyed": "anger",
    "rage": "anger",
    "fury": "anger",
    "furious": "anger",
    "hostile": "anger",
    "hostility": "anger",
    "defiance": "anger",
    "defiant": "anger",
    "aggressive": "anger",
    "aggression": "anger",
    "outrage": "anger",
    "outraged": "anger",
    "indignation": "anger",
    "indignant": "anger",
    "resentment": "anger",
    "resentful": "anger",
    "angry": "anger",  # common LLM typo/variant
    "controlled_anger": "anger",
    "threatening": "anger",
    "accusatory": "anger",
    "defensive": "anger",

    # Sad variants
    "resignation": "sad",
    "resigned": "sad",
    "melancholy": "sad",
    "sorrow": "sad",
    "sorrowful": "sad",
    "grief": "sad",
    "grieving": "sad",
    "despair": "sad",
    "despairing": "sad",
    "hopeless": "sad",
    "hopelessness": "sad",
    "dejected": "sad",
    "dejection": "sad",
    "disappointed": "sad",
    "disappointment": "sad",
    "regret": "sad",
    "regretful": "sad",
    "weary": "sad",
    "weariness": "sad",
    "tired": "sad",
    "exhausted": "sad",
    "beaten resignation": "sad",
    "resigned_acceptance": "sad",
    "cynical_resignation": "sad",
    "giving up": "sad",

    # Disgust variants
    "contempt": "disgust",
    "contemptuous": "disgust",
    "revulsion": "disgust",
    "repulsed": "disgust",
    "disdain": "disgust",
    "disdainful": "disgust",
    "scorn": "disgust",
    "scornful": "disgust",
    "distaste": "disgust",
    "aversion": "disgust",
    "loathing": "disgust",
    "dismissive": "disgust",
    "mocking": "disgust",
    "exasperation": "disgust",
    "exasperated": "disgust",

    # Happy variants
    "pleased": "happy",
    "pleasure": "happy",
    "joy": "happy",
    "joyful": "happy",
    "satisfied": "happy",
    "satisfaction": "happy",
    "content": "happy",
    "contentment": "happy",
    "amused": "happy",
    "amusement": "happy",
    "delighted": "happy",
    "delight": "happy",
    "relieved": "happy",
    "relief": "happy",
    "excited": "happy",
    "excitement": "happy",
    "hopeful": "happy",
    "hope": "happy",
    "eager": "happy",
    "enthusiasm": "happy",
    "enthusiastic": "happy",

    # Surprise variants
    "shocked": "surprise",
    "shock": "surprise",
    "astonished": "surprise",
    "astonishment": "surprise",
    "amazed": "surprise",
    "amazement": "surprise",
    "startled": "surprise",
    "bewildered": "surprise",
    "bewilderment": "surprise",
    "stunned": "surprise",
    "disbelief": "surprise",
    "surprised": "surprise",  # common LLM variant
    "realization": "surprise",

    # Neutral variants (contemplative/cognitive states)
    "contemplative": "neutral",
    "contemplation": "neutral",
    "thoughtful": "neutral",
    "pensive": "neutral",
    "reflective": "neutral",
    "curious": "neutral",
    "curiosity": "neutral",
    "interested": "neutral",
    "interest": "neutral",
    "cautious": "neutral",
    "caution": "neutral",
    "suspicious": "neutral",  # could be fear, but neutral is safer
    "suspicion": "neutral",
    "uncertain": "neutral",
    "uncertainty": "neutral",
    "confused": "neutral",
    "confusion": "neutral",
    "calm": "neutral",
    "composed": "neutral",
    "stoic": "neutral",
    "indifferent": "neutral",
    "detached": "neutral",
    "matter-of-fact": "neutral",
    "professional": "neutral",
    "formal": "neutral",
    "businesslike": "neutral",
    "guarded": "neutral",
    "wary": "neutral",  # borderline fear, but neutral safer
    "measured": "neutral",
    "reserved": "neutral",
    "impassive": "neutral",
    "analytical": "neutral",
    "calculating": "neutral",
    "cold": "neutral",
    "confident": "neutral",
    "determined": "neutral",
    "urgent": "neutral",
    "serious": "neutral",
    "stern": "neutral",
    "flat": "neutral",
    "monotone": "neutral",
    "intrigued": "neutral",
    "unsure": "neutral",
    "cold_formality": "neutral",
    "neutral_with_hint": "neutral",

    # Pained variants
    "hurt": "sad",
    "anguish": "sad",
    "anguished": "sad",
    "suffering": "sad",
    "tormented": "sad",
    "distressed": "fear",
    "distress": "fear",
    "wounded": "sad",
    "pained": "sad",
}


@dataclass
class ValidationResult:
    """Result of validating an emotion value."""
    original: str
    canonical: str
    was_valid: bool
    was_in_map: bool = False  # True if non-canonical but found in EMOTION_MAP

    @property
    def was_mapped(self) -> bool:
        """True if emotion was non-canonical but successfully mapped via EMOTION_MAP."""
        return not self.was_valid and self.was_in_map

    @property
    def was_defaulted(self) -> bool:
        """True if emotion was unknown and defaulted."""
        return not self.was_valid and not self.was_in_map


@dataclass
class ValidationStats:
    """Aggregated validation statistics."""
    total: int = 0
    valid: int = 0
    mapped: int = 0
    defaulted: int = 0
    unknown_emotions: dict[str, int] = field(default_factory=dict)

    @property
    def defect_rate(self) -> float:
        """Fraction of emotions that were non-canonical."""
        return (self.total - self.valid) / self.total if self.total > 0 else 0.0

    @property
    def recovery_rate(self) -> float:
        """Fraction of non-canonical emotions that were successfully mapped."""
        non_canonical = self.total - self.valid
        return self.mapped / non_canonical if non_canonical > 0 else 1.0

    def record(self, result: ValidationResult) -> None:
        """Record a validation result."""
        self.total += 1
        if result.was_valid:
            self.valid += 1
        elif result.was_mapped:
            self.mapped += 1
        else:
            self.defaulted += 1
            original_lower = result.original.lower()
            self.unknown_emotions[original_lower] = self.unknown_emotions.get(original_lower, 0) + 1

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Emotion validation: {self.total} total",
            f"  Valid (canonical): {self.valid} ({self.valid/self.total*100:.1f}%)" if self.total else "  No emotions processed",
            f"  Mapped (known variant): {self.mapped} ({self.mapped/self.total*100:.1f}%)" if self.total else "",
            f"  Defaulted (unknown): {self.defaulted} ({self.defaulted/self.total*100:.1f}%)" if self.total else "",
            f"  Defect rate: {self.defect_rate*100:.1f}%",
            f"  Recovery rate: {self.recovery_rate*100:.1f}%",
        ]
        if self.unknown_emotions:
            lines.append("  Unknown emotions (consider adding to map):")
            for emo, count in sorted(self.unknown_emotions.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"    {emo}: {count}")
        return "\n".join(lines)


def validate_emotion(emotion: str, default: str = "neutral") -> ValidationResult:
    """
    Validate and potentially map an emotion to canonical form.

    Args:
        emotion: The emotion string to validate
        default: Fallback if emotion is unknown (must be canonical)

    Returns:
        ValidationResult with original, canonical, and validity info
    """
    if default not in CANONICAL_EMOTIONS:
        raise ValueError(f"Default must be canonical, got: {default}")

    emotion_lower = emotion.lower().strip()

    # Already canonical
    if emotion_lower in CANONICAL_EMOTIONS:
        return ValidationResult(
            original=emotion,
            canonical=emotion_lower,
            was_valid=True,
        )

    # Known variant - map it
    if emotion_lower in EMOTION_MAP:
        return ValidationResult(
            original=emotion,
            canonical=EMOTION_MAP[emotion_lower],
            was_valid=False,
            was_in_map=True,
        )

    # Unknown - default
    return ValidationResult(
        original=emotion,
        canonical=default,
        was_valid=False,
    )


def validate_emotions_batch(
    emotions: list[str],
    default: str = "neutral"
) -> tuple[list[str], ValidationStats]:
    """
    Validate a batch of emotions, returning canonical versions and stats.

    Args:
        emotions: List of emotion strings
        default: Fallback for unknown emotions

    Returns:
        Tuple of (canonical_emotions, validation_stats)
    """
    stats = ValidationStats()
    canonical = []

    for emotion in emotions:
        result = validate_emotion(emotion, default)
        stats.record(result)
        canonical.append(result.canonical)

    return canonical, stats


# Type for canonical emotions (useful for type hints)
CanonicalEmotion = Literal[
    "neutral", "anger", "fear", "happy",
    "sad", "disgust", "surprise"
]


if __name__ == "__main__":
    # Demo/test
    test_emotions = [
        "neutral",  # valid
        "anger",    # valid
        "anxiety",  # -> fear
        "bureaucratic_dread",  # -> fear
        "contemplative",  # -> neutral
        "resigned",  # -> sad
        "cosmic_horror",  # unknown -> neutral
        "professional_detachment",  # unknown -> neutral
    ]

    print("Emotion Schema Validator Demo")
    print("=" * 40)

    for emo in test_emotions:
        result = validate_emotion(emo)
        status = "valid" if result.was_valid else ("mapped" if result.was_mapped else "defaulted")
        print(f"  {emo:25} -> {result.canonical:10} ({status})")

    print("\nBatch validation:")
    canonical, stats = validate_emotions_batch(test_emotions)
    print(stats.summary())
