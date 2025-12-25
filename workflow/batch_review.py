"""
Batch Review: Orchestrator reviews wave statistics, dispatches corrections.

The orchestrator doesn't read every translated line - it reviews aggregate
statistics of the translation wave and decides:
1. Metadata assignments (topic category, hub vs leaf, etc.)
2. In-fill tasks (missing edge types within the batch)
3. Out-fill tasks (connecting this batch to future/past batches)

Reference statistics from FNV:
  - Emotion self-loop rate: 74.6% global, 32.5% stdev locally
  - Topic avg degree: 2.12 (sparse)
  - Combat topics are hubs (degree 45-56)
  - Most topics are leaves (degree 1-2)
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics


@dataclass
class ReferenceStats:
    """Global statistics from reference corpus."""
    # Emotion transitions
    emotion_self_loop_rate: float = 0.746
    emotion_self_loop_local_stdev: float = 0.325
    emotions: list[str] = field(default_factory=lambda: [
        'neutral', 'anger', 'happy', 'fear', 'sad', 'disgust', 'surprise', 'pained'
    ])

    # Topic structure
    topic_avg_degree: float = 2.12
    topic_hub_threshold: int = 20  # degree > this = hub
    hub_topics: list[str] = field(default_factory=lambda: [
        'Attack', 'Assault', 'Death', 'AcceptYield', 'AlertToCombat',
        'CombatToLost', 'NormalToAlert', 'Flee', 'GREETING', 'GOODBYE'
    ])

    # Walk structure
    avg_walk_length: float = 4.4
    walk_length_stdev: float = 2.5


@dataclass
class WaveStatistics:
    """Statistics computed from a wave of translations."""
    wave_id: str
    synthetic_count: int

    # Emotion distribution
    emotion_counts: dict[str, int] = field(default_factory=dict)
    emotion_self_loop_rate: float = 0.0

    # Arc shapes
    arc_shape_counts: dict[str, int] = field(default_factory=dict)

    # Transitions produced
    transition_pairs: list[tuple[str, str]] = field(default_factory=list)  # (from_emotion, to_emotion)

    # Topic coverage (what semantic categories are represented)
    topic_categories: dict[str, int] = field(default_factory=dict)

    # Gaps detected
    missing_transitions: list[tuple[str, str]] = field(default_factory=list)
    underrepresented_categories: list[str] = field(default_factory=list)


@dataclass
class CorrectionTask:
    """A task dispatched by orchestrator to fill gaps."""
    task_type: str  # 'in_fill' | 'out_fill' | 'hub_connect' | 'leaf_extend'
    priority: int
    specification: dict
    # e.g., {"generate": "crime_bark", "emotion": "anger", "connect_to": ["alert_topic"]}


def compute_wave_statistics(translations: list[dict], ref: ReferenceStats) -> WaveStatistics:
    """
    Compute aggregate statistics from a wave of translations.

    This is what the orchestrator sees - NOT individual lines.
    """
    stats = WaveStatistics(
        wave_id="",
        synthetic_count=len(translations)
    )

    # Count emotions
    for t in translations:
        for text_idx, text in enumerate(t.get('translated_texts', [])):
            # We'd need emotion annotations here
            pass

    # Count arc shapes
    for t in translations:
        shape = t.get('arc_shape', 'unknown')
        stats.arc_shape_counts[shape] = stats.arc_shape_counts.get(shape, 0) + 1

    # Identify transition pairs from structural triplets
    for t in translations:
        arc = t.get('triplet', {}).get('arc', [])
        for i in range(len(arc) - 1):
            e1 = arc[i].get('emotion', 'neutral')
            e2 = arc[i+1].get('emotion', 'neutral')
            stats.transition_pairs.append((e1, e2))

    # Compute self-loop rate
    if stats.transition_pairs:
        self_loops = sum(1 for e1, e2 in stats.transition_pairs if e1 == e2)
        stats.emotion_self_loop_rate = self_loops / len(stats.transition_pairs)

    return stats


def identify_gaps(wave_stats: WaveStatistics, ref: ReferenceStats) -> list[CorrectionTask]:
    """
    Compare wave statistics to reference, identify gaps, generate correction tasks.

    This is the orchestrator's decision-making step.
    """
    tasks = []

    # Check self-loop rate deviation
    deviation = abs(wave_stats.emotion_self_loop_rate - ref.emotion_self_loop_rate)
    if deviation > ref.emotion_self_loop_local_stdev:
        # Wave is statistically unusual - might be fine (local variance) or might need correction
        if wave_stats.emotion_self_loop_rate > ref.emotion_self_loop_rate + ref.emotion_self_loop_local_stdev:
            # Too monotone - need more emotional transitions
            tasks.append(CorrectionTask(
                task_type='in_fill',
                priority=2,
                specification={
                    'goal': 'add_emotional_transitions',
                    'current_rate': wave_stats.emotion_self_loop_rate,
                    'target_rate': ref.emotion_self_loop_rate,
                    'suggest': 'Generate 2-3 walks with emotion changes (anger->neutral, neutral->fear)'
                }
            ))

    # Check arc shape diversity
    shapes = set(wave_stats.arc_shape_counts.keys())
    expected_shapes = {'escalating_threat', 'de_escalation', 'negotiation_arc', 'information_dump', 'ambient_chatter'}
    missing_shapes = expected_shapes - shapes
    if missing_shapes:
        tasks.append(CorrectionTask(
            task_type='in_fill',
            priority=1,
            specification={
                'goal': 'add_arc_shapes',
                'missing': list(missing_shapes),
                'suggest': f'Generate walks with arc shapes: {missing_shapes}'
            }
        ))

    # Check for hub topic coverage
    # (In a real implementation, we'd track which hub-equivalent topics are covered)

    return tasks


def format_orchestrator_prompt(wave_stats: WaveStatistics, gaps: list[CorrectionTask]) -> str:
    """
    Format the batch review for orchestrator consumption.

    The orchestrator sees this summary, NOT individual translations.
    """
    prompt = f"""## Wave Statistics Summary

**Synthetics generated**: {wave_stats.synthetic_count}
**Emotion self-loop rate**: {wave_stats.emotion_self_loop_rate*100:.1f}% (reference: 74.6% ± 32.5%)

**Arc shape distribution**:
"""
    for shape, count in sorted(wave_stats.arc_shape_counts.items(), key=lambda x: -x[1]):
        prompt += f"  - {shape}: {count}\n"

    if gaps:
        prompt += "\n## Gaps Detected\n"
        for gap in gaps:
            prompt += f"\n**{gap.task_type}** (priority {gap.priority}):\n"
            prompt += f"  {gap.specification.get('suggest', '')}\n"

    prompt += """
## Decision Required

Based on these statistics, should I:
1. Accept this wave as-is (statistics within expected variance)
2. Dispatch in-fill tasks to address gaps
3. Dispatch out-fill tasks to connect this wave to existing corpus
4. Flag for human review (unusual patterns)

Respond with action and any specific generation prompts needed.
"""
    return prompt
