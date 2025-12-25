"""
Stats-Guided Sampling Integration

Bridges stats_guided_growth.py into the ticket queue pipeline.
Provides guided sampling for translation runs and linking runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stats_guided_growth import ReferenceCorpus, StatsGuidedGrower, StatisticalGap, ReferenceStats


# Singleton reference corpus (expensive to load)
_reference_corpus: Optional[ReferenceCorpus] = None


def get_reference_corpus() -> ReferenceCorpus:
    """Get or initialize the reference corpus singleton."""
    global _reference_corpus
    if _reference_corpus is None:
        _reference_corpus = ReferenceCorpus()
        _reference_corpus.load(["oblivion", "falloutnv"])
    return _reference_corpus


def get_grower(setting: str, version: int) -> StatsGuidedGrower:
    """Create a StatsGuidedGrower for a target setting."""
    ref = get_reference_corpus()
    return StatsGuidedGrower(
        reference=ref,
        target_setting=setting,
        target_version=version,
    )


@dataclass
class GuidedSampleResult:
    """Result of guided sampling."""
    walks: List[dict]
    gaps_targeted: List[StatisticalGap]
    stats_before: Optional[ReferenceStats] = None
    stats_after: Optional[ReferenceStats] = None


def sample_walks_guided(
    setting: str,
    version: int,
    count: int,
    top_gaps: int = 10,
) -> GuidedSampleResult:
    """
    Sample walks from reference corpus guided by statistical gaps.

    Instead of uniform random sampling, identifies underrepresented
    transitions/arcs in the target graph and samples walks that would
    close those gaps.

    Args:
        setting: Target setting (e.g., 'gallia')
        version: Target version number
        count: Number of walks to sample
        top_gaps: Number of top gaps to target

    Returns:
        GuidedSampleResult with walks and gap information
    """
    grower = get_grower(setting, version)

    # Get current stats
    stats_before = grower._compute_target_stats()

    # Identify gaps
    gaps = grower.identify_gaps(top_n=top_gaps)

    if not gaps:
        # No gaps - fall back to random sampling
        ref = get_reference_corpus()
        walks = []
        for game in ["oblivion", "falloutnv"]:
            if game in ref.dialogue_graphs:
                graph = ref.dialogue_graphs[game]
                for _ in range(count // 2):
                    walk = graph.random_walk(max_steps=6)
                    if len(walk) >= 2:
                        walks.append({
                            'game': game,
                            'nodes': [_node_to_dict(n) for n in walk],
                            'emotions': [_get_emotion(n) for n in walk],
                            'texts': [_get_text(n) for n in walk],
                        })
        return GuidedSampleResult(walks=walks[:count], gaps_targeted=[], stats_before=stats_before)

    # Distribute count across gaps (weighted by gap magnitude)
    total_magnitude = sum(g.magnitude for g in gaps)
    walks = []
    gaps_used = []

    for gap in gaps:
        # Allocate samples proportional to gap magnitude
        n_for_gap = max(1, int(count * gap.magnitude / total_magnitude))

        gap_walks = grower.sample_for_gap(gap, n=n_for_gap)
        walks.extend(gap_walks)

        if gap_walks:
            gaps_used.append(gap)

    # Trim to requested count
    walks = walks[:count]

    return GuidedSampleResult(
        walks=walks,
        gaps_targeted=gaps_used,
        stats_before=stats_before,
    )


def sample_link_targets_guided(
    setting: str,
    version: int,
    source_node: dict,
    candidate_pool: List[dict],
    n_choices: int = 5,
) -> List[dict]:
    """
    Sample link targets guided by reference topology.

    Instead of random target selection, chooses targets that would
    create emotion transitions matching the reference distribution.

    Args:
        setting: Target setting
        version: Target version
        source_node: The source node for linking
        candidate_pool: Pool of potential target nodes
        n_choices: Number of targets to select

    Returns:
        List of selected target nodes with transition info
    """
    ref = get_reference_corpus()
    ref_stats = ref.compute_stats()

    source_emotion = source_node.get('emotion', 'neutral')

    # Get reference transition probabilities from this emotion
    transition_probs = {}
    for (from_emo, to_emo), prob in ref_stats.emotion_transitions.items():
        if from_emo == source_emotion:
            transition_probs[to_emo] = prob

    if not transition_probs:
        # No reference data for this emotion - fall back to uniform
        import random
        random.shuffle(candidate_pool)
        return candidate_pool[:n_choices]

    # Score candidates by how much they match reference distribution
    scored_candidates = []
    for candidate in candidate_pool:
        if candidate['id'] == source_node.get('id'):
            continue

        target_emotion = candidate.get('emotion', 'neutral')

        # Score = reference probability of this transition
        score = transition_probs.get(target_emotion, 0.01)

        scored_candidates.append((score, candidate))

    # Sample proportionally to scores (not just top-k)
    import random
    selected = []
    remaining = scored_candidates.copy()

    for _ in range(min(n_choices, len(remaining))):
        if not remaining:
            break

        # Weighted random selection
        total_score = sum(s for s, _ in remaining)
        if total_score <= 0:
            # All zero scores - pick randomly
            idx = random.randint(0, len(remaining) - 1)
        else:
            r = random.random() * total_score
            cumulative = 0
            idx = 0
            for i, (score, _) in enumerate(remaining):
                cumulative += score
                if cumulative >= r:
                    idx = i
                    break

        _, candidate = remaining.pop(idx)

        # Add transition info
        candidate['transition_required'] = f"{source_emotion}→{candidate.get('emotion', 'neutral')}"
        candidate['context'] = []

        selected.append(candidate)

    return selected


def compute_topology_score(setting: str, version: int) -> dict:
    """
    Compute how well target topology matches reference.

    Returns metrics comparing target to reference statistics.
    """
    grower = get_grower(setting, version)
    ref_stats = grower.ref_stats.normalize()
    target_stats = grower._compute_target_stats().normalize()

    # Compute divergences
    emotion_trans_divergence = 0
    for trans in set(ref_stats.emotion_transitions.keys()) | set(target_stats.emotion_transitions.keys()):
        ref_p = ref_stats.emotion_transitions.get(trans, 0)
        tgt_p = target_stats.emotion_transitions.get(trans, 0)
        if ref_p > 0:
            emotion_trans_divergence += abs(ref_p - tgt_p)

    arc_shape_divergence = 0
    for shape in set(ref_stats.arc_shapes.keys()) | set(target_stats.arc_shapes.keys()):
        ref_p = ref_stats.arc_shapes.get(shape, 0)
        tgt_p = target_stats.arc_shapes.get(shape, 0)
        if ref_p > 0:
            arc_shape_divergence += abs(ref_p - tgt_p)

    return {
        'emotion_transition_divergence': emotion_trans_divergence,
        'arc_shape_divergence': arc_shape_divergence,
        'total_divergence': emotion_trans_divergence + arc_shape_divergence,
        'gaps': [str(g) for g in grower.identify_gaps(top_n=5)],
    }


def _node_to_dict(node) -> dict:
    """Convert a node object to dict."""
    if hasattr(node, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(node)
    elif isinstance(node, dict):
        return node
    else:
        return {'id': str(node)}


def _get_emotion(node) -> str:
    """Get emotion from node."""
    if hasattr(node, 'emotion'):
        return node.emotion
    elif isinstance(node, dict):
        return node.get('emotion', 'neutral')
    return 'neutral'


def _get_text(node) -> str:
    """Get text from node."""
    if hasattr(node, 'text'):
        return node.text
    elif isinstance(node, dict):
        return node.get('text', '')
    return ''
