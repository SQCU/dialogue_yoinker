#!/usr/bin/env python3
"""
Stats-Guided Graph Growth

The reference corpus IS the model. Sample from its statistics to grow
synthetic graphs that are statistically similar but topologically different.

Algorithm:
    while target_graph needs growth:
        1. MEASURE: Compute stats of current target graph
        2. COMPARE: Find gaps between target stats and reference stats
        3. SAMPLE: Query reference corpus for walks that would close the gap
        4. GENERATE: Use sampled walk as template, generate target-setting analogue
        5. ATTACH: Connect new content to target graph

Local similarity is acceptable - the divergence comes from WHERE we attach,
creating different global topology even with similar local texture.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import hashlib

from dialogue_graph import DialogueGraph
from topic_graph import TopicGraph
from cross_game import CrossGameLinker
from synthetic_versioning import GraphVersion, extend_graph


@dataclass
class ReferenceStats:
    """Statistics computed from reference corpus."""

    # Emotion transitions: (from_emotion, to_emotion) -> count
    emotion_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    emotion_self_loop_rate: float = 0.0

    # Topic degree distribution: degree -> count
    topic_out_degrees: Dict[int, int] = field(default_factory=dict)
    topic_in_degrees: Dict[int, int] = field(default_factory=dict)

    # Arc shapes: shape_name -> count
    arc_shapes: Dict[str, int] = field(default_factory=dict)

    # Emotion distribution
    emotion_counts: Dict[str, int] = field(default_factory=dict)

    # Walk length distribution
    walk_lengths: Dict[int, int] = field(default_factory=dict)

    def normalize(self) -> 'ReferenceStats':
        """Convert counts to probabilities."""
        normalized = ReferenceStats()

        # Emotion transitions
        total_trans = sum(self.emotion_transitions.values()) or 1
        normalized.emotion_transitions = {
            k: v / total_trans for k, v in self.emotion_transitions.items()
        }
        normalized.emotion_self_loop_rate = self.emotion_self_loop_rate

        # Topic degrees
        total_out = sum(self.topic_out_degrees.values()) or 1
        normalized.topic_out_degrees = {
            k: v / total_out for k, v in self.topic_out_degrees.items()
        }
        total_in = sum(self.topic_in_degrees.values()) or 1
        normalized.topic_in_degrees = {
            k: v / total_in for k, v in self.topic_in_degrees.items()
        }

        # Arc shapes
        total_arcs = sum(self.arc_shapes.values()) or 1
        normalized.arc_shapes = {
            k: v / total_arcs for k, v in self.arc_shapes.items()
        }

        # Emotions
        total_emo = sum(self.emotion_counts.values()) or 1
        normalized.emotion_counts = {
            k: v / total_emo for k, v in self.emotion_counts.items()
        }

        # Walk lengths
        total_walks = sum(self.walk_lengths.values()) or 1
        normalized.walk_lengths = {
            k: v / total_walks for k, v in self.walk_lengths.items()
        }

        return normalized


@dataclass
class StatisticalGap:
    """Represents a gap between target and reference statistics."""
    stat_type: str  # "emotion_transition", "arc_shape", "topic_degree", etc.
    key: Any        # The specific stat key (e.g., ("anger", "neutral") or "negotiation_arc")
    reference_value: float
    target_value: float
    gap_size: float  # reference - target (positive = underrepresented in target)

    def __str__(self):
        return f"{self.stat_type}:{self.key} (ref={self.reference_value:.3f}, tgt={self.target_value:.3f}, gap={self.gap_size:+.3f})"


class ReferenceCorpus:
    """
    Manages the reference corpus and provides stat-aware sampling.
    """

    def __init__(self, data_dir: Path = Path("dialogue_data")):
        self.data_dir = data_dir
        self.dialogue_graphs: Dict[str, DialogueGraph] = {}
        self.topic_graphs: Dict[str, TopicGraph] = {}
        self.linker = CrossGameLinker()
        self._stats: Optional[ReferenceStats] = None
        self._walks_by_emotion_transition: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        self._walks_by_arc_shape: Dict[str, List[dict]] = defaultdict(list)

    def load(self, games: List[str] = None):
        """Load reference corpus from dialogue data."""
        patterns = ["*_full_dialogue.json", "*_dialogue.json"]

        loaded = set()
        for pattern in patterns:
            for path in sorted(self.data_dir.glob(pattern)):
                game = path.stem.replace('_full_dialogue', '').replace('_dialogue', '')
                if game in loaded:
                    continue
                if games and game not in games:
                    continue

                print(f"Loading {game}...")

                with open(path) as f:
                    data = json.load(f)

                dialogue = data.get('dialogue', [])

                # Build dialogue graph
                self.dialogue_graphs[game] = DialogueGraph.from_dialogue_data(dialogue)

                # Build topic graph
                self.topic_graphs[game] = TopicGraph.from_dialogue(dialogue)

                # Add to cross-game linker
                self.linker.load_game(path, game)

                loaded.add(game)
                print(f"  {len(dialogue)} lines, {len(self.dialogue_graphs[game].edges)} edges")

        self.linker.build_emotion_clusters()
        self._build_walk_indexes()
        print(f"\nLoaded {len(loaded)} games: {list(loaded)}")

    def _build_walk_indexes(self):
        """Index walks by emotion transitions and arc shapes for fast sampling."""
        for game, graph in self.dialogue_graphs.items():
            # Sample walks and index them
            for _ in range(1000):  # Sample representative walks
                walk = graph.random_walk(max_steps=6)
                if len(walk) < 2:
                    continue

                walk_data = {
                    'game': game,
                    'nodes': [asdict(n) if hasattr(n, '__dataclass_fields__') else n for n in walk],
                    'emotions': [n.emotion if hasattr(n, 'emotion') else n.get('emotion', 'neutral') for n in walk],
                    'texts': [n.text if hasattr(n, 'text') else n.get('text', '') for n in walk],
                }

                # Index by emotion transitions
                emotions = walk_data['emotions']
                for i in range(len(emotions) - 1):
                    trans = (emotions[i], emotions[i + 1])
                    self._walks_by_emotion_transition[trans].append(walk_data)

                # Classify and index by arc shape
                arc_shape = self._classify_arc_shape(emotions)
                self._walks_by_arc_shape[arc_shape].append(walk_data)

    def _classify_arc_shape(self, emotions: List[str]) -> str:
        """Classify an emotion sequence into an arc shape."""
        if len(emotions) < 2:
            return "single"

        # Simple arc shape classification
        unique = list(dict.fromkeys(emotions))  # Preserve order, remove adjacent dupes

        if len(unique) == 1:
            return f"flat_{unique[0]}"

        if unique[0] == 'neutral' and unique[-1] == 'neutral':
            if len(unique) > 2:
                return f"neutral_peak_{unique[len(unique)//2]}"
            return "neutral_flat"

        if unique[0] == unique[-1]:
            return f"return_to_{unique[0]}"

        # Detect escalation/de-escalation
        intensity_map = {'neutral': 0, 'happy': 1, 'sad': 1, 'surprise': 1, 'fear': 2, 'anger': 2, 'disgust': 2}
        intensities = [intensity_map.get(e, 0) for e in unique]

        if intensities == sorted(intensities):
            return f"escalation_{unique[0]}_to_{unique[-1]}"
        if intensities == sorted(intensities, reverse=True):
            return f"de_escalation_{unique[0]}_to_{unique[-1]}"

        return f"mixed_{unique[0]}_to_{unique[-1]}"

    def compute_stats(self) -> ReferenceStats:
        """Compute comprehensive statistics from reference corpus."""
        if self._stats:
            return self._stats

        stats = ReferenceStats()

        # Emotion transitions from all dialogue graphs
        total_self_loops = 0
        total_transitions = 0

        for game, graph in self.dialogue_graphs.items():
            for edge in graph.edges:
                src = graph.nodes.get(edge.source)
                tgt = graph.nodes.get(edge.target)
                if src and tgt:
                    src_emo = src.emotion if hasattr(src, 'emotion') else 'neutral'
                    tgt_emo = tgt.emotion if hasattr(tgt, 'emotion') else 'neutral'

                    trans = (src_emo, tgt_emo)
                    stats.emotion_transitions[trans] = stats.emotion_transitions.get(trans, 0) + 1

                    total_transitions += 1
                    if src_emo == tgt_emo:
                        total_self_loops += 1

        if total_transitions > 0:
            stats.emotion_self_loop_rate = total_self_loops / total_transitions

        # Topic degree distribution
        for game, tgraph in self.topic_graphs.items():
            for topic_id in tgraph.nodes:
                out_deg = len(tgraph._adjacency.get(topic_id, set()))
                in_deg = len(tgraph._reverse_adjacency.get(topic_id, set()))
                stats.topic_out_degrees[out_deg] = stats.topic_out_degrees.get(out_deg, 0) + 1
                stats.topic_in_degrees[in_deg] = stats.topic_in_degrees.get(in_deg, 0) + 1

        # Arc shapes from indexed walks
        for arc_shape, walks in self._walks_by_arc_shape.items():
            stats.arc_shapes[arc_shape] = len(walks)

        # Emotion distribution from linker
        dist = self.linker.get_emotion_distribution()
        for game_dist in dist.values():
            for emotion, count in game_dist.items():
                stats.emotion_counts[emotion] = stats.emotion_counts.get(emotion, 0) + count

        # Walk length distribution
        for walks in self._walks_by_emotion_transition.values():
            for walk in walks:
                length = len(walk['nodes'])
                stats.walk_lengths[length] = stats.walk_lengths.get(length, 0) + 1

        self._stats = stats
        return stats

    def sample_walks_for_gap(self, gap: StatisticalGap, n: int = 5) -> List[dict]:
        """Sample walks from reference that would help close a statistical gap."""
        walks = []

        if gap.stat_type == "emotion_transition":
            # Sample walks containing this transition
            trans = gap.key
            candidates = self._walks_by_emotion_transition.get(trans, [])
            if candidates:
                walks = random.sample(candidates, min(n, len(candidates)))

        elif gap.stat_type == "arc_shape":
            # Sample walks with this arc shape
            shape = gap.key
            candidates = self._walks_by_arc_shape.get(shape, [])
            if candidates:
                walks = random.sample(candidates, min(n, len(candidates)))

        elif gap.stat_type == "emotion":
            # Sample walks starting with this emotion
            emotion = gap.key
            candidates = []
            for trans, trans_walks in self._walks_by_emotion_transition.items():
                if trans[0] == emotion:
                    candidates.extend(trans_walks)
            if candidates:
                walks = random.sample(candidates, min(n, len(candidates)))

        return walks


class StatsGuidedGrowth:
    """
    Grows synthetic graphs by sampling from reference corpus statistics.

    The reference statistics ARE the model. We sample new edges/nodes to
    match these statistics, producing graphs that are statistically similar
    to reference but topologically different.
    """

    def __init__(
        self,
        reference: ReferenceCorpus,
        target_version: GraphVersion,
        target_bible_path: Path = None,
    ):
        self.reference = reference
        self.target_version = target_version
        self.target_bible_path = target_bible_path

        # Load existing target graph if any
        self._load_target_graph()

        # Compute reference stats
        self.ref_stats = reference.compute_stats().normalize()

    def _load_target_graph(self):
        """Load existing target graph from version."""
        graph_file = self.target_version.path() / "graph.json"
        if graph_file.exists():
            with open(graph_file) as f:
                data = json.load(f)
            self.target_nodes = data.get('nodes', [])
            self.target_edges = data.get('edges', [])
        else:
            self.target_nodes = []
            self.target_edges = []

    def _compute_target_stats(self) -> ReferenceStats:
        """Compute current statistics of target graph."""
        stats = ReferenceStats()

        if not self.target_edges:
            return stats.normalize() if self.target_nodes else stats

        # Build emotion transition counts
        node_map = {n['id']: n for n in self.target_nodes}
        total_self_loops = 0

        for edge in self.target_edges:
            src = node_map.get(edge.get('source'))
            tgt = node_map.get(edge.get('target'))
            if src and tgt:
                src_emo = src.get('emotion', 'neutral')
                tgt_emo = tgt.get('emotion', 'neutral')
                trans = (src_emo, tgt_emo)
                stats.emotion_transitions[trans] = stats.emotion_transitions.get(trans, 0) + 1
                if src_emo == tgt_emo:
                    total_self_loops += 1

        total_trans = sum(stats.emotion_transitions.values())
        if total_trans > 0:
            stats.emotion_self_loop_rate = total_self_loops / total_trans

        # Emotion distribution
        for node in self.target_nodes:
            emo = node.get('emotion', 'neutral')
            stats.emotion_counts[emo] = stats.emotion_counts.get(emo, 0) + 1

        return stats.normalize()

    def identify_gaps(self, top_n: int = 5) -> List[StatisticalGap]:
        """Find the largest statistical gaps between target and reference."""
        target_stats = self._compute_target_stats()
        gaps = []

        # Compare emotion transitions
        all_trans = set(self.ref_stats.emotion_transitions.keys()) | set(target_stats.emotion_transitions.keys())
        for trans in all_trans:
            ref_val = self.ref_stats.emotion_transitions.get(trans, 0)
            tgt_val = target_stats.emotion_transitions.get(trans, 0)
            if ref_val > 0.01:  # Only significant transitions
                gap = ref_val - tgt_val
                if gap > 0.01:  # Underrepresented in target
                    gaps.append(StatisticalGap(
                        stat_type="emotion_transition",
                        key=trans,
                        reference_value=ref_val,
                        target_value=tgt_val,
                        gap_size=gap,
                    ))

        # Compare arc shapes
        all_shapes = set(self.ref_stats.arc_shapes.keys()) | set(target_stats.arc_shapes.keys())
        for shape in all_shapes:
            ref_val = self.ref_stats.arc_shapes.get(shape, 0)
            tgt_val = target_stats.arc_shapes.get(shape, 0)
            if ref_val > 0.01:
                gap = ref_val - tgt_val
                if gap > 0.01:
                    gaps.append(StatisticalGap(
                        stat_type="arc_shape",
                        key=shape,
                        reference_value=ref_val,
                        target_value=tgt_val,
                        gap_size=gap,
                    ))

        # Compare emotion distribution
        all_emo = set(self.ref_stats.emotion_counts.keys()) | set(target_stats.emotion_counts.keys())
        for emo in all_emo:
            ref_val = self.ref_stats.emotion_counts.get(emo, 0)
            tgt_val = target_stats.emotion_counts.get(emo, 0)
            if ref_val > 0.01:
                gap = ref_val - tgt_val
                if gap > 0.02:  # Higher threshold for emotions
                    gaps.append(StatisticalGap(
                        stat_type="emotion",
                        key=emo,
                        reference_value=ref_val,
                        target_value=tgt_val,
                        gap_size=gap,
                    ))

        # Sort by gap size and return top N
        gaps.sort(key=lambda g: g.gap_size, reverse=True)
        return gaps[:top_n]

    def sample_for_gap(self, gap: StatisticalGap, n: int = 3) -> List[dict]:
        """Sample reference walks that would help close a gap."""
        return self.reference.sample_walks_for_gap(gap, n)

    def find_attachment_points(self, walk: dict) -> List[dict]:
        """
        Find good attachment points in target graph for a new walk.

        Chooses points that would improve overall statistics.
        """
        if not self.target_nodes:
            # No existing nodes - this will be the root
            return [None]

        # Find nodes with matching emotion to walk start
        walk_start_emotion = walk['emotions'][0] if walk.get('emotions') else 'neutral'

        candidates = []
        for node in self.target_nodes:
            node_emo = node.get('emotion', 'neutral')

            # Prefer nodes where attaching would create a transition we need more of
            target_stats = self._compute_target_stats()
            potential_trans = (node_emo, walk_start_emotion)

            ref_val = self.ref_stats.emotion_transitions.get(potential_trans, 0)
            tgt_val = target_stats.emotion_transitions.get(potential_trans, 0)

            score = ref_val - tgt_val  # Higher = more underrepresented = better attachment
            candidates.append({
                'node': node,
                'transition': potential_trans,
                'score': score,
            })

        # Sort by score and return top candidates
        candidates.sort(key=lambda c: c['score'], reverse=True)
        return [c['node'] for c in candidates[:3]]

    def generate_id(self, text: str, prefix: str = "syn") -> str:
        """Generate a unique ID for a synthetic node."""
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]
        return f"{prefix}_{content_hash}"

    def grow_step(self, walk: dict, attachment_point: Optional[dict] = None) -> dict:
        """
        Add a walk to the target graph.

        For now, copies walk structure with new IDs.
        TODO: Hook up translation engine for novel content generation.
        """
        new_nodes = []
        new_edges = []
        prev_id = attachment_point['id'] if attachment_point else None

        for i, text in enumerate(walk.get('texts', [])):
            emotion = walk['emotions'][i] if i < len(walk.get('emotions', [])) else 'neutral'

            node_id = self.generate_id(f"{text}_{i}_{random.random()}")
            node = {
                'id': node_id,
                'text': text,  # TODO: Replace with translated text
                'emotion': emotion,
                'source_game': walk.get('game', 'unknown'),
                'source_ref': walk.get('nodes', [{}])[i].get('form_id') if i < len(walk.get('nodes', [])) else None,
            }
            new_nodes.append(node)

            if prev_id:
                new_edges.append({
                    'source': prev_id,
                    'target': node_id,
                    'type': 'synthetic',
                })

            prev_id = node_id

        # Update target graph
        self.target_nodes.extend(new_nodes)
        self.target_edges.extend(new_edges)

        return {
            'nodes_added': len(new_nodes),
            'edges_added': len(new_edges),
            'attachment': attachment_point['id'] if attachment_point else None,
        }

    def grow(self, target_size: int = 100, max_iterations: int = 50) -> dict:
        """
        Grow the target graph toward reference statistics.

        Args:
            target_size: Target number of nodes
            max_iterations: Maximum growth iterations

        Returns:
            Growth report with statistics
        """
        report = {
            'initial_nodes': len(self.target_nodes),
            'initial_edges': len(self.target_edges),
            'iterations': [],
            'gaps_closed': [],
        }

        for i in range(max_iterations):
            if len(self.target_nodes) >= target_size:
                break

            # Find gaps
            gaps = self.identify_gaps(top_n=3)
            if not gaps:
                print("No significant gaps remaining")
                break

            # Pick a gap to close (weighted by gap size)
            gap = random.choices(gaps, weights=[g.gap_size for g in gaps])[0]

            # Sample walks for this gap
            walks = self.sample_for_gap(gap, n=2)
            if not walks:
                print(f"No walks available for gap: {gap}")
                continue

            walk = random.choice(walks)

            # Find attachment point
            attachment_points = self.find_attachment_points(walk)
            attachment = attachment_points[0] if attachment_points else None

            # Grow
            result = self.grow_step(walk, attachment)

            report['iterations'].append({
                'iteration': i,
                'gap_targeted': str(gap),
                'walk_length': len(walk.get('texts', [])),
                'attached_to': result['attachment'],
                'nodes_added': result['nodes_added'],
            })

            print(f"[{i+1}] Targeted {gap.stat_type}:{gap.key}, added {result['nodes_added']} nodes")

        # Save updated graph
        self._save_target_graph()

        report['final_nodes'] = len(self.target_nodes)
        report['final_edges'] = len(self.target_edges)
        report['gaps_remaining'] = [str(g) for g in self.identify_gaps()]

        return report

    def _save_target_graph(self):
        """Save target graph to version directory."""
        graph_file = self.target_version.path() / "graph.json"
        data = {
            'nodes': self.target_nodes,
            'edges': self.target_edges,
            'state_changes': [],  # TODO: track state changes
        }
        graph_file.write_text(json.dumps(data, indent=2))

        # Update version metadata
        self.target_version.total_nodes = len(self.target_nodes)
        self.target_version.total_edges = len(self.target_edges)

        meta_file = self.target_version.path() / "metadata.json"
        meta_file.write_text(json.dumps(asdict(self.target_version), indent=2))


def print_stats_comparison(ref_stats: ReferenceStats, target_stats: ReferenceStats = None):
    """Pretty-print statistics comparison."""
    print("\n" + "="*60)
    print("Reference Corpus Statistics")
    print("="*60)

    print(f"\nEmotion Self-Loop Rate: {ref_stats.emotion_self_loop_rate:.1%}")

    print("\nTop Emotion Transitions:")
    sorted_trans = sorted(ref_stats.emotion_transitions.items(), key=lambda x: -x[1])
    for (src, tgt), val in sorted_trans[:10]:
        marker = ""
        if target_stats:
            tgt_val = target_stats.emotion_transitions.get((src, tgt), 0)
            gap = val - tgt_val
            if gap > 0.02:
                marker = f" [gap: +{gap:.1%}]"
        print(f"  {src:10} -> {tgt:10}: {val:.1%}{marker}")

    print("\nArc Shapes:")
    sorted_shapes = sorted(ref_stats.arc_shapes.items(), key=lambda x: -x[1])
    for shape, val in sorted_shapes[:10]:
        print(f"  {shape:30}: {val:.1%}")

    print("\nEmotion Distribution:")
    sorted_emo = sorted(ref_stats.emotion_counts.items(), key=lambda x: -x[1])
    for emo, val in sorted_emo:
        print(f"  {emo:10}: {val:.1%}")


def main():
    import argparse
    from synthetic_versioning import new_graph, get_latest_version

    parser = argparse.ArgumentParser(description='Stats-guided synthetic graph growth')
    parser.add_argument('--setting', type=str, default='gallia',
                       help='Target setting name')
    parser.add_argument('--version', type=int, default=None,
                       help='Version to extend (default: create new)')
    parser.add_argument('--target-size', type=int, default=50,
                       help='Target number of nodes')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics, no growth')
    parser.add_argument('--games', type=str, default=None,
                       help='Comma-separated list of games to use as reference')

    args = parser.parse_args()

    # Load reference corpus
    print("Loading reference corpus...")
    reference = ReferenceCorpus()
    games = args.games.split(',') if args.games else None
    reference.load(games)

    # Compute and show stats
    stats = reference.compute_stats()
    print_stats_comparison(stats.normalize())

    if args.stats_only:
        return

    # Get or create target version
    if args.version:
        versions = list_versions(args.setting)
        target_version = None
        for v in versions:
            if v.version == args.version:
                target_version = v
                break
        if not target_version:
            print(f"Version {args.version} not found for {args.setting}")
            return
    else:
        target_version = new_graph(
            setting=args.setting,
            description=f"Stats-guided growth ({args.target_size} nodes target)",
            approach="stats_guided_growth",
        )
        print(f"Created {args.setting}_v{target_version.version}")

    # Initialize growth engine
    grower = StatsGuidedGrowth(reference, target_version)

    # Show initial gaps
    print("\n" + "="*60)
    print("Initial Statistical Gaps")
    print("="*60)
    for gap in grower.identify_gaps(top_n=10):
        print(f"  {gap}")

    # Grow
    print("\n" + "="*60)
    print(f"Growing to {args.target_size} nodes")
    print("="*60)

    report = grower.grow(target_size=args.target_size)

    print("\n" + "="*60)
    print("Growth Report")
    print("="*60)
    print(f"Initial: {report['initial_nodes']} nodes, {report['initial_edges']} edges")
    print(f"Final: {report['final_nodes']} nodes, {report['final_edges']} edges")
    print(f"Iterations: {len(report['iterations'])}")

    print("\nRemaining gaps:")
    for gap in report['gaps_remaining'][:5]:
        print(f"  {gap}")


if __name__ == '__main__':
    from synthetic_versioning import list_versions
    main()
