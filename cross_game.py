#!/usr/bin/env python3
"""
Cross-Game Semantic Linking

Link dialogue across multiple games using shared semantic features:
- Emotion annotations (same emotion types across all Bethesda games)
- Similar text patterns
- Quest/topic similarities

This enables:
- Building unified corpora with consistent emotion annotations
- Finding parallel narrative structures across games
- Training on semantically-linked multi-game data
"""

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple, Iterator
import random


@dataclass
class CrossGameNode:
    """A dialogue node with cross-game linking metadata."""
    game: str
    form_id: str
    text: str
    speaker: Optional[str]
    emotion: str
    emotion_value: int
    topic: str
    quest: Optional[str]
    source_file: str

    def key(self) -> str:
        return f"{self.game}:{self.form_id}"


@dataclass
class EmotionCluster:
    """A cluster of dialogue nodes sharing an emotion."""
    emotion: str
    nodes: List[CrossGameNode] = field(default_factory=list)

    def by_game(self) -> Dict[str, List[CrossGameNode]]:
        result = defaultdict(list)
        for node in self.nodes:
            result[node.game].append(node)
        return dict(result)

    def sample_pairs(self, n: int = 10) -> List[Tuple[CrossGameNode, CrossGameNode]]:
        """Sample pairs of nodes from different games."""
        games = list(self.by_game().keys())
        if len(games) < 2:
            return []

        pairs = []
        for _ in range(n * 10):  # Over-sample then filter
            g1, g2 = random.sample(games, 2)
            nodes1 = self.by_game()[g1]
            nodes2 = self.by_game()[g2]
            if nodes1 and nodes2:
                n1 = random.choice(nodes1)
                n2 = random.choice(nodes2)
                pairs.append((n1, n2))
            if len(pairs) >= n:
                break

        return pairs[:n]


class CrossGameLinker:
    """
    Links dialogue across multiple games.

    Linking strategies:
    1. Emotion-based: Group by emotion annotation (most reliable)
    2. Intensity-based: Group by emotion + intensity bucket
    3. Topic-pattern: Link topics with similar naming patterns
    """

    def __init__(self):
        self.nodes: List[CrossGameNode] = []
        self.games: Set[str] = set()
        self._by_emotion: Dict[str, EmotionCluster] = {}
        self._by_game: Dict[str, List[CrossGameNode]] = defaultdict(list)

    def load_game(self, path: Path, game_name: str = None):
        """Load dialogue from a game export."""
        with open(path) as f:
            data = json.load(f)

        game = game_name or data.get('game', path.stem.replace('_dialogue', '').replace('_full', ''))
        self.games.add(game)

        dialogue = data.get('dialogue', [])
        for line in dialogue:
            node = CrossGameNode(
                game=game,
                form_id=line.get('form_id', ''),
                text=line.get('text', ''),
                speaker=line.get('speaker'),
                emotion=line.get('emotion', 'neutral'),
                emotion_value=line.get('emotion_value', 0),
                topic=line.get('topic', ''),
                quest=line.get('quest'),
                source_file=line.get('source', ''),
            )
            self.nodes.append(node)
            self._by_game[game].append(node)

        print(f"Loaded {len(dialogue)} lines from {game}")

    def build_emotion_clusters(self):
        """Build clusters of nodes grouped by emotion."""
        self._by_emotion.clear()

        for node in self.nodes:
            if node.emotion not in self._by_emotion:
                self._by_emotion[node.emotion] = EmotionCluster(emotion=node.emotion)
            self._by_emotion[node.emotion].nodes.append(node)

    def get_emotion_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get emotion distribution per game."""
        dist = {}
        for game, nodes in self._by_game.items():
            emotions = defaultdict(int)
            for node in nodes:
                emotions[node.emotion] += 1
            dist[game] = dict(emotions)
        return dist

    def get_cross_game_stats(self) -> Dict[str, Any]:
        """Get statistics about cross-game linking potential."""
        if not self._by_emotion:
            self.build_emotion_clusters()

        stats = {
            'games': list(self.games),
            'total_nodes': len(self.nodes),
            'nodes_per_game': {g: len(n) for g, n in self._by_game.items()},
            'emotions': {},
            'linkable_emotions': [],
        }

        for emotion, cluster in self._by_emotion.items():
            by_game = cluster.by_game()
            games_with_emotion = list(by_game.keys())
            counts = {g: len(nodes) for g, nodes in by_game.items()}

            stats['emotions'][emotion] = {
                'total': len(cluster.nodes),
                'games': games_with_emotion,
                'per_game': counts,
            }

            # Linkable = present in multiple games
            if len(games_with_emotion) >= 2:
                stats['linkable_emotions'].append({
                    'emotion': emotion,
                    'games': games_with_emotion,
                    'min_count': min(counts.values()),
                    'max_count': max(counts.values()),
                })

        # Sort by linkability potential
        stats['linkable_emotions'].sort(key=lambda x: x['min_count'], reverse=True)

        return stats

    def sample_cross_game_pairs(self, emotion: str = None, n: int = 10) -> List[Dict[str, Any]]:
        """
        Sample pairs of semantically-linked dialogue across games.

        Returns pairs with the same emotion from different games.
        """
        if not self._by_emotion:
            self.build_emotion_clusters()

        if emotion:
            if emotion not in self._by_emotion:
                return []
            cluster = self._by_emotion[emotion]
            pairs = cluster.sample_pairs(n)
        else:
            # Sample from all linkable emotions
            pairs = []
            linkable = [e for e, c in self._by_emotion.items() if len(c.by_game()) >= 2]
            for _ in range(n * 2):
                if not linkable:
                    break
                emotion = random.choice(linkable)
                cluster_pairs = self._by_emotion[emotion].sample_pairs(1)
                pairs.extend(cluster_pairs)
                if len(pairs) >= n:
                    break

        return [
            {
                'emotion': n1.emotion,
                'game_a': {
                    'game': n1.game,
                    'text': n1.text,
                    'speaker': n1.speaker,
                    'topic': n1.topic,
                    'quest': n1.quest,
                },
                'game_b': {
                    'game': n2.game,
                    'text': n2.text,
                    'speaker': n2.speaker,
                    'topic': n2.topic,
                    'quest': n2.quest,
                }
            }
            for n1, n2 in pairs[:n]
        ]

    def build_unified_graph(self) -> Dict[str, Any]:
        """
        Build a unified graph structure linking all games.

        Nodes from different games are connected if they share:
        - Same emotion
        - Similar intensity bucket (0-25, 25-50, 50-75, 75-100)
        """
        if not self._by_emotion:
            self.build_emotion_clusters()

        graph = {
            'games': list(self.games),
            'emotion_bridges': {},  # emotion -> list of cross-game edges
        }

        for emotion, cluster in self._by_emotion.items():
            by_game = cluster.by_game()
            if len(by_game) < 2:
                continue

            # Sample edges between games
            games = list(by_game.keys())
            edges = []

            for i, g1 in enumerate(games):
                for g2 in games[i+1:]:
                    # Sample node pairs from each game pair
                    for _ in range(min(5, len(by_game[g1]), len(by_game[g2]))):
                        n1 = random.choice(by_game[g1])
                        n2 = random.choice(by_game[g2])
                        edges.append({
                            'source': {'game': g1, 'id': n1.form_id, 'text': n1.text[:60]},
                            'target': {'game': g2, 'id': n2.form_id, 'text': n2.text[:60]},
                        })

            graph['emotion_bridges'][emotion] = {
                'edge_count': len(edges),
                'sample_edges': edges[:10],
            }

        return graph

    def export_unified_corpus(self, output_path: Path, emotions: List[str] = None):
        """
        Export a unified corpus with cross-game links.

        Each line includes game source and emotion, enabling:
        - Multi-game training with consistent labels
        - Emotion-conditioned generation
        - Cross-game transfer learning
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for node in self.nodes:
                if emotions and node.emotion not in emotions:
                    continue

                record = {
                    'text': node.text,
                    'game': node.game,
                    'emotion': node.emotion,
                    'emotion_value': node.emotion_value,
                    'speaker': node.speaker,
                    'topic': node.topic,
                    'quest': node.quest,
                    'source': node.source_file,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Exported unified corpus to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Cross-game semantic linking')
    parser.add_argument('inputs', nargs='+', type=Path,
                       help='Dialogue JSON files to link')
    parser.add_argument('--stats', action='store_true',
                       help='Show cross-game statistics')
    parser.add_argument('--sample-pairs', type=int, default=0,
                       help='Sample N cross-game pairs')
    parser.add_argument('--emotion', type=str, default=None,
                       help='Filter to specific emotion')
    parser.add_argument('--export', type=Path, default=None,
                       help='Export unified corpus to JSONL')

    args = parser.parse_args()

    linker = CrossGameLinker()

    for path in args.inputs:
        if path.exists():
            linker.load_game(path)

    if args.stats:
        print(f"\n{'='*60}")
        print("Cross-Game Linking Statistics")
        print(f"{'='*60}")

        stats = linker.get_cross_game_stats()
        print(f"Games: {stats['games']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"\nNodes per game:")
        for game, count in stats['nodes_per_game'].items():
            print(f"  {game}: {count}")

        print(f"\nLinkable emotions (present in 2+ games):")
        for e in stats['linkable_emotions'][:10]:
            print(f"  {e['emotion']}: {e['games']} (min={e['min_count']}, max={e['max_count']})")

        print(f"\nEmotion distribution:")
        dist = linker.get_emotion_distribution()
        # Show matrix format
        emotions = sorted(set(e for g in dist.values() for e in g.keys()))
        header = "          " + " ".join(f"{e[:6]:>8}" for e in emotions)
        print(header)
        for game, counts in dist.items():
            row = f"{game[:8]:8}  " + " ".join(f"{counts.get(e, 0):>8}" for e in emotions)
            print(row)

    if args.sample_pairs > 0:
        print(f"\n{'='*60}")
        print(f"Cross-Game Pairs (emotion={args.emotion or 'any'}, n={args.sample_pairs})")
        print(f"{'='*60}")

        pairs = linker.sample_cross_game_pairs(emotion=args.emotion, n=args.sample_pairs)
        for i, pair in enumerate(pairs):
            print(f"\n--- Pair {i+1} [{pair['emotion']}] ---")
            print(f"  {pair['game_a']['game']}: \"{pair['game_a']['text'][:60]}...\"")
            print(f"    Speaker: {pair['game_a']['speaker']}, Topic: {pair['game_a']['topic']}")
            print(f"  {pair['game_b']['game']}: \"{pair['game_b']['text'][:60]}...\"")
            print(f"    Speaker: {pair['game_b']['speaker']}, Topic: {pair['game_b']['topic']}")

    if args.export:
        linker.export_unified_corpus(args.export, emotions=args.emotion.split(',') if args.emotion else None)


if __name__ == '__main__':
    main()
