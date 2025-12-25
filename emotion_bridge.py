#!/usr/bin/env python3
"""
Emotion Bridge - Cross-Game Dialogue Graph via Emotion Transitions

Links dialogue graphs from different games by treating matching emotion
transitions as "wormholes" between datasets. A happy->disgust transition
in Skyrim can bridge to a happy->disgust transition in Fallout NV.

This enables:
- Cross-game trajectory sampling
- Unified emotion flow analysis
- Coverage sampling across the full corpus
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import json


@dataclass
class BridgeNode:
    """A node in the cross-game graph."""
    id: str  # game:form_id
    game: str
    form_id: str
    text: str
    speaker: Optional[str]
    emotion: str
    topic: str
    quest: Optional[str]


@dataclass
class BridgeEdge:
    """An edge in the cross-game graph."""
    source: str  # node id
    target: str  # node id
    edge_type: str  # 'intra' (within game) or 'bridge' (cross-game)
    emotion_key: Tuple[str, str]  # (source_emotion, target_emotion)
    weight: float = 1.0


@dataclass
class EmotionCell:
    """Represents one cell in the emotion transition matrix."""
    source_emotion: str
    target_emotion: str
    edges_by_game: Dict[str, List[BridgeEdge]] = field(default_factory=dict)

    @property
    def total_edges(self) -> int:
        return sum(len(edges) for edges in self.edges_by_game.values())

    @property
    def games(self) -> List[str]:
        return list(self.edges_by_game.keys())

    def is_cross_game(self) -> bool:
        """True if this cell has edges from multiple games."""
        return len(self.edges_by_game) > 1


class EmotionBridgeGraph:
    """
    Cross-game dialogue graph linked by emotion transitions.

    The key insight: emotion transitions (happy->sad, anger->fear, etc.)
    represent similar narrative moments across games. By linking these,
    we can sample trajectories that cross game boundaries while
    maintaining emotional coherence.
    """

    def __init__(self):
        self.nodes: Dict[str, BridgeNode] = {}
        self.edges: List[BridgeEdge] = []
        self.games: Set[str] = set()

        # Index: (src_emotion, tgt_emotion) -> EmotionCell
        self.transition_matrix: Dict[Tuple[str, str], EmotionCell] = {}

        # Adjacency for fast traversal
        self.outgoing: Dict[str, List[BridgeEdge]] = {}
        self.incoming: Dict[str, List[BridgeEdge]] = {}

    def add_game(self, game_name: str, dialogue_data: List[Dict]):
        """Add a game's dialogue to the bridge graph."""
        self.games.add(game_name)

        # First pass: create nodes
        for line in dialogue_data:
            form_id = line.get('form_id', '')
            if not form_id:
                continue

            node_id = f"{game_name}:{form_id}"
            self.nodes[node_id] = BridgeNode(
                id=node_id,
                game=game_name,
                form_id=form_id,
                text=line.get('text', ''),
                speaker=line.get('speaker'),
                emotion=line.get('emotion', 'neutral'),
                topic=line.get('topic', ''),
                quest=line.get('quest'),
            )
            self.outgoing[node_id] = []
            self.incoming[node_id] = []

        # Second pass: create intra-game edges based on topic adjacency
        # Group by topic
        by_topic: Dict[str, List[str]] = {}
        for node_id, node in self.nodes.items():
            if node.game == game_name:
                topic = node.topic
                if topic not in by_topic:
                    by_topic[topic] = []
                by_topic[topic].append(node_id)

        # Create edges within topics (sequential dialogue)
        for topic, node_ids in by_topic.items():
            for i in range(len(node_ids) - 1):
                src_id = node_ids[i]
                tgt_id = node_ids[i + 1]
                src_node = self.nodes[src_id]
                tgt_node = self.nodes[tgt_id]

                emotion_key = (src_node.emotion, tgt_node.emotion)
                edge = BridgeEdge(
                    source=src_id,
                    target=tgt_id,
                    edge_type='intra',
                    emotion_key=emotion_key,
                )

                self.edges.append(edge)
                self.outgoing[src_id].append(edge)
                self.incoming[tgt_id].append(edge)

                # Index by emotion transition
                if emotion_key not in self.transition_matrix:
                    self.transition_matrix[emotion_key] = EmotionCell(
                        source_emotion=emotion_key[0],
                        target_emotion=emotion_key[1],
                    )
                cell = self.transition_matrix[emotion_key]
                if game_name not in cell.edges_by_game:
                    cell.edges_by_game[game_name] = []
                cell.edges_by_game[game_name].append(edge)

    def build_bridges(self):
        """
        Build cross-game bridge edges.

        For each emotion transition cell that has edges from multiple games,
        we create virtual "bridge" edges that allow jumping between games.
        """
        bridge_count = 0

        for emotion_key, cell in self.transition_matrix.items():
            if not cell.is_cross_game():
                continue

            # This cell has edges from multiple games - it's a bridge point
            games = cell.games

            # For each pair of games, we can bridge
            # We don't create explicit edges, but mark this as bridgeable
            # The walk algorithm will handle the cross-game jump
            bridge_count += 1

        return bridge_count

    def get_bridge_cells(self) -> List[Dict]:
        """Get all emotion transition cells that bridge multiple games."""
        bridges = []
        for emotion_key, cell in self.transition_matrix.items():
            if cell.is_cross_game():
                bridges.append({
                    'source_emotion': emotion_key[0],
                    'target_emotion': emotion_key[1],
                    'games': cell.games,
                    'edges_per_game': {g: len(e) for g, e in cell.edges_by_game.items()},
                    'total_edges': cell.total_edges,
                })
        return sorted(bridges, key=lambda x: x['total_edges'], reverse=True)

    def cross_game_walk(
        self,
        max_steps: int = 10,
        start_game: str = None,
        start_node: str = None,
        cross_probability: float = 0.3,
        prefer_off_diagonal: bool = True,
    ) -> List[Dict]:
        """
        Random walk that can jump between games at emotion transitions.

        Args:
            max_steps: Maximum walk length
            start_game: Starting game (random if None)
            start_node: Starting node (random if None)
            cross_probability: Probability of crossing to another game at bridge points
            prefer_off_diagonal: Prefer non-neutral emotion transitions for bridges

        Returns:
            List of nodes visited, with game transitions marked
        """
        if not self.nodes:
            return []

        # Pick starting point
        if start_node and start_node in self.nodes:
            current = start_node
        elif start_game and start_game in self.games:
            game_nodes = [n for n in self.nodes if self.nodes[n].game == start_game]
            if not game_nodes:
                return []
            current = random.choice(game_nodes)
        else:
            current = random.choice(list(self.nodes.keys()))

        path = []
        visited = set()

        for step in range(max_steps):
            node = self.nodes[current]
            visited.add(current)

            path.append({
                'step': step,
                'node_id': current,
                'game': node.game,
                'form_id': node.form_id,
                'text': node.text,
                'speaker': node.speaker,
                'emotion': node.emotion,
                'topic': node.topic,
                'crossed_from': None,
            })

            # Get outgoing edges
            out_edges = self.outgoing.get(current, [])
            if not out_edges:
                break

            # Pick next edge
            edge = random.choice(out_edges)
            emotion_key = edge.emotion_key

            # Check if we should cross games
            cell = self.transition_matrix.get(emotion_key)
            should_cross = (
                cell and
                cell.is_cross_game() and
                random.random() < cross_probability
            )

            # Prefer off-diagonal (non-neutral->neutral) for more interesting bridges
            if prefer_off_diagonal and emotion_key[0] == emotion_key[1] == 'neutral':
                should_cross = should_cross and random.random() < 0.3

            if should_cross:
                # Jump to another game with same emotion transition
                current_game = node.game
                other_games = [g for g in cell.games if g != current_game]

                if other_games:
                    target_game = random.choice(other_games)
                    target_edges = cell.edges_by_game[target_game]

                    if target_edges:
                        # Pick a random edge from the target game and use its target
                        bridge_edge = random.choice(target_edges)
                        next_node = bridge_edge.target

                        if next_node not in visited:
                            path[-1]['crossed_to'] = target_game
                            path[-1]['bridge_emotion'] = f"{emotion_key[0]}->{emotion_key[1]}"
                            current = next_node
                            continue

            # Normal within-game traversal
            next_node = edge.target
            if next_node in visited:
                # Try to find unvisited
                unvisited = [e.target for e in out_edges if e.target not in visited]
                if unvisited:
                    next_node = random.choice(unvisited)
                else:
                    break

            current = next_node

        return path

    def get_transition_matrix_data(self) -> Dict:
        """Get the full transition matrix with cross-game bridge info."""
        emotions = ['neutral', 'happy', 'anger', 'sad', 'fear', 'surprise', 'disgust', 'pained']

        matrix = {}
        for src in emotions:
            matrix[src] = {}
            for tgt in emotions:
                key = (src, tgt)
                cell = self.transition_matrix.get(key)
                if cell:
                    matrix[src][tgt] = {
                        'total': cell.total_edges,
                        'by_game': {g: len(e) for g, e in cell.edges_by_game.items()},
                        'is_bridge': cell.is_cross_game(),
                    }
                else:
                    matrix[src][tgt] = {'total': 0, 'by_game': {}, 'is_bridge': False}

        return {
            'emotions': emotions,
            'matrix': matrix,
            'games': list(self.games),
        }

    def get_visualization_data(self, max_nodes_per_cell: int = 5) -> Dict:
        """
        Get data for D3.js force-directed visualization.

        Returns a graph where:
        - Nodes are emotion transition cells (matrix entries)
        - Edges connect cells that share games (can traverse within game)
        - Bridge nodes are highlighted
        """
        # Create nodes for each active emotion cell
        vis_nodes = []
        vis_edges = []

        emotions = ['neutral', 'happy', 'anger', 'sad', 'fear', 'surprise', 'disgust', 'pained']

        cell_ids = {}
        for src in emotions:
            for tgt in emotions:
                key = (src, tgt)
                cell = self.transition_matrix.get(key)
                if cell and cell.total_edges > 0:
                    cell_id = f"{src}->{tgt}"
                    cell_ids[key] = cell_id

                    vis_nodes.append({
                        'id': cell_id,
                        'source_emotion': src,
                        'target_emotion': tgt,
                        'total': cell.total_edges,
                        'games': cell.games,
                        'is_bridge': cell.is_cross_game(),
                        'is_diagonal': src == tgt,
                    })

        # Create edges between cells that can flow (target of one = source of another)
        for key1, id1 in cell_ids.items():
            for key2, id2 in cell_ids.items():
                if key1[1] == key2[0]:  # target emotion matches source emotion
                    # Check if they share at least one game
                    cell1 = self.transition_matrix[key1]
                    cell2 = self.transition_matrix[key2]
                    shared_games = set(cell1.games) & set(cell2.games)

                    if shared_games:
                        vis_edges.append({
                            'source': id1,
                            'target': id2,
                            'shared_games': list(shared_games),
                            'weight': len(shared_games),
                        })

        return {
            'nodes': vis_nodes,
            'edges': vis_edges,
            'games': list(self.games),
        }

    def sample_coverage_trajectory(
        self,
        target_games: List[str] = None,
        min_length: int = 5,
        max_length: int = 20,
    ) -> List[Dict]:
        """
        Sample a trajectory that tries to cover multiple games.

        This is useful for generating training data that spans
        the full corpus while maintaining emotional coherence.
        """
        if target_games is None:
            target_games = list(self.games)

        games_visited = set()
        path = []

        # Start from a random game
        start_game = random.choice(target_games)
        current = random.choice([
            n for n in self.nodes
            if self.nodes[n].game == start_game and self.outgoing.get(n)
        ])

        for step in range(max_length):
            node = self.nodes[current]
            games_visited.add(node.game)

            path.append({
                'step': step,
                'node_id': current,
                'game': node.game,
                'text': node.text,
                'speaker': node.speaker,
                'emotion': node.emotion,
            })

            # Check if we've covered all target games
            if len(games_visited) >= len(target_games) and step >= min_length:
                break

            # Get outgoing edges
            out_edges = self.outgoing.get(current, [])
            if not out_edges:
                break

            # If we haven't visited all games, try to bridge
            unvisited_games = set(target_games) - games_visited
            if unvisited_games:
                # Look for bridge opportunities
                for edge in out_edges:
                    cell = self.transition_matrix.get(edge.emotion_key)
                    if cell and cell.is_cross_game():
                        for game in unvisited_games:
                            if game in cell.edges_by_game:
                                # Bridge to this game
                                target_edges = cell.edges_by_game[game]
                                bridge_edge = random.choice(target_edges)
                                current = bridge_edge.target
                                path[-1]['bridged_to'] = game
                                break
                        else:
                            continue
                        break
                else:
                    # No bridge found, continue within game
                    edge = random.choice(out_edges)
                    current = edge.target
            else:
                # All games visited, just walk
                edge = random.choice(out_edges)
                current = edge.target

        return path

    def get_statistics(self) -> Dict:
        """Get overall statistics about the bridge graph."""
        bridge_cells = [c for c in self.transition_matrix.values() if c.is_cross_game()]

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'games': list(self.games),
            'nodes_per_game': {
                g: sum(1 for n in self.nodes.values() if n.game == g)
                for g in self.games
            },
            'transition_cells': len(self.transition_matrix),
            'bridge_cells': len(bridge_cells),
            'bridgeable_transitions': [
                f"{c.source_emotion}->{c.target_emotion}"
                for c in bridge_cells
            ],
        }


def load_emotion_bridge(data_dir: Path) -> EmotionBridgeGraph:
    """Load all dialogue files and build the emotion bridge graph."""
    graph = EmotionBridgeGraph()

    # Load full dialogue files (they have DLC content)
    for path in sorted(data_dir.glob("*_full_dialogue.json")):
        game_name = path.stem.replace('_full_dialogue', '')
        print(f"Loading {game_name}...")

        with open(path) as f:
            data = json.load(f)

        dialogue = data.get('dialogue', [])
        graph.add_game(game_name, dialogue)
        print(f"  Added {len(dialogue)} lines")

    # Build bridges
    bridges = graph.build_bridges()
    print(f"Built {bridges} bridge cells across {len(graph.games)} games")

    return graph


if __name__ == '__main__':
    # Test the emotion bridge graph
    data_dir = Path('dialogue_data')
    graph = load_emotion_bridge(data_dir)

    print("\n=== Statistics ===")
    stats = graph.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== Bridge Cells ===")
    bridges = graph.get_bridge_cells()
    for b in bridges[:10]:
        print(f"  {b['source_emotion']}->{b['target_emotion']}: {b['games']} ({b['total_edges']} edges)")

    print("\n=== Cross-Game Walk ===")
    path = graph.cross_game_walk(max_steps=8, cross_probability=0.5)
    for node in path:
        bridge = f" [BRIDGE to {node.get('crossed_to')}]" if node.get('crossed_to') else ""
        print(f"  [{node['game']}] [{node['emotion']}] {node['text'][:60]}...{bridge}")
