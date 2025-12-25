#!/usr/bin/env python3
"""
Topic Graph - Model dialogue as topic→topic transitions.

Unlike the DialogueGraph (which uses INFO records as nodes), this graph uses
dialogue TOPICS as nodes. Edges represent conversational flow between topics.

This is useful for:
- Understanding narrative structure at a higher level
- Finding topic clusters that form conversation arcs
- Filtering out hub topics (GREETING, GOODBYE) that dominate raw graphs
- Building topic→text→topic traversals for training data
"""

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple, Iterator
import random


# Common hub topics that may need filtering
COMMON_HUB_TOPICS = {
    # Generic
    'GREETING', 'HELLO', 'GOODBYE', 'BYE',
    # Oblivion-style
    'YOURDEFAULTTOPIC', 'YOURDEFAULTTOPIC2',
    # Quest-generic
    'INFOGENERAL',
}


@dataclass
class TopicNode:
    """A topic node containing aggregated statistics."""
    topic_id: str                          # Topic editor ID
    line_count: int = 0                    # Number of dialogue lines
    speakers: Set[str] = field(default_factory=set)
    quests: Set[str] = field(default_factory=set)
    emotions: Dict[str, int] = field(default_factory=dict)  # emotion -> count
    sample_lines: List[str] = field(default_factory=list)   # Sample texts

    def dominant_emotion(self) -> str:
        if not self.emotions:
            return 'neutral'
        return max(self.emotions.items(), key=lambda x: x[1])[0]

    def emotion_distribution(self) -> Dict[str, float]:
        total = sum(self.emotions.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.emotions.items()}


@dataclass
class TopicEdge:
    """An edge representing transition between topics."""
    source: str
    target: str
    weight: int = 1                        # Transition count
    quest_context: Set[str] = field(default_factory=set)
    speaker_context: Set[str] = field(default_factory=set)

    def strength(self) -> float:
        """Normalized strength based on weight and context overlap."""
        context_bonus = len(self.quest_context) * 0.1 + len(self.speaker_context) * 0.05
        return self.weight + context_bonus


class TopicGraph:
    """
    A graph of topic→topic transitions.

    Construction:
    1. Group dialogue lines by topic
    2. Identify topic transitions within quest/speaker context
    3. Build weighted edges based on transition frequency
    """

    def __init__(self):
        self.nodes: Dict[str, TopicNode] = {}
        self.edges: Dict[Tuple[str, str], TopicEdge] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_topic(self, topic_id: str, line: Dict[str, Any]):
        """Add or update a topic node from a dialogue line."""
        if topic_id not in self.nodes:
            self.nodes[topic_id] = TopicNode(topic_id=topic_id)

        node = self.nodes[topic_id]
        node.line_count += 1

        if line.get('speaker'):
            node.speakers.add(line['speaker'])
        if line.get('quest'):
            node.quests.add(line['quest'])

        emotion = line.get('emotion', 'neutral')
        node.emotions[emotion] = node.emotions.get(emotion, 0) + 1

        # Keep sample lines (up to 5)
        if len(node.sample_lines) < 5 and line.get('text'):
            node.sample_lines.append(line['text'][:100])

    def add_transition(self, source: str, target: str,
                       quest: Optional[str] = None,
                       speaker: Optional[str] = None):
        """Add or strengthen a topic transition edge."""
        if source == target:
            return  # Skip self-loops

        key = (source, target)
        if key not in self.edges:
            self.edges[key] = TopicEdge(source=source, target=target)

        edge = self.edges[key]
        edge.weight += 1
        if quest:
            edge.quest_context.add(quest)
        if speaker:
            edge.speaker_context.add(speaker)

        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)

    @classmethod
    def from_dialogue(cls, dialogue: List[Dict[str, Any]]) -> 'TopicGraph':
        """Build topic graph from dialogue data."""
        graph = cls()

        # Group by (quest, speaker) for transition detection
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        for line in dialogue:
            topic = line.get('topic', '')
            if not topic:
                continue

            graph.add_topic(topic, line)

            # Group for transition detection
            quest = line.get('quest') or '_no_quest'
            speaker = line.get('speaker') or '_no_speaker'
            groups[(quest, speaker)].append(line)

        # Detect transitions within groups
        for (quest, speaker), lines in groups.items():
            # Sort by form_id to approximate ordering
            sorted_lines = sorted(lines, key=lambda x: x.get('form_id', ''))

            for i in range(len(sorted_lines) - 1):
                src_topic = sorted_lines[i].get('topic', '')
                tgt_topic = sorted_lines[i + 1].get('topic', '')

                if src_topic and tgt_topic:
                    graph.add_transition(
                        src_topic, tgt_topic,
                        quest=quest if quest != '_no_quest' else None,
                        speaker=speaker if speaker != '_no_speaker' else None
                    )

        return graph

    # =========================================================================
    # Filtering
    # =========================================================================

    def filter_topics(self,
                      exclude_topics: Set[str] = None,
                      min_lines: int = 1,
                      top_k_exclude: int = 0,
                      top_p_exclude: float = 0.0) -> 'TopicGraph':
        """
        Create filtered copy of graph.

        Args:
            exclude_topics: Set of topic IDs to exclude
            min_lines: Minimum lines per topic to include
            top_k_exclude: Exclude top K most connected topics
            top_p_exclude: Exclude topics in top P percentile by degree

        Returns:
            New filtered TopicGraph
        """
        exclude = set(exclude_topics or [])

        # Add top-k exclusions
        if top_k_exclude > 0:
            by_degree = sorted(
                self.nodes.keys(),
                key=lambda t: len(self._adjacency.get(t, set())) + len(self._reverse_adjacency.get(t, set())),
                reverse=True
            )
            exclude.update(by_degree[:top_k_exclude])

        # Add top-p exclusions
        if top_p_exclude > 0:
            degrees = [
                len(self._adjacency.get(t, set())) + len(self._reverse_adjacency.get(t, set()))
                for t in self.nodes
            ]
            if degrees:
                threshold = sorted(degrees, reverse=True)[int(len(degrees) * top_p_exclude)]
                for topic in self.nodes:
                    deg = len(self._adjacency.get(topic, set())) + len(self._reverse_adjacency.get(topic, set()))
                    if deg >= threshold:
                        exclude.add(topic)

        # Build filtered graph
        filtered = TopicGraph()

        for topic_id, node in self.nodes.items():
            if topic_id in exclude:
                continue
            if node.line_count < min_lines:
                continue

            filtered.nodes[topic_id] = node

        for (src, tgt), edge in self.edges.items():
            if src in filtered.nodes and tgt in filtered.nodes:
                filtered.edges[(src, tgt)] = edge
                filtered._adjacency[src].add(tgt)
                filtered._reverse_adjacency[tgt].add(src)

        return filtered

    def filter_common_hubs(self, additional: Set[str] = None) -> 'TopicGraph':
        """Filter out common hub topics like GREETING, GOODBYE."""
        exclude = COMMON_HUB_TOPICS.copy()
        if additional:
            exclude.update(additional)
        return self.filter_topics(exclude_topics=exclude)

    # =========================================================================
    # Analysis
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.nodes:
            return {'topics': 0, 'edges': 0}

        degrees = [
            len(self._adjacency.get(t, set())) + len(self._reverse_adjacency.get(t, set()))
            for t in self.nodes
        ]

        return {
            'topics': len(self.nodes),
            'edges': len(self.edges),
            'total_lines': sum(n.line_count for n in self.nodes.values()),
            'avg_lines_per_topic': sum(n.line_count for n in self.nodes.values()) / len(self.nodes),
            'avg_degree': sum(degrees) / len(degrees),
            'max_degree': max(degrees),
            'isolated_topics': len([d for d in degrees if d == 0]),
        }

    def find_hubs(self, min_degree: int = 10) -> List[Tuple[str, int, TopicNode]]:
        """Find hub topics with high connectivity."""
        hubs = []
        for topic_id, node in self.nodes.items():
            degree = len(self._adjacency.get(topic_id, set())) + len(self._reverse_adjacency.get(topic_id, set()))
            if degree >= min_degree:
                hubs.append((topic_id, degree, node))
        return sorted(hubs, key=lambda x: -x[1])

    def find_chains(self, min_length: int = 3, max_length: int = 10) -> List[List[str]]:
        """Find linear chains of topics (paths with no branching)."""
        chains = []
        visited = set()

        for topic in self.nodes:
            if topic in visited:
                continue

            # Try to extend chain from this topic
            chain = [topic]
            current = topic

            # Extend forward
            while len(chain) < max_length:
                neighbors = self._adjacency.get(current, set())
                # Only continue if single outgoing edge to unvisited node
                unvisited = [n for n in neighbors if n not in visited and n not in chain]
                if len(unvisited) != 1:
                    break
                current = unvisited[0]
                chain.append(current)

            if len(chain) >= min_length:
                chains.append(chain)
                visited.update(chain)

        return sorted(chains, key=len, reverse=True)

    def topic_text_topic_paths(self, topic: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get topic→text→topic paths starting from a topic.

        Returns paths showing: topic → [sample lines] → next_topic
        """
        paths = []

        if topic not in self.nodes:
            return paths

        node = self.nodes[topic]
        neighbors = self._adjacency.get(topic, set())

        for neighbor in list(neighbors)[:10]:  # Limit for performance
            edge = self.edges.get((topic, neighbor))
            neighbor_node = self.nodes.get(neighbor)

            if not neighbor_node:
                continue

            paths.append({
                'source_topic': topic,
                'source_lines': node.sample_lines[:3],
                'source_emotion': node.dominant_emotion(),
                'transition_weight': edge.weight if edge else 1,
                'target_topic': neighbor,
                'target_lines': neighbor_node.sample_lines[:3],
                'target_emotion': neighbor_node.dominant_emotion(),
                'shared_quests': list(edge.quest_context)[:3] if edge else [],
            })

        return paths

    # =========================================================================
    # NetworkX Integration
    # =========================================================================

    def to_networkx(self):
        """Convert to NetworkX DiGraph."""
        import networkx as nx

        G = nx.DiGraph()

        for topic_id, node in self.nodes.items():
            G.add_node(
                topic_id,
                line_count=node.line_count,
                dominant_emotion=node.dominant_emotion(),
                quest_count=len(node.quests),
                speaker_count=len(node.speakers),
            )

        for (src, tgt), edge in self.edges.items():
            G.add_edge(src, tgt, weight=edge.weight)

        return G

    def pagerank(self, top_n: int = 20) -> List[Tuple[str, float, TopicNode]]:
        """Compute PageRank on topic graph."""
        import networkx as nx

        G = self.to_networkx()
        scores = nx.pagerank(G, alpha=0.85)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        return [(topic, score, self.nodes[topic]) for topic, score in ranked if topic in self.nodes]

    def communities(self) -> List[Set[str]]:
        """Detect communities in topic graph."""
        import networkx as nx

        G = self.to_networkx().to_undirected()
        try:
            communities = nx.community.louvain_communities(G, seed=42)
        except:
            communities = nx.community.greedy_modularity_communities(G)

        return sorted(communities, key=len, reverse=True)


def load_dialogue(path: Path) -> List[Dict[str, Any]]:
    """Load dialogue from JSON export."""
    with open(path) as f:
        data = json.load(f)
    return data.get('dialogue', [])


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build and analyze topic graphs')
    parser.add_argument('input', type=Path, help='Path to *_dialogue.json file')
    parser.add_argument('--stats', action='store_true', help='Print statistics')
    parser.add_argument('--filter-hubs', action='store_true',
                       help='Filter common hub topics (GREETING, GOODBYE, etc.)')
    parser.add_argument('--top-k-exclude', type=int, default=0,
                       help='Exclude top K most connected topics')
    parser.add_argument('--min-lines', type=int, default=1,
                       help='Minimum lines per topic')
    parser.add_argument('--chains', type=int, default=0,
                       help='Find linear chains (specify min length)')
    parser.add_argument('--pagerank', type=int, default=0,
                       help='Show top N topics by PageRank')
    parser.add_argument('--paths-from', type=str, default=None,
                       help='Show topic→text→topic paths from specified topic')

    args = parser.parse_args()

    # Load and build graph
    print(f"Loading {args.input}...")
    dialogue = load_dialogue(args.input)
    print(f"Loaded {len(dialogue)} lines")

    print("Building topic graph...")
    graph = TopicGraph.from_dialogue(dialogue)
    print(f"Built graph with {len(graph.nodes)} topics, {len(graph.edges)} edges")

    # Apply filters
    if args.filter_hubs:
        graph = graph.filter_common_hubs()
        print(f"After hub filtering: {len(graph.nodes)} topics, {len(graph.edges)} edges")

    if args.top_k_exclude > 0:
        graph = graph.filter_topics(top_k_exclude=args.top_k_exclude)
        print(f"After top-{args.top_k_exclude} exclusion: {len(graph.nodes)} topics, {len(graph.edges)} edges")

    if args.min_lines > 1:
        graph = graph.filter_topics(min_lines=args.min_lines)
        print(f"After min-lines={args.min_lines}: {len(graph.nodes)} topics, {len(graph.edges)} edges")

    if args.stats:
        print(f"\n{'='*60}")
        print("Topic Graph Statistics")
        print(f"{'='*60}")
        stats = graph.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

        print("\nTop 10 Hub Topics:")
        for topic, degree, node in graph.find_hubs(min_degree=1)[:10]:
            print(f"  [{degree:3d}] {topic}: {node.dominant_emotion()}, {node.line_count} lines")

    if args.chains > 0:
        print(f"\n{'='*60}")
        print(f"Topic Chains (min length {args.chains})")
        print(f"{'='*60}")
        chains = graph.find_chains(min_length=args.chains)
        for i, chain in enumerate(chains[:10]):
            print(f"\nChain {i+1} ({len(chain)} topics):")
            for topic in chain:
                node = graph.nodes.get(topic)
                if node:
                    print(f"  → {topic} [{node.dominant_emotion()}]: {node.sample_lines[0][:50] if node.sample_lines else ''}...")

    if args.pagerank > 0:
        print(f"\n{'='*60}")
        print(f"Top {args.pagerank} Topics by PageRank")
        print(f"{'='*60}")
        for topic, score, node in graph.pagerank(top_n=args.pagerank):
            print(f"  {score:.5f} {topic} [{node.dominant_emotion()}]")
            if node.sample_lines:
                print(f"           \"{node.sample_lines[0][:60]}...\"")

    if args.paths_from:
        print(f"\n{'='*60}")
        print(f"Topic→Text→Topic Paths from: {args.paths_from}")
        print(f"{'='*60}")
        paths = graph.topic_text_topic_paths(args.paths_from)
        for path in paths:
            print(f"\n{path['source_topic']} [{path['source_emotion']}]")
            for line in path['source_lines'][:2]:
                print(f"    \"{line[:60]}...\"")
            print(f"  ↓ (weight: {path['transition_weight']}, quests: {path['shared_quests']})")
            print(f"{path['target_topic']} [{path['target_emotion']}]")
            for line in path['target_lines'][:2]:
                print(f"    \"{line[:60]}...\"")


if __name__ == '__main__':
    main()
