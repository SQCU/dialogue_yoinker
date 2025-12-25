#!/usr/bin/env python3
"""
Query Graph - Topics as Edges/Gaps Between Dialogue

Models dialogue as a bipartite graph where:
- Text nodes: Actual dialogue content (responses)
- Topic nodes: Queries/gaps that elicit responses

This captures the semantic structure where a "topic" represents
the conversational move or question that bridges dialogue points.

Example:
    Topic "GREETING" --> [text1, text2, text3, ...]
    Topic "QuestAccept" --> [text4, text5, ...]

The topic IS the edge label - it represents the semantic "gap"
or query that the text nodes respond to.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict


@dataclass
class TextNode:
    """A dialogue text response node."""
    id: str  # game:form_id
    game: str
    form_id: str
    text: str
    speaker: Optional[str]
    emotion: str
    topic: str  # The query/gap this responds to
    quest: Optional[str] = None


@dataclass
class TopicNode:
    """A topic/query node representing a semantic gap."""
    id: str  # The topic name
    category: Optional[str] = None  # Semantic category (GREETING, COMBAT, etc.)
    games: Set[str] = field(default_factory=set)
    response_count: int = 0
    sample_texts: List[str] = field(default_factory=list)
    emotion_distribution: Dict[str, int] = field(default_factory=dict)

    def dominant_emotion(self) -> str:
        if not self.emotion_distribution:
            return 'neutral'
        return max(self.emotion_distribution.items(), key=lambda x: x[1])[0]


@dataclass
class QueryEdge:
    """Edge from topic (query) to text (response)."""
    topic_id: str
    text_id: str
    game: str
    weight: float = 1.0


class QueryGraph:
    """
    Bipartite graph: Topics (queries/gaps) <-> Texts (responses)

    This models dialogue structure as:
    - Topics are the "questions" or conversational moves
    - Texts are the "answers" or responses
    - The topic-text relationship captures semantic gaps
    """

    def __init__(self):
        self.text_nodes: Dict[str, TextNode] = {}
        self.topic_nodes: Dict[str, TopicNode] = {}
        self.edges: List[QueryEdge] = []

        # Indexes for fast lookup
        self.topic_to_texts: Dict[str, List[str]] = defaultdict(list)
        self.text_to_topic: Dict[str, str] = {}
        self.games: Set[str] = set()

    def add_dialogue(self, game: str, dialogue: List[Dict]):
        """Add dialogue from a game."""
        self.games.add(game)

        for line in dialogue:
            topic = line.get('topic', '')
            text = line.get('text', '')
            form_id = line.get('form_id', '')

            if not topic or not form_id:
                continue

            # Create text node
            text_id = f"{game}:{form_id}"
            text_node = TextNode(
                id=text_id,
                game=game,
                form_id=form_id,
                text=text,
                speaker=line.get('speaker'),
                emotion=line.get('emotion', 'neutral'),
                topic=topic,
                quest=line.get('quest'),
            )
            self.text_nodes[text_id] = text_node

            # Create or update topic node
            if topic not in self.topic_nodes:
                self.topic_nodes[topic] = TopicNode(
                    id=topic,
                    category=self._categorize_topic(topic),
                )

            topic_node = self.topic_nodes[topic]
            topic_node.games.add(game)
            topic_node.response_count += 1

            # Track emotion distribution
            emo = line.get('emotion', 'neutral')
            topic_node.emotion_distribution[emo] = topic_node.emotion_distribution.get(emo, 0) + 1

            # Sample texts for the topic
            if len(topic_node.sample_texts) < 5 and text:
                topic_node.sample_texts.append(text[:100])

            # Create edge
            edge = QueryEdge(topic_id=topic, text_id=text_id, game=game)
            self.edges.append(edge)

            # Update indexes
            self.topic_to_texts[topic].append(text_id)
            self.text_to_topic[text_id] = topic

    def _categorize_topic(self, topic: str) -> Optional[str]:
        """Categorize a topic by semantic meaning."""
        t = topic.lower()

        if 'greeting' in t or 'hello' in t:
            return 'GREETING'
        elif 'goodbye' in t or 'farewell' in t or 'bye' in t:
            return 'FAREWELL'
        elif 'attack' in t or 'combat' in t or 'fight' in t or 'hostile' in t:
            return 'COMBAT'
        elif 'trade' in t or 'barter' in t or 'buy' in t or 'sell' in t or 'shop' in t:
            return 'TRADE'
        elif 'rumor' in t or 'news' in t or 'gossip' in t:
            return 'RUMORS'
        elif 'quest' in t or 'task' in t or 'mission' in t or 'job' in t:
            return 'QUEST'
        elif 'idle' in t or 'ambient' in t:
            return 'AMBIENT'
        elif 'service' in t or 'train' in t or 'repair' in t:
            return 'SERVICE'
        elif 'follow' in t or 'wait' in t or 'dismiss' in t:
            return 'COMPANION'
        elif 'crime' in t or 'bounty' in t or 'arrest' in t or 'guard' in t:
            return 'CRIME'

        return None

    def get_topic_responses(self, topic: str, game: str = None) -> List[TextNode]:
        """Get all text responses for a topic."""
        text_ids = self.topic_to_texts.get(topic, [])
        nodes = [self.text_nodes[tid] for tid in text_ids if tid in self.text_nodes]

        if game:
            nodes = [n for n in nodes if n.game == game]

        return nodes

    def get_cross_game_topics(self) -> List[TopicNode]:
        """Get topics that appear in multiple games."""
        return [t for t in self.topic_nodes.values() if len(t.games) > 1]

    def get_topics_by_category(self, category: str) -> List[TopicNode]:
        """Get all topics in a semantic category."""
        return [t for t in self.topic_nodes.values() if t.category == category]

    def sample_topic_responses(
        self,
        topic: str = None,
        category: str = None,
        n: int = 5,
        cross_game: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample responses for a topic/category.

        If cross_game=True, tries to get responses from different games.
        """
        # Select topic
        if topic:
            if topic not in self.topic_nodes:
                return {'error': f'Topic not found: {topic}'}
            selected_topic = topic
        elif category:
            cat_topics = self.get_topics_by_category(category)
            if not cat_topics:
                return {'error': f'No topics in category: {category}'}
            # Pick one with multiple responses
            cat_topics = [t for t in cat_topics if t.response_count >= n]
            if not cat_topics:
                cat_topics = self.get_topics_by_category(category)
            selected_topic = random.choice(cat_topics).id
        else:
            # Random topic with enough responses
            candidates = [t for t in self.topic_nodes.values() if t.response_count >= n]
            if not candidates:
                candidates = list(self.topic_nodes.values())
            selected_topic = random.choice(candidates).id

        topic_node = self.topic_nodes[selected_topic]
        responses = self.get_topic_responses(selected_topic)

        if cross_game and len(topic_node.games) > 1:
            # Sample from different games
            by_game = defaultdict(list)
            for r in responses:
                by_game[r.game].append(r)

            sampled = []
            games = list(by_game.keys())
            random.shuffle(games)
            for game in games:
                if len(sampled) >= n:
                    break
                game_responses = by_game[game]
                take = min(len(game_responses), n - len(sampled), 2)  # Max 2 per game
                sampled.extend(random.sample(game_responses, take))
            responses = sampled
        else:
            responses = random.sample(responses, min(n, len(responses)))

        return {
            'topic': selected_topic,
            'category': topic_node.category,
            'games': list(topic_node.games),
            'total_responses': topic_node.response_count,
            'dominant_emotion': topic_node.dominant_emotion(),
            'sampled_responses': [
                {
                    'id': r.id,
                    'game': r.game,
                    'text': r.text,
                    'speaker': r.speaker,
                    'emotion': r.emotion,
                }
                for r in responses
            ],
        }

    def walk_topic_chain(
        self,
        start_topic: str = None,
        max_steps: int = 5,
        cross_game: bool = True,
    ) -> List[Dict]:
        """
        Walk through topics by following semantic connections.

        At each step:
        1. Sample a response from current topic
        2. Find a related topic (same category, shared words, or random)
        3. Move to that topic
        """
        if not self.topic_nodes:
            return []

        # Start topic
        if start_topic and start_topic in self.topic_nodes:
            current_topic = start_topic
        else:
            current_topic = random.choice(list(self.topic_nodes.keys()))

        path = []
        visited_topics = set()

        for step in range(max_steps):
            visited_topics.add(current_topic)
            topic_node = self.topic_nodes[current_topic]

            # Sample a response
            responses = self.get_topic_responses(current_topic)
            if not responses:
                break

            response = random.choice(responses)

            path.append({
                'step': step,
                'topic': current_topic,
                'category': topic_node.category,
                'response': {
                    'game': response.game,
                    'text': response.text,
                    'speaker': response.speaker,
                    'emotion': response.emotion,
                },
            })

            # Find next topic
            next_topic = self._find_related_topic(current_topic, visited_topics, cross_game)
            if not next_topic:
                break

            current_topic = next_topic

        return path

    def _find_related_topic(
        self,
        current: str,
        visited: Set[str],
        cross_game: bool,
    ) -> Optional[str]:
        """
        Find a related topic to transition to using REAL structural edges.

        Only returns topics that have an actual dialogue edge from current topic.
        """
        # Use real topic transitions if available
        if hasattr(self, '_topic_transitions') and self._topic_transitions:
            real_neighbors = self._topic_transitions.get(current, set())
            candidates = [t for t in real_neighbors if t not in visited]

            if candidates:
                # If cross_game requested, prefer topics from different games
                if cross_game:
                    current_games = self.topic_nodes[current].games if current in self.topic_nodes else set()
                    cross_game_candidates = [
                        t for t in candidates
                        if t in self.topic_nodes and self.topic_nodes[t].games - current_games
                    ]
                    if cross_game_candidates:
                        return random.choice(cross_game_candidates)

                return random.choice(candidates)

            # No unvisited real neighbors - walk ends
            return None

        # Fallback if no transitions built (shouldn't happen)
        return None

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        categories = defaultdict(int)
        for t in self.topic_nodes.values():
            if t.category:
                categories[t.category] += 1

        cross_game = len(self.get_cross_game_topics())

        return {
            'text_nodes': len(self.text_nodes),
            'topic_nodes': len(self.topic_nodes),
            'edges': len(self.edges),
            'games': list(self.games),
            'cross_game_topics': cross_game,
            'categories': dict(categories),
            'avg_responses_per_topic': len(self.edges) / max(len(self.topic_nodes), 1),
        }

    def build_topic_transitions_from_graph(self, dialogue_graphs: Dict[str, 'DialogueGraph']) -> Dict[str, Set[str]]:
        """
        Build topic→topic transition graph from actual DialogueGraph edges.

        This lifts the real text→text structural edges to topic→topic.
        Only creates topic transitions where there's an actual dialogue edge.

        Args:
            dialogue_graphs: {game_name: DialogueGraph} for each game

        Returns: {topic_id: set of next_topic_ids}
        """
        topic_transitions: Dict[str, Set[str]] = defaultdict(set)
        edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        for game, graph in dialogue_graphs.items():
            for edge in graph.edges:
                # Map text IDs to our node IDs
                src_text_id = f"{game}:{edge.source}"
                tgt_text_id = f"{game}:{edge.target}"

                src_topic = self.text_to_topic.get(src_text_id)
                tgt_topic = self.text_to_topic.get(tgt_text_id)

                if src_topic and tgt_topic and src_topic != tgt_topic:
                    topic_transitions[src_topic].add(tgt_topic)
                    edge_counts[(src_topic, tgt_topic)] += 1

        # Store edge counts for weighting
        self._topic_edge_counts = dict(edge_counts)
        self._topic_transitions = dict(topic_transitions)

        return dict(topic_transitions)

    def build_topic_transitions(self, dialogue_graph_edges: List[Tuple[str, str, str]] = None) -> Dict[str, Set[str]]:
        """
        Build topic→topic transition graph from text-level edges.

        Args:
            dialogue_graph_edges: List of (game, src_form_id, tgt_form_id) tuples
                                  from actual DialogueGraph edges

        Returns: {topic_id: set of next_topic_ids}
        """
        if hasattr(self, '_topic_transitions') and self._topic_transitions:
            return self._topic_transitions

        topic_transitions: Dict[str, Set[str]] = defaultdict(set)

        if dialogue_graph_edges:
            for game, src_id, tgt_id in dialogue_graph_edges:
                src_text_id = f"{game}:{src_id}"
                tgt_text_id = f"{game}:{tgt_id}"

                src_topic = self.text_to_topic.get(src_text_id)
                tgt_topic = self.text_to_topic.get(tgt_text_id)

                if src_topic and tgt_topic and src_topic != tgt_topic:
                    topic_transitions[src_topic].add(tgt_topic)

        self._topic_transitions = dict(topic_transitions)
        return dict(topic_transitions)

    def find_topic_cycles(
        self,
        max_cycle_length: int = 6,
        max_cycles: int = 20,
    ) -> List[List[str]]:
        """
        Find cycles in the topic transition graph using DFS.

        Returns list of cycles, where each cycle is a list of topic IDs.
        """
        transitions = self.build_topic_transitions()
        cycles = []
        visited_cycles = set()  # To avoid duplicates

        def dfs(start: str, current: str, path: List[str], depth: int):
            if depth > max_cycle_length:
                return
            if len(cycles) >= max_cycles:
                return

            for next_topic in transitions.get(current, []):
                if next_topic == start and len(path) >= 2:
                    # Found a cycle back to start
                    cycle = path + [next_topic]
                    # Normalize cycle (start from lexically smallest)
                    min_idx = cycle.index(min(cycle[:-1]))
                    normalized = tuple(cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]])
                    if normalized not in visited_cycles:
                        visited_cycles.add(normalized)
                        cycles.append(list(normalized))
                elif next_topic not in path:
                    dfs(start, next_topic, path + [next_topic], depth + 1)

        # Start DFS from each topic
        for topic in list(transitions.keys())[:500]:  # Limit search space
            if len(cycles) >= max_cycles:
                break
            dfs(topic, topic, [topic], 0)

        return cycles

    def get_cycle_examples(
        self,
        max_cycles: int = 10,
        max_cycle_length: int = 5,
    ) -> List[Dict]:
        """
        Get example topic cycles with sample text for each step.

        Returns cycles showing: topic → text → topic → text → ... → (back to start)
        """
        cycles = self.find_topic_cycles(
            max_cycle_length=max_cycle_length,
            max_cycles=max_cycles,
        )

        examples = []
        for cycle in cycles:
            steps = []
            for i, topic_id in enumerate(cycle[:-1]):  # Exclude the repeated end
                topic_node = self.topic_nodes.get(topic_id)
                if not topic_node:
                    continue

                # Get sample response for this topic
                responses = self.get_topic_responses(topic_id)
                sample_text = random.choice(responses) if responses else None

                next_topic = cycle[i + 1] if i + 1 < len(cycle) else cycle[0]

                steps.append({
                    'topic': topic_id,
                    'category': topic_node.category,
                    'next_topic': next_topic,
                    'sample_response': {
                        'game': sample_text.game if sample_text else None,
                        'text': sample_text.text[:100] if sample_text and sample_text.text else '',
                        'speaker': sample_text.speaker if sample_text else None,
                        'emotion': sample_text.emotion if sample_text else 'neutral',
                    } if sample_text else None,
                })

            if steps:
                # Determine games involved
                games_in_cycle = set()
                for s in steps:
                    if s['sample_response'] and s['sample_response']['game']:
                        games_in_cycle.add(s['sample_response']['game'])

                examples.append({
                    'cycle_length': len(cycle) - 1,
                    'topics': [s['topic'] for s in steps],
                    'games': list(games_in_cycle),
                    'steps': steps,
                })

        return examples

    def get_topic_transition_stats(self) -> Dict:
        """Get statistics about topic-to-topic transitions (from real edges)."""
        if not hasattr(self, '_topic_transitions') or not self._topic_transitions:
            return {
                'error': 'No topic transitions built. Load with DialogueGraphs.',
                'total_transitions': 0,
            }

        transitions = self._topic_transitions

        # Count transitions
        total_edges = sum(len(targets) for targets in transitions.values())
        topics_with_outgoing = len(transitions)

        # Find highly connected topics
        out_degree = [(t, len(targets)) for t, targets in transitions.items()]
        out_degree.sort(key=lambda x: -x[1])

        # Find topics with most incoming
        in_degree: Dict[str, int] = defaultdict(int)
        for targets in transitions.values():
            for t in targets:
                in_degree[t] += 1
        in_sorted = sorted(in_degree.items(), key=lambda x: -x[1])

        # Get edge counts if available
        edge_weights = []
        if hasattr(self, '_topic_edge_counts'):
            edge_weights = sorted(self._topic_edge_counts.items(), key=lambda x: -x[1])[:10]

        return {
            'total_transitions': total_edges,
            'topics_with_outgoing': topics_with_outgoing,
            'avg_out_degree': total_edges / max(topics_with_outgoing, 1),
            'source': 'DialogueGraph structural edges',
            'top_sources': [
                {'topic': t, 'out_degree': d, 'category': self.topic_nodes[t].category if t in self.topic_nodes else None}
                for t, d in out_degree[:10]
            ],
            'top_sinks': [
                {'topic': t, 'in_degree': d, 'category': self.topic_nodes[t].category if t in self.topic_nodes else None}
                for t, d in in_sorted[:10]
            ],
            'strongest_edges': [
                {'from': e[0][0], 'to': e[0][1], 'text_edges': e[1]}
                for e in edge_weights
            ] if edge_weights else [],
        }

    def get_visualization_data(self, max_topics: int = 50) -> Dict:
        """Get D3.js visualization data for the bipartite graph."""
        # Select top topics by response count
        sorted_topics = sorted(
            self.topic_nodes.values(),
            key=lambda t: t.response_count,
            reverse=True
        )[:max_topics]

        nodes = []
        edges = []

        # Add topic nodes
        for topic in sorted_topics:
            nodes.append({
                'id': topic.id,
                'type': 'topic',
                'category': topic.category,
                'games': list(topic.games),
                'response_count': topic.response_count,
                'emotion': topic.dominant_emotion(),
            })

        # Add sample text nodes and edges for each topic
        topic_ids = {t.id for t in sorted_topics}
        text_sample_count = 3  # texts per topic to show

        for topic in sorted_topics:
            responses = self.get_topic_responses(topic.id)
            sampled = random.sample(responses, min(text_sample_count, len(responses)))

            for text_node in sampled:
                nodes.append({
                    'id': text_node.id,
                    'type': 'text',
                    'game': text_node.game,
                    'text': text_node.text[:80] if text_node.text else '',
                    'emotion': text_node.emotion,
                })
                edges.append({
                    'source': topic.id,
                    'target': text_node.id,
                    'game': text_node.game,
                })

        return {
            'nodes': nodes,
            'edges': edges,
            'topic_count': len(sorted_topics),
        }


def load_query_graph(data_dir: Path, load_dialogue_graphs: bool = True) -> QueryGraph:
    """
    Load all dialogue and build query graph.

    Args:
        data_dir: Path to dialogue data directory
        load_dialogue_graphs: If True, also loads DialogueGraphs to get real
                              text→text edges for topic transitions
    """
    from dialogue_graph import DialogueGraph

    graph = QueryGraph()
    dialogue_graphs = {}

    for path in sorted(data_dir.glob("*_full_dialogue.json")):
        game = path.stem.replace('_full_dialogue', '')
        print(f"Loading {game}...")

        with open(path) as f:
            data = json.load(f)

        dialogue_list = data.get('dialogue', [])
        graph.add_dialogue(game, dialogue_list)
        print(f"  {len(dialogue_list)} lines")

        # Build DialogueGraph for real edges
        if load_dialogue_graphs:
            dg = DialogueGraph.from_dialogue_data(dialogue_list)
            dialogue_graphs[game] = dg
            print(f"  {len(dg.edges)} structural edges")

    # Build topic transitions from real edges
    if dialogue_graphs:
        graph.build_topic_transitions_from_graph(dialogue_graphs)
        trans = graph._topic_transitions
        print(f"\nBuilt {sum(len(v) for v in trans.values())} topic→topic transitions from {len(trans)} topics")

    print(f"\nQuery graph stats:")
    stats = graph.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return graph


if __name__ == '__main__':
    data_dir = Path('dialogue_data')
    graph = load_query_graph(data_dir)

    print("\n=== Cross-Game Topics ===")
    cross = graph.get_cross_game_topics()
    for t in cross[:10]:
        print(f"  {t.id}: {list(t.games)} ({t.response_count} responses)")

    print("\n=== Sample by Category ===")
    for cat in ['GREETING', 'COMBAT', 'QUEST']:
        result = graph.sample_topic_responses(category=cat, n=3, cross_game=True)
        print(f"\n{cat}: {result['topic']}")
        for r in result['sampled_responses']:
            print(f"  [{r['game']}] {r['text'][:60]}...")

    print("\n=== Topic Chain Walk ===")
    path = graph.walk_topic_chain(max_steps=5, cross_game=True)
    for step in path:
        print(f"  [{step['topic']}] ({step['category'] or 'misc'})")
        print(f"    [{step['response']['game']}] {step['response']['text'][:60]}...")
