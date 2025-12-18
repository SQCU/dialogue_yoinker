#!/usr/bin/env python3
"""
Dialogue Graph - Model dialogue as a directed graph with state annotations.

Unlike linear chains, this represents dialogue as a multigraph where:
- Nodes are dialogue lines (INFO records)
- Edges represent transitions (condition-gated, topic-linked, or sequential)
- Cycles are allowed and expected (repeatable content, conversation loops)
- State annotations on edges encode reachability conditions

This enables sampling non-linear paths through dialogue space,
finding interesting substructures (hubs, cycles, bottlenecks),
and analyzing the "shape" of interactive narrative.
"""

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple, Iterator
from enum import Enum
import random


# =============================================================================
# Condition Decoding
# =============================================================================

class ConditionFunction(Enum):
    """Known condition functions relevant to dialogue structure."""
    GET_STAGE = 43           # Quest stage check
    GET_STAGE_FNV = 79       # FNV variant
    GET_IS_ID = 72           # Speaker/target identity
    GET_FACTION_RANK = 73    # Faction membership
    GET_DISPOSITION = 56     # NPC disposition toward player
    GET_QUEST_RUNNING = 38   # Quest active check
    GET_QUEST_COMPLETED = 161
    GET_IN_CELL = 117        # Location check
    GET_IS_SEX = 70          # Gender check
    GET_IS_RACE = 77         # Race check
    GET_ACTOR_VALUE = 14     # Skill/attribute check
    GET_RANDOM_PERCENT = 60  # Random gating


@dataclass
class StateCondition:
    """A decoded state condition for dialogue reachability."""
    function: str           # Human-readable function name
    target: Optional[int]   # Form ID being checked (quest, NPC, faction, etc.)
    comparison: str         # ==, !=, <, >, <=, >=
    value: float           # Comparison value
    run_on: str            # 'subject', 'target', 'reference', etc.

    def __str__(self):
        return f"{self.function}({self.target or '?'}) {self.comparison} {self.value}"

    @classmethod
    def from_ctda(cls, ctda: Dict[str, Any]) -> 'StateCondition':
        """Decode a raw CTDA condition dict."""
        func_idx = ctda.get('function', 0)

        # Decode function name
        func_names = {
            43: 'GetStage', 79: 'GetStage',
            72: 'GetIsID',
            73: 'GetFactionRank',
            56: 'GetDisposition',
            38: 'GetQuestRunning',
            161: 'GetQuestCompleted',
            117: 'GetInCell',
            70: 'GetIsSex',
            77: 'GetIsRace',
            14: 'GetActorValue',
            60: 'GetRandomPercent',
        }
        func_name = func_names.get(func_idx, f'Function_{func_idx}')

        # Decode comparison type from condition type byte
        cond_type = ctda.get('type', 0)
        comparisons = {
            0: '==', 1: '!=', 2: '>', 3: '>=', 4: '<', 5: '<='
        }
        comparison = comparisons.get(cond_type & 0x0F, '?')

        # Decode run_on
        run_on_val = ctda.get('run_on', 0)
        run_on_names = {0: 'subject', 1: 'target', 2: 'reference', 3: 'combat_target'}
        run_on = run_on_names.get(run_on_val, f'run_on_{run_on_val}')

        return cls(
            function=func_name,
            target=ctda.get('param1'),
            comparison=comparison,
            value=ctda.get('value', 0),
            run_on=run_on
        )


@dataclass
class DialogueNode:
    """A node in the dialogue graph (one INFO record)."""
    id: str                 # form_id as string
    text: str
    speaker: Optional[str]
    emotion: str
    topic: str
    quest: Optional[str]
    conditions: List[StateCondition] = field(default_factory=list)

    # Graph connectivity (populated during graph construction)
    outgoing: Set[str] = field(default_factory=set)  # IDs of reachable nodes
    incoming: Set[str] = field(default_factory=set)  # IDs that can reach this

    def state_signature(self) -> frozenset:
        """Get a hashable representation of required state conditions."""
        return frozenset(str(c) for c in self.conditions if c.function in
                        ('GetStage', 'GetQuestRunning', 'GetQuestCompleted'))


@dataclass
class DialogueEdge:
    """An edge in the dialogue graph."""
    source: str             # Source node ID
    target: str             # Target node ID
    edge_type: str          # 'sequential', 'topic_branch', 'condition_gate', 'response_link'
    conditions: List[StateCondition] = field(default_factory=list)
    weight: float = 1.0     # For weighted sampling

    def label(self) -> str:
        if self.conditions:
            return f"{self.edge_type}: {', '.join(str(c) for c in self.conditions[:2])}"
        return self.edge_type


class DialogueGraph:
    """
    A directed multigraph of dialogue with state annotations.

    Construction strategy:
    1. Create nodes from all INFO records
    2. Add sequential edges within topics (ordered by quest stage)
    3. Add topic branch edges (GREETING → specific topics)
    4. Add condition-gated edges (same topic, different state requirements)
    5. Identify cycles and strongly connected components
    """

    def __init__(self):
        self.nodes: Dict[str, DialogueNode] = {}
        self.edges: List[DialogueEdge] = []
        self._adjacency: Dict[str, List[DialogueEdge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[DialogueEdge]] = defaultdict(list)

        # Indices for fast lookup
        self._by_topic: Dict[str, List[str]] = defaultdict(list)
        self._by_quest: Dict[str, List[str]] = defaultdict(list)
        self._by_speaker: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: DialogueNode):
        self.nodes[node.id] = node
        self._by_topic[node.topic].append(node.id)
        if node.quest:
            self._by_quest[node.quest].append(node.id)
        if node.speaker:
            self._by_speaker[node.speaker].append(node.id)

    def add_edge(self, edge: DialogueEdge):
        self.edges.append(edge)
        self._adjacency[edge.source].append(edge)
        self._reverse_adjacency[edge.target].append(edge)

        # Update node connectivity
        if edge.source in self.nodes:
            self.nodes[edge.source].outgoing.add(edge.target)
        if edge.target in self.nodes:
            self.nodes[edge.target].incoming.add(edge.source)

    def neighbors(self, node_id: str) -> List[Tuple[str, DialogueEdge]]:
        """Get all neighbors with their connecting edges."""
        return [(e.target, e) for e in self._adjacency.get(node_id, [])]

    def predecessors(self, node_id: str) -> List[Tuple[str, DialogueEdge]]:
        """Get all predecessors with their connecting edges."""
        return [(e.source, e) for e in self._reverse_adjacency.get(node_id, [])]

    # =========================================================================
    # Graph Construction
    # =========================================================================

    @classmethod
    def from_dialogue_data(cls, dialogue: List[Dict[str, Any]]) -> 'DialogueGraph':
        """Build graph from parsed dialogue JSON."""
        graph = cls()

        # Phase 1: Create all nodes
        for line in dialogue:
            conditions = [StateCondition.from_ctda(c) for c in line.get('conditions', [])]
            node = DialogueNode(
                id=line.get('form_id', ''),
                text=line.get('text', ''),
                speaker=line.get('speaker'),
                emotion=line.get('emotion', 'neutral'),
                topic=line.get('topic', ''),
                quest=line.get('quest'),
                conditions=conditions
            )
            graph.add_node(node)

        # Phase 2: Build edges within topics
        graph._build_topic_edges()

        # Phase 3: Build cross-topic edges (greeting → specific topics)
        graph._build_topic_branch_edges()

        # Phase 4: Build state-gated edges (same content, different conditions)
        graph._build_condition_edges()

        return graph

    def _build_topic_edges(self):
        """Build sequential edges within each topic based on quest stage."""
        for topic, node_ids in self._by_topic.items():
            if len(node_ids) < 2:
                continue

            # Sort by quest stage conditions
            def stage_key(nid: str) -> float:
                node = self.nodes[nid]
                for cond in node.conditions:
                    if cond.function == 'GetStage':
                        return cond.value
                return 0

            sorted_ids = sorted(node_ids, key=stage_key)

            # Add sequential edges
            for i in range(len(sorted_ids) - 1):
                # Check if transition has stage gate
                source = self.nodes[sorted_ids[i]]
                target = self.nodes[sorted_ids[i + 1]]

                # Find conditions that differ (the "gate" to the next line)
                gate_conditions = [c for c in target.conditions
                                  if c.function == 'GetStage']

                self.add_edge(DialogueEdge(
                    source=sorted_ids[i],
                    target=sorted_ids[i + 1],
                    edge_type='sequential',
                    conditions=gate_conditions
                ))

    def _build_topic_branch_edges(self):
        """Build edges from GREETING topics to specific topics."""
        greeting_ids = self._by_topic.get('GREETING', [])

        for gid in greeting_ids:
            greeting = self.nodes[gid]
            if not greeting.quest:
                continue

            # Find other topics in same quest
            quest_node_ids = self._by_quest.get(greeting.quest, [])
            other_topics = set()
            for nid in quest_node_ids:
                node = self.nodes[nid]
                if node.topic != 'GREETING':
                    other_topics.add(node.topic)

            # Add branch edges to first node of each topic
            for topic in other_topics:
                topic_nodes = [nid for nid in self._by_topic[topic]
                              if self.nodes[nid].quest == greeting.quest]
                if topic_nodes:
                    self.add_edge(DialogueEdge(
                        source=gid,
                        target=topic_nodes[0],
                        edge_type='topic_branch',
                        weight=0.5  # Lower weight for exploration
                    ))

    def _build_condition_edges(self):
        """Build edges between nodes with overlapping but different conditions."""
        # Group by quest + topic
        groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for nid, node in self.nodes.items():
            key = (node.quest or '', node.topic)
            groups[key].append(nid)

        for (quest, topic), node_ids in groups.items():
            if len(node_ids) < 2:
                continue

            # Find nodes with different condition signatures
            by_signature: Dict[frozenset, List[str]] = defaultdict(list)
            for nid in node_ids:
                sig = self.nodes[nid].state_signature()
                by_signature[sig].append(nid)

            # If multiple signatures exist, create condition-gated edges
            if len(by_signature) > 1:
                sigs = list(by_signature.keys())
                for i, sig1 in enumerate(sigs):
                    for sig2 in sigs[i+1:]:
                        # Add bidirectional edges (cycle allowed)
                        for nid1 in by_signature[sig1][:1]:  # Just first of each
                            for nid2 in by_signature[sig2][:1]:
                                self.add_edge(DialogueEdge(
                                    source=nid1, target=nid2,
                                    edge_type='condition_gate',
                                    weight=0.3
                                ))
                                self.add_edge(DialogueEdge(
                                    source=nid2, target=nid1,
                                    edge_type='condition_gate',
                                    weight=0.3
                                ))

    # =========================================================================
    # Graph Analysis
    # =========================================================================

    def find_cycles(self, max_length: int = 10) -> List[List[str]]:
        """Find cycles in the graph using DFS."""
        cycles = []
        visited = set()

        def dfs(node: str, path: List[str], path_set: Set[str]):
            if len(path) > max_length:
                return

            for neighbor, edge in self.neighbors(node):
                if neighbor in path_set:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if len(cycle) >= 3:  # Minimum interesting cycle
                        cycles.append(cycle)
                elif neighbor not in visited:
                    path.append(neighbor)
                    path_set.add(neighbor)
                    dfs(neighbor, path, path_set)
                    path.pop()
                    path_set.remove(neighbor)

            visited.add(node)

        for node_id in list(self.nodes.keys())[:1000]:  # Limit for performance
            if node_id not in visited:
                dfs(node_id, [node_id], {node_id})

        return cycles

    def find_hubs(self, min_degree: int = 5) -> List[Tuple[str, int, int]]:
        """Find hub nodes with high connectivity."""
        hubs = []
        for nid, node in self.nodes.items():
            in_deg = len(node.incoming)
            out_deg = len(node.outgoing)
            total = in_deg + out_deg
            if total >= min_degree:
                hubs.append((nid, in_deg, out_deg))

        return sorted(hubs, key=lambda x: x[1] + x[2], reverse=True)

    def find_bottlenecks(self) -> List[str]:
        """Find nodes that many paths must pass through."""
        # Simple heuristic: high in-degree AND high out-degree
        bottlenecks = []
        for nid, node in self.nodes.items():
            if len(node.incoming) >= 3 and len(node.outgoing) >= 3:
                bottlenecks.append(nid)
        return bottlenecks

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        in_degrees = [len(n.incoming) for n in self.nodes.values()]
        out_degrees = [len(n.outgoing) for n in self.nodes.values()]

        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'topics': len(self._by_topic),
            'quests': len(self._by_quest),
            'speakers': len(self._by_speaker),
            'edge_types': {
                'sequential': len([e for e in self.edges if e.edge_type == 'sequential']),
                'topic_branch': len([e for e in self.edges if e.edge_type == 'topic_branch']),
                'condition_gate': len([e for e in self.edges if e.edge_type == 'condition_gate']),
            },
            'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'isolated_nodes': len([n for n in self.nodes.values()
                                  if not n.incoming and not n.outgoing]),
        }

    # =========================================================================
    # NetworkX Analysis (requires networkx)
    # =========================================================================

    def _ensure_networkx(self):
        """Lazily convert to NetworkX graph."""
        if not hasattr(self, '_nx_graph'):
            self._nx_graph = self.to_networkx()
        return self._nx_graph

    def pagerank(self, top_n: int = 20) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Compute PageRank to find "important" dialogue nodes.

        High PageRank = many paths lead here, or important paths lead here.
        These are often narrative bottlenecks or key conversation hubs.

        Returns list of (node_id, score, node_info) tuples.
        """
        import networkx as nx
        G = self._ensure_networkx()

        scores = nx.pagerank(G, alpha=0.85)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        results = []
        for node_id, score in ranked:
            node = self.nodes.get(node_id)
            if node:
                results.append((node_id, score, {
                    'text': node.text,
                    'speaker': node.speaker,
                    'emotion': node.emotion,
                    'topic': node.topic,
                    'quest': node.quest,
                    'in_degree': len(node.incoming),
                    'out_degree': len(node.outgoing),
                }))
        return results

    def communities(self, algorithm: str = 'louvain') -> List[Dict[str, Any]]:
        """
        Detect communities (clusters) of related dialogue.

        Algorithms:
        - 'louvain': Fast, good for large graphs (default)
        - 'label_propagation': Very fast, less accurate
        - 'greedy_modularity': Slower but deterministic

        Returns list of community dicts with members and stats.
        """
        import networkx as nx
        G = self._ensure_networkx().to_undirected()

        if algorithm == 'louvain':
            try:
                communities = nx.community.louvain_communities(G, seed=42)
            except:
                communities = nx.community.greedy_modularity_communities(G)
        elif algorithm == 'label_propagation':
            communities = list(nx.community.label_propagation_communities(G))
        else:
            communities = nx.community.greedy_modularity_communities(G)

        results = []
        for i, community in enumerate(sorted(communities, key=len, reverse=True)[:20]):
            # Get community characteristics
            members = list(community)[:10]  # Sample members
            emotions = {}
            quests = {}
            for nid in community:
                node = self.nodes.get(nid)
                if node:
                    emotions[node.emotion] = emotions.get(node.emotion, 0) + 1
                    if node.quest:
                        quests[node.quest] = quests.get(node.quest, 0) + 1

            # Dominant emotion and quest
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'unknown'
            dominant_quest = max(quests.items(), key=lambda x: x[1])[0] if quests else None

            results.append({
                'id': i,
                'size': len(community),
                'dominant_emotion': dominant_emotion,
                'dominant_quest': dominant_quest,
                'emotion_distribution': emotions,
                'sample_members': [
                    {
                        'id': nid,
                        'text': self.nodes[nid].text[:100] if nid in self.nodes else '',
                        'emotion': self.nodes[nid].emotion if nid in self.nodes else '',
                    }
                    for nid in members if nid in self.nodes
                ]
            })

        return results

    def find_path(self, source: str, target: str, max_length: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path between two dialogue nodes.

        Returns list of node dicts along the path, or None if no path exists.
        """
        import networkx as nx
        G = self._ensure_networkx()

        if source not in G or target not in G:
            return None

        try:
            path = nx.shortest_path(G, source, target)
            if len(path) > max_length:
                return None

            return [
                {
                    'id': nid,
                    'text': self.nodes[nid].text if nid in self.nodes else '',
                    'speaker': self.nodes[nid].speaker if nid in self.nodes else None,
                    'emotion': self.nodes[nid].emotion if nid in self.nodes else '',
                    'topic': self.nodes[nid].topic if nid in self.nodes else '',
                }
                for nid in path
            ]
        except nx.NetworkXNoPath:
            return None

    def centrality_analysis(self, top_n: int = 10) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Compute multiple centrality measures to find important nodes.

        Returns dict with different centrality rankings:
        - degree: Most connected nodes
        - betweenness: Nodes on many shortest paths (bottlenecks)
        - closeness: Nodes close to all others (central)
        - eigenvector: Nodes connected to important nodes
        """
        import networkx as nx
        G = self._ensure_networkx()

        results = {}

        # Degree centrality
        degree = nx.degree_centrality(G)
        results['degree'] = [
            (nid, score, self.nodes[nid].text[:60] if nid in self.nodes else '')
            for nid, score in sorted(degree.items(), key=lambda x: -x[1])[:top_n]
        ]

        # Betweenness (sample for large graphs)
        if len(G) > 5000:
            betweenness = nx.betweenness_centrality(G, k=min(500, len(G)))
        else:
            betweenness = nx.betweenness_centrality(G)
        results['betweenness'] = [
            (nid, score, self.nodes[nid].text[:60] if nid in self.nodes else '')
            for nid, score in sorted(betweenness.items(), key=lambda x: -x[1])[:top_n]
        ]

        # Closeness
        closeness = nx.closeness_centrality(G)
        results['closeness'] = [
            (nid, score, self.nodes[nid].text[:60] if nid in self.nodes else '')
            for nid, score in sorted(closeness.items(), key=lambda x: -x[1])[:top_n]
        ]

        return results

    def strongly_connected_components(self) -> List[Dict[str, Any]]:
        """
        Find strongly connected components (dialogue cycles/loops).

        An SCC is a set of nodes where every node can reach every other node.
        These represent repeatable conversation structures.
        """
        import networkx as nx
        G = self._ensure_networkx()

        sccs = list(nx.strongly_connected_components(G))
        # Filter to non-trivial SCCs (size > 1)
        sccs = [scc for scc in sccs if len(scc) > 1]
        sccs = sorted(sccs, key=len, reverse=True)[:20]

        results = []
        for i, scc in enumerate(sccs):
            members = list(scc)[:5]
            quests = {}
            for nid in scc:
                node = self.nodes.get(nid)
                if node and node.quest:
                    quests[node.quest] = quests.get(node.quest, 0) + 1

            results.append({
                'id': i,
                'size': len(scc),
                'dominant_quest': max(quests.items(), key=lambda x: x[1])[0] if quests else None,
                'sample_nodes': [
                    {
                        'id': nid,
                        'text': self.nodes[nid].text[:80] if nid in self.nodes else '',
                        'topic': self.nodes[nid].topic if nid in self.nodes else '',
                    }
                    for nid in members if nid in self.nodes
                ]
            })

        return results

    # =========================================================================
    # Graph Sampling
    # =========================================================================

    def random_walk(self, start: Optional[str] = None,
                    max_steps: int = 20,
                    allow_cycles: bool = True,
                    state: Optional[Dict[str, float]] = None) -> List[DialogueNode]:
        """
        Perform a random walk through the dialogue graph.

        Args:
            start: Starting node ID (random if None)
            max_steps: Maximum walk length
            allow_cycles: Whether to allow revisiting nodes
            state: Simulated game state for condition checking

        Returns:
            List of visited DialogueNodes
        """
        if not self.nodes:
            return []

        # Pick start node
        if start is None:
            # Prefer nodes with outgoing edges
            candidates = [nid for nid, node in self.nodes.items() if node.outgoing]
            if not candidates:
                candidates = list(self.nodes.keys())
            start = random.choice(candidates)

        path = [self.nodes[start]]
        visited = {start} if not allow_cycles else set()
        current = start

        for _ in range(max_steps - 1):
            neighbors = self.neighbors(current)
            if not neighbors:
                break

            # Filter by visited if no cycles
            if not allow_cycles:
                neighbors = [(n, e) for n, e in neighbors if n not in visited]
                if not neighbors:
                    break

            # Weighted random selection
            weights = [e.weight for _, e in neighbors]
            total = sum(weights)
            if total == 0:
                break

            r = random.random() * total
            cumsum = 0
            selected = neighbors[0][0]
            for nid, edge in neighbors:
                cumsum += edge.weight
                if r <= cumsum:
                    selected = nid
                    break

            current = selected
            visited.add(current)
            path.append(self.nodes[current])

        return path

    def sample_subgraph(self, center: str, radius: int = 2) -> 'DialogueGraph':
        """Extract a subgraph around a center node."""
        subgraph = DialogueGraph()

        # BFS to find nodes within radius
        frontier = {center}
        visited = set()

        for _ in range(radius):
            next_frontier = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)

                if nid in self.nodes:
                    subgraph.add_node(self.nodes[nid])

                for neighbor, _ in self.neighbors(nid):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)

                for pred, _ in self.predecessors(nid):
                    if pred not in visited:
                        next_frontier.add(pred)

            frontier = next_frontier

        # Add edges between included nodes
        for edge in self.edges:
            if edge.source in subgraph.nodes and edge.target in subgraph.nodes:
                subgraph.add_edge(edge)

        return subgraph

    # =========================================================================
    # Export
    # =========================================================================

    def to_dot(self, max_nodes: int = 100) -> str:
        """Export to GraphViz DOT format."""
        lines = ['digraph dialogue {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')

        # Limit nodes
        node_ids = list(self.nodes.keys())[:max_nodes]
        node_set = set(node_ids)

        for nid in node_ids:
            node = self.nodes[nid]
            label = node.text[:40].replace('"', '\\"')
            emotion = f"\\n[{node.emotion}]" if node.emotion != 'neutral' else ''
            lines.append(f'  "{nid}" [label="{label}{emotion}"];')

        for edge in self.edges:
            if edge.source in node_set and edge.target in node_set:
                style = {
                    'sequential': '',
                    'topic_branch': 'style=dashed',
                    'condition_gate': 'style=dotted,color=red'
                }.get(edge.edge_type, '')
                lines.append(f'  "{edge.source}" -> "{edge.target}" [{style}];')

        lines.append('}')
        return '\n'.join(lines)

    def to_networkx(self):
        """Convert to NetworkX DiGraph (requires networkx)."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required: uv add networkx")

        G = nx.DiGraph()

        for nid, node in self.nodes.items():
            G.add_node(nid,
                      text=node.text,
                      speaker=node.speaker,
                      emotion=node.emotion,
                      topic=node.topic,
                      quest=node.quest)

        for edge in self.edges:
            G.add_edge(edge.source, edge.target,
                      edge_type=edge.edge_type,
                      weight=edge.weight)

        return G


def load_dialogue(path: Path) -> List[Dict[str, Any]]:
    """Load dialogue from JSON export."""
    with open(path) as f:
        data = json.load(f)
    return data.get('dialogue', [])


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Build and analyze dialogue graph'
    )
    parser.add_argument('input', type=Path,
                        help='Path to *_dialogue.json file')
    parser.add_argument('--stats', action='store_true',
                        help='Print graph statistics')
    parser.add_argument('--cycles', type=int, default=0,
                        help='Find cycles up to N length')
    parser.add_argument('--hubs', type=int, default=0,
                        help='Find hubs with degree >= N')
    parser.add_argument('--walk', type=int, default=0,
                        help='Perform N random walks')
    parser.add_argument('--walk-length', type=int, default=10,
                        help='Maximum walk length')
    parser.add_argument('--dot', type=Path,
                        help='Export to DOT file')
    parser.add_argument('--subgraph', type=str,
                        help='Extract subgraph around node ID')

    args = parser.parse_args()

    # Load and build graph
    print(f"Loading {args.input}...")
    dialogue = load_dialogue(args.input)
    print(f"Loaded {len(dialogue)} lines")

    print("Building graph...")
    graph = DialogueGraph.from_dialogue_data(dialogue)
    print(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    if args.stats:
        print(f"\n{'='*60}")
        print("Graph Statistics")
        print(f"{'='*60}")
        stats = graph.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

    if args.cycles > 0:
        print(f"\n{'='*60}")
        print(f"Cycles (max length {args.cycles})")
        print(f"{'='*60}")
        cycles = graph.find_cycles(args.cycles)
        print(f"Found {len(cycles)} cycles")
        for i, cycle in enumerate(cycles[:5]):
            print(f"\nCycle {i+1} ({len(cycle)} nodes):")
            for nid in cycle[:5]:
                node = graph.nodes.get(nid)
                if node:
                    print(f"  [{node.emotion}] {node.text[:60]}...")

    if args.hubs > 0:
        print(f"\n{'='*60}")
        print(f"Hub Nodes (degree >= {args.hubs})")
        print(f"{'='*60}")
        hubs = graph.find_hubs(args.hubs)
        for nid, in_deg, out_deg in hubs[:10]:
            node = graph.nodes[nid]
            print(f"\n[in={in_deg}, out={out_deg}] {node.topic}")
            print(f"  {node.text[:70]}...")

    if args.walk > 0:
        print(f"\n{'='*60}")
        print(f"Random Walks (n={args.walk}, max_length={args.walk_length})")
        print(f"{'='*60}")
        for i in range(args.walk):
            path = graph.random_walk(max_steps=args.walk_length, allow_cycles=False)
            print(f"\n--- Walk {i+1} ({len(path)} steps) ---")
            for node in path:
                emo = f"({node.emotion})" if node.emotion != 'neutral' else ''
                speaker = node.speaker or 'NPC'
                print(f"  {speaker}{emo}: {node.text[:60]}...")

    if args.dot:
        print(f"\nExporting to {args.dot}...")
        dot = graph.to_dot()
        args.dot.write_text(dot)
        print(f"Wrote {len(dot)} bytes")

    if args.subgraph:
        print(f"\n{'='*60}")
        print(f"Subgraph around {args.subgraph}")
        print(f"{'='*60}")
        sub = graph.sample_subgraph(args.subgraph, radius=2)
        print(f"Subgraph: {len(sub.nodes)} nodes, {len(sub.edges)} edges")
        for nid, node in sub.nodes.items():
            print(f"  [{node.topic}] {node.text[:50]}...")


if __name__ == '__main__':
    main()
