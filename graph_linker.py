#!/usr/bin/env python3
"""
Graph Linker: Add cross-links to synthetic dialogue graphs.

Identifies candidate link points (chain termini → potential entry points),
generates bridge node requests, and applies completed bridges to the graph.

Usage:
    # Analyze and generate link candidates
    python graph_linker.py --setting gallia --version 3 --analyze

    # Generate N link requests
    python graph_linker.py --setting gallia --version 3 --sample 50

    # Check status of pending links
    python graph_linker.py --setting gallia --version 3 --status

    # Apply completed bridges
    python graph_linker.py --setting gallia --version 3 --apply
"""

import json
import random
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


RUNS_DIR = Path("runs")
SYNTHETIC_DIR = Path("synthetic")
BIBLES_DIR = Path("bibles")


@dataclass
class LinkCandidate:
    """A potential link between two nodes."""
    terminus_id: str          # End of source chain
    terminus_text: str
    terminus_emotion: str
    terminus_context: List[dict]  # Preceding nodes for context

    entry_id: str             # Start of target chain
    entry_text: str
    entry_emotion: str
    entry_context: List[dict]     # Following nodes for context

    compatibility_score: float    # How well emotions align

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LinkState:
    """Tracks linking job state."""
    run_id: str
    setting: str
    version: int
    created_at: str

    total_candidates: int = 0
    sampled_count: int = 0
    completed_count: int = 0
    applied_count: int = 0

    def path(self) -> Path:
        return RUNS_DIR / self.run_id

    def save(self):
        self.path().mkdir(parents=True, exist_ok=True)
        state_file = self.path() / "link_state.json"
        state_file.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, run_id: str) -> Optional['LinkState']:
        state_file = RUNS_DIR / run_id / "link_state.json"
        if not state_file.exists():
            return None
        data = json.loads(state_file.read_text())
        return cls(**data)

    @classmethod
    def find_latest(cls, setting: str, version: int) -> Optional['LinkState']:
        pattern = f"link_*_{setting}_v{version}"
        candidates = sorted(RUNS_DIR.glob(pattern), reverse=True)
        for run_dir in candidates:
            state = cls.load(run_dir.name)
            if state:
                return state
        return None


def load_synthetic_graph(setting: str, version: int) -> Tuple[List[dict], List[dict]]:
    """Load synthetic graph nodes and edges."""
    graph_path = SYNTHETIC_DIR / f"{setting}_v{version}" / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph at {graph_path}")

    data = json.loads(graph_path.read_text())
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    return nodes, edges


def load_bible_excerpt(setting: str, max_tokens: int = 2000) -> str:
    """Load lore bible for the setting."""
    bible_path = BIBLES_DIR / f"{setting}.yaml"
    if not bible_path.exists():
        bible_path = BIBLES_DIR / f"{setting}.md"
    if not bible_path.exists():
        return f"# {setting.title()} Setting\n\nNo bible found."

    content = bible_path.read_text()
    # Rough truncation to stay within token budget
    if len(content) > max_tokens * 4:
        content = content[:max_tokens * 4]
    return content


def build_graph_index(nodes: List[dict], edges: List[dict]) -> dict:
    """Build index structures for graph analysis."""
    node_by_id = {n['id']: n for n in nodes}

    out_edges = defaultdict(list)  # node_id -> [target_ids]
    in_edges = defaultdict(list)   # node_id -> [source_ids]

    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src and tgt:
            out_edges[src].append(tgt)
            in_edges[tgt].append(src)

    return {
        'node_by_id': node_by_id,
        'out_edges': out_edges,
        'in_edges': in_edges,
    }


def find_chain_termini(index: dict) -> List[str]:
    """Find nodes with out_degree=0 (chain endpoints)."""
    node_by_id = index['node_by_id']
    out_edges = index['out_edges']

    termini = []
    for node_id in node_by_id:
        if len(out_edges.get(node_id, [])) == 0:
            termini.append(node_id)

    return termini


def find_potential_entries(index: dict) -> List[str]:
    """Find nodes that could serve as link targets (low in_degree, not termini)."""
    node_by_id = index['node_by_id']
    in_edges = index['in_edges']
    out_edges = index['out_edges']

    entries = []
    for node_id in node_by_id:
        in_deg = len(in_edges.get(node_id, []))
        out_deg = len(out_edges.get(node_id, []))
        # Good entry points: have outgoing edges (not leaves) and low incoming
        if out_deg > 0 and in_deg <= 2:
            entries.append(node_id)

    return entries


def get_node_context(node_id: str, index: dict, direction: str, depth: int = 3) -> List[dict]:
    """Get context nodes before (direction='back') or after (direction='forward') a node."""
    node_by_id = index['node_by_id']
    in_edges = index['in_edges']
    out_edges = index['out_edges']

    context = []
    current = node_id

    for _ in range(depth):
        if direction == 'back':
            sources = in_edges.get(current, [])
            if not sources:
                break
            current = sources[0]  # Follow first incoming edge
        else:  # forward
            targets = out_edges.get(current, [])
            if not targets:
                break
            current = targets[0]  # Follow first outgoing edge

        if current in node_by_id:
            node = node_by_id[current]
            context.append({
                'id': current,
                'text': node.get('text', ''),
                'emotion': node.get('emotion', 'neutral'),
            })

    if direction == 'back':
        context.reverse()  # Put earliest first

    return context


# Emotion compatibility matrix (simplified)
EMOTION_COMPATIBILITY = {
    ('neutral', 'neutral'): 1.0,
    ('neutral', 'happy'): 0.8,
    ('neutral', 'sad'): 0.7,
    ('neutral', 'anger'): 0.6,
    ('neutral', 'fear'): 0.6,
    ('neutral', 'surprise'): 0.7,
    ('neutral', 'disgust'): 0.5,
    ('happy', 'happy'): 1.0,
    ('happy', 'neutral'): 0.8,
    ('happy', 'surprise'): 0.7,
    ('sad', 'sad'): 1.0,
    ('sad', 'neutral'): 0.7,
    ('anger', 'anger'): 1.0,
    ('anger', 'neutral'): 0.6,
    ('anger', 'disgust'): 0.7,
    ('fear', 'fear'): 1.0,
    ('fear', 'neutral'): 0.6,
    ('fear', 'surprise'): 0.6,
    ('surprise', 'surprise'): 1.0,
    ('surprise', 'neutral'): 0.7,
    ('disgust', 'disgust'): 1.0,
    ('disgust', 'anger'): 0.7,
    ('disgust', 'neutral'): 0.5,
}


def get_emotion_compatibility(emo1: str, emo2: str) -> float:
    """Get compatibility score for emotion transition."""
    key = (emo1.lower(), emo2.lower())
    if key in EMOTION_COMPATIBILITY:
        return EMOTION_COMPATIBILITY[key]
    # Try reverse
    key_rev = (emo2.lower(), emo1.lower())
    if key_rev in EMOTION_COMPATIBILITY:
        return EMOTION_COMPATIBILITY[key_rev] * 0.9  # Slight penalty for reverse
    return 0.4  # Default low compatibility


def generate_link_candidates(
    nodes: List[dict],
    edges: List[dict],
    max_candidates: int = 500
) -> List[LinkCandidate]:
    """Generate candidate links between chain termini and potential entries."""
    index = build_graph_index(nodes, edges)
    node_by_id = index['node_by_id']

    termini = find_chain_termini(index)
    entries = find_potential_entries(index)

    print(f"Found {len(termini)} chain termini, {len(entries)} potential entry points")

    # Shuffle for variety
    random.shuffle(termini)
    random.shuffle(entries)

    candidates = []

    # Sample terminus-entry pairs
    for terminus_id in termini:
        terminus = node_by_id.get(terminus_id, {})
        terminus_emo = terminus.get('emotion', 'neutral')
        terminus_text = terminus.get('text', '')

        if not terminus_text:
            continue

        terminus_context = get_node_context(terminus_id, index, 'back', depth=3)

        # Find compatible entries
        for entry_id in entries:
            if entry_id == terminus_id:
                continue

            entry = node_by_id.get(entry_id, {})
            entry_emo = entry.get('emotion', 'neutral')
            entry_text = entry.get('text', '')

            if not entry_text:
                continue

            # Skip same-chain check for now - rely on emotion compatibility
            # (the graph is becoming more connected, making context checks too broad)

            compatibility = get_emotion_compatibility(terminus_emo, entry_emo)

            if compatibility >= 0.5:  # Threshold for viability
                entry_context = get_node_context(entry_id, index, 'forward', depth=3)

                candidates.append(LinkCandidate(
                    terminus_id=terminus_id,
                    terminus_text=terminus_text,
                    terminus_emotion=terminus_emo,
                    terminus_context=terminus_context,
                    entry_id=entry_id,
                    entry_text=entry_text,
                    entry_emotion=entry_emo,
                    entry_context=entry_context,
                    compatibility_score=compatibility,
                ))

                # Sample multiple compatible entries per terminus for variety
                if len([c for c in candidates if c.terminus_id == terminus_id]) >= 3:
                    break  # Move to next terminus after finding a few options

        # Early exit if we've processed enough termini (check at terminus level)
        if len(candidates) >= max_candidates * 2:
            break

    # Sort by compatibility and sample
    candidates.sort(key=lambda c: -c.compatibility_score)

    # Deduplicate: don't link same terminus twice, but entries CAN be reused (become hubs)
    seen_termini = set()
    entry_usage = defaultdict(int)  # Track how many times each entry is used
    max_entry_reuse = 20  # Allow entries to become larger hubs
    filtered = []

    for c in candidates:
        if c.terminus_id in seen_termini:
            continue
        if entry_usage[c.entry_id] >= max_entry_reuse:
            continue

        filtered.append(c)
        seen_termini.add(c.terminus_id)
        entry_usage[c.entry_id] += 1

        if len(filtered) >= max_candidates:
            break

    return filtered


def create_link_request(
    candidate: LinkCandidate,
    index: int,
    bible_excerpt: str,
    output_dir: Path
) -> Path:
    """Create a request file for bridge generation."""
    request = {
        'index': index,
        'link': candidate.to_dict(),
        'bible_excerpt': bible_excerpt,
        'status': 'pending',
    }

    request_path = output_dir / f"link_{index:04d}.json"
    request_path.write_text(json.dumps(request, indent=2))
    return request_path


def sample_and_save_requests(
    setting: str,
    version: int,
    sample_size: int
) -> LinkState:
    """Sample link candidates and save as request files."""
    nodes, edges = load_synthetic_graph(setting, version)
    bible = load_bible_excerpt(setting)

    print(f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges")

    candidates = generate_link_candidates(nodes, edges, max_candidates=sample_size)
    print(f"Generated {len(candidates)} link candidates")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"link_{timestamp}_{setting}_v{version}"

    state = LinkState(
        run_id=run_id,
        setting=setting,
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_candidates=len(candidates),
        sampled_count=len(candidates),
    )

    requests_dir = state.path() / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)

    for i, candidate in enumerate(candidates):
        create_link_request(candidate, i, bible, requests_dir)

    state.save()

    print(f"\nCreated linking run: {run_id}")
    print(f"  {len(candidates)} link requests saved to: {requests_dir}")

    return state


def get_link_status(state: LinkState) -> dict:
    """Get current status of link requests."""
    requests_dir = state.path() / "requests"

    pending = 0
    completed = 0
    applied = 0

    for request_file in requests_dir.glob("link_*.json"):
        request = json.loads(request_file.read_text())
        status = request.get('status', 'pending')

        if status == 'pending':
            pending += 1
        elif status == 'completed':
            if request.get('applied'):
                applied += 1
            else:
                completed += 1

    return {
        'run_id': state.run_id,
        'pending': pending,
        'completed_not_applied': completed,
        'applied': applied,
        'total': pending + completed + applied,
    }


def apply_completed_bridges(state: LinkState) -> dict:
    """Apply completed bridge nodes to the graph."""
    nodes, edges = load_synthetic_graph(state.setting, state.version)
    requests_dir = state.path() / "requests"

    applied = 0
    bridges_added = 0
    edges_added = 0

    for request_file in sorted(requests_dir.glob("link_*.json")):
        request = json.loads(request_file.read_text())

        if request.get('status') != 'completed':
            continue
        if request.get('applied'):
            continue

        # Get bridge data (handle both flat and nested formats)
        bridge_text = request.get('bridge_text')
        bridge_emotion = request.get('bridge_emotion', 'neutral')

        if not bridge_text and 'bridge' in request:
            bridge_text = request['bridge'].get('text')
            bridge_emotion = request['bridge'].get('emotion', 'neutral')

        if not bridge_text:
            continue

        link = request['link']
        terminus_id = link['terminus_id']
        entry_id = link['entry_id']

        # Create bridge node
        bridge_id = f"bridge_{hashlib.sha256(f'{terminus_id}_{entry_id}'.encode()).hexdigest()[:12]}"

        bridge_node = {
            'id': bridge_id,
            'text': bridge_text,
            'emotion': bridge_emotion,
            'is_bridge': True,
            'source_terminus': terminus_id,
            'target_entry': entry_id,
        }

        nodes.append(bridge_node)
        bridges_added += 1

        # Add edges: terminus -> bridge -> entry
        edges.append({
            'source': terminus_id,
            'target': bridge_id,
            'type': 'bridge_out',
        })
        edges.append({
            'source': bridge_id,
            'target': entry_id,
            'type': 'bridge_in',
        })
        edges_added += 2

        # Mark as applied
        request['applied'] = True
        request_file.write_text(json.dumps(request, indent=2))
        applied += 1

    # Save updated graph
    if applied > 0:
        graph_path = SYNTHETIC_DIR / f"{state.setting}_v{state.version}" / "graph.json"
        graph_data = {'nodes': nodes, 'edges': edges}
        graph_path.write_text(json.dumps(graph_data, indent=2))

        state.applied_count += applied
        state.save()

    return {
        'applied': applied,
        'bridges_added': bridges_added,
        'edges_added': edges_added,
        'total_nodes': len(nodes),
        'total_edges': len(edges),
    }


def analyze_graph(setting: str, version: int):
    """Print analysis of current graph structure."""
    nodes, edges = load_synthetic_graph(setting, version)
    index = build_graph_index(nodes, edges)

    termini = find_chain_termini(index)
    entries = find_potential_entries(index)

    # Count bridges
    bridges = [n for n in nodes if n.get('is_bridge')]

    print(f"\nGraph Analysis: {setting}_v{version}")
    print(f"{'='*50}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Edges/Node: {len(edges)/len(nodes):.3f}")
    print(f"\nChain termini (leaves): {len(termini)}")
    print(f"Potential entry points: {len(entries)}")
    print(f"Bridge nodes: {len(bridges)}")

    # Sample some candidates
    candidates = generate_link_candidates(nodes, edges, max_candidates=10)
    print(f"\nSample link candidates (top 5 by compatibility):")
    for c in candidates[:5]:
        print(f"  [{c.terminus_emotion}] \"{c.terminus_text[:40]}...\"")
        print(f"    -> [{c.entry_emotion}] \"{c.entry_text[:40]}...\" (compat={c.compatibility_score:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Graph linker for synthetic dialogue')
    parser.add_argument('--setting', type=str, default='gallia', help='Setting name')
    parser.add_argument('--version', type=int, default=3, help='Graph version')
    parser.add_argument('--analyze', action='store_true', help='Analyze graph structure')
    parser.add_argument('--sample', type=int, help='Generate N link requests')
    parser.add_argument('--status', action='store_true', help='Check link job status')
    parser.add_argument('--apply', action='store_true', help='Apply completed bridges')
    args = parser.parse_args()

    if args.analyze:
        analyze_graph(args.setting, args.version)
        return

    if args.sample:
        state = sample_and_save_requests(args.setting, args.version, args.sample)
        return

    # Find existing run
    state = LinkState.find_latest(args.setting, args.version)

    if args.status:
        if not state:
            print(f"No linking run found for {args.setting}_v{args.version}")
            return
        status = get_link_status(state)
        print(f"Link status: {args.setting}_v{args.version}")
        print(f"  run_id: {status['run_id']}")
        print(f"  pending: {status['pending']}")
        print(f"  completed_not_applied: {status['completed_not_applied']}")
        print(f"  applied: {status['applied']}")
        print(f"  total: {status['total']}")
        return

    if args.apply:
        if not state:
            print(f"No linking run found for {args.setting}_v{args.version}")
            return
        result = apply_completed_bridges(state)
        print(f"Applied {result['applied']} bridges")
        print(f"  Bridges added: {result['bridges_added']}")
        print(f"  Edges added: {result['edges_added']}")
        print(f"  Total nodes: {result['total_nodes']}")
        print(f"  Total edges: {result['total_edges']}")
        return

    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()
