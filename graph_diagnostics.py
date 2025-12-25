#!/usr/bin/env python3
"""
Graph Diagnostics: Compare topology between reference and synthetic graphs.

Measures:
- Branching factor (in-degree, out-degree distributions)
- Hub detection (nodes with high connectivity)
- Vine detection (linear chains ending in leaves)
- Cycle presence
- Connected component structure
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import statistics


def load_graph(path: Path) -> tuple[dict, list]:
    """Load graph from JSON, return (nodes, edges)."""
    data = json.loads(path.read_text())

    # Handle both dict and list formats for nodes
    if isinstance(data.get('nodes'), dict):
        nodes = data['nodes']
    elif isinstance(data.get('nodes'), list):
        nodes = {n.get('id', str(i)): n for i, n in enumerate(data['nodes'])}
    else:
        nodes = {}

    edges = data.get('edges', [])
    return nodes, edges


def compute_degree_stats(nodes: dict, edges: list) -> dict:
    """Compute in-degree and out-degree statistics."""
    in_degree = Counter()
    out_degree = Counter()

    node_ids = set(nodes.keys()) if isinstance(nodes, dict) else set(range(len(nodes)))

    for edge in edges:
        src = edge.get('source') or edge.get('from')
        tgt = edge.get('target') or edge.get('to')
        if src:
            out_degree[src] += 1
        if tgt:
            in_degree[tgt] += 1

    # Include nodes with degree 0
    for nid in node_ids:
        if nid not in in_degree:
            in_degree[nid] = 0
        if nid not in out_degree:
            out_degree[nid] = 0

    in_vals = list(in_degree.values())
    out_vals = list(out_degree.values())

    return {
        'in_degree': {
            'mean': statistics.mean(in_vals) if in_vals else 0,
            'median': statistics.median(in_vals) if in_vals else 0,
            'max': max(in_vals) if in_vals else 0,
            'distribution': dict(Counter(in_vals).most_common(10)),
        },
        'out_degree': {
            'mean': statistics.mean(out_vals) if out_vals else 0,
            'median': statistics.median(out_vals) if out_vals else 0,
            'max': max(out_vals) if out_vals else 0,
            'distribution': dict(Counter(out_vals).most_common(10)),
        },
        'branching_factor': statistics.mean(out_vals) if out_vals else 0,
        'raw_in': in_degree,
        'raw_out': out_degree,
    }


def find_hubs(degree_stats: dict, threshold_percentile: float = 95) -> list:
    """Find hub nodes (high out-degree)."""
    out_deg = degree_stats['raw_out']
    if not out_deg:
        return []

    vals = sorted(out_deg.values())
    threshold = vals[int(len(vals) * threshold_percentile / 100)] if vals else 0

    hubs = [(nid, deg) for nid, deg in out_deg.items() if deg >= threshold and deg > 1]
    return sorted(hubs, key=lambda x: -x[1])[:20]


def find_leaves(degree_stats: dict) -> dict:
    """Find leaf nodes (out-degree 0) and entry points (in-degree 0)."""
    out_deg = degree_stats['raw_out']
    in_deg = degree_stats['raw_in']

    leaves = [nid for nid, deg in out_deg.items() if deg == 0]
    entry_points = [nid for nid, deg in in_deg.items() if deg == 0]

    return {
        'leaf_count': len(leaves),
        'entry_point_count': len(entry_points),
        'leaf_ratio': len(leaves) / len(out_deg) if out_deg else 0,
    }


def find_linear_chains(nodes: dict, edges: list) -> dict:
    """Detect linear chain segments (in-degree=1, out-degree=1)."""
    degree_stats = compute_degree_stats(nodes, edges)
    in_deg = degree_stats['raw_in']
    out_deg = degree_stats['raw_out']

    chain_nodes = [
        nid for nid in in_deg.keys()
        if in_deg.get(nid, 0) == 1 and out_deg.get(nid, 0) == 1
    ]

    return {
        'chain_node_count': len(chain_nodes),
        'chain_ratio': len(chain_nodes) / len(nodes) if nodes else 0,
    }


def detect_components(nodes: dict, edges: list) -> dict:
    """Find weakly connected components."""
    # Build adjacency (undirected)
    adj = defaultdict(set)
    node_ids = set(nodes.keys()) if isinstance(nodes, dict) else set(range(len(nodes)))

    for edge in edges:
        src = edge.get('source') or edge.get('from')
        tgt = edge.get('target') or edge.get('to')
        if src and tgt:
            adj[src].add(tgt)
            adj[tgt].add(src)

    # Add isolated nodes
    for nid in node_ids:
        if nid not in adj:
            adj[nid] = set()

    # BFS to find components
    visited = set()
    components = []

    for start in adj.keys():
        if start in visited:
            continue
        component = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adj[node] - visited)
        if component:
            components.append(component)

    sizes = sorted([len(c) for c in components], reverse=True)

    return {
        'component_count': len(components),
        'largest_component': sizes[0] if sizes else 0,
        'size_distribution': sizes[:10],
        'isolated_nodes': sizes.count(1),
    }


def analyze_graph(path: Path, name: str) -> dict:
    """Full analysis of a single graph."""
    nodes, edges = load_graph(path)

    degree_stats = compute_degree_stats(nodes, edges)
    hubs = find_hubs(degree_stats)
    leaves = find_leaves(degree_stats)
    chains = find_linear_chains(nodes, edges)
    components = detect_components(nodes, edges)

    return {
        'name': name,
        'node_count': len(nodes),
        'edge_count': len(edges),
        'edges_per_node': len(edges) / len(nodes) if nodes else 0,
        'branching_factor': degree_stats['branching_factor'],
        'in_degree': {k: v for k, v in degree_stats['in_degree'].items() if k != 'raw'},
        'out_degree': {k: v for k, v in degree_stats['out_degree'].items() if k != 'raw'},
        'hub_count': len(hubs),
        'top_hubs': hubs[:5],
        'leaves': leaves,
        'chains': chains,
        'components': components,
    }


def print_analysis(analysis: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"Graph: {analysis['name']}")
    print(f"{'='*60}")
    print(f"Nodes: {analysis['node_count']:,}")
    print(f"Edges: {analysis['edge_count']:,}")
    print(f"Edges/Node: {analysis['edges_per_node']:.3f}")
    print(f"Branching Factor (mean out-degree): {analysis['branching_factor']:.3f}")

    print(f"\nOut-degree distribution:")
    for deg, count in sorted(analysis['out_degree']['distribution'].items()):
        print(f"  {deg}: {count}")

    print(f"\nIn-degree distribution:")
    for deg, count in sorted(analysis['in_degree']['distribution'].items()):
        print(f"  {deg}: {count}")

    print(f"\nTopology:")
    print(f"  Hubs (high out-degree): {analysis['hub_count']}")
    print(f"  Leaves (out=0): {analysis['leaves']['leaf_count']} ({analysis['leaves']['leaf_ratio']:.1%})")
    print(f"  Entry points (in=0): {analysis['leaves']['entry_point_count']}")
    print(f"  Chain nodes (in=1,out=1): {analysis['chains']['chain_node_count']} ({analysis['chains']['chain_ratio']:.1%})")

    print(f"\nConnected Components:")
    print(f"  Count: {analysis['components']['component_count']}")
    print(f"  Largest: {analysis['components']['largest_component']}")
    print(f"  Isolated: {analysis['components']['isolated_nodes']}")
    if analysis['components']['size_distribution']:
        print(f"  Top sizes: {analysis['components']['size_distribution']}")


def main():
    parser = argparse.ArgumentParser(description='Compare graph topology')
    parser.add_argument('--reference', type=Path, help='Reference graph JSON')
    parser.add_argument('--synthetic', type=Path, help='Synthetic graph JSON')
    parser.add_argument('--all-refs', action='store_true', help='Analyze all reference graphs')
    args = parser.parse_args()

    results = []

    if args.all_refs:
        # Analyze all available reference graphs
        data_dir = Path('dialogue_data')
        for game in ['oblivion', 'falloutnv', 'skyrim']:
            graph_path = data_dir / f'{game}_dialogue_graph.json'
            if graph_path.exists():
                results.append(analyze_graph(graph_path, f'ref:{game}'))

    if args.reference:
        results.append(analyze_graph(args.reference, f'ref:{args.reference.stem}'))

    if args.synthetic:
        results.append(analyze_graph(args.synthetic, f'syn:{args.synthetic.stem}'))

    # Default: compare gallia_v3 to combined reference
    if not results:
        syn_path = Path('synthetic/gallia_v3/graph.json')
        if syn_path.exists():
            results.append(analyze_graph(syn_path, 'synthetic:gallia_v3'))

        # Try to find reference graphs
        data_dir = Path('dialogue_data')
        for game in ['oblivion', 'falloutnv']:
            graph_path = data_dir / f'{game}_dialogue_graph.json'
            if graph_path.exists():
                results.append(analyze_graph(graph_path, f'reference:{game}'))

    for r in results:
        print_analysis(r)

    # Comparison summary if we have both
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"{r['name']:30} BF={r['branching_factor']:.3f}  chains={r['chains']['chain_ratio']:.1%}  components={r['components']['component_count']}")


if __name__ == '__main__':
    main()
