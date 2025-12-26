#!/usr/bin/env python3
"""
Compare graph TOPOLOGY between reference games and synthetic corpora.

This compares the actual graph structure - the heterogeneous local statistics
that make narrative graphs interesting:

- Degree distributions (not averages - the full distribution)
- Hub/spur/bridge node ratios
- Branching factor at each node
- Loop participation (nodes in cycles vs linear spurs)
- Walk statistics (typical lengths, branching rates along walks)

The insight: real narrative graphs are NOT uniform. Some nodes are hubs,
some are dead ends, some are in loops. The DISTRIBUTION of these patterns
matters, not just the mean.

Usage:
    python scripts/compare_topology.py
    python scripts/compare_topology.py --json
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import httpx

API_BASE = "http://127.0.0.1:8000"


class NodeStats(NamedTuple):
    """Local statistics for a single node."""
    in_degree: int
    out_degree: int
    is_hub: bool      # out_degree > 2
    is_spur: bool     # out_degree == 0 (dead end)
    is_source: bool   # in_degree == 0 (entry point)
    in_cycle: bool    # participates in a loop


class GraphTopology(NamedTuple):
    """Topology statistics for a graph."""
    node_count: int
    edge_count: int

    # Degree distributions (Counter: degree -> count)
    in_degree_dist: Counter
    out_degree_dist: Counter

    # Node type ratios
    hub_ratio: float       # fraction with out_degree > 2
    spur_ratio: float      # fraction with out_degree == 0
    source_ratio: float    # fraction with in_degree == 0
    cycle_ratio: float     # fraction participating in cycles

    # Branching statistics
    mean_out_degree: float
    median_out_degree: int
    max_out_degree: int
    branching_entropy: float  # entropy of out-degree distribution

    # Connectivity
    component_count: int   # number of weakly connected components
    largest_component_ratio: float  # fraction of nodes in largest component


def compute_topology_from_api(game: str) -> GraphTopology | None:
    """Compute topology stats from reference game via API."""
    try:
        # Get basic stats
        stats_resp = httpx.get(f"{API_BASE}/api/stats/{game}")
        stats = stats_resp.json()

        # Get centrality data (has degree info)
        centrality_resp = httpx.get(f"{API_BASE}/api/centrality/{game}?top_n=10000")
        centrality = centrality_resp.json()

        # Get component info
        components_resp = httpx.get(f"{API_BASE}/api/components/{game}")
        components = components_resp.json()

        node_count = stats.get("nodes", 0)
        edge_count = stats.get("edges", 0)

        # Extract degree distribution from centrality data
        # The API returns top nodes by degree - we need full distribution
        # For now, approximate from what we have
        degree_data = centrality.get("degree", [])

        in_degree_dist = Counter()
        out_degree_dist = Counter()

        # We don't have full degree data from API, so estimate from edges/nodes
        avg_degree = edge_count / node_count if node_count > 0 else 0

        # Get SCC info for cycle detection
        sccs = components.get("strongly_connected_components", [])
        nodes_in_cycles = sum(len(scc.get("nodes", [])) for scc in sccs if len(scc.get("nodes", [])) > 1)
        cycle_ratio = nodes_in_cycles / node_count if node_count > 0 else 0

        # Component info
        wcc_count = components.get("weakly_connected_count", 1)
        largest_wcc = components.get("largest_weakly_connected_size", node_count)
        largest_component_ratio = largest_wcc / node_count if node_count > 0 else 1.0

        return GraphTopology(
            node_count=node_count,
            edge_count=edge_count,
            in_degree_dist=in_degree_dist,  # Empty - API doesn't expose full distribution
            out_degree_dist=out_degree_dist,
            hub_ratio=0.0,  # Can't compute without full data
            spur_ratio=0.0,
            source_ratio=0.0,
            cycle_ratio=cycle_ratio,
            mean_out_degree=avg_degree,
            median_out_degree=int(avg_degree),
            max_out_degree=max((d.get("degree", 0) for d in degree_data), default=0),
            branching_entropy=0.0,
            component_count=wcc_count,
            largest_component_ratio=largest_component_ratio,
        )
    except Exception as e:
        print(f"Error fetching {game}: {e}")
        return None


def compute_topology_from_graph(graph_path: Path) -> GraphTopology | None:
    """Compute full topology stats from a graph.json file."""
    if not graph_path.exists():
        return None

    data = json.loads(graph_path.read_text())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    node_ids = {n["id"] for n in nodes}
    node_count = len(nodes)
    edge_count = len(edges)

    if node_count == 0:
        return None

    # Compute degrees
    in_degree = Counter()
    out_degree = Counter()

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in node_ids:
            out_degree[src] += 1
        if tgt in node_ids:
            in_degree[tgt] += 1

    # Initialize nodes with 0 degree if they have no edges
    for nid in node_ids:
        if nid not in in_degree:
            in_degree[nid] = 0
        if nid not in out_degree:
            out_degree[nid] = 0

    # Degree distributions
    in_degree_dist = Counter(in_degree.values())
    out_degree_dist = Counter(out_degree.values())

    # Node type classification
    hubs = sum(1 for d in out_degree.values() if d > 2)
    spurs = sum(1 for d in out_degree.values() if d == 0)
    sources = sum(1 for d in in_degree.values() if d == 0)

    # Cycle detection via Tarjan's SCC
    nodes_in_cycles = find_nodes_in_cycles(node_ids, edges)

    # Connected components (simple BFS)
    components = find_connected_components(node_ids, edges)
    largest_component = max(len(c) for c in components) if components else 0

    # Branching statistics
    out_degrees = list(out_degree.values())
    mean_out = sum(out_degrees) / len(out_degrees)
    sorted_out = sorted(out_degrees)
    median_out = sorted_out[len(sorted_out) // 2]
    max_out = max(out_degrees)

    # Entropy of out-degree distribution (measures heterogeneity)
    total = sum(out_degree_dist.values())
    entropy = 0.0
    for count in out_degree_dist.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return GraphTopology(
        node_count=node_count,
        edge_count=edge_count,
        in_degree_dist=in_degree_dist,
        out_degree_dist=out_degree_dist,
        hub_ratio=hubs / node_count,
        spur_ratio=spurs / node_count,
        source_ratio=sources / node_count,
        cycle_ratio=len(nodes_in_cycles) / node_count,
        mean_out_degree=mean_out,
        median_out_degree=median_out,
        max_out_degree=max_out,
        branching_entropy=entropy,
        component_count=len(components),
        largest_component_ratio=largest_component / node_count,
    )


def find_nodes_in_cycles(node_ids: set, edges: list) -> set:
    """Find all nodes that participate in cycles (SCCs of size > 1)."""
    # Build adjacency list
    adj = defaultdict(list)
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        if src in node_ids and tgt in node_ids:
            adj[src].append(tgt)

    # Tarjan's algorithm for SCCs
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []

    def strongconnect(node):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in adj.get(node, []):
            if neighbor not in index:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack.get(neighbor, False):
                lowlinks[node] = min(lowlinks[node], index[neighbor])

        if lowlinks[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)

    for node in node_ids:
        if node not in index:
            strongconnect(node)

    # Nodes in cycles are in SCCs of size > 1
    cycle_nodes = set()
    for scc in sccs:
        if len(scc) > 1:
            cycle_nodes.update(scc)

    return cycle_nodes


def find_connected_components(node_ids: set, edges: list) -> list[set]:
    """Find weakly connected components."""
    # Build undirected adjacency
    adj = defaultdict(set)
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        if src in node_ids and tgt in node_ids:
            adj[src].add(tgt)
            adj[tgt].add(src)

    visited = set()
    components = []

    for start in node_ids:
        if start in visited:
            continue

        component = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adj.get(node, set()) - visited)

        components.append(component)

    return components


def distribution_divergence(p: Counter, q: Counter) -> dict:
    """Compare two degree distributions."""
    # Normalize to probabilities
    p_total = sum(p.values())
    q_total = sum(q.values())

    if p_total == 0 or q_total == 0:
        return {"js_divergence": 1.0, "tv_distance": 1.0, "shared_degrees": 0}

    all_degrees = set(p.keys()) | set(q.keys())

    p_prob = {k: p.get(k, 0) / p_total for k in all_degrees}
    q_prob = {k: q.get(k, 0) / q_total for k in all_degrees}

    # Jensen-Shannon divergence
    epsilon = 1e-10
    m_prob = {k: (p_prob[k] + q_prob[k]) / 2 for k in all_degrees}

    kl_pm = sum(p_prob[k] * math.log((p_prob[k] + epsilon) / (m_prob[k] + epsilon))
                for k in all_degrees if p_prob[k] > 0)
    kl_qm = sum(q_prob[k] * math.log((q_prob[k] + epsilon) / (m_prob[k] + epsilon))
                for k in all_degrees if q_prob[k] > 0)
    js = (kl_pm + kl_qm) / 2

    # Total variation
    tv = sum(abs(p_prob[k] - q_prob[k]) for k in all_degrees) / 2

    # Shared degrees
    shared = len(set(p.keys()) & set(q.keys()))

    return {
        "js_divergence": js,
        "tv_distance": tv,
        "shared_degrees": shared,
        "p_unique_degrees": len(p),
        "q_unique_degrees": len(q),
    }


def print_degree_histogram(dist: Counter, label: str, max_width: int = 40):
    """Print a simple ASCII histogram of degree distribution."""
    if not dist:
        print(f"  {label}: (no data)")
        return

    total = sum(dist.values())
    max_count = max(dist.values())

    print(f"  {label}:")
    for degree in sorted(dist.keys())[:15]:  # Show first 15 degrees
        count = dist[degree]
        pct = count / total * 100
        bar_len = int(count / max_count * max_width)
        bar = "█" * bar_len
        print(f"    {degree:3d}: {bar} ({pct:5.1f}%, n={count})")

    if len(dist) > 15:
        print(f"    ... ({len(dist) - 15} more degree values)")


def main():
    parser = argparse.ArgumentParser(description="Compare graph topology")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    synthetic_dir = Path("synthetic")

    # Reference games
    references = ["oblivion", "falloutnv"]

    # Synthetic versions
    synthetics = [
        ("gallia", 3),
        ("gallia", 4),
        ("gallia", 5),
        ("marmotte", 1),
        ("marmotte", 2),
        ("marmotte", 3),
    ]

    # Load reference topology (from API - limited data)
    ref_topology = {}
    for game in references:
        topo = compute_topology_from_api(game)
        if topo:
            ref_topology[game] = topo

    # Load synthetic topology (from files - full data)
    syn_topology = {}
    for setting, version in synthetics:
        graph_path = synthetic_dir / f"{setting}_v{version}" / "graph.json"
        topo = compute_topology_from_graph(graph_path)
        if topo:
            syn_topology[f"{setting}_v{version}"] = topo

    if args.json:
        output = {
            "references": {
                name: {
                    "nodes": t.node_count,
                    "edges": t.edge_count,
                    "mean_out_degree": t.mean_out_degree,
                    "cycle_ratio": t.cycle_ratio,
                    "component_count": t.component_count,
                }
                for name, t in ref_topology.items()
            },
            "synthetics": {
                name: {
                    "nodes": t.node_count,
                    "edges": t.edge_count,
                    "hub_ratio": t.hub_ratio,
                    "spur_ratio": t.spur_ratio,
                    "source_ratio": t.source_ratio,
                    "cycle_ratio": t.cycle_ratio,
                    "mean_out_degree": t.mean_out_degree,
                    "median_out_degree": t.median_out_degree,
                    "max_out_degree": t.max_out_degree,
                    "branching_entropy": t.branching_entropy,
                    "component_count": t.component_count,
                    "largest_component_ratio": t.largest_component_ratio,
                    "out_degree_distribution": dict(t.out_degree_dist),
                    "in_degree_distribution": dict(t.in_degree_dist),
                }
                for name, t in syn_topology.items()
            },
        }
        print(json.dumps(output, indent=2))
        return

    # Pretty print
    print("=" * 80)
    print("REFERENCE CORPUS TOPOLOGY")
    print("=" * 80)

    for name, t in ref_topology.items():
        print(f"\n{name}:")
        print(f"  Nodes: {t.node_count:,}")
        print(f"  Edges: {t.edge_count:,}")
        print(f"  Mean out-degree: {t.mean_out_degree:.2f}")
        print(f"  Nodes in cycles: {t.cycle_ratio*100:.1f}%")
        print(f"  Connected components: {t.component_count}")
        print(f"  Largest component: {t.largest_component_ratio*100:.1f}%")

    print("\n" + "=" * 80)
    print("SYNTHETIC GRAPH TOPOLOGY")
    print("=" * 80)

    print("\nNode Type Distribution:")
    print(f"{'Graph':<15} {'Nodes':>8} {'Hubs':>8} {'Spurs':>8} {'Sources':>8} {'InCycle':>8} {'Components':>10}")
    print("-" * 75)

    for name, t in syn_topology.items():
        print(f"{name:<15} {t.node_count:>8} {t.hub_ratio*100:>7.1f}% {t.spur_ratio*100:>7.1f}% "
              f"{t.source_ratio*100:>7.1f}% {t.cycle_ratio*100:>7.1f}% {t.component_count:>10}")

    print("\nBranching Statistics:")
    print(f"{'Graph':<15} {'Mean':>8} {'Median':>8} {'Max':>8} {'Entropy':>10} {'LargestCC':>10}")
    print("-" * 70)

    for name, t in syn_topology.items():
        print(f"{name:<15} {t.mean_out_degree:>8.2f} {t.median_out_degree:>8} {t.max_out_degree:>8} "
              f"{t.branching_entropy:>10.3f} {t.largest_component_ratio*100:>9.1f}%")

    print("\n" + "=" * 80)
    print("OUT-DEGREE DISTRIBUTIONS (local heterogeneity)")
    print("=" * 80)
    print("\nThis shows the DISTRIBUTION of branching factors, not just the mean.")
    print("A healthy narrative graph has varied branching - some hubs, some linear, some dead ends.\n")

    for name, t in syn_topology.items():
        print(f"\n{name}:")
        print_degree_histogram(t.out_degree_dist, "out-degree")

    print("\n" + "=" * 80)
    print("DISTRIBUTION COMPARISON (synthetic vs synthetic)")
    print("=" * 80)
    print("\nComparing out-degree distributions between versions:")

    # Compare random vs guided for each setting
    comparisons = [
        ("gallia_v4", "gallia_v5", "Gallia: Random (v4) vs Guided (v5)"),
        ("marmotte_v2", "marmotte_v3", "Marmotte: Random (v2) vs Guided (v3)"),
        ("gallia_v3", "gallia_v4", "Gallia: Overgrown (v3) vs Controlled (v4)"),
    ]

    for a_name, b_name, label in comparisons:
        if a_name in syn_topology and b_name in syn_topology:
            a_topo = syn_topology[a_name]
            b_topo = syn_topology[b_name]

            div = distribution_divergence(a_topo.out_degree_dist, b_topo.out_degree_dist)

            print(f"\n{label}:")
            print(f"  Out-degree JS divergence: {div['js_divergence']:.4f}")
            print(f"  Out-degree TV distance: {div['tv_distance']:.4f}")
            print(f"  Shared degree values: {div['shared_degrees']}")
            print(f"  Hub ratio: {a_name}={a_topo.hub_ratio*100:.1f}%, {b_name}={b_topo.hub_ratio*100:.1f}%")
            print(f"  Spur ratio: {a_name}={a_topo.spur_ratio*100:.1f}%, {b_name}={b_topo.spur_ratio*100:.1f}%")
            print(f"  Branching entropy: {a_name}={a_topo.branching_entropy:.3f}, {b_name}={b_topo.branching_entropy:.3f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Key metrics for narrative graph health:

1. HUB RATIO (out-degree > 2): Higher = more branching dialogue trees
   - Too low: linear, no player choice
   - Too high: shallow, all nodes branch

2. SPUR RATIO (out-degree = 0): Dead ends / conversation terminators
   - Some needed for natural endings
   - Too many = fragmented, disconnected

3. CYCLE RATIO: Nodes participating in loops
   - Loops enable revisitable dialogue
   - Too many = confusing circular conversations

4. BRANCHING ENTROPY: Heterogeneity of out-degrees
   - Higher = more varied structure (good)
   - Lower = uniform structure (potentially artificial)

5. LARGEST COMPONENT: Graph connectivity
   - Should be high (>90%) for coherent narrative
   - Low = many disconnected dialogue islands

The goal: synthetic graphs should match reference distributions,
not just reference means. A graph with mean degree 2 could be
all linear chains OR a mix of hubs and spurs - very different!
""")


if __name__ == "__main__":
    main()
