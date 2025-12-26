#!/usr/bin/env python3
"""
Compute structural metrics for dialogue graphs.

Used by the growth controller to measure progress toward reference baselines.

Reference baselines (Dec 2024):
  Metric          | Oblivion | FalloutNV
  ----------------+----------+----------
  chain_ratio     |   53.6%  |   27.5%
  leaf_ratio      |   15.0%  |   39.8%
  hub_density     |   34.6%  |   51.4%
  entry_ratio     |    1.4%  |    4.0%
  edges_per_node  |    5.5   |   20.6

Usage:
    python scripts/structural_metrics.py gallia:6
    python scripts/structural_metrics.py --reference
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StructuralMetrics:
    """Structural metrics for a dialogue graph."""
    nodes: int
    edges: int
    chain_ratio: float  # nodes with in=out=1
    leaf_ratio: float   # nodes with out=0
    hub_density: float  # nodes with degree > 5
    entry_ratio: float  # nodes with in=0
    edges_per_node: float
    max_out_degree: int
    max_in_degree: int

    def summary(self) -> str:
        return (
            f"Nodes: {self.nodes}, Edges: {self.edges}\n"
            f"  chain_ratio:     {self.chain_ratio*100:5.1f}%\n"
            f"  leaf_ratio:      {self.leaf_ratio*100:5.1f}%\n"
            f"  hub_density:     {self.hub_density*100:5.1f}%\n"
            f"  entry_ratio:     {self.entry_ratio*100:5.1f}%\n"
            f"  edges_per_node:  {self.edges_per_node:5.1f}\n"
            f"  max_out_degree:  {self.max_out_degree}\n"
            f"  max_in_degree:   {self.max_in_degree}"
        )

    def as_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "chain_ratio": self.chain_ratio,
            "leaf_ratio": self.leaf_ratio,
            "hub_density": self.hub_density,
            "entry_ratio": self.entry_ratio,
            "edges_per_node": self.edges_per_node,
            "max_out_degree": self.max_out_degree,
            "max_in_degree": self.max_in_degree,
        }


def compute_metrics(nodes: list, edges: list) -> StructuralMetrics:
    """Compute structural metrics from nodes/edges lists."""
    out_degree = Counter()
    in_degree = Counter()

    for e in edges:
        src = e.get("source") or e.get("from")
        tgt = e.get("target") or e.get("to")
        if src and tgt:
            out_degree[src] += 1
            in_degree[tgt] += 1

    chain = leaf = hub = entry = 0
    node_ids = set()

    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        node_ids.add(nid)
        in_d = in_degree.get(nid, 0)
        out_d = out_degree.get(nid, 0)

        if in_d == 1 and out_d == 1:
            chain += 1
        if out_d == 0:
            leaf += 1
        if out_d > 5 or in_d > 5:
            hub += 1
        if in_d == 0:
            entry += 1

    total = len(node_ids)
    if total == 0:
        return StructuralMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    return StructuralMetrics(
        nodes=total,
        edges=len(edges),
        chain_ratio=chain / total,
        leaf_ratio=leaf / total,
        hub_density=hub / total,
        entry_ratio=entry / total,
        edges_per_node=len(edges) / total,
        max_out_degree=max(out_degree.values()) if out_degree else 0,
        max_in_degree=max(in_degree.values()) if in_degree else 0,
    )


def compute_from_graph_path(graph_path: Path) -> StructuralMetrics:
    """Compute metrics from a graph.json file."""
    graph = json.loads(graph_path.read_text())
    return compute_metrics(graph["nodes"], graph["edges"])


def compute_reference_metrics(game: str) -> StructuralMetrics:
    """Compute metrics for a reference game's dialogue graph."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dialogue_graph import DialogueGraph, load_dialogue

    path = Path(f"dialogue_data/{game}_full_dialogue.json")
    if not path.exists():
        raise FileNotFoundError(f"Reference dialogue not found: {path}")

    dialogue = load_dialogue(path)
    dg = DialogueGraph.from_dialogue_data(dialogue)

    # Calculate degrees from adjacency
    out_degree = {nid: len(dg._adjacency.get(nid, [])) for nid in dg.nodes}
    in_degree = {nid: len(dg._reverse_adjacency.get(nid, [])) for nid in dg.nodes}

    chain = leaf = hub = entry = 0
    total = len(dg.nodes)

    for nid in dg.nodes:
        in_d = in_degree.get(nid, 0)
        out_d = out_degree.get(nid, 0)
        if in_d == 1 and out_d == 1:
            chain += 1
        if out_d == 0:
            leaf += 1
        if out_d > 5 or in_d > 5:
            hub += 1
        if in_d == 0:
            entry += 1

    return StructuralMetrics(
        nodes=total,
        edges=len(dg.edges),
        chain_ratio=chain / total if total else 0,
        leaf_ratio=leaf / total if total else 0,
        hub_density=hub / total if total else 0,
        entry_ratio=entry / total if total else 0,
        edges_per_node=len(dg.edges) / total if total else 0,
        max_out_degree=max(out_degree.values()) if out_degree else 0,
        max_in_degree=max(in_degree.values()) if in_degree else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute structural metrics")
    parser.add_argument("setting", nargs="?", help="Setting spec (e.g., 'gallia:6')")
    parser.add_argument("--reference", action="store_true",
                        help="Compute reference game metrics")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    if args.reference:
        print("Reference baselines:\n")
        for game in ["oblivion", "falloutnv"]:
            try:
                metrics = compute_reference_metrics(game)
                print(f"[{game}]")
                print(metrics.summary())
                print()
            except FileNotFoundError as e:
                print(f"[{game}] {e}")
                print()
        return

    if not args.setting:
        parser.print_help()
        return

    # Parse setting spec
    if ":" in args.setting:
        setting, version = args.setting.split(":")
        version = int(version)
    else:
        setting = args.setting
        synthetic_dir = Path("synthetic")
        versions = sorted([
            int(p.name.split("_v")[1])
            for p in synthetic_dir.glob(f"{setting}_v*")
            if p.is_dir()
        ])
        if versions:
            version = versions[-1]
        else:
            print(f"No versions found for {setting}")
            return

    graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
    if not graph_path.exists():
        print(f"Graph not found: {graph_path}")
        return

    metrics = compute_from_graph_path(graph_path)

    if args.json:
        print(json.dumps(metrics.as_dict(), indent=2))
    else:
        print(f"[{setting}_v{version}]")
        print(metrics.summary())


if __name__ == "__main__":
    main()
