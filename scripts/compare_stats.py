#!/usr/bin/env python3
"""
Compare graph statistics between reference games and synthetic corpora.

Usage:
    python scripts/compare_stats.py
    python scripts/compare_stats.py --json  # Machine-readable output
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import httpx

API_BASE = "http://127.0.0.1:8000"


def get_reference_transitions(game: str) -> dict[tuple[str, str], float]:
    """Get normalized emotion transition probabilities from reference game."""
    resp = httpx.get(f"{API_BASE}/api/transitions/{game}")
    data = resp.json()

    transitions = {}
    total = 0

    for from_emo, targets in data.get("transitions", {}).items():
        for to_emo, count in targets.items():
            transitions[(from_emo, to_emo)] = count
            total += count

    # Normalize
    if total > 0:
        transitions = {k: v / total for k, v in transitions.items()}

    return transitions


def get_reference_stats(game: str) -> dict:
    """Get basic stats from reference game."""
    resp = httpx.get(f"{API_BASE}/api/stats/{game}")
    return resp.json()


def compute_synthetic_transitions(graph_path: Path) -> dict[tuple[str, str], float]:
    """Compute normalized emotion transitions from synthetic graph.json."""
    if not graph_path.exists():
        return {}

    data = json.loads(graph_path.read_text())
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    edges = data.get("edges", [])

    transitions = defaultdict(int)
    total = 0

    for edge in edges:
        src_id = edge.get("source")
        tgt_id = edge.get("target")

        if src_id in nodes and tgt_id in nodes:
            src_emo = nodes[src_id].get("emotion", "neutral")
            tgt_emo = nodes[tgt_id].get("emotion", "neutral")
            transitions[(src_emo, tgt_emo)] += 1
            total += 1

    # Normalize
    if total > 0:
        transitions = {k: v / total for k, v in transitions.items()}

    return dict(transitions)


def compute_synthetic_stats(graph_path: Path) -> dict:
    """Compute basic stats from synthetic graph.json."""
    if not graph_path.exists():
        return {"nodes": 0, "edges": 0}

    data = json.loads(graph_path.read_text())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Compute degree distribution
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)

    for edge in edges:
        out_degree[edge.get("source")] += 1
        in_degree[edge.get("target")] += 1

    # Emotion distribution
    emotions = defaultdict(int)
    for node in nodes:
        emotions[node.get("emotion", "neutral")] += 1

    total_emo = sum(emotions.values())
    emotion_dist = {k: v / total_emo for k, v in emotions.items()} if total_emo > 0 else {}

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "avg_out_degree": sum(out_degree.values()) / len(nodes) if nodes else 0,
        "avg_in_degree": sum(in_degree.values()) / len(nodes) if nodes else 0,
        "emotion_distribution": emotion_dist,
    }


def kl_divergence(p: dict, q: dict, epsilon: float = 1e-10) -> float:
    """Compute KL divergence D(P || Q) with smoothing."""
    all_keys = set(p.keys()) | set(q.keys())

    kl = 0.0
    for k in all_keys:
        p_val = p.get(k, epsilon)
        q_val = q.get(k, epsilon)
        if p_val > 0:
            kl += p_val * math.log(p_val / q_val)

    return kl


def total_variation_distance(p: dict, q: dict) -> float:
    """Compute total variation distance (L1/2)."""
    all_keys = set(p.keys()) | set(q.keys())

    tv = 0.0
    for k in all_keys:
        tv += abs(p.get(k, 0) - q.get(k, 0))

    return tv / 2


def jensen_shannon_divergence(p: dict, q: dict, epsilon: float = 1e-10) -> float:
    """Compute Jensen-Shannon divergence (symmetric, bounded 0-1)."""
    all_keys = set(p.keys()) | set(q.keys())

    # Create mixture distribution
    m = {}
    for k in all_keys:
        m[k] = (p.get(k, 0) + q.get(k, 0)) / 2

    return (kl_divergence(p, m, epsilon) + kl_divergence(q, m, epsilon)) / 2


def compare_transitions(ref_trans: dict, syn_trans: dict) -> dict:
    """Compare two transition distributions."""
    return {
        "kl_divergence": kl_divergence(ref_trans, syn_trans),
        "total_variation": total_variation_distance(ref_trans, syn_trans),
        "jensen_shannon": jensen_shannon_divergence(ref_trans, syn_trans),
        "ref_transitions": len(ref_trans),
        "syn_transitions": len(syn_trans),
        "shared_transitions": len(set(ref_trans.keys()) & set(syn_trans.keys())),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare graph statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    synthetic_dir = Path("synthetic")

    # Reference games
    references = ["oblivion", "falloutnv"]

    # Synthetic versions to compare
    synthetics = [
        ("gallia", 3),
        ("gallia", 4),
        ("gallia", 5),
        ("marmotte", 1),
        ("marmotte", 2),
        ("marmotte", 3),
    ]

    # Load reference data
    ref_transitions = {}
    ref_stats = {}

    for game in references:
        try:
            ref_transitions[game] = get_reference_transitions(game)
            ref_stats[game] = get_reference_stats(game)
        except Exception as e:
            print(f"Warning: Could not load {game}: {e}")

    # Combine reference transitions (weighted average)
    combined_ref = defaultdict(float)
    total_weight = 0
    for game, trans in ref_transitions.items():
        weight = ref_stats.get(game, {}).get("edges", 1)
        total_weight += weight
        for k, v in trans.items():
            combined_ref[k] += v * weight

    if total_weight > 0:
        combined_ref = {k: v / total_weight for k, v in combined_ref.items()}

    # Load and compare synthetic graphs
    results = []

    for setting, version in synthetics:
        graph_path = synthetic_dir / f"{setting}_v{version}" / "graph.json"

        syn_trans = compute_synthetic_transitions(graph_path)
        syn_stats = compute_synthetic_stats(graph_path)

        if not syn_trans:
            continue

        # Compare to each reference and combined
        comparisons = {}
        for game in references:
            if game in ref_transitions:
                comparisons[game] = compare_transitions(ref_transitions[game], syn_trans)

        comparisons["combined"] = compare_transitions(dict(combined_ref), syn_trans)

        results.append({
            "name": f"{setting}_v{version}",
            "stats": syn_stats,
            "comparisons": comparisons,
        })

    if args.json:
        print(json.dumps({
            "references": {
                game: {
                    "nodes": ref_stats.get(game, {}).get("nodes", 0),
                    "edges": ref_stats.get(game, {}).get("edges", 0),
                    "transitions": len(ref_transitions.get(game, {})),
                }
                for game in references
            },
            "synthetics": results,
        }, indent=2))
        return
    else:
        # Pretty print
        print("=" * 80)
        print("REFERENCE CORPORA")
        print("=" * 80)
        for game in references:
            stats = ref_stats.get(game, {})
            trans = ref_transitions.get(game, {})
            print(f"\n{game}:")
            print(f"  Nodes: {stats.get('nodes', 0):,}")
            print(f"  Edges: {stats.get('edges', 0):,}")
            print(f"  Unique transitions: {len(trans)}")

        print(f"\nCombined reference: {len(combined_ref)} unique transitions")

        print("\n" + "=" * 80)
        print("SYNTHETIC GRAPHS vs COMBINED REFERENCE")
        print("=" * 80)
        print("\nMetrics:")
        print("  JS Div  = Jensen-Shannon divergence (0=identical, 1=disjoint)")
        print("  TV Dist = Total Variation distance (0=identical, 1=disjoint)")
        print("  Shared  = Transition types present in both distributions")
        print("  Eff     = Transitions covered per 100 nodes (coverage efficiency)")
        print(f"\n{'Graph':<15} {'Nodes':>8} {'Edges':>8} {'JS Div':>8} {'TV Dist':>8} {'Shared':>8} {'Eff':>8}")
        print("-" * 75)

        for r in results:
            name = r["name"]
            stats = r["stats"]
            cmp = r["comparisons"].get("combined", {})
            nodes = stats['nodes']
            shared = cmp.get('shared_transitions', 0)
            efficiency = (shared / nodes * 100) if nodes > 0 else 0

            print(f"{name:<15} {nodes:>8} {stats['edges']:>8} "
                  f"{cmp.get('jensen_shannon', 0):>8.4f} {cmp.get('total_variation', 0):>8.4f} "
                  f"{shared:>8} {efficiency:>8.2f}")

        # Comparison analysis
        print("\n" + "=" * 80)
        print("ANALYSIS: Random vs Guided Sampling")
        print("=" * 80)

        # Group by setting
        gallia_results = [r for r in results if "gallia" in r["name"]]
        marmotte_results = [r for r in results if "marmotte" in r["name"]]

        for setting_name, setting_results in [("Gallia", gallia_results), ("Marmotte", marmotte_results)]:
            if len(setting_results) < 2:
                continue
            print(f"\n{setting_name}:")

            # Find random (v4/v2) and guided (v5/v3) versions
            random_v = None
            guided_v = None
            for r in setting_results:
                if "_v4" in r["name"] or "_v2" in r["name"]:
                    random_v = r
                elif "_v5" in r["name"] or "_v3" in r["name"]:
                    guided_v = r

            if random_v and guided_v:
                r_js = random_v["comparisons"]["combined"]["jensen_shannon"]
                g_js = guided_v["comparisons"]["combined"]["jensen_shannon"]
                r_nodes = random_v["stats"]["nodes"]
                g_nodes = guided_v["stats"]["nodes"]

                # Per-node JS (lower is better)
                r_js_per_k = r_js / (r_nodes / 1000) if r_nodes > 0 else 0
                g_js_per_k = g_js / (g_nodes / 1000) if g_nodes > 0 else 0

                print(f"  Random ({random_v['name']}): JS={r_js:.4f} with {r_nodes} nodes")
                print(f"  Guided ({guided_v['name']}): JS={g_js:.4f} with {g_nodes} nodes")

                if g_js < r_js:
                    pct = (1 - g_js/r_js) * 100
                    print(f"  → Guided has {pct:.1f}% lower divergence (better match)")
                else:
                    pct = (g_js/r_js - 1) * 100
                    print(f"  → Guided has {pct:.1f}% higher divergence")

                # Size-normalized comparison
                print(f"  → Guided uses {g_nodes/r_nodes*100:.1f}% of the nodes for similar quality")

        print("\n" + "=" * 80)
        print("DETAILED COMPARISON (per reference)")
        print("=" * 80)

        for r in results:
            name = r["name"]
            print(f"\n{name}:")
            print(f"  Emotion distribution: {r['stats'].get('emotion_distribution', {})}")

            for ref_game in references + ["combined"]:
                cmp = r["comparisons"].get(ref_game, {})
                if cmp:
                    print(f"  vs {ref_game}:")
                    print(f"    Jensen-Shannon: {cmp.get('jensen_shannon', 0):.4f}")
                    print(f"    Total Variation: {cmp.get('total_variation', 0):.4f}")
                    print(f"    KL Divergence: {cmp.get('kl_divergence', 0):.4f}")
                    print(f"    Shared transitions: {cmp.get('shared_transitions', 0)} / {cmp.get('ref_transitions', 0)}")


if __name__ == "__main__":
    main()
