#!/usr/bin/env python3
"""
Growth Oracle - LLM-based constraint solver for stats-guided graph growth.

Queries an LLM with full context about the current graph state and returns
the next action to take.

Usage:
    DEEPSEEK_API_KEY="sk-..." python scripts/growth_oracle.py gallia:6
    DEEPSEEK_API_KEY="sk-..." python scripts/growth_oracle.py gallia:6 --execute
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import httpx

# Load the oracle prompt
ORACLE_PROMPT_PATH = Path(__file__).parent.parent / ".claude/agents/growth-oracle.md"
DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"


def load_oracle_prompt() -> str:
    """Load the growth oracle system prompt."""
    return ORACLE_PROMPT_PATH.read_text()


def compute_structural_metrics(graph: dict) -> dict:
    """Compute structural metrics from graph dict."""
    nodes = graph["nodes"]
    edges = graph["edges"]

    out_degree = Counter()
    in_degree = Counter()
    for e in edges:
        out_degree[e["source"]] += 1
        in_degree[e["target"]] += 1

    chain = leaf = hub = entry = 0
    for n in nodes:
        nid = n["id"]
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

    total = len(nodes)
    return {
        "nodes": total,
        "edges": len(edges),
        "chain_ratio": chain / total if total else 0,
        "leaf_ratio": leaf / total if total else 0,
        "hub_density": hub / total if total else 0,
        "entry_ratio": entry / total if total else 0,
        "edges_per_node": len(edges) / total if total else 0,
        "max_out_degree": max(out_degree.values()) if out_degree else 0,
        "max_in_degree": max(in_degree.values()) if in_degree else 0,
    }


def compute_arc_inventory(graph: dict) -> dict:
    """Compute arc shape distribution."""
    arc_shapes = Counter()
    arc_count = 0

    for n in graph["nodes"]:
        if n.get("beat_index") == 0:  # arc entry
            arc_count += 1
            shape = n.get("arc_shape") or "unknown"
            arc_shapes[shape] += 1

    return {
        "total_arcs": arc_count,
        "arc_shapes": dict(arc_shapes.most_common()),
    }


def count_hubs(graph: dict) -> int:
    """Count existing topic hub nodes."""
    return sum(1 for n in graph["nodes"] if n.get("node_type") == "topic_hub")


def count_extension_candidates(graph: dict) -> int:
    """Count nodes marked as extension candidates."""
    return sum(1 for n in graph["nodes"] if n.get("extension_candidate"))


def load_action_history(setting: str, version: int) -> list:
    """Load action history from metadata if available."""
    meta_path = Path(f"synthetic/{setting}_v{version}/metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("action_history", [])[-5:]  # Last 5 actions
    return []


def format_context(setting: str, version: int, graph: dict) -> str:
    """Format the context for the oracle."""
    metrics = compute_structural_metrics(graph)
    inventory = compute_arc_inventory(graph)
    hubs = count_hubs(graph)
    ext_candidates = count_extension_candidates(graph)
    history = load_action_history(setting, version)

    # Format metrics with target comparison
    def fmt_metric(name, value, target, is_pct=True):
        if is_pct:
            val_str = f"{value*100:.1f}%"
        else:
            val_str = f"{value:.1f}"

        if isinstance(target, tuple):
            in_range = target[0] <= value <= target[1]
            target_str = f"{target[0]*100:.0f}-{target[1]*100:.0f}%" if is_pct else f"{target[0]}-{target[1]}"
        else:
            in_range = value >= target
            target_str = f">{target*100:.0f}%" if is_pct else f">{target}"

        status = "✓" if in_range else "← gap"
        return f"{name}: {val_str:>6} (target: {target_str}) {status}"

    metrics_lines = [
        fmt_metric("chain_ratio", metrics["chain_ratio"], (0.27, 0.54)),
        fmt_metric("leaf_ratio", metrics["leaf_ratio"], (0.15, 0.40)),
        fmt_metric("hub_density", metrics["hub_density"], 0.25),
        fmt_metric("entry_ratio", metrics["entry_ratio"], (0.01, 0.04)),
        fmt_metric("edges_per_node", metrics["edges_per_node"], 4.0, is_pct=False),
    ]

    # Format arc shapes
    arc_shapes_str = ", ".join(f"{k}: {v}" for k, v in list(inventory["arc_shapes"].items())[:8])
    if len(inventory["arc_shapes"]) > 8:
        arc_shapes_str += f", ... (+{len(inventory['arc_shapes'])-8} more)"

    # Format history
    if history:
        history_lines = [f"{i+1}. {h}" for i, h in enumerate(history)]
        history_str = "\n".join(history_lines)
    else:
        history_str = "(no previous actions recorded)"

    context = f"""## Current State
Setting: {setting}_v{version}
Nodes: {metrics['nodes']}, Edges: {metrics['edges']}

## Structural Metrics
{chr(10).join(metrics_lines)}

## Arc Inventory
Total arcs: {inventory['total_arcs']}
Arc shapes: {{{arc_shapes_str}}}
Existing hubs: {hubs} (aggregated by arc_shape)

## Action History (last 5)
{history_str}

## Extension Candidates Available: {ext_candidates}
"""
    return context


def query_oracle(context: str, api_key: str) -> dict:
    """Query the growth oracle LLM."""
    system_prompt = load_oracle_prompt()

    response = httpx.post(
        DEEPSEEK_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            "temperature": 0.3,  # Lower temp for more deterministic decisions
            "max_tokens": 1000,
        },
        timeout=60.0,
    )

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text[:200]}")

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Parse JSON from response
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    return json.loads(content.strip())


def record_action(setting: str, version: int, action: str, params: dict, result: dict):
    """Record action to metadata history."""
    meta_path = Path(f"synthetic/{setting}_v{version}/metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}

    if "action_history" not in meta:
        meta["action_history"] = []

    # Format action string
    params_str = " ".join(f"{k}={v}" for k, v in params.items())
    result_str = " ".join(f"{k}={v}" for k, v in result.items())
    action_str = f"{action} {params_str} → {result_str}"

    meta["action_history"].append(action_str)
    meta_path.write_text(json.dumps(meta, indent=2))


def execute_translate(setting: str, version: int, params: dict, api_key: str) -> dict:
    """Execute TRANSLATE action via run_batch.py."""
    import subprocess

    batch_size = params.get("batch_size", 100)
    guided_weights = params.get("guided_weights", {})

    # Build command
    cmd = [
        "uv", "run", "python", "scripts/run_batch.py",
        "translate", f"{setting}:{version}", str(batch_size),
        "--guided"
    ]

    print(f"  Running: {' '.join(cmd)}")

    # Get before stats
    graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
    graph_before = json.loads(graph_path.read_text())
    nodes_before = len(graph_before["nodes"])
    edges_before = len(graph_before["edges"])

    # Run translate
    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = api_key

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500]}")
        return {"success": False, "error": result.stderr[:200]}

    print(result.stdout)

    # Get after stats
    graph_after = json.loads(graph_path.read_text())
    nodes_after = len(graph_after["nodes"])
    edges_after = len(graph_after["edges"])

    return {
        "success": True,
        "nodes_added": nodes_after - nodes_before,
        "edges_added": edges_after - edges_before,
    }


def execute_link(setting: str, version: int, params: dict, api_key: str) -> dict:
    """Execute LINK action via run_batch.py link."""
    import subprocess

    batch_size = params.get("batch_size", 50)

    cmd = [
        "uv", "run", "python", "scripts/run_batch.py",
        "link", f"{setting}:{version}", str(batch_size),
    ]

    print(f"  Running: {' '.join(cmd)}")

    graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
    graph_before = json.loads(graph_path.read_text())
    nodes_before = len(graph_before["nodes"])
    edges_before = len(graph_before["edges"])

    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = api_key

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500]}")
        return {"success": False, "error": result.stderr[:200]}

    print(result.stdout)

    graph_after = json.loads(graph_path.read_text())
    nodes_after = len(graph_after["nodes"])
    edges_after = len(graph_after["edges"])

    return {
        "success": True,
        "nodes_added": nodes_after - nodes_before,
        "edges_added": edges_after - edges_before,
    }


def execute_aggregate(setting: str, version: int, params: dict) -> dict:
    """Execute AGGREGATE action via aggregate_topic_hubs.py."""
    import subprocess

    group_by = params.get("group_by", "arc_shape")
    min_entries = params.get("min_entries", 2)

    cmd = [
        "uv", "run", "python", "scripts/aggregate_topic_hubs.py",
        f"{setting}:{version}",
        "--group-by", group_by,
        "--min-entries", str(min_entries),
    ]

    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500]}")
        return {"success": False, "error": result.stderr[:200]}

    print(result.stdout)

    # Parse output for hub count
    output = result.stdout
    hubs_created = 0
    edges_added = 0

    for line in output.split("\n"):
        if "Created" in line and "hub nodes" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "Created":
                    hubs_created = int(parts[i + 1])
                    break
        if "Created" in line and "topic_branch" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "Created":
                    edges_added = int(parts[i + 1])
                    break

    return {
        "success": True,
        "hubs_created": hubs_created,
        "edges_added": edges_added,
    }


def execute_extend(setting: str, version: int, params: dict, api_key: str) -> dict:
    """Execute EXTEND action via run_batch.py extend."""
    import subprocess

    batch_size = params.get("batch_size", 50)

    cmd = [
        "uv", "run", "python", "scripts/run_batch.py",
        "extend", f"{setting}:{version}", str(batch_size),
    ]

    print(f"  Running: {' '.join(cmd)}")

    graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
    graph_before = json.loads(graph_path.read_text())
    nodes_before = len(graph_before["nodes"])
    edges_before = len(graph_before["edges"])

    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = api_key

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500]}")
        return {"success": False, "error": result.stderr[:200]}

    print(result.stdout)

    graph_after = json.loads(graph_path.read_text())
    nodes_after = len(graph_after["nodes"])
    edges_after = len(graph_after["edges"])

    return {
        "success": True,
        "nodes_added": nodes_after - nodes_before,
        "edges_added": edges_after - edges_before,
    }


def execute_action(action: str, setting: str, version: int, params: dict, api_key: str) -> dict:
    """Execute the oracle's recommended action."""
    print(f"\n{'='*50}")
    print(f"Executing {action}...")
    print(f"{'='*50}\n")

    if action == "TRANSLATE":
        return execute_translate(setting, version, params, api_key)
    elif action == "LINK":
        return execute_link(setting, version, params, api_key)
    elif action == "AGGREGATE":
        return execute_aggregate(setting, version, params)
    elif action == "EXTEND":
        return execute_extend(setting, version, params, api_key)
    else:
        return {"success": False, "error": f"Unknown action: {action}"}


def main():
    parser = argparse.ArgumentParser(description="Growth Oracle - decide next action")
    parser.add_argument("setting", help="Setting spec (e.g., 'gallia:6')")
    parser.add_argument("--execute", action="store_true",
                        help="Execute the recommended action (not just display)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON response")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        sys.exit(1)

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
            sys.exit(1)

    graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
    if not graph_path.exists():
        print(f"Graph not found: {graph_path}")
        sys.exit(1)

    graph = json.loads(graph_path.read_text())

    # Format context
    context = format_context(setting, version, graph)

    print(f"[{setting}_v{version}] Querying growth oracle...\n")
    print("Context:")
    print(context)
    print("-" * 50)

    # Query oracle
    decision = query_oracle(context, api_key)

    if args.json:
        print(json.dumps(decision, indent=2))
        return

    # Display decision
    print(f"\nRecommended Action: {decision['action']}")
    print(f"Parameters: {json.dumps(decision.get('parameters', {}), indent=2)}")
    print(f"\nReasoning: {decision.get('reasoning', 'N/A')}")

    if "expected_effect" in decision:
        print(f"\nExpected Effect:")
        for k, v in decision["expected_effect"].items():
            print(f"  {k}: {v}")

    if "next_likely_action" in decision:
        print(f"\nNext Likely Action: {decision['next_likely_action']}")

    if args.execute:
        action = decision["action"]
        params = decision.get("parameters", {})

        result = execute_action(action, setting, version, params, api_key)

        # Record to history
        record_action(setting, version, action, params, result)

        print(f"\n{'='*50}")
        print("Execution Result:")
        print(f"{'='*50}")
        for k, v in result.items():
            print(f"  {k}: {v}")

        # Show updated metrics
        print(f"\nUpdated Metrics:")
        graph = json.loads(graph_path.read_text())
        metrics = compute_structural_metrics(graph)
        print(f"  Nodes: {metrics['nodes']}")
        print(f"  Edges: {metrics['edges']}")
        print(f"  chain_ratio:     {metrics['chain_ratio']*100:.1f}%")
        print(f"  hub_density:     {metrics['hub_density']*100:.1f}%")
        print(f"  edges_per_node:  {metrics['edges_per_node']:.2f}")

        if args.loop:
            print(f"\n{'='*50}")
            print("Looping for next action...")
            print(f"{'='*50}")


def run_loop(setting: str, version: int, api_key: str, target_nodes: int = 500):
    """Run the oracle loop until target size reached."""
    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}")
        print(f"{'#'*60}")

        graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
        graph = json.loads(graph_path.read_text())
        metrics = compute_structural_metrics(graph)

        print(f"\nCurrent: {metrics['nodes']} nodes, {metrics['edges']} edges")
        print(f"Target: {target_nodes} nodes")

        if metrics["nodes"] >= target_nodes:
            print(f"\n✓ Target reached! ({metrics['nodes']} >= {target_nodes})")
            break

        # Format context and query oracle
        context = format_context(setting, version, graph)
        print("\nQuerying oracle...")

        try:
            decision = query_oracle(context, api_key)
        except Exception as e:
            print(f"Oracle error: {e}")
            break

        action = decision["action"]
        params = decision.get("parameters", {})

        print(f"\nOracle recommends: {action}")
        print(f"Reasoning: {decision.get('reasoning', 'N/A')[:200]}...")

        # Execute
        result = execute_action(action, setting, version, params, api_key)
        record_action(setting, version, action, params, result)

        if not result.get("success"):
            print(f"Action failed: {result.get('error')}")
            break

        # Brief pause to avoid rate limits
        import time
        time.sleep(2)

    # Final stats
    print(f"\n{'='*60}")
    print("FINAL STATE")
    print(f"{'='*60}")
    graph = json.loads(graph_path.read_text())
    metrics = compute_structural_metrics(graph)
    print(f"Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
    print(f"chain_ratio:     {metrics['chain_ratio']*100:.1f}% (target: 27-54%)")
    print(f"hub_density:     {metrics['hub_density']*100:.1f}% (target: >25%)")
    print(f"edges_per_node:  {metrics['edges_per_node']:.2f} (target: >4.0)")


def main():
    parser = argparse.ArgumentParser(description="Growth Oracle - decide next action")
    parser.add_argument("setting", help="Setting spec (e.g., 'gallia:6')")
    parser.add_argument("--execute", action="store_true",
                        help="Execute the recommended action (not just display)")
    parser.add_argument("--loop", action="store_true",
                        help="Keep executing until target reached")
    parser.add_argument("--target", type=int, default=500,
                        help="Target node count for --loop mode (default: 500)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON response")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        sys.exit(1)

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
            sys.exit(1)

    graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
    if not graph_path.exists():
        print(f"Graph not found: {graph_path}")
        sys.exit(1)

    # Loop mode
    if args.loop:
        run_loop(setting, version, api_key, args.target)
        return

    graph = json.loads(graph_path.read_text())

    # Format context
    context = format_context(setting, version, graph)

    print(f"[{setting}_v{version}] Querying growth oracle...\n")
    print("Context:")
    print(context)
    print("-" * 50)

    # Query oracle
    decision = query_oracle(context, api_key)

    if args.json:
        print(json.dumps(decision, indent=2))
        return

    # Display decision
    print(f"\nRecommended Action: {decision['action']}")
    print(f"Parameters: {json.dumps(decision.get('parameters', {}), indent=2)}")
    print(f"\nReasoning: {decision.get('reasoning', 'N/A')}")

    if "expected_effect" in decision:
        print(f"\nExpected Effect:")
        for k, v in decision["expected_effect"].items():
            print(f"  {k}: {v}")

    if "next_likely_action" in decision:
        print(f"\nNext Likely Action: {decision['next_likely_action']}")

    if args.execute:
        action = decision["action"]
        params = decision.get("parameters", {})

        result = execute_action(action, setting, version, params, api_key)

        # Record to history
        record_action(setting, version, action, params, result)

        print(f"\n{'='*50}")
        print("Execution Result:")
        print(f"{'='*50}")
        for k, v in result.items():
            print(f"  {k}: {v}")

        # Show updated metrics
        print(f"\nUpdated Metrics:")
        graph = json.loads(graph_path.read_text())
        metrics = compute_structural_metrics(graph)
        print(f"  Nodes: {metrics['nodes']}")
        print(f"  Edges: {metrics['edges']}")
        print(f"  chain_ratio:     {metrics['chain_ratio']*100:.1f}%")
        print(f"  hub_density:     {metrics['hub_density']*100:.1f}%")
        print(f"  edges_per_node:  {metrics['edges_per_node']:.2f}")


if __name__ == "__main__":
    main()
