#!/usr/bin/env python3
"""
Apply link_stitch results to synthetic graph.

Usage:
    python scripts/apply_link_stitch_results.py <run_id> <setting> <version>
    python scripts/apply_link_stitch_results.py link_20251225_032316_gallia_v3 gallia 3
"""

import sys
import json
import hashlib
from pathlib import Path

RUNS_DIR = Path("runs")
SYNTHETIC_DIR = Path("synthetic")


def apply_link_stitch_results(run_id: str, setting: str, version: int) -> dict:
    """Apply completed link_stitch results to the graph."""

    # Load queue
    queue_path = RUNS_DIR / run_id / "queue.json"
    if not queue_path.exists():
        raise FileNotFoundError(f"Queue not found: {queue_path}")

    queue_data = json.loads(queue_path.read_text())
    link_stitch_tickets = queue_data.get("link_stitch_tickets", [])

    # Load graph
    graph_path = SYNTHETIC_DIR / f"{setting}_v{version}" / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    graph_data = json.loads(graph_path.read_text())
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Track existing node IDs
    existing_ids = {n["id"] for n in nodes}

    # Stats
    links_added = 0
    bridges_added = 0
    tickets_applied = 0
    skipped_exists = 0

    for ticket in link_stitch_tickets:
        if ticket.get("status") != "completed":
            continue
        if ticket.get("applied"):
            continue

        output_data = ticket.get("output_data", {})
        if not output_data:
            continue

        # Add generated bridge nodes
        for gen_node in output_data.get("generated_nodes", []):
            node_id = gen_node.get("id")
            if not node_id or node_id in existing_ids:
                continue

            nodes.append({
                "id": node_id,
                "text": gen_node.get("text", ""),
                "emotion": gen_node.get("emotion", "neutral"),
                "is_bridge": True,
                "source_ticket": ticket.get("ticket_id"),
            })
            existing_ids.add(node_id)
            bridges_added += 1

        # Add links
        for link in output_data.get("links", []):
            status = link.get("status")
            if status == "declined":
                continue

            from_id = link.get("from")
            to_id = link.get("to")
            via_id = link.get("via")

            if not from_id or not to_id:
                continue

            # Check if source node exists
            if from_id not in existing_ids:
                continue

            # For bridged links, create bridge node if not already added
            if status == "bridged" and via_id:
                if via_id not in existing_ids:
                    # Create bridge node from link data
                    bridge_text = link.get("bridge_text", "")
                    bridge_emotion = link.get("bridge_emotion", "neutral")
                    if bridge_text:
                        nodes.append({
                            "id": via_id,
                            "text": bridge_text,
                            "emotion": bridge_emotion,
                            "is_bridge": True,
                            "source_ticket": ticket.get("ticket_id"),
                        })
                        existing_ids.add(via_id)
                        bridges_added += 1

                # Add edges: from -> via -> to
                edges.append({
                    "source": from_id,
                    "target": via_id,
                    "type": "bridge_out",
                    "source_ticket": ticket.get("ticket_id"),
                })
                edges.append({
                    "source": via_id,
                    "target": to_id,
                    "type": "bridge_in",
                    "source_ticket": ticket.get("ticket_id"),
                })
                links_added += 2

            elif status == "direct":
                # Direct link: from -> to
                # Only add if target exists (or create placeholder)
                edges.append({
                    "source": from_id,
                    "target": to_id,
                    "type": "direct_link",
                    "source_ticket": ticket.get("ticket_id"),
                })
                links_added += 1

        # Mark ticket as applied
        ticket["applied"] = True
        tickets_applied += 1

    # Save updated graph
    graph_data = {"nodes": nodes, "edges": edges}
    graph_path.write_text(json.dumps(graph_data, indent=2))

    # Save updated queue
    queue_path.write_text(json.dumps(queue_data, indent=2))

    return {
        "tickets_applied": tickets_applied,
        "bridges_added": bridges_added,
        "links_added": links_added,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/apply_link_stitch_results.py <run_id> <setting> <version>")
        sys.exit(1)

    run_id = sys.argv[1]
    setting = sys.argv[2]
    version = int(sys.argv[3])

    print(f"Applying link_stitch results from {run_id} to {setting}_v{version}")

    result = apply_link_stitch_results(run_id, setting, version)

    print(f"\nResults:")
    print(f"  Tickets applied: {result['tickets_applied']}")
    print(f"  Bridges added: {result['bridges_added']}")
    print(f"  Links added: {result['links_added']}")
    print(f"  Total nodes: {result['total_nodes']}")
    print(f"  Total edges: {result['total_edges']}")


if __name__ == "__main__":
    main()
