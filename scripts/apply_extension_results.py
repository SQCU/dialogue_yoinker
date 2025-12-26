#!/usr/bin/env python3
"""
Apply extension_resolve results to synthetic graph.

Usage:
    python scripts/apply_extension_results.py <run_id> <setting> <version>
    python scripts/apply_extension_results.py ext_20251225_... gallia 4
"""

import sys
import json
from pathlib import Path

from scripts.emotion_schema import ValidationStats, validate_emotion

RUNS_DIR = Path("runs")
SYNTHETIC_DIR = Path("synthetic")


def apply_extension_results(run_id: str, setting: str, version: int) -> dict:
    """Apply completed extension_resolve results to the graph."""

    # Load queue
    queue_path = RUNS_DIR / run_id / "queue.json"
    if not queue_path.exists():
        raise FileNotFoundError(f"Queue not found: {queue_path}")

    queue_data = json.loads(queue_path.read_text())
    extension_resolve_tickets = queue_data.get("extension_resolve_tickets", [])

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
    tickets_applied = 0
    bridges_added = 0
    edges_added = 0
    failed_resolutions = 0
    emotion_stats = ValidationStats()

    for ticket in extension_resolve_tickets:
        if ticket.get("status") != "completed":
            continue
        if ticket.get("applied"):
            continue

        output_data = ticket.get("output_data", {})
        if not output_data:
            continue

        # Check if resolution was successful
        if not output_data.get("success"):
            failed_resolutions += 1
            ticket["applied"] = True  # Mark as processed even if failed
            continue

        # Add bridge nodes
        for bridge in output_data.get("bridge_nodes", []):
            node_id = bridge.get("id")
            if not node_id or node_id in existing_ids:
                continue

            # Validate emotion against canonical schema
            raw_emotion = bridge.get("emotion", "neutral")
            validation = validate_emotion(raw_emotion)
            emotion_stats.record(validation)

            nodes.append({
                "id": node_id,
                "text": bridge.get("text", ""),
                "emotion": validation.canonical,
                "emotion_raw": raw_emotion if not validation.was_valid else None,
                "is_extension_bridge": True,
                "arc_type": output_data.get("arc_realized"),
                "source_ticket": ticket.get("ticket_id"),
            })
            existing_ids.add(node_id)
            bridges_added += 1

        # Add edges
        for edge in output_data.get("edges", []):
            source_id = edge.get("source")
            target_id = edge.get("target")
            if source_id and target_id:
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "type": edge.get("type", "extension_bridge"),
                    "source_ticket": ticket.get("ticket_id"),
                })
                edges_added += 1

        ticket["applied"] = True
        tickets_applied += 1

    # Save updated graph
    graph_data = {"nodes": nodes, "edges": edges}
    graph_path.write_text(json.dumps(graph_data, indent=2))

    # Save updated queue
    queue_path.write_text(json.dumps(queue_data, indent=2))

    return {
        "tickets_applied": tickets_applied,
        "failed_resolutions": failed_resolutions,
        "bridges_added": bridges_added,
        "edges_added": edges_added,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "emotion_validation": {
            "total": emotion_stats.total,
            "valid": emotion_stats.valid,
            "mapped": emotion_stats.mapped,
            "defaulted": emotion_stats.defaulted,
            "defect_rate": emotion_stats.defect_rate,
            "recovery_rate": emotion_stats.recovery_rate,
            "unknown_emotions": emotion_stats.unknown_emotions,
        },
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/apply_extension_results.py <run_id> <setting> <version>")
        sys.exit(1)

    run_id = sys.argv[1]
    setting = sys.argv[2]
    version = int(sys.argv[3])

    print(f"Applying extension results from {run_id} to {setting}_v{version}")

    result = apply_extension_results(run_id, setting, version)

    print(f"\nResults:")
    print(f"  Tickets applied: {result['tickets_applied']}")
    print(f"  Failed resolutions: {result['failed_resolutions']}")
    print(f"  Bridges added: {result['bridges_added']}")
    print(f"  Edges added: {result['edges_added']}")
    print(f"  Total nodes: {result['total_nodes']}")
    print(f"  Total edges: {result['total_edges']}")

    # Emotion validation stats
    ev = result.get("emotion_validation", {})
    if ev.get("total", 0) > 0:
        print(f"\nEmotion schema validation:")
        print(f"  Valid (canonical): {ev['valid']}/{ev['total']} ({ev['valid']/ev['total']*100:.1f}%)")
        print(f"  Mapped (known variant): {ev['mapped']}")
        print(f"  Defaulted (unknown): {ev['defaulted']}")
        print(f"  Defect rate: {ev['defect_rate']*100:.1f}%")
        print(f"  Recovery rate: {ev['recovery_rate']*100:.1f}%")
        if ev.get("unknown_emotions"):
            print(f"  Unknown emotions (consider adding to map):")
            for emo, count in sorted(ev["unknown_emotions"].items(), key=lambda x: -x[1])[:5]:
                print(f"    {emo}: {count}")


if __name__ == "__main__":
    main()
