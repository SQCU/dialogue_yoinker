#!/usr/bin/env python3
"""
Topic Hub Aggregation

Creates hub nodes that aggregate arcs by synthetic_topic, mimicking the
DIAL→INFO structure in reference games where topics are natural hubs.

Before:
  arc1: A → B → C  (topic: "debt_collection")
  arc2: D → E → F  (topic: "debt_collection")
  arc3: G → H      (topic: "property_dispute")

After:
  hub_debt → A, D  (topic hub fans out to arc entries)
  A → B → C
  D → E → F
  hub_prop → G
  G → H

Usage:
    python scripts/aggregate_topic_hubs.py gallia:6
    python scripts/aggregate_topic_hubs.py gallia:6 --min-entries 2
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def aggregate_topic_hubs(
    graph_path: Path,
    min_entries: int = 2,
    group_by: str = "arc_shape",
    dry_run: bool = False,
) -> dict:
    """
    Create hub nodes that aggregate arcs by a grouping field.

    Args:
        graph_path: Path to graph.json
        min_entries: Minimum arc entries per group to create a hub
        group_by: Field to group by ("arc_shape", "topic", or "both")
        dry_run: If True, don't write changes

    Returns:
        Stats about hubs created
    """
    graph = json.loads(graph_path.read_text())
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Track existing hub IDs to avoid duplicates
    existing_ids = {n["id"] for n in nodes}
    existing_hub_keys = {
        n.get("hub_key") for n in nodes if n.get("node_type") == "topic_hub"
    }

    # Group arc entry nodes by the specified field
    # Arc entries are nodes with beat_index == 0
    groups = defaultdict(list)
    for node in nodes:
        if node.get("beat_index") == 0:  # arc entry point
            if group_by == "arc_shape":
                key = node.get("arc_shape") or "unknown"
            elif group_by == "topic":
                key = node.get("topic") or node.get("arc_shape") or "unknown"
            elif group_by == "both":
                # Combine arc_shape and topic for finer grouping
                shape = node.get("arc_shape") or "unknown"
                topic = node.get("topic") or "unknown"
                key = f"{shape}:{topic}"
            else:
                key = node.get(group_by) or "unknown"

            # Skip if we already have a hub for this key
            if key not in existing_hub_keys:
                groups[key].append(node["id"])

    # Create hubs for groups with multiple entries
    new_nodes = []
    new_edges = []
    groups_aggregated = []

    for key, entry_ids in sorted(groups.items()):
        if len(entry_ids) < min_entries:
            continue

        # Generate deterministic hub ID
        hub_id = f"hub_{hashlib.sha256(key.encode()).hexdigest()[:12]}"

        # Skip if hub already exists
        if hub_id in existing_ids:
            continue

        hub_node = {
            "id": hub_id,
            "text": f"[{key}]",
            "emotion": "neutral",
            "node_type": "topic_hub",
            "hub_key": key,
            "group_by": group_by,
            "aggregates": len(entry_ids),
        }
        new_nodes.append(hub_node)
        existing_ids.add(hub_id)

        # Create edges from hub to each arc entry
        for entry_id in entry_ids:
            new_edges.append({
                "source": hub_id,
                "target": entry_id,
                "type": "topic_branch",
            })

        groups_aggregated.append({
            "key": key,
            "entries": len(entry_ids),
            "hub_id": hub_id,
        })

    result = {
        "hubs_created": len(new_nodes),
        "edges_added": len(new_edges),
        "groups_aggregated": groups_aggregated,
        "groups_skipped": len([k for k, ids in groups.items() if len(ids) < min_entries]),
        "group_by": group_by,
    }

    if not dry_run and new_nodes:
        graph["nodes"].extend(new_nodes)
        graph["edges"].extend(new_edges)
        graph_path.write_text(json.dumps(graph, indent=2))

        # Invalidate cache
        try:
            import httpx
            setting = graph_path.parent.name
            httpx.post(f"http://127.0.0.1:8000/api/cache/clear/synthetic/{setting}", timeout=5.0)
        except:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Aggregate arc entries into topic hubs")
    parser.add_argument("setting", help="Setting spec (e.g., 'gallia:6')")
    parser.add_argument("--min-entries", type=int, default=2,
                        help="Minimum entries per group to create hub (default: 2)")
    parser.add_argument("--group-by", default="arc_shape",
                        choices=["arc_shape", "topic", "both"],
                        help="Field to group by (default: arc_shape)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created without writing")

    args = parser.parse_args()

    # Parse setting spec
    if ":" in args.setting:
        setting, version = args.setting.split(":")
        version = int(version)
    else:
        setting = args.setting
        version = None
        # Find latest version
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

    # Get before stats
    graph = json.loads(graph_path.read_text())
    nodes_before = len(graph["nodes"])
    edges_before = len(graph["edges"])

    # Run aggregation
    result = aggregate_topic_hubs(graph_path, args.min_entries, args.group_by, args.dry_run)

    # Report
    action = "Would create" if args.dry_run else "Created"
    print(f"\n[{setting}_v{version}] Topic Hub Aggregation (group_by={args.group_by})")
    print(f"  {action} {result['hubs_created']} hub nodes")
    print(f"  {action} {result['edges_added']} topic_branch edges")
    print(f"  Skipped {result['groups_skipped']} groups with < {args.min_entries} entries")

    if result['groups_aggregated']:
        print(f"\n  Groups aggregated:")
        for g in sorted(result['groups_aggregated'], key=lambda x: -x['entries'])[:10]:
            print(f"    {g['key']}: {g['entries']} entries → {g['hub_id']}")
        if len(result['groups_aggregated']) > 10:
            print(f"    ... and {len(result['groups_aggregated']) - 10} more")

    if not args.dry_run and result['hubs_created'] > 0:
        graph = json.loads(graph_path.read_text())
        print(f"\n  Graph: {nodes_before} → {len(graph['nodes'])} nodes, "
              f"{edges_before} → {len(graph['edges'])} edges")


if __name__ == "__main__":
    main()
