#!/usr/bin/env python3
"""
Retroactively fix non-canonical emotions in synthetic graphs.

Reads each graph.json, validates/maps emotions, writes back with:
- `emotion` field corrected to canonical value
- `emotion_raw` field preserving original if non-canonical

Usage:
    python scripts/fix_emotions.py              # Dry run
    python scripts/fix_emotions.py --apply      # Actually modify files
    python scripts/fix_emotions.py gallia:5     # Fix specific version
"""

import argparse
import json
from pathlib import Path

from scripts.emotion_schema import ValidationStats, validate_emotion

SYNTHETIC_DIR = Path("synthetic")


def fix_graph_emotions(graph_path: Path, apply: bool = False) -> dict:
    """Fix emotions in a single graph file."""
    if not graph_path.exists():
        return {"error": f"Not found: {graph_path}"}

    data = json.loads(graph_path.read_text())
    nodes = data.get("nodes", [])

    stats = ValidationStats()
    modified = 0

    for node in nodes:
        raw_emotion = node.get("emotion", "neutral")
        result = validate_emotion(raw_emotion)
        stats.record(result)

        if not result.was_valid:
            modified += 1
            node["emotion"] = result.canonical
            # Preserve original for debugging/analysis
            if "emotion_raw" not in node:
                node["emotion_raw"] = raw_emotion

    if apply and modified > 0:
        data["nodes"] = nodes
        graph_path.write_text(json.dumps(data, indent=2))

    return {
        "path": str(graph_path),
        "total_nodes": len(nodes),
        "modified": modified,
        "applied": apply,
        "stats": {
            "total": stats.total,
            "valid": stats.valid,
            "mapped": stats.mapped,
            "defaulted": stats.defaulted,
            "defect_rate": stats.defect_rate,
            "recovery_rate": stats.recovery_rate,
            "unknown_emotions": stats.unknown_emotions,
        },
    }


def find_all_graphs() -> list[Path]:
    """Find all synthetic graph.json files."""
    graphs = []
    for version_dir in sorted(SYNTHETIC_DIR.iterdir()):
        if version_dir.is_dir():
            graph_path = version_dir / "graph.json"
            if graph_path.exists():
                graphs.append(graph_path)
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Fix non-canonical emotions in synthetic graphs")
    parser.add_argument("targets", nargs="*", help="Specific targets (e.g., 'gallia:5')")
    parser.add_argument("--apply", action="store_true", help="Actually modify files (default: dry run)")
    args = parser.parse_args()

    if args.targets:
        # Specific targets
        graphs = []
        for target in args.targets:
            if ":" in target:
                setting, version = target.split(":")
                graph_path = SYNTHETIC_DIR / f"{setting}_v{version}" / "graph.json"
            else:
                graph_path = SYNTHETIC_DIR / f"{target}" / "graph.json"
            graphs.append(graph_path)
    else:
        # All graphs
        graphs = find_all_graphs()

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Emotion Schema Fixer [{mode}]")
    print("=" * 60)

    total_modified = 0
    all_unknown = {}

    for graph_path in graphs:
        result = fix_graph_emotions(graph_path, apply=args.apply)

        if "error" in result:
            print(f"\n{result['error']}")
            continue

        stats = result["stats"]
        name = graph_path.parent.name

        print(f"\n{name}:")
        print(f"  Nodes: {result['total_nodes']}")
        print(f"  Emotions: {stats['valid']} valid, {stats['mapped']} mapped, {stats['defaulted']} defaulted")
        print(f"  Defect rate: {stats['defect_rate']*100:.1f}%")

        if result["modified"] > 0:
            print(f"  Modified: {result['modified']} nodes {'(applied)' if args.apply else '(would change)'}")
            total_modified += result["modified"]

        # Aggregate unknown emotions
        for emo, count in stats.get("unknown_emotions", {}).items():
            all_unknown[emo] = all_unknown.get(emo, 0) + count

    print(f"\n{'=' * 60}")
    print(f"Total nodes that need fixing: {total_modified}")

    if all_unknown:
        print(f"\nUnknown emotions (consider adding to EMOTION_MAP):")
        for emo, count in sorted(all_unknown.items(), key=lambda x: -x[1])[:15]:
            print(f"  {emo}: {count}")

    if not args.apply and total_modified > 0:
        print(f"\nRun with --apply to actually modify the files")


if __name__ == "__main__":
    main()
