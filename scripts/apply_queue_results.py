#!/usr/bin/env python3
"""
Apply Ticket Queue Results to Synthetic Graph

Reads completed translations from ticket queue runs and applies them
to the synthetic graph, matching what batch_growth.py does for file-based runs.

Usage:
    python scripts/apply_queue_results.py run_20251224_061535_gallia
    python scripts/apply_queue_results.py --all  # Apply all unapplied runs
    python scripts/apply_queue_results.py --list  # Show runs with status
"""

import json
import argparse
from pathlib import Path
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from growth_engine import GrowthEngine


RUNS_DIR = Path("runs")


def load_queue(run_id: str) -> Optional[dict]:
    """Load queue.json for a run."""
    queue_file = RUNS_DIR / run_id / "queue.json"
    if not queue_file.exists():
        return None
    return json.loads(queue_file.read_text())


def save_queue(run_id: str, queue: dict):
    """Save queue.json back with applied markers."""
    queue_file = RUNS_DIR / run_id / "queue.json"
    queue_file.write_text(json.dumps(queue, indent=2))


def extract_translation_data(translate_ticket: dict) -> Optional[dict]:
    """
    Extract the data needed for add_translated_walk from a translate ticket.

    Returns None if ticket isn't usable (missing data, not completed, etc.)
    """
    if translate_ticket.get("status") != "completed":
        return None

    if translate_ticket.get("applied"):
        return None  # Already applied

    output = translate_ticket.get("output_data")
    if not output:
        return None

    translated_texts = output.get("translated_texts", [])
    if not translated_texts:
        return None

    # Check for beat metadata vs actual prose
    if translated_texts and isinstance(translated_texts[0], dict):
        # Output was beat metadata, not prose - skip
        return None

    input_data = translate_ticket.get("input_data", {})
    triplet = input_data.get("triplet", {})

    # Build arc info from triplet
    arc = triplet.get("arc", [])
    emotions = [beat.get("emotion", "neutral") for beat in arc]
    source_texts = [beat.get("text", "") for beat in arc]

    # If source_texts came from original_walk passthrough
    if not source_texts or not any(source_texts):
        source_texts = input_data.get("source_texts", [])

    arc_shape = triplet.get("arc_shape", "information_dump")
    source_game = input_data.get("source_game", "unknown")
    confidence = output.get("confidence", 0.7)

    # Validate beat count matches
    if len(translated_texts) != len(emotions) and len(emotions) > 0:
        print(f"  Warning: Beat count mismatch in {translate_ticket.get('ticket_id')}: "
              f"{len(translated_texts)} translations vs {len(emotions)} emotions")
        # Try to use shorter list
        min_len = min(len(translated_texts), len(emotions))
        translated_texts = translated_texts[:min_len]
        emotions = emotions[:min_len]
        source_texts = source_texts[:min_len] if source_texts else []

    return {
        "arc_shape": arc_shape,
        "emotions": emotions,
        "source_texts": source_texts,
        "translated_texts": translated_texts,
        "source_game": source_game,
        "confidence": confidence,
        "gap_targeted": triplet.get("gap_targeted", ""),
        "worker_backend": translate_ticket.get("worker_backend"),
    }


def apply_queue_translations(run_id: str, dry_run: bool = False) -> dict:
    """
    Apply completed translations from a ticket queue run to the synthetic graph.

    Returns summary of what was applied.
    """
    queue = load_queue(run_id)
    if not queue:
        return {"error": f"Run not found: {run_id}"}

    target_bible = queue.get("target_bible", "gallia")

    # Infer version from config or default based on setting
    config = queue.get("config", {})
    if "version" in config:
        version = config["version"]
    elif target_bible == "gallia":
        version = 3  # gallia is on v3
    else:
        version = 1  # new settings start at v1

    translate_tickets = queue.get("translate_tickets", [])

    # Count what we have
    completed = [t for t in translate_tickets if t.get("status") == "completed"]
    already_applied = [t for t in completed if t.get("applied")]
    to_apply = [t for t in completed if not t.get("applied")]

    print(f"Run: {run_id}")
    print(f"  Target: {target_bible} v{version}")
    print(f"  Translate tickets: {len(translate_tickets)} total, "
          f"{len(completed)} completed, {len(already_applied)} already applied")
    print(f"  To apply: {len(to_apply)}")

    if not to_apply:
        return {
            "run_id": run_id,
            "applied": 0,
            "message": "Nothing to apply",
        }

    if dry_run:
        return {
            "run_id": run_id,
            "would_apply": len(to_apply),
            "dry_run": True,
        }

    # Load growth engine
    engine = GrowthEngine(setting=target_bible, version=version)
    print(f"  Loaded graph: {len(engine.nodes)} nodes, {len(engine.edges)} edges")

    applied = 0
    nodes_added = 0
    edges_added = 0
    skipped = 0
    backends_used = set()

    for ticket in to_apply:
        data = extract_translation_data(ticket)
        if not data:
            skipped += 1
            continue

        if data.get("worker_backend"):
            backends_used.add(data["worker_backend"])

        # Find attachment point
        attachment = engine.find_attachment_point({"emotions": data["emotions"]})

        # Add to graph
        try:
            result = engine.add_translated_walk(data, attachment)
            nodes_added += result.get("nodes_added", 0)
            edges_added += result.get("edges_added", 0)

            # Mark as applied
            ticket["applied"] = True
            applied += 1

        except Exception as e:
            print(f"  Error applying {ticket['ticket_id']}: {e}")
            skipped += 1

    # Save graph
    if applied > 0:
        engine.save()
        print(f"  Applied {applied} translations, added {nodes_added} nodes, {edges_added} edges")
        print(f"  Graph now: {len(engine.nodes)} nodes, {len(engine.edges)} edges")

    # Save queue with applied markers
    save_queue(run_id, queue)

    return {
        "run_id": run_id,
        "target": target_bible,
        "version": version,
        "applied": applied,
        "skipped": skipped,
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "total_nodes": len(engine.nodes),
        "total_edges": len(engine.edges),
        "backends_used": list(backends_used),
    }


def list_queue_runs() -> list[dict]:
    """List all ticket queue runs with their status."""
    runs = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        queue_file = run_dir / "queue.json"
        if not queue_file.exists():
            continue

        # Skip batch runs (they have batch_state.json)
        if (run_dir / "batch_state.json").exists():
            continue

        queue = json.loads(queue_file.read_text())

        translate_tickets = queue.get("translate_tickets", [])
        completed = sum(1 for t in translate_tickets if t.get("status") == "completed")
        applied = sum(1 for t in translate_tickets if t.get("applied"))

        # Get backends used
        backends = set()
        for t in translate_tickets:
            if t.get("worker_backend"):
                backends.add(t["worker_backend"])

        runs.append({
            "run_id": queue.get("run_id", run_dir.name),
            "target_bible": queue.get("target_bible"),
            "created_at": queue.get("created_at", ""),
            "translate_total": len(translate_tickets),
            "translate_completed": completed,
            "translate_applied": applied,
            "unapplied": completed - applied,
            "backends": list(backends),
        })

    return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Apply ticket queue results to synthetic graph")
    parser.add_argument("run_id", nargs="?", help="Run ID to apply")
    parser.add_argument("--all", action="store_true", help="Apply all unapplied runs")
    parser.add_argument("--list", action="store_true", help="List runs with status")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be applied")

    args = parser.parse_args()

    if args.list:
        runs = list_queue_runs()
        print(f"{'Run ID':<45} {'Target':<10} {'Completed':<10} {'Applied':<10} {'Pending':<10} {'Backends'}")
        print("-" * 115)
        for r in runs:
            print(f"{r['run_id']:<45} {r['target_bible']:<10} "
                  f"{r['translate_completed']:<10} {r['translate_applied']:<10} "
                  f"{r['unapplied']:<10} {', '.join(r['backends']) or '-'}")
        return

    if args.all:
        runs = list_queue_runs()
        unapplied = [r for r in runs if r["unapplied"] > 0]

        if not unapplied:
            print("No unapplied runs found")
            return

        print(f"Found {len(unapplied)} runs with unapplied translations")

        for r in unapplied:
            print(f"\n{'='*60}")
            result = apply_queue_translations(r["run_id"], dry_run=args.dry_run)
            print(json.dumps(result, indent=2))
        return

    if not args.run_id:
        parser.print_help()
        return

    result = apply_queue_translations(args.run_id, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
