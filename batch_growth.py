#!/usr/bin/env python3
"""
Batch Growth Runner

Runs stats-guided growth in batches, sampling many walks upfront
then processing translations. Designed to be resumable.

Usage:
    # Initial run - sample walks and save queue
    python batch_growth.py --setting gallia --version 3 --target-nodes 900 --sample-only

    # Process batch of translations (call repeatedly)
    python batch_growth.py --setting gallia --version 3 --process-batch 10

    # Check status
    python batch_growth.py --setting gallia --version 3 --status
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional

from growth_engine import GrowthEngine
from synthetic_versioning import get_latest_version, list_versions


RUNS_DIR = Path("runs")


@dataclass
class BatchState:
    """Tracks batch growth state."""
    run_id: str
    setting: str
    version: int
    target_nodes: int
    created_at: str

    # Queued walks waiting for translation
    pending_walks: List[dict]

    # Completed translations
    completed_count: int = 0
    total_nodes_added: int = 0
    total_edges_added: int = 0

    def path(self) -> Path:
        return RUNS_DIR / self.run_id

    def save(self):
        self.path().mkdir(parents=True, exist_ok=True)
        state_file = self.path() / "batch_state.json"
        state_file.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, run_id: str) -> Optional['BatchState']:
        state_file = RUNS_DIR / run_id / "batch_state.json"
        if not state_file.exists():
            return None
        data = json.loads(state_file.read_text())
        return cls(**data)

    @classmethod
    def find_latest(cls, setting: str, version: int) -> Optional['BatchState']:
        """Find the latest batch state for a setting/version."""
        pattern = f"batch_*_{setting}_v{version}"
        candidates = sorted(RUNS_DIR.glob(pattern), reverse=True)
        for run_dir in candidates:
            state = cls.load(run_dir.name)
            if state:
                return state
        return None


def make_json_serializable(obj):
    """Convert sets and other non-JSON types to serializable forms."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    return obj


def sample_walks_for_target(engine: GrowthEngine, target_nodes: int) -> List[dict]:
    """Sample enough walks to reach target node count."""
    walks = []
    estimated_nodes = 0

    # Get all gaps
    gaps = engine.identify_gaps(top_n=20)

    print(f"Sampling walks to reach {target_nodes} nodes...")
    print(f"Top gaps: {[str(g)[:50] for g in gaps[:5]]}")

    iteration = 0
    while estimated_nodes < target_nodes and iteration < 500:
        # Weight gaps by size
        if not gaps:
            break

        gap = random.choices(gaps, weights=[g.gap_size for g in gaps])[0]

        # Sample walk for this gap
        sampled = engine.reference.sample_walks_for_gap(gap, n=1)
        if not sampled:
            continue

        walk = sampled[0]
        walk['gap_targeted'] = str(gap)

        # Make JSON serializable
        walk = make_json_serializable(walk)

        # Estimate nodes this will add
        walk_nodes = len(walk.get('texts', []))
        if walk_nodes < 2:
            continue

        walks.append(walk)
        estimated_nodes += walk_nodes
        iteration += 1

        if iteration % 50 == 0:
            print(f"  Sampled {len(walks)} walks, ~{estimated_nodes} nodes")

    print(f"Sampled {len(walks)} walks, estimated {estimated_nodes} nodes")
    return walks


def create_batch_run(setting: str, version: int, target_nodes: int) -> BatchState:
    """Create a new batch run with sampled walks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"batch_{timestamp}_{setting}_v{version}"

    # Load engine
    engine = GrowthEngine(setting=setting, version=version)

    # Sample walks
    walks = sample_walks_for_target(engine, target_nodes)

    # Create state
    state = BatchState(
        run_id=run_id,
        setting=setting,
        version=version,
        target_nodes=target_nodes,
        created_at=datetime.now(timezone.utc).isoformat(),
        pending_walks=walks,
    )
    state.save()

    # Save walks as individual request files for agent processing
    requests_dir = state.path() / "requests"
    requests_dir.mkdir(exist_ok=True)

    bible_path = Path("bibles") / f"{setting}.yaml"
    bible_content = bible_path.read_text() if bible_path.exists() else ""

    for i, walk in enumerate(walks):
        request = {
            "index": i,
            "triplet": {
                "arc_shape": engine._classify_arc_shape(walk.get('emotions', [])),
                "emotions": walk.get('emotions', []),
                "source_texts": walk.get('texts', []),
                "source_game": walk.get('game', 'unknown'),
                "gap_targeted": walk.get('gap_targeted', ''),
            },
            "bible_excerpt": bible_content[:3000],
            "status": "pending",
        }
        (requests_dir / f"request_{i:04d}.json").write_text(json.dumps(request, indent=2))

    print(f"\nCreated batch run: {run_id}")
    print(f"  {len(walks)} walks queued")
    print(f"  Requests saved to: {requests_dir}/")

    return state


def get_next_pending_requests(state: BatchState, batch_size: int) -> List[dict]:
    """Get the next batch of pending requests."""
    requests_dir = state.path() / "requests"
    pending = []

    for request_file in sorted(requests_dir.glob("request_*.json")):
        request = json.loads(request_file.read_text())
        if request.get("status") == "pending":
            request["file_path"] = str(request_file)
            pending.append(request)
            if len(pending) >= batch_size:
                break

    return pending


def mark_request_complete(request_path: str, translated_texts: List[str], confidence: float):
    """Mark a request as completed with translation results."""
    request = json.loads(Path(request_path).read_text())
    request["status"] = "completed"
    request["translated_texts"] = translated_texts
    request["confidence"] = confidence
    request["completed_at"] = datetime.now(timezone.utc).isoformat()
    Path(request_path).write_text(json.dumps(request, indent=2))


def apply_completed_translations(state: BatchState) -> dict:
    """Apply all completed translations to the graph."""
    from growth_engine import GrowthEngine

    engine = GrowthEngine(setting=state.setting, version=state.version)
    requests_dir = state.path() / "requests"

    applied = 0
    nodes_added = 0
    edges_added = 0

    for request_file in sorted(requests_dir.glob("request_*.json")):
        request = json.loads(request_file.read_text())

        if request.get("status") != "completed":
            continue
        if request.get("applied"):
            continue

        triplet = request["triplet"]
        # Handle both flat and nested translation formats
        translated_texts = request.get("translated_texts", [])
        if not translated_texts and "translation" in request:
            translated_texts = request["translation"].get("translated_texts", [])
        confidence = request.get("confidence", 0.5)
        if not confidence and "translation" in request:
            confidence = request["translation"].get("confidence", 0.5)

        if not translated_texts:
            continue

        # Build translated walk
        translated = {
            'arc_shape': triplet['arc_shape'],
            'emotions': triplet['emotions'],
            'source_texts': triplet['source_texts'],
            'translated_texts': translated_texts,
            'source_game': triplet['source_game'],
            'gap_targeted': triplet['gap_targeted'],
            'confidence': confidence,
        }

        # Find attachment point (prefer variety)
        attachment = engine.find_attachment_point({'emotions': triplet['emotions']})

        # Add to graph
        result = engine.add_translated_walk(translated, attachment)
        nodes_added += result['nodes_added']
        edges_added += result['edges_added']

        # Mark as applied
        request["applied"] = True
        request_file.write_text(json.dumps(request, indent=2))
        applied += 1

    # Save graph
    if applied > 0:
        engine.save()
        state.completed_count += applied
        state.total_nodes_added += nodes_added
        state.total_edges_added += edges_added
        state.save()

    return {
        "applied": applied,
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "total_nodes": len(engine.nodes),
        "total_edges": len(engine.edges),
    }


def get_status(state: BatchState) -> dict:
    """Get current batch status."""
    requests_dir = state.path() / "requests"

    pending = 0
    completed = 0
    applied = 0

    for request_file in requests_dir.glob("request_*.json"):
        request = json.loads(request_file.read_text())
        status = request.get("status", "pending")
        if status == "pending":
            pending += 1
        elif status == "completed":
            if request.get("applied"):
                applied += 1
            else:
                completed += 1

    return {
        "run_id": state.run_id,
        "setting": state.setting,
        "version": state.version,
        "target_nodes": state.target_nodes,
        "pending": pending,
        "completed_not_applied": completed,
        "applied": applied,
        "total_walks": pending + completed + applied,
        "current_nodes": state.total_nodes_added,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch growth runner")
    parser.add_argument("--setting", default="gallia")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--target-nodes", type=int, default=900)
    parser.add_argument("--sample-only", action="store_true",
                       help="Only sample walks, don't translate")
    parser.add_argument("--process-batch", type=int, default=0,
                       help="Process N translations")
    parser.add_argument("--apply", action="store_true",
                       help="Apply completed translations to graph")
    parser.add_argument("--status", action="store_true")

    args = parser.parse_args()

    # Find or create batch state
    state = BatchState.find_latest(args.setting, args.version)

    if args.status:
        if not state:
            print(f"No batch run found for {args.setting}_v{args.version}")
            return
        status = get_status(state)
        print(f"Batch status: {args.setting}_v{args.version}")
        for k, v in status.items():
            print(f"  {k}: {v}")
        return

    if args.sample_only or not state:
        state = create_batch_run(args.setting, args.version, args.target_nodes)
        if args.sample_only:
            return

    if args.process_batch > 0:
        pending = get_next_pending_requests(state, args.process_batch)
        print(f"Next {len(pending)} requests to translate:")
        for req in pending:
            triplet = req['triplet']
            print(f"  [{req['index']}] {triplet['source_game']} - {triplet['arc_shape']} - {len(triplet['source_texts'])} beats")
            print(f"      First: {triplet['source_texts'][0][:60]}...")
        return

    if args.apply:
        result = apply_completed_translations(state)
        print(f"Applied {result['applied']} translations")
        print(f"  Nodes: {result['nodes_added']} added, {result['total_nodes']} total")
        print(f"  Edges: {result['edges_added']} added, {result['total_edges']} total")
        return


if __name__ == "__main__":
    main()
