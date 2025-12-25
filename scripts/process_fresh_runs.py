#!/usr/bin/env python3
"""
Process fresh translation runs through DeepSeek, then create versioned synthetic graphs.

Usage:
    DEEPSEEK_API_KEY="sk-..." python scripts/process_fresh_runs.py
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from workflow.multi_backend import WorkerDispatcher
from synthetic_versioning import new_graph, extend_graph

API_BASE = "http://127.0.0.1:8000"


async def process_run(run_id: str, dispatcher: WorkerDispatcher) -> dict:
    """Process a run through parse→translate phases."""
    print(f"\nProcessing run: {run_id}")

    worker = dispatcher.get_worker("deepseek-chat")

    # Run parse phase
    print("  Running structural parser...")
    parse_result = await worker.process_tickets(run_id, "structural_parser")
    print(f"    Parse: {parse_result}")

    # Run translate phase
    print("  Running translation engine...")
    translate_result = await worker.process_tickets(run_id, "translation_engine")
    print(f"    Translate: {translate_result}")

    # Get final status
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{API_BASE}/api/runs/{run_id}/status")
        status = resp.json()

    return {
        "run_id": run_id,
        "parse_completed": status["parse"]["completed"],
        "translate_completed": status["translate"]["completed"],
    }


def compile_translations_to_graph(run_id: str, setting: str) -> dict:
    """Extract translations from run and compile into synthetic graph."""
    from pathlib import Path

    queue_path = Path("runs") / run_id / "queue.json"
    if not queue_path.exists():
        raise FileNotFoundError(f"Queue not found: {queue_path}")

    queue_data = json.loads(queue_path.read_text())
    translate_tickets = queue_data.get("translate_tickets", [])

    nodes = []
    edges = []

    for ticket in translate_tickets:
        if ticket.get("status") != "completed":
            continue

        output = ticket.get("output_data", {})
        if not output:
            continue

        # Get generated texts and emotions
        generated_texts = output.get("generated_texts", [])
        arc = ticket.get("input_data", {}).get("triplet", {}).get("arc", [])

        if not generated_texts or not arc:
            continue

        # Create nodes for each beat
        prev_node_id = None
        for i, (text, beat) in enumerate(zip(generated_texts, arc)):
            import hashlib
            ticket_id = ticket["ticket_id"]
            node_id = f"syn_{hashlib.sha256(f'{run_id}_{ticket_id}_{i}'.encode()).hexdigest()[:12]}"

            nodes.append({
                "id": node_id,
                "text": text,
                "emotion": beat.get("emotion", "neutral"),
                "beat_function": beat.get("function"),
                "archetype_relation": beat.get("archetype_relation"),
                "source_run": run_id,
                "source_ticket": ticket["ticket_id"],
            })

            if prev_node_id:
                edges.append({
                    "source": prev_node_id,
                    "target": node_id,
                    "type": "sequential",
                })

            prev_node_id = node_id

    return {"nodes": nodes, "edges": edges}


async def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    # Define runs to process
    runs = [
        {"run_id": "run_20251225_041927_gallia", "setting": "gallia", "target_version": 4},
        {"run_id": "run_20251225_043935_marmotte", "setting": "marmotte", "target_version": 2},
    ]

    dispatcher = WorkerDispatcher(API_BASE)

    for run_info in runs:
        run_id = run_info["run_id"]
        setting = run_info["setting"]
        target_version = run_info["target_version"]

        # Process through DeepSeek
        result = await process_run(run_id, dispatcher)
        print(f"\nRun {run_id} completed:")
        print(f"  Parse: {result['parse_completed']}")
        print(f"  Translate: {result['translate_completed']}")

        # Compile to graph
        print(f"\nCompiling translations to {setting}_v{target_version}...")
        graph_data = compile_translations_to_graph(run_id, setting)

        # Create new version
        version = new_graph(
            setting=setting,
            description=f"Fresh v{target_version} with new two-layer pipeline. Run: {run_id}",
            approach="two_layer_pipeline",
            source_games=["oblivion", "falloutnv"],
        )

        # Save graph
        graph_path = version.path() / "graph.json"
        graph_path.write_text(json.dumps(graph_data, indent=2))

        # Update metadata
        version.total_nodes = len(graph_data["nodes"])
        version.total_edges = len(graph_data["edges"])
        version.run_ids = [run_id]

        meta_path = version.path() / "metadata.json"
        from dataclasses import asdict
        meta_path.write_text(json.dumps(asdict(version), indent=2))

        print(f"  Created {setting}_v{version.version}:")
        print(f"    Nodes: {version.total_nodes}")
        print(f"    Edges: {version.total_edges}")


if __name__ == "__main__":
    asyncio.run(main())
