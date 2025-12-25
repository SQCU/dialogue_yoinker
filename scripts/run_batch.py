#!/usr/bin/env python3
"""
High-concurrency batch runner for dialogue generation pipeline.

Usage:
    # Translate 100 samples for gallia
    python scripts/run_batch.py translate gallia 100

    # Link 100 nodes in gallia_v4
    python scripts/run_batch.py link gallia 4 100

    # Extend 100 candidates in gallia_v4
    python scripts/run_batch.py extend gallia 4 100 --source-run link_20251225_...

    # Full 100/100/100 pipeline
    python scripts/run_batch.py full gallia 4 100

    # Process multiple settings in parallel
    python scripts/run_batch.py translate gallia,marmotte 100 --parallel

Environment:
    DEEPSEEK_API_KEY - Required API key
    CONCURRENCY - Concurrent requests (default: 25)
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_CONCURRENCY = int(os.environ.get("CONCURRENCY", "25"))


async def create_translation_run(
    client: httpx.AsyncClient, setting: str, count: int, version: int = 0, guided: bool = False
) -> str:
    """Create a translation run and return run_id."""
    payload = {
        "target_bible": setting,
        "sample_count": count,
        "target_version": version,
        "guided": guided,
    }
    resp = await client.post(f"{API_BASE}/api/runs", json=payload)
    resp.raise_for_status()
    data = resp.json()
    mode = "[guided]" if guided else "[random]"
    print(f"{mode}[{setting}] Created {data['run_id']} with {data['tickets_created']} parse tickets")
    return data["run_id"]


async def create_linking_run(
    client: httpx.AsyncClient, setting: str, version: int, count: int, guided: bool = False
) -> str:
    """Create a linking run and return run_id."""
    payload = {
        "target_setting": setting,
        "version": version,
        "sample_count": count,
        "guided": guided,
    }
    resp = await client.post(f"{API_BASE}/api/runs/linking", json=payload)
    resp.raise_for_status()
    data = resp.json()
    mode = "[guided]" if guided else "[random]"
    print(f"{mode}[{setting}_v{version}] Created {data['run_id']} with {data['tickets_created']} link tickets")
    return data["run_id"]


async def create_extension_run(
    client: httpx.AsyncClient, source_run_id: str, setting: str, version: int, count: int
) -> str:
    """Create an extension run and return run_id."""
    resp = await client.post(
        f"{API_BASE}/api/runs/extension",
        json={
            "source_run_id": source_run_id,
            "sample_count": count,
            "target_setting": setting,
            "version": version,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"[{setting}_v{version}] Created {data['run_id']} with {data['tickets_created']} extension tickets")
    return data["run_id"]


async def process_tickets_via_dispatcher(
    run_id: str, worker_type: str, concurrency: int = DEFAULT_CONCURRENCY
) -> dict:
    """Process tickets using the multi-backend dispatcher."""
    from workflow.multi_backend import WorkerDispatcher

    dispatcher = WorkerDispatcher(API_BASE)
    worker = dispatcher.get_worker("deepseek-chat")

    start = time.time()
    result = await worker.process_tickets(run_id, worker_type, concurrency)
    elapsed = time.time() - start

    completed = result.get("completed", 0)
    rate = completed / elapsed if elapsed > 0 else 0
    print(f"[{run_id}] {worker_type}: {completed} completed in {elapsed:.1f}s ({rate:.1f}/s)")

    return result


async def process_link_stitch_direct(
    run_id: str, api_key: str, concurrency: int = DEFAULT_CONCURRENCY
) -> dict:
    """Process link_stitch tickets directly via DeepSeek API."""
    from scripts.run_link_stitch_batch import main as link_main

    start = time.time()
    await link_main(run_id, concurrency)
    elapsed = time.time() - start
    print(f"[{run_id}] link_stitch completed in {elapsed:.1f}s")
    return {"elapsed": elapsed}


async def process_extension_direct(
    run_id: str, api_key: str, concurrency: int = DEFAULT_CONCURRENCY
) -> dict:
    """Process extension_resolve tickets directly via DeepSeek API."""
    from scripts.run_extension_resolve_batch import main as ext_main

    start = time.time()
    await ext_main(run_id, concurrency)
    elapsed = time.time() - start
    print(f"[{run_id}] extension_resolve completed in {elapsed:.1f}s")
    return {"elapsed": elapsed}


def apply_link_results(run_id: str, setting: str, version: int) -> int:
    """Apply link results to graph. Returns extension candidates count."""
    from scripts.apply_link_stitch_results import apply_link_stitch_results

    result = apply_link_stitch_results(run_id, setting, version)
    print(f"[{setting}_v{version}] Links: +{result['bridges_added']} bridges, +{result['links_added']} edges")
    print(f"[{setting}_v{version}] Total: {result['total_nodes']} nodes, {result['total_edges']} edges")

    # Get extension candidates count from queue
    queue_path = Path("runs") / run_id / "queue.json"
    queue_data = json.loads(queue_path.read_text())
    return len(queue_data.get("extension_candidates", []))


def apply_extension_results(run_id: str, setting: str, version: int):
    """Apply extension results to graph."""
    from scripts.apply_extension_results import apply_extension_results as _apply

    result = _apply(run_id, setting, version)
    print(f"[{setting}_v{version}] Extensions: +{result['bridges_added']} bridges, +{result['edges_added']} edges")
    print(f"[{setting}_v{version}] Total: {result['total_nodes']} nodes, {result['total_edges']} edges")


def compile_translations(run_id: str, setting: str, version: int):
    """Compile translations from run into synthetic graph."""
    queue_path = Path("runs") / run_id / "queue.json"
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

        translated_texts = output.get("translated_texts", [])
        arc = ticket.get("input_data", {}).get("triplet", {}).get("arc", [])

        if not translated_texts or not arc:
            continue

        prev_node_id = None
        for i, (text, beat) in enumerate(zip(translated_texts, arc)):
            ticket_id = ticket["ticket_id"]
            node_id = f"syn_{hashlib.sha256(f'{run_id}_{ticket_id}_{i}'.encode()).hexdigest()[:12]}"

            nodes.append({
                "id": node_id,
                "text": text,
                "emotion": beat.get("emotion", "neutral"),
                "beat_function": beat.get("function"),
                "source_run": run_id,
            })

            if prev_node_id:
                edges.append({
                    "source": prev_node_id,
                    "target": node_id,
                    "type": "sequential",
                })

            prev_node_id = node_id

    # Load and merge into existing graph
    graph_path = Path(f"synthetic/{setting}_v{version}/graph.json")
    if graph_path.exists():
        existing = json.loads(graph_path.read_text())
        existing_ids = {n["id"] for n in existing["nodes"]}

        new_count = 0
        for node in nodes:
            if node["id"] not in existing_ids:
                existing["nodes"].append(node)
                existing_ids.add(node["id"])
                new_count += 1

        for edge in edges:
            existing["edges"].append(edge)

        graph_path.write_text(json.dumps(existing, indent=2))
        print(f"[{setting}_v{version}] Added {new_count} nodes, {len(edges)} edges")
        print(f"[{setting}_v{version}] Total: {len(existing['nodes'])} nodes, {len(existing['edges'])} edges")
    else:
        print(f"Graph not found: {graph_path}")



async def run_translate(
    setting_specs: list[tuple[str, int]], count: int, concurrency: int,
    parallel: bool = False, guided: bool = False
):
    """Run translation pipeline for one or more settings."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        if parallel:
            # Create all runs first
            run_ids = await asyncio.gather(*[
                create_translation_run(client, s, count, v, guided)
                for s, v in setting_specs
            ])

            # Process parse phase for all
            await asyncio.gather(*[
                process_tickets_via_dispatcher(rid, "structural_parser", concurrency)
                for rid in run_ids
            ])

            # Process translate phase for all
            await asyncio.gather(*[
                process_tickets_via_dispatcher(rid, "translation_engine", concurrency)
                for rid in run_ids
            ])

            return dict(zip([s for s, _ in setting_specs], run_ids))
        else:
            results = {}
            for setting, version in setting_specs:
                run_id = await create_translation_run(client, setting, count, version, guided)
                await process_tickets_via_dispatcher(run_id, "structural_parser", concurrency)
                await process_tickets_via_dispatcher(run_id, "translation_engine", concurrency)
                results[setting] = run_id
            return results


async def run_link(
    setting: str, version: int, count: int, concurrency: int, api_key: str,
    guided: bool = False
) -> tuple[str, int]:
    """Run linking pipeline and return (run_id, extension_candidates_count)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        run_id = await create_linking_run(client, setting, version, count, guided)

    await process_link_stitch_direct(run_id, api_key, concurrency)
    ext_count = apply_link_results(run_id, setting, version)

    return run_id, ext_count


async def run_extend(
    source_run_id: str, setting: str, version: int, count: int, concurrency: int, api_key: str
):
    """Run extension pipeline."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        run_id = await create_extension_run(client, source_run_id, setting, version, count)

    await process_extension_direct(run_id, api_key, concurrency)
    apply_extension_results(run_id, setting, version)


async def run_full_pipeline(
    setting: str, version: int, count: int, concurrency: int, api_key: str,
    guided: bool = False
):
    """Run full 100/100/100 pipeline: translate -> link -> extend."""
    mode = "GUIDED" if guided else "RANDOM"
    print(f"\n{'='*60}")
    print(f"FULL PIPELINE [{mode}]: {setting}_v{version} ({count}/{count}/{count})")
    print(f"{'='*60}\n")

    start = time.time()

    # Phase 1: Translate
    print(f"\n--- Phase 1: Translate {count} samples ---")
    async with httpx.AsyncClient(timeout=30.0) as client:
        run_id = await create_translation_run(client, setting, count, version, guided)

    await process_tickets_via_dispatcher(run_id, "structural_parser", concurrency)
    await process_tickets_via_dispatcher(run_id, "translation_engine", concurrency)
    compile_translations(run_id, setting, version)

    # Phase 2: Link
    print(f"\n--- Phase 2: Link {count} nodes ---")
    link_run_id, ext_count = await run_link(setting, version, count, concurrency, api_key, guided)
    print(f"[{setting}_v{version}] Extension candidates available: {ext_count}")

    # Phase 3: Extend
    ext_to_process = min(count, ext_count)
    print(f"\n--- Phase 3: Extend {ext_to_process} candidates ---")
    if ext_to_process > 0:
        await run_extend(link_run_id, setting, version, ext_to_process, concurrency, api_key)
    else:
        print(f"[{setting}_v{version}] No extension candidates to process")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"COMPLETE: {setting}_v{version} in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}\n")


def parse_setting_spec(spec: str) -> tuple[str, int | None]:
    """Parse setting:version spec (e.g., 'gallia:4' or 'gallia')."""
    if ":" in spec:
        parts = spec.split(":")
        return parts[0], int(parts[1])
    return spec, None


def get_latest_version(setting: str) -> int:
    """Find the latest version for a setting."""
    synthetic_dir = Path("synthetic")
    versions = []
    for p in synthetic_dir.glob(f"{setting}_v*"):
        if p.is_dir():
            try:
                v = int(p.name.split("_v")[1])
                versions.append(v)
            except (ValueError, IndexError):
                pass
    return max(versions) if versions else 1


def main():
    parser = argparse.ArgumentParser(
        description="High-concurrency batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Translate 100 samples for gallia (random sampling)
    python scripts/run_batch.py translate gallia 100

    # Translate with stats-guided sampling (closes topology gaps)
    python scripts/run_batch.py translate gallia:4 100 --guided

    # Full pipeline on gallia_v4 and marmotte_v2
    python scripts/run_batch.py full gallia:4,marmotte:2 100 --parallel

    # Full pipeline with guided sampling
    python scripts/run_batch.py full gallia:4 100 --guided

    # Compile translations from a specific run
    python scripts/run_batch.py compile run_20251225_gallia gallia:4
"""
    )
    parser.add_argument("command", choices=["translate", "link", "extend", "full", "compile"])
    parser.add_argument("settings", help="Setting specs (e.g., 'gallia:4,marmotte:2' or 'gallia,marmotte')")
    parser.add_argument("count", type=int, nargs="?", default=100, help="Sample count")
    parser.add_argument("--source-run", help="Source run ID for extend/compile commands")
    parser.add_argument("--parallel", action="store_true", help="Process settings in parallel")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--guided", action="store_true",
                        help="Use stats-guided sampling (closes topology gaps vs reference corpus)")

    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and args.command != "compile":
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    # Parse setting:version specs
    setting_specs = []
    for spec in args.settings.split(","):
        setting, version = parse_setting_spec(spec.strip())
        if version is None:
            version = get_latest_version(setting)
        setting_specs.append((setting, version))

    if args.command == "translate":
        asyncio.run(run_translate(setting_specs, args.count, args.concurrency, args.parallel, args.guided))

    elif args.command == "link":
        for setting, version in setting_specs:
            asyncio.run(run_link(setting, version, args.count, args.concurrency, api_key, args.guided))

    elif args.command == "extend":
        if not args.source_run:
            print("ERROR: --source-run required for extend command")
            sys.exit(1)
        for setting, version in setting_specs:
            asyncio.run(run_extend(args.source_run, setting, version, args.count, args.concurrency, api_key))

    elif args.command == "compile":
        if not args.source_run:
            print("ERROR: --source-run required for compile command")
            sys.exit(1)
        for setting, version in setting_specs:
            compile_translations(args.source_run, setting, version)

    elif args.command == "full":
        if args.parallel and len(setting_specs) > 1:
            async def run_all():
                await asyncio.gather(*[
                    run_full_pipeline(s, v, args.count, args.concurrency, api_key, args.guided)
                    for s, v in setting_specs
                ])
            asyncio.run(run_all())
        else:
            for setting, version in setting_specs:
                asyncio.run(run_full_pipeline(setting, version, args.count, args.concurrency, api_key, args.guided))


if __name__ == "__main__":
    main()
