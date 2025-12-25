#!/usr/bin/env python3
"""
DeepSeek Orchestration Script

Runs translate + link + extend pipeline using DeepSeek API.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python scripts/run_deepseek_orchestration.py --setting gallia --translate 25
    python scripts/run_deepseek_orchestration.py --setting marmotte --translate 10 --version 1
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import httpx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflow.multi_backend import WorkerDispatcher, OpenAICompatibleWorker, BACKEND_CONFIGS


API_BASE = "http://127.0.0.1:8000"


async def run_translate_phase(count: int, setting: str, dispatcher: WorkerDispatcher, api_base: str) -> dict:
    """Run translation phase: sample walks, parse, translate."""
    print(f"\n{'='*60}")
    print(f"TRANSLATE PHASE: {count} samples -> {setting}")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Create run
        resp = await client.post(f"{api_base}/api/runs", json={
            "target_bible": setting,
            "sample_count": count,
            "source_games": ["oblivion", "falloutnv"],
        })
        run_id = resp.json()["run_id"]
        print(f"Created run: {run_id}")

        # Run parse phase
        print("Running structural parser...")
        worker = dispatcher.get_worker("deepseek-chat")
        result = await worker.process_tickets(run_id, "structural_parser")
        print(f"  Parse: {result}")

        # Run translate phase
        print("Running translation engine...")
        result = await worker.process_tickets(run_id, "translation_engine")
        print(f"  Translate: {result}")

        # Get final status
        resp = await client.get(f"{api_base}/api/runs/{run_id}/status")
        status = resp.json()

    return {
        "run_id": run_id,
        "setting": setting,
        "parse_completed": status["parse"]["completed"],
        "translate_completed": status["translate"]["completed"],
        "curate_pending": status["curate"]["pending"],
    }


async def run_link_phase(count: int, setting: str, version: int, dispatcher: WorkerDispatcher) -> dict:
    """Run linking phase: generate bridge nodes."""
    print(f"\n{'='*60}")
    print(f"LINK PHASE: {count} bridges for {setting}_v{version}")
    print(f"{'='*60}")

    # Import graph linker
    try:
        from graph_linker import (
            load_synthetic_graph, load_bible_excerpt,
            generate_link_candidates, LinkState
        )
    except ImportError as e:
        return {"error": f"graph_linker not available: {e}"}

    # Load graph and generate candidates
    try:
        nodes, edges = load_synthetic_graph(setting, version)
        bible = load_bible_excerpt(setting)
        print(f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges")
    except FileNotFoundError:
        return {"error": f"No graph found for {setting}_v{version}"}

    candidates = generate_link_candidates(nodes, edges, max_candidates=count)
    print(f"Generated {len(candidates)} link candidates")

    if not candidates:
        return {"completed": 0, "message": "No link candidates found"}

    # Process each candidate with DeepSeek
    worker = dispatcher.get_worker("deepseek-chat")
    completed = 0
    bridges = []

    for i, candidate in enumerate(candidates[:count]):
        input_data = {
            "link": candidate.to_dict(),
            "bible_excerpt": bible[:2000],
        }

        # Format and call
        task = worker._format_task("bridge_generator", input_data)

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )

            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You generate bridge dialogue for synthetic graphs."},
                    {"role": "user", "content": task}
                ],
                max_tokens=256,
                temperature=0.7,
            )

            output = worker._parse_response(response.choices[0].message.content)

            if "bridge_text" in output:
                bridges.append({
                    "terminus_id": candidate.terminus_id,
                    "entry_id": candidate.entry_id,
                    "text": output["bridge_text"],
                    "emotion": output.get("bridge_emotion", "neutral"),
                })
                completed += 1
                print(f"  [{i+1}/{count}] Bridge: {output['bridge_text'][:50]}...")

        except Exception as e:
            print(f"  [{i+1}/{count}] Error: {e}")

    # Save bridges for later application
    if bridges:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bridges_path = Path("runs") / f"bridges_{setting}_{timestamp}.json"
        bridges_path.parent.mkdir(exist_ok=True)
        bridges_path.write_text(json.dumps(bridges, indent=2))
        print(f"Saved {len(bridges)} bridges to {bridges_path}")

    return {"completed": completed, "bridges": len(bridges), "setting": setting}


async def run_extend_phase(count: int, setting: str, dispatcher: WorkerDispatcher, api_base: str) -> dict:
    """Run extension phase: more translations from different source."""
    print(f"\n{'='*60}")
    print(f"EXTEND PHASE: {count} additional samples -> {setting}")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Create run with different source mix
        resp = await client.post(f"{api_base}/api/runs", json={
            "target_bible": setting,
            "sample_count": count,
            "source_games": ["skyrim_full"],  # Different source for variety
        })

        if resp.status_code != 200:
            # Fallback to other games
            resp = await client.post(f"{api_base}/api/runs", json={
                "target_bible": setting,
                "sample_count": count,
                "source_games": ["oblivion"],
            })

        run_id = resp.json()["run_id"]
        print(f"Created extension run: {run_id}")

        # Run parse phase
        print("Running structural parser...")
        worker = dispatcher.get_worker("deepseek-chat")
        result = await worker.process_tickets(run_id, "structural_parser")
        print(f"  Parse: {result}")

        # Run translate phase
        print("Running translation engine...")
        result = await worker.process_tickets(run_id, "translation_engine")
        print(f"  Translate: {result}")

        # Get final status
        resp = await client.get(f"{api_base}/api/runs/{run_id}/status")
        status = resp.json()

    return {
        "run_id": run_id,
        "setting": setting,
        "parse_completed": status["parse"]["completed"],
        "translate_completed": status["translate"]["completed"],
    }


async def main():
    parser = argparse.ArgumentParser(description="DeepSeek synthetic dialogue orchestration")
    parser.add_argument("--setting", default="gallia", help="Target setting (gallia, marmotte)")
    parser.add_argument("--version", type=int, default=1, help="Target version number")
    parser.add_argument("--translate", type=int, default=0, help="Number of translations")
    parser.add_argument("--link", type=int, default=0, help="Number of bridge links")
    parser.add_argument("--extend", type=int, default=0, help="Number of extensions")
    parser.add_argument("--api", default=API_BASE, help="API base URL")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    if args.translate == 0 and args.link == 0 and args.extend == 0:
        print("ERROR: Specify at least one of --translate, --link, or --extend")
        sys.exit(1)

    api_base = args.api
    dispatcher = WorkerDispatcher(api_base)

    print(f"Target setting: {args.setting} v{args.version}")

    results = {}

    # Phase 1: Translate
    if args.translate > 0:
        results["translate"] = await run_translate_phase(args.translate, args.setting, dispatcher, api_base)

    # Phase 2: Link
    if args.link > 0:
        results["link"] = await run_link_phase(args.link, args.setting, args.version, dispatcher)

    # Phase 3: Extend
    if args.extend > 0:
        results["extend"] = await run_extend_phase(args.extend, args.setting, dispatcher, api_base)

    # Summary
    print(f"\n{'='*60}")
    print("ORCHESTRATION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
