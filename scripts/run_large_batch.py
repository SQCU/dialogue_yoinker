#!/usr/bin/env python3
"""
Large-scale batch processing through DeepSeek API.

Runs many translation batches concurrently for maximum throughput.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python scripts/run_large_batch.py --setting gallia --translate 2100 --batch-size 50 --concurrency 10
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "http://127.0.0.1:8000"


async def create_run(client: httpx.AsyncClient, setting: str, sample_count: int, source_games: list) -> str:
    """Create a new translation run."""
    resp = await client.post(f"{API_BASE}/api/runs", json={
        "target_bible": setting,
        "sample_count": sample_count,
        "source_games": source_games,
    })
    return resp.json()["run_id"]


async def process_run_tickets(
    run_id: str,
    api_key: str,
    http_client: httpx.AsyncClient,
    concurrency: int = 5
) -> dict:
    """Process all tickets for a run (parse + translate)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    results = {"parse": 0, "translate": 0, "errors": 0}

    for worker_type in ["structural_parser", "translation_engine"]:
        while True:
            # Claim batch of tickets
            claimed = []
            for _ in range(concurrency):
                try:
                    resp = await http_client.post(
                        f"{API_BASE}/api/runs/{run_id}/claim",
                        json={"worker_type": worker_type}
                    )
                    claim = resp.json()
                    if not claim.get("success"):
                        break
                    claimed.append(claim["ticket"])
                except Exception:
                    break

            if not claimed:
                break

            # Process concurrently
            async def process_one(ticket: dict) -> bool:
                try:
                    input_data = ticket["input_data"]

                    if worker_type == "structural_parser":
                        prompt = format_parse_prompt(input_data)
                    else:
                        prompt = format_translate_prompt(input_data)

                    response = await client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": get_system_prompt(worker_type)},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1024 if worker_type == "translation_engine" else 512,
                        temperature=0.7,
                    )

                    content = response.choices[0].message.content.strip()
                    output = parse_json_response(content)

                    await http_client.post(
                        f"{API_BASE}/api/runs/{run_id}/submit",
                        json={
                            "ticket_id": ticket["ticket_id"],
                            "output_data": output,
                            "worker_concerns": [],
                            "worker_backend": "deepseek-chat",
                        }
                    )
                    return True
                except Exception as e:
                    return False

            batch_results = await asyncio.gather(*[process_one(t) for t in claimed])
            successes = sum(1 for r in batch_results if r)

            if worker_type == "structural_parser":
                results["parse"] += successes
            else:
                results["translate"] += successes
            results["errors"] += len(batch_results) - successes

    return results


def get_system_prompt(worker_type: str) -> str:
    if worker_type == "structural_parser":
        return "You extract structural arcs from dialogue. Output valid JSON only."
    else:
        return "You translate dialogue to new settings while preserving emotional structure. Output valid JSON only."


def format_parse_prompt(input_data: dict) -> str:
    walk = input_data.get("walk", [])
    walk_text = "\n".join([f"[{n.get('emotion', 'neutral')}] {n.get('text', '')}" for n in walk])

    return f"""Analyze this dialogue walk and extract its structural arc.

DIALOGUE:
{walk_text}

Output JSON with:
- arc_shape: one of (information_dump, escalating_threat, negotiation, confession, ambient_chatter, quest_hook, skill_check_reward)
- arc: list of beats, each with {{text, emotion, beat_function}}
- beat_functions: establish_context, deliver_information, build_tension, climax, resolution, etc.

Output ONLY valid JSON."""


def format_translate_prompt(input_data: dict) -> str:
    triplet = input_data.get("triplet", {})
    bible = input_data.get("bible_excerpt", "")[:2000]
    arc = triplet.get("arc", [])

    arc_text = "\n".join([f"[{b.get('emotion', 'neutral')}] {b.get('text', '')}" for b in arc])

    return f"""Translate this dialogue to the target setting while preserving emotional arc.

TARGET SETTING:
{bible}

SOURCE DIALOGUE (preserve emotions and beat structure):
{arc_text}

Output JSON with:
- translated_texts: list of translated lines (same count as source)
- confidence: 0.0-1.0

Output ONLY valid JSON."""


def parse_json_response(content: str) -> dict:
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content)


async def run_batch(
    batch_id: int,
    setting: str,
    sample_count: int,
    source_games: list,
    api_key: str,
    concurrency: int
) -> dict:
    """Run a single batch of translations."""
    # Stagger start to avoid timestamp collisions
    await asyncio.sleep(batch_id % 5 * 0.5)

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        try:
            run_id = await create_run(http_client, setting, sample_count, source_games)
            results = await process_run_tickets(run_id, api_key, http_client, concurrency)
            return {
                "batch_id": batch_id,
                "run_id": run_id,
                "success": True,
                **results
            }
        except Exception as e:
            return {
                "batch_id": batch_id,
                "success": False,
                "error": str(e)
            }


async def main():
    parser = argparse.ArgumentParser(description="Large-scale DeepSeek batch processing")
    parser.add_argument("--setting", default="gallia", help="Target setting")
    parser.add_argument("--translate", type=int, required=True, help="Total translations to run")
    parser.add_argument("--batch-size", type=int, default=50, help="Samples per batch/run")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent ticket processing")
    parser.add_argument("--parallel-batches", type=int, default=3, help="Parallel batch runs")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    total = args.translate
    batch_size = args.batch_size
    num_batches = (total + batch_size - 1) // batch_size

    print(f"Running {total} translations for {args.setting}")
    print(f"  {num_batches} batches of {batch_size}")
    print(f"  {args.parallel_batches} parallel batches, {args.concurrency} concurrent tickets each")
    print(f"  Started: {datetime.now().isoformat()}")
    print()

    source_games = ["oblivion", "falloutnv"]

    total_parse = 0
    total_translate = 0
    total_errors = 0
    completed_batches = 0

    # Process batches in parallel groups
    for group_start in range(0, num_batches, args.parallel_batches):
        group_end = min(group_start + args.parallel_batches, num_batches)
        group_size = group_end - group_start

        print(f"Processing batches {group_start+1}-{group_end} of {num_batches}...")

        # Calculate samples for this group (last batch may be smaller)
        batch_tasks = []
        for i in range(group_start, group_end):
            remaining = total - (i * batch_size)
            samples = min(batch_size, remaining)
            if samples > 0:
                batch_tasks.append(
                    run_batch(i, args.setting, samples, source_games, api_key, args.concurrency)
                )

        # Run group concurrently
        results = await asyncio.gather(*batch_tasks)

        for r in results:
            if r["success"]:
                total_parse += r.get("parse", 0)
                total_translate += r.get("translate", 0)
                total_errors += r.get("errors", 0)
                completed_batches += 1
                print(f"  Batch {r['batch_id']+1}: {r['run_id']} - parse={r['parse']}, translate={r['translate']}")
            else:
                print(f"  Batch {r['batch_id']+1}: FAILED - {r.get('error', 'unknown')}")

    print()
    print(f"{'='*60}")
    print(f"COMPLETE: {datetime.now().isoformat()}")
    print(f"  Batches: {completed_batches}/{num_batches}")
    print(f"  Parse tickets: {total_parse}")
    print(f"  Translate tickets: {total_translate}")
    print(f"  Errors: {total_errors}")
    print()
    print("Run `python scripts/apply_queue_results.py --all` to apply to graph")


if __name__ == "__main__":
    asyncio.run(main())
