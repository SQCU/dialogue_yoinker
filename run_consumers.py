#!/usr/bin/env python3
"""
Consumer Orchestrator

Runs FK-normed stories and brainrot-aesops generation against
dialogue walks from reference or synthetic corpora.

Usage:
    # FK-normed stories from synthetic graph
    DEEPSEEK_API_KEY="sk-..." uv run python run_consumers.py fk-normed \
        --source synthetic/gallia_v4/graph.json \
        --output output/gallia_fk_stories.jsonl

    # Brainrot-aesops from reference corpus
    DEEPSEEK_API_KEY="sk-..." uv run python run_consumers.py brainrot \
        --source dialogue_data/falloutnv_full_dialogue.json \
        --output output/fnv_aesops.jsonl

    # Both tiers
    DEEPSEEK_API_KEY="sk-..." uv run python run_consumers.py all \
        --source synthetic/gallia_v4/graph.json \
        --output output/gallia_training.jsonl
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List, Any
from dataclasses import asdict

import httpx

from prose_wrapper import Walk, extract_walks_from_graph
from fk_normed_stories import (
    process_walks_batch as fk_batch,
    write_training_jsonl as fk_write,
    write_stats as fk_stats,
    FlattenedWalk,
    FKNormedStory,
)
from brainrot_aesops import (
    generate_aesops_batch,
    BrainrotAesop,
)
from brainrot_aesops_v4 import (
    generate_aesops_v4_batch,
    AesopV4,
)


# =============================================================================
# DeepSeek API Client
# =============================================================================

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"


async def call_deepseek(
    prompt: str,
    api_key: str,
    model: str = "deepseek-chat",
    max_tokens: int = 800,
    temperature: float = 0.7,
) -> str:
    """Call DeepSeek API for text generation."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]["content"]


def make_llm_call(api_key: str, model: str = "deepseek-chat"):
    """Create a closure for LLM calls with the API key baked in."""
    async def llm_call(prompt: str) -> str:
        return await call_deepseek(prompt, api_key, model)
    return llm_call


# =============================================================================
# Walk Loading
# =============================================================================

def load_walks_from_synthetic(path: Path, num_walks: int = 100) -> List[Walk]:
    """Load walks from a synthetic graph."""
    print(f"Loading synthetic graph: {path}")
    with open(path) as f:
        graph = json.load(f)

    walks = extract_walks_from_graph(graph, walk_length=5, num_walks=num_walks)
    print(f"Extracted {len(walks)} walks")
    return walks


def load_walks_from_reference(path: Path, num_walks: int = 100) -> List[Walk]:
    """Load walks from reference dialogue data."""
    print(f"Loading reference corpus: {path}")

    # Reference format: {"game": ..., "dialogue": [...]}
    with open(path) as f:
        data = json.load(f)

    # Extract dialogue array from wrapper dict
    if isinstance(data, dict) and "dialogue" in data:
        entries = data["dialogue"]
    else:
        entries = data  # Fallback for flat arrays

    # Group by quest/topic to form walks
    from collections import defaultdict
    by_topic = defaultdict(list)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        topic = entry.get("topic") or entry.get("quest_context") or "misc"
        by_topic[topic].append(entry)

    walks = []
    for topic, entries in by_topic.items():
        if len(entries) < 2:
            continue

        # Take sequences of 3-6 entries
        for i in range(0, len(entries) - 2, 3):
            chunk = entries[i:i+5]
            beats = [
                {
                    "text": e.get("text", ""),
                    "emotion": e.get("emotion", "neutral"),
                    "speaker": e.get("speaker"),
                }
                for e in chunk
            ]
            walks.append(Walk(
                beats=beats,
                source=str(path.stem),
            ))

        if len(walks) >= num_walks:
            break

    print(f"Extracted {len(walks)} walks from {len(by_topic)} topics")
    return walks[:num_walks]


def load_walks(path: Path, num_walks: int = 100) -> List[Walk]:
    """Auto-detect format and load walks."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "nodes" in data and "edges" in data:
        return load_walks_from_synthetic(path, num_walks)
    else:
        return load_walks_from_reference(path, num_walks)


# =============================================================================
# Bible Loading
# =============================================================================

def load_bible_excerpt(setting: str, max_chars: int = 2000) -> Optional[str]:
    """Load a lore bible excerpt for a setting."""
    bible_path = Path(f"bibles/{setting}.yaml")

    if not bible_path.exists():
        # Try to infer from path
        for name in ["gallia", "marmotte", "cyrodiil", "mojave", "skyrim"]:
            if name in setting.lower():
                bible_path = Path(f"bibles/{name}.yaml")
                break

    if not bible_path.exists():
        return None

    content = bible_path.read_text()

    # Extract key sections
    import re
    sections = []

    # Get proper noun clusters
    noun_match = re.search(r'proper_noun_clusters:(.*?)(?=\n[a-z_]+:|$)', content, re.DOTALL)
    if noun_match:
        sections.append("## Proper Nouns\n" + noun_match.group(1)[:800])

    # Get faction templates
    faction_match = re.search(r'faction_templates:(.*?)(?=\n[a-z_]+:|$)', content, re.DOTALL)
    if faction_match:
        sections.append("## Factions\n" + faction_match.group(1)[:800])

    excerpt = "\n\n".join(sections)
    return excerpt[:max_chars] if excerpt else None


# =============================================================================
# Main Runners
# =============================================================================

async def run_fk_normed(
    walks: List[Walk],
    llm_call: callable,
    output_path: Path,
    fk_levels: List[int] = [0, 3, 6, 9],
    bible_excerpt: Optional[str] = None,
    concurrency: int = 5,
):
    """Run FK-normed stories generation."""
    print(f"\n{'='*60}")
    print("FK-NORMED STORIES GENERATION")
    print(f"{'='*60}")
    print(f"Walks: {len(walks)}")
    print(f"FK levels: {fk_levels}")
    print(f"Expected outputs: {len(walks)} flattened + {len(walks) * len(fk_levels)} FK-normed")

    flattened, stories = await fk_batch(
        walks,
        llm_call,
        fk_levels=fk_levels,
        bible_excerpt=bible_excerpt,
        concurrency=concurrency,
    )

    # Write outputs
    fk_write(flattened, stories, output_path)
    print(f"\nWritten to {output_path}")

    # Stats
    stats_path = output_path.with_suffix('.stats.json')
    stats = fk_stats(flattened, stories, stats_path)
    print(f"Stats: {json.dumps(stats, indent=2)}")

    return flattened, stories


async def run_brainrot(
    walks: List[Walk],
    llm_call: callable,
    output_path: Path,
    n_aesops: int = 10,
    source_corpus: str = "unknown",
    concurrency: int = 5,
    use_cleaner: bool = False,
    version: str = "v3",
):
    """Run brainrot-aesops generation."""
    print(f"\n{'='*60}")

    if version == "v4":
        print("BRAINROT-AESOPS GENERATION (v4 - many small outputs)")
        print(f"{'='*60}")
        print(f"Walks available: {len(walks)}")
        # v4: n_aesops controls total batches, 12 words per batch
        n_batches = max(1, n_aesops // 12)
        words_per_batch = 12
        print(f"Batches: {n_batches} x {words_per_batch} words = {n_batches * words_per_batch} outputs")
        print(f"Each output: 1 word, 1-2 walks, 60-120 words")

        aesops = await generate_aesops_v4_batch(
            walks,
            llm_call,
            n_batches=n_batches,
            words_per_batch=words_per_batch,
            source_corpus=source_corpus,
            concurrency=concurrency,
        )

        # Write outputs
        passed = [a for a in aesops if a.passed_filters]
        failed = [a for a in aesops if not a.passed_filters]

        with open(output_path, 'w') as f:
            for aesop in passed:
                f.write(json.dumps(asdict(aesop)) + '\n')

        print(f"\nWritten {len(passed)} aesops to {output_path}")
        print(f"Failed filters: {len(failed)}")

        # Word coverage
        if passed:
            words_taught = [a.word for a in passed]
            print(f"Words taught: {words_taught}")
            avg_fk = sum(a.fk_measured for a in passed) / len(passed)
            avg_wc = sum(a.word_count for a in passed) / len(passed)
            print(f"Avg FK: {avg_fk:.1f}, Avg word count: {avg_wc:.0f}")

        # Failure analysis
        if failed:
            reasons = {}
            for a in failed:
                r = (a.reject_reason or "unknown").split("_")[0]
                reasons[r] = reasons.get(r, 0) + 1
            print(f"Failure reasons: {reasons}")

        return aesops

    else:
        # v3: model-driven pairing (original behavior)
        print("BRAINROT-AESOPS GENERATION (v3 - model-driven pairing)")
        print(f"{'='*60}")
        print(f"Walks available: {len(walks)}")
        print(f"Target aesops: {n_aesops}")
        print(f"Each aesop: 8 walks offered, 12 words offered")
        print(f"Minimums: 4 walks used, 5 words defined")
        if use_cleaner:
            print(f"Cleaner: enabled (2x LLM calls per aesop)")

        aesops = await generate_aesops_batch(
            walks,
            llm_call,
            n_aesops=n_aesops,
            source_corpus=source_corpus,
            concurrency=concurrency,
            use_cleaner=use_cleaner,
        )

        # Write outputs
        passed = [a for a in aesops if a.passed_filters]
        failed = [a for a in aesops if not a.passed_filters]

        with open(output_path, 'w') as f:
            for aesop in passed:
                f.write(json.dumps(asdict(aesop)) + '\n')

        print(f"\nWritten {len(passed)} aesops to {output_path}")
        print(f"Failed filters: {len(failed)}")

        # Word coverage
        if passed:
            all_words = set()
            for a in passed:
                all_words.update(a.words_used)
            print(f"Unique words taught: {sorted(all_words)}")
            print(f"Avg walks used: {sum(a.walks_used for a in passed) / len(passed):.1f}")
            print(f"Avg words defined: {sum(len(a.words_used) for a in passed) / len(passed):.1f}")

        # Failure analysis
        if failed:
            reasons = {}
            for a in failed:
                r = (a.reject_reason or "unknown").split("_")[0]
                reasons[r] = reasons.get(r, 0) + 1
            print(f"Failure reasons: {reasons}")

        return aesops


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run training data consumers")
    parser.add_argument("mode", choices=["fk-normed", "brainrot", "all"],
                        help="Which consumer to run")
    parser.add_argument("--source", required=True,
                        help="Path to source data (synthetic graph or reference corpus)")
    parser.add_argument("--output", required=True,
                        help="Output path for training data")
    parser.add_argument("--num-walks", type=int, default=50,
                        help="Number of walks to extract from source")
    parser.add_argument("--num-aesops", type=int, default=10,
                        help="Number of aesops to generate")
    parser.add_argument("--fk-levels", type=str, default="0,3,6,9",
                        help="Comma-separated FK levels")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent API calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock LLM instead of API")
    parser.add_argument("--use-cleaner", action="store_true",
                        help="Use second LLM call to strip assistant boilerplate (v3 only)")
    parser.add_argument("--version", choices=["v3", "v4"], default="v4",
                        help="Aesop version: v3 (mega-passage) or v4 (many small)")

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: DEEPSEEK_API_KEY not set. Use --dry-run for testing.")
        sys.exit(1)

    # Setup paths
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"ERROR: Source not found: {source_path}")
        sys.exit(1)

    # Load walks
    walks = load_walks(source_path, args.num_walks)

    if not walks:
        print("ERROR: No walks extracted from source")
        sys.exit(1)

    # Load bible if available
    bible = load_bible_excerpt(source_path.stem)

    # Setup LLM call
    if args.dry_run:
        from fk_normed_stories import mock_llm_call as fk_mock
        if args.version == "v4":
            from brainrot_aesops_v4 import mock_llm_call as aesop_mock
        else:
            from brainrot_aesops import mock_llm_call as aesop_mock
        fk_llm = fk_mock
        aesop_llm = aesop_mock
    else:
        llm = make_llm_call(api_key)
        fk_llm = llm
        aesop_llm = llm

    # Parse FK levels
    fk_levels = [int(x) for x in args.fk_levels.split(",")]

    # Run
    async def run():
        if args.mode in ["fk-normed", "all"]:
            fk_output = output_path.with_stem(f"{output_path.stem}_fk")
            await run_fk_normed(
                walks,
                fk_llm,
                fk_output,
                fk_levels=fk_levels,
                bible_excerpt=bible,
                concurrency=args.concurrency,
            )

        if args.mode in ["brainrot", "all"]:
            aesop_output = output_path.with_stem(f"{output_path.stem}_aesops")
            await run_brainrot(
                walks,
                aesop_llm,
                aesop_output,
                n_aesops=args.num_aesops,
                source_corpus=source_path.stem,
                concurrency=args.concurrency,
                use_cleaner=args.use_cleaner,
                version=args.version,
            )

    asyncio.run(run())
    print("\nDone!")


if __name__ == "__main__":
    main()
