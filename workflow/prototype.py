#!/usr/bin/env python3
"""
Minimal Prototype: End-to-End Synthetic Dialogue Generation

Demonstrates the full pipeline:
1. Sample a dialogue walk from the graph API
2. Extract structural triplet (Haiku)
3. Translate to target setting (Sonnet)
4. Validate new proper nouns (Opus) - optional
5. Persist to synthetic corpus
6. Log everything for observability

Usage:
    # Make sure the API server is running first:
    # uv run uvicorn api_server:app --host 127.0.0.1 --port 8000

    # Run the prototype:
    python workflow/prototype.py

    # Or with options:
    python workflow/prototype.py --game falloutnv --count 3 --no-curator
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from subagent_orchestrator.observability import (
    WorkflowTrace,
    TraceStore,
)
from subagent_orchestrator.subagent import SubagentCaller
from subagent_orchestrator.models import StructuralTriplet, SyntheticEntry


# Configuration
API_BASE = "http://127.0.0.1:8000"
BIBLES_DIR = Path("bibles")
SYNTHETIC_DIR = Path("synthetic")
TRACES_DIR = Path("traces")


def sample_walk_from_api(game: str = "oblivion", method: str = "walk", max_length: int = 4) -> list[dict]:
    """Fetch a dialogue walk from the running API server."""
    print(f"\n[1/5] Sampling walk from {game}...")

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{API_BASE}/api/sample",
            json={
                "game": game,
                "method": method,
                "count": 1,
                "max_length": max_length,
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code} - {response.text}")

        data = response.json()
        if not data.get("samples"):
            raise RuntimeError("No samples returned from API")

        walk = data["samples"][0]["nodes"]
        print(f"      Got walk with {len(walk)} nodes")
        for i, node in enumerate(walk):
            print(f"      [{i+1}] {node.get('emotion', '?'):8} | {node.get('text', '')[:60]}...")

        return walk


def load_target_bible(bible_id: str = "gallia") -> str:
    """Load target bible YAML content."""
    path = BIBLES_DIR / f"{bible_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Bible not found: {path}")
    return path.read_text()


def run_pipeline(
    game: str = "oblivion",
    target_bible: str = "gallia",
    use_curator: bool = True,
    persist: bool = True,
    walk: list[dict] | None = None,
) -> WorkflowTrace:
    """
    Run the complete synthetic generation pipeline.

    Returns the workflow trace for inspection/debugging.
    """
    # Initialize infrastructure
    caller = SubagentCaller(prompts_dir="claudefiles/subagents")
    store = TraceStore(TRACES_DIR)

    # Get walk if not provided
    if walk is None:
        walk = sample_walk_from_api(game)

    # Initialize trace
    trace = WorkflowTrace(
        source_game=game,
        source_walk={"walk": walk},
        target_bible=target_bible,
    )

    # Step 2: Extract triplet
    print(f"\n[2/5] Extracting structural triplet (Haiku)...")
    try:
        extractor_log = caller.call_triplet_extractor(
            walk=walk,
            reference_bible="mojave" if game == "falloutnv" else "cyrodiil",
        )
        trace.add_extractor_log(extractor_log)

        if not extractor_log.parse_success:
            trace.fail("extraction", f"Parse failed: {extractor_log.parse_errors}")
            store.save(trace)
            print(f"      FAILED: {extractor_log.parse_errors}")
            return trace

        triplet = extractor_log.output_parsed
        print(f"      Arc shape: {triplet.get('arc_shape')}")
        print(f"      Barrier: {triplet.get('barrier_type')}")
        print(f"      Proper nouns: {triplet.get('proper_nouns_used', [])}")
        print(f"      Latency: {extractor_log.latency_ms}ms | Cost: ${extractor_log.cost_usd:.4f}")

    except Exception as e:
        trace.fail("extraction", str(e))
        store.save(trace)
        print(f"      ERROR: {e}")
        return trace

    # Step 3: Translate
    print(f"\n[3/5] Translating to {target_bible} (Sonnet)...")
    try:
        target_yaml = load_target_bible(target_bible)

        # Load few-shot examples
        few_shot_path = BIBLES_DIR / "few_shot_translations.yaml"
        few_shot = None
        if few_shot_path.exists():
            import yaml
            examples_data = yaml.safe_load(few_shot_path.read_text())
            few_shot = [v for k, v in examples_data.items() if k.startswith("example_")][:3]

        translator_log = caller.call_translation_engine(
            triplet=triplet,
            source_bible="mojave" if game == "falloutnv" else "cyrodiil",
            target_bible=target_bible,
            target_bible_content=target_yaml,
            few_shot_examples=few_shot,
        )
        trace.add_translator_log(translator_log)

        if not translator_log.parse_success:
            trace.fail("translation", f"Parse failed: {translator_log.parse_errors}")
            store.save(trace)
            print(f"      FAILED: {translator_log.parse_errors}")
            return trace

        translation = translator_log.output_parsed
        print(f"      Confidence: {translation.get('confidence', 0):.2f}")
        print(f"      New nouns: {translation.get('proper_nouns_introduced', [])}")
        print(f"      Latency: {translator_log.latency_ms}ms | Cost: ${translator_log.cost_usd:.4f}")

        print("\n      Translated texts:")
        for i, text in enumerate(translation.get("translated_texts", [])):
            print(f"      [{i+1}] {text}")

    except Exception as e:
        trace.fail("translation", str(e))
        store.save(trace)
        print(f"      ERROR: {e}")
        return trace

    # Step 4: Curator validation (optional)
    new_nouns = translation.get("proper_nouns_introduced", [])
    if use_curator and new_nouns:
        print(f"\n[4/5] Validating {len(new_nouns)} new nouns with curator (Opus)...")
        try:
            for noun in new_nouns:
                curator_log = caller.call_lore_curator(
                    proposal_type="proper_noun",
                    proposal={
                        "proposed_noun": noun,
                        "context": "Appeared in translated dialogue",
                    },
                    bible_content=target_yaml,
                )
                trace.add_curator_log(curator_log)

                if curator_log.parse_success:
                    decision = curator_log.output_parsed
                    status = "APPROVED" if decision.get("approved") else "REJECTED"
                    print(f"      {noun}: {status}")
                    print(f"        Reason: {decision.get('reasoning', 'N/A')[:100]}...")
                    print(f"        Latency: {curator_log.latency_ms}ms | Cost: ${curator_log.cost_usd:.4f}")
                else:
                    print(f"      {noun}: PARSE ERROR - {curator_log.parse_errors}")

        except Exception as e:
            print(f"      WARNING: Curator failed - {e}")
    else:
        print(f"\n[4/5] Skipping curator validation (no new nouns or --no-curator)")

    # Step 5: Persist
    if persist:
        print(f"\n[5/5] Persisting to synthetic corpus...")
        try:
            # Build synthetic entry
            triplet_model = StructuralTriplet(**triplet)
            synthetic = SyntheticEntry(
                source_walk_id=trace.workflow_id,
                source_bible="mojave" if game == "falloutnv" else "cyrodiil",
                target_bible=target_bible,
                source_triplet=triplet_model,
                translated_texts=translation.get("translated_texts", []),
                proper_nouns_introduced=new_nouns,
                validation_score=translation.get("confidence", 0.5),
                workflow_id=trace.workflow_id,
            )

            # Save to corpus
            target_dir = SYNTHETIC_DIR / target_bible
            target_dir.mkdir(parents=True, exist_ok=True)
            corpus_file = target_dir / "corpus.jsonl"

            with open(corpus_file, "a") as f:
                f.write(json.dumps(synthetic.model_dump()) + "\n")

            trace.complete(synthetic.model_dump())
            print(f"      Saved: {synthetic.synthetic_id}")
            print(f"      File: {corpus_file}")

        except Exception as e:
            trace.fail("persistence", str(e))
            print(f"      ERROR: {e}")
    else:
        print(f"\n[5/5] Skipping persistence (--no-persist)")
        trace.complete()

    # Save trace
    store.save(trace)

    # Summary
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"  Workflow ID: {trace.workflow_id}")
    print(f"  Status: {trace.status.value}")
    print(f"  Total latency: {trace.total_latency_ms}ms")
    print(f"  Total cost: ${trace.total_cost_usd:.4f}")
    print(f"  Trace file: {TRACES_DIR}/traces_*.jsonl")

    return trace


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic dialogue generation prototype"
    )
    parser.add_argument(
        "--game", "-g",
        default="oblivion",
        choices=["oblivion", "falloutnv"],
        help="Source game to sample from"
    )
    parser.add_argument(
        "--target", "-t",
        default="gallia",
        help="Target bible/setting for translation"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=1,
        help="Number of synthetics to generate"
    )
    parser.add_argument(
        "--no-curator",
        action="store_true",
        help="Skip curator validation"
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip persisting to corpus"
    )
    parser.add_argument(
        "--walk-json",
        type=str,
        help="Use a specific walk from JSON file instead of sampling"
    )

    args = parser.parse_args()

    print("="*60)
    print("SYNTHETIC DIALOGUE GENERATION PROTOTYPE")
    print("="*60)
    print(f"Source: {args.game}")
    print(f"Target: {args.target}")
    print(f"Count: {args.count}")
    print(f"Curator: {'disabled' if args.no_curator else 'enabled'}")
    print(f"Persist: {'disabled' if args.no_persist else 'enabled'}")

    # Load custom walk if provided
    custom_walk = None
    if args.walk_json:
        with open(args.walk_json) as f:
            custom_walk = json.load(f)

    # Run pipeline
    for i in range(args.count):
        if args.count > 1:
            print(f"\n{'='*60}")
            print(f"GENERATING SYNTHETIC {i+1}/{args.count}")
            print("="*60)

        trace = run_pipeline(
            game=args.game,
            target_bible=args.target,
            use_curator=not args.no_curator,
            persist=not args.no_persist,
            walk=custom_walk,
        )

        if trace.status.value != "completed":
            print(f"\nWARNING: Workflow {trace.workflow_id} ended with status: {trace.status.value}")
            if trace.failure_reason:
                print(f"  Reason: {trace.failure_reason}")


if __name__ == "__main__":
    main()
