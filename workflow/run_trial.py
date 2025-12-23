#!/usr/bin/env python3
"""
Trial Run Executor

Runs a trial (e.g., 1% of corpus) through the synthetic generation pipeline
with full tracing and persistence to the TrialRun structure.

Usage:
    # 1% trial run
    python workflow/run_trial.py --rate 0.01 --target gallia

    # Fixed count
    python workflow/run_trial.py --count 50 --target gallia --game falloutnv

    # Dry run (no API calls, just samples walks)
    python workflow/run_trial.py --rate 0.01 --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from trial_runs import RunManager, TrialRun, LinkedSynthetic, RunConfig
from subagent_orchestrator import (
    SubagentCaller,
    WorkflowTrace,
    TraceStore,
    validate_subagent_output,
    ValidationStats,
)


API_BASE = "http://127.0.0.1:8000"
BIBLES_DIR = Path("bibles")


def get_corpus_size(game: str) -> int:
    """Get total dialogue count for a game."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{API_BASE}/api/games")
        games = response.json()
        for g in games:
            if g["name"] == game:
                return g["dialogue_count"]
    return 0


def sample_walks(game: str, count: int, method: str = "walk", max_length: int = 6) -> list[dict]:
    """Sample dialogue walks from the API."""
    walks = []
    batch_size = min(count, 20)  # API might limit batch size

    with httpx.Client(timeout=60.0) as client:
        remaining = count
        while remaining > 0:
            batch = min(remaining, batch_size)
            response = client.post(
                f"{API_BASE}/api/sample",
                json={
                    "game": game,
                    "method": method,
                    "count": batch,
                    "max_length": max_length,
                }
            )

            if response.status_code != 200:
                print(f"Warning: API error sampling walks: {response.status_code}")
                break

            data = response.json()
            for sample in data.get("samples", []):
                walks.append(sample["nodes"])

            remaining -= batch

            if remaining > 0:
                time.sleep(0.1)  # Be nice to the API

    return walks


def load_bible(bible_id: str) -> str:
    """Load target bible content."""
    path = BIBLES_DIR / f"{bible_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Bible not found: {path}")
    return path.read_text()


def run_pipeline_step(
    caller: SubagentCaller,
    walk: list[dict],
    source_game: str,
    target_bible: str,
    bible_content: str,
    use_curator: bool = True,
) -> dict:
    """
    Run a single walk through the pipeline.

    Returns a dict with all intermediate results for TrialRun persistence.
    """
    result = {
        "success": False,
        "walk": walk,
        "source_game": source_game,
        "target_bible": target_bible,
        "triplet": None,
        "translation": None,
        "curator_decisions": [],
        "validation_reports": [],
        "total_latency_ms": 0,
        "total_cost_usd": 0.0,
        "failure_stage": None,
        "failure_reason": None,
    }

    # Step 1: Extract triplet
    try:
        extractor_log = caller.call_triplet_extractor(
            walk=walk,
            reference_bible="mojave" if source_game == "falloutnv" else "cyrodiil",
        )
        result["total_latency_ms"] += extractor_log.latency_ms
        result["total_cost_usd"] += extractor_log.cost_usd

        if not extractor_log.parse_success:
            result["failure_stage"] = "extraction"
            result["failure_reason"] = str(extractor_log.parse_errors)
            return result

        # Validate schema
        report = validate_subagent_output(extractor_log.output_parsed, "structural_parser")
        result["validation_reports"].append(report.to_dict())

        if not report.valid:
            result["failure_stage"] = "extraction_validation"
            result["failure_reason"] = report.summary()
            return result

        result["triplet"] = extractor_log.output_parsed

    except Exception as e:
        result["failure_stage"] = "extraction"
        result["failure_reason"] = str(e)
        return result

    # Step 2: Translate
    try:
        translator_log = caller.call_translation_engine(
            triplet=result["triplet"],
            source_bible="mojave" if source_game == "falloutnv" else "cyrodiil",
            target_bible=target_bible,
            target_bible_content=bible_content,
        )
        result["total_latency_ms"] += translator_log.latency_ms
        result["total_cost_usd"] += translator_log.cost_usd

        if not translator_log.parse_success:
            result["failure_stage"] = "translation"
            result["failure_reason"] = str(translator_log.parse_errors)
            return result

        # Validate schema
        report = validate_subagent_output(translator_log.output_parsed, "translation_engine")
        result["validation_reports"].append(report.to_dict())

        if not report.valid:
            result["failure_stage"] = "translation_validation"
            result["failure_reason"] = report.summary()
            return result

        result["translation"] = translator_log.output_parsed

    except Exception as e:
        result["failure_stage"] = "translation"
        result["failure_reason"] = str(e)
        return result

    # Step 3: Curator validation (if new nouns introduced)
    new_nouns = result["translation"].get("proper_nouns_introduced", [])
    if use_curator and new_nouns:
        try:
            for noun in new_nouns:
                curator_log = caller.call_lore_curator(
                    proposal_type="proper_noun",
                    proposal={
                        "proposed_noun": noun,
                        "context": "From translated dialogue",
                    },
                    bible_content=bible_content,
                )
                result["total_latency_ms"] += curator_log.latency_ms
                result["total_cost_usd"] += curator_log.cost_usd

                if curator_log.parse_success:
                    decision = curator_log.output_parsed
                    result["curator_decisions"].append({
                        "noun": noun,
                        "approved": decision.get("approved", False),
                        "reasoning": decision.get("reasoning", ""),
                    })
                else:
                    result["curator_decisions"].append({
                        "noun": noun,
                        "approved": False,
                        "reasoning": f"Parse error: {curator_log.parse_errors}",
                    })

            # Check if any nouns were rejected
            rejections = [d for d in result["curator_decisions"] if not d["approved"]]
            if rejections:
                result["failure_stage"] = "curator"
                result["failure_reason"] = f"{len(rejections)} nouns rejected"
                return result

        except Exception as e:
            result["failure_stage"] = "curator"
            result["failure_reason"] = str(e)
            return result

    # Success!
    result["success"] = True
    return result


def create_linked_synthetic(pipeline_result: dict, run_id: str) -> LinkedSynthetic:
    """Convert pipeline result to LinkedSynthetic."""
    walk = pipeline_result["walk"]
    triplet = pipeline_result["triplet"]
    translation = pipeline_result["translation"]

    return LinkedSynthetic(
        synthetic_id=f"s_{datetime.now(timezone.utc).strftime('%H%M%S%f')[:10]}",
        run_id=run_id,
        source_game=pipeline_result["source_game"],
        source_walk_ids=[n.get("id", "") for n in walk],
        source_walk_texts=[n.get("text", "") for n in walk],
        arc_shape=triplet.get("arc_shape", "unknown"),
        barrier_type=triplet.get("barrier_type", "unknown"),
        attractor_type=triplet.get("attractor_type", "unknown"),
        beat_count=len(triplet.get("arc", [])),
        emotion_sequence=[b.get("emotion", "neutral") for b in triplet.get("arc", [])],
        function_sequence=[b.get("function", "unknown") for b in triplet.get("arc", [])],
        target_bible=pipeline_result["target_bible"],
        translated_texts=translation.get("translated_texts", []),
        proper_nouns_introduced=translation.get("proper_nouns_introduced", []),
        translation_confidence=translation.get("confidence", 0.0),
        schema_valid=all(r.get("valid", False) for r in pipeline_result["validation_reports"]),
        curator_approved=all(d.get("approved", True) for d in pipeline_result["curator_decisions"]),
        latency_ms=pipeline_result["total_latency_ms"],
        cost_usd=pipeline_result["total_cost_usd"],
    )


def run_trial(
    target_bible: str,
    source_games: list[str],
    sample_count: int = 0,
    sample_rate: float = 0.0,
    use_curator: bool = True,
    dry_run: bool = False,
    prompts_dir: str = "claudefiles/subagents",
) -> TrialRun:
    """
    Execute a trial run.

    Args:
        target_bible: Target setting (e.g., "gallia")
        source_games: Source games to sample from
        sample_count: Fixed sample count (overrides rate)
        sample_rate: Fraction of corpus to sample (e.g., 0.01 for 1%)
        use_curator: Whether to validate new proper nouns
        dry_run: If True, only sample walks without API calls
        prompts_dir: Directory containing subagent CLAUDE.md files

    Returns:
        TrialRun with all results
    """
    # Initialize
    mgr = RunManager()
    run = mgr.create_run(
        target_bible=target_bible,
        source_games=source_games,
        sample_count=sample_count,
        sample_rate=sample_rate,
        use_curator=use_curator,
        notes=f"Trial run {'(dry)' if dry_run else ''}",
    )

    print(f"Created run: {run.config.run_id}")
    print(f"  Target: {target_bible}")
    print(f"  Sources: {source_games}")
    print(f"  Rate: {sample_rate:.1%}" if sample_rate else f"  Count: {sample_count}")
    print()

    # Load bible
    try:
        bible_content = load_bible(target_bible)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return run

    # Calculate sample counts per game
    game_samples = {}
    for game in source_games:
        corpus_size = get_corpus_size(game)
        if sample_rate > 0:
            count = max(1, int(corpus_size * sample_rate))
        else:
            count = sample_count // len(source_games)
        game_samples[game] = count
        print(f"  {game}: {count} samples (from {corpus_size} total)")

    total_samples = sum(game_samples.values())
    print(f"\nTotal samples: {total_samples}")

    if dry_run:
        print("\n[DRY RUN - Sampling walks only, no API calls]")
        for game, count in game_samples.items():
            walks = sample_walks(game, count)
            print(f"\n{game}: sampled {len(walks)} walks")
            for i, walk in enumerate(walks[:3]):
                print(f"  [{i+1}] {len(walk)} nodes: {walk[0].get('text', '')[:50]}...")
        return run

    # Initialize caller
    try:
        caller = SubagentCaller(prompts_dir=prompts_dir)
    except Exception as e:
        print(f"Error initializing SubagentCaller: {e}")
        print("Make sure ANTHROPIC_API_KEY is set and prompts exist")
        return run

    # Run pipeline
    validation_stats = ValidationStats()
    success_count = 0
    failure_counts = {}

    print("\nRunning pipeline...")
    print("-" * 60)

    for game, count in game_samples.items():
        print(f"\nProcessing {game}...")
        walks = sample_walks(game, count)

        for i, walk in enumerate(walks):
            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(walks)}] ", end="", flush=True)

            # Run pipeline
            result = run_pipeline_step(
                caller=caller,
                walk=walk,
                source_game=game,
                target_bible=target_bible,
                bible_content=bible_content,
                use_curator=use_curator,
            )

            # Record validation stats
            for report in result["validation_reports"]:
                if "valid" in report:
                    # Reconstruct report for stats
                    class FakeReport:
                        def __init__(self, d):
                            self.valid = d.get("valid", False)
                            self.enum_violations = d.get("enum_violations", [])
                            self.missing_fields = d.get("missing_fields", [])
                            self.field_errors = d.get("field_errors", [])
                    validation_stats.add_report(FakeReport(report))

            if result["success"]:
                # Create and persist synthetic
                synthetic = create_linked_synthetic(result, run.config.run_id)
                run.append_synthetic(synthetic)
                success_count += 1
                print("✓", end="", flush=True)
            else:
                # Record failure
                stage = result["failure_stage"] or "unknown"
                failure_counts[stage] = failure_counts.get(stage, 0) + 1
                print("✗", end="", flush=True)

                # Persist trace for debugging
                run.append_trace({
                    "walk": walk,
                    "source_game": game,
                    "failure_stage": result["failure_stage"],
                    "failure_reason": result["failure_reason"],
                })

            # Rate limit
            time.sleep(0.5)

        print()  # Newline after game

    # Summary
    print("\n" + "=" * 60)
    print("TRIAL RUN COMPLETE")
    print("=" * 60)
    print(f"Run ID: {run.config.run_id}")
    print(f"Successes: {success_count}/{total_samples}")
    print(f"Failures by stage: {failure_counts}")
    print()

    # Stats
    stats = run.stats()
    print(f"Arc shapes: {stats.get('by_arc_shape', {})}")
    print(f"Emotion transitions: {len(stats.get('emotion_transitions', {}))} unique")
    print(f"New proper nouns: {stats.get('total_new_nouns', 0)} total, {len(stats.get('unique_new_nouns', []))} unique")
    print(f"Total cost: ${stats.get('total_cost_usd', 0):.4f}")
    print()

    # Validation stats
    print(validation_stats.summary())

    return run


def main():
    parser = argparse.ArgumentParser(description="Run synthetic generation trial")
    parser.add_argument("--target", "-t", default="gallia", help="Target bible")
    parser.add_argument("--game", "-g", action="append", help="Source game(s)")
    parser.add_argument("--rate", "-r", type=float, default=0.0, help="Sample rate (e.g., 0.01 for 1%%)")
    parser.add_argument("--count", "-n", type=int, default=0, help="Fixed sample count")
    parser.add_argument("--no-curator", action="store_true", help="Skip curator validation")
    parser.add_argument("--dry-run", action="store_true", help="Sample walks only, no API calls")
    parser.add_argument("--prompts-dir", default="claudefiles/subagents", help="Subagent prompts directory")

    args = parser.parse_args()

    # Defaults
    source_games = args.game or ["oblivion", "falloutnv"]

    if args.rate == 0 and args.count == 0:
        print("Error: Must specify --rate or --count")
        return 1

    run = run_trial(
        target_bible=args.target,
        source_games=source_games,
        sample_count=args.count,
        sample_rate=args.rate,
        use_curator=not args.no_curator,
        dry_run=args.dry_run,
        prompts_dir=args.prompts_dir,
    )

    print(f"\nResults saved to: runs/{run.config.run_id}/")
    return 0


if __name__ == "__main__":
    exit(main())
