"""
Compile Synthetic Dialogue from Translation Runs

Takes completed translation runs from runs/ and compiles them into
the shareable synthetic/ directory in formats matching reference data.

IMPORTANT: Source text is replaced with content hashes to avoid including
copyrighted material in the shareable output. A separate reference table
maps hashes back to source text for local development.

Output formats:
    synthetic/{setting}_dialogue.json     - Shareable: hashed source refs
    synthetic/{setting}_training.jsonl    - Shareable: hashed source refs
    synthetic/{setting}_trajectories.json - Shareable: hashed source refs

    runs/{run_id}/source_reference.json   - NOT shareable: hash→source mapping
"""

import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import argparse


# =============================================================================
# Hash-Based Reference System
# =============================================================================

class SourceReferenceTable:
    """
    Maps content hashes to source text for local development.

    The reference table stays in runs/ (gitignored) while synthetics
    contain only hashes, making them shareable without copyright concerns.
    """

    def __init__(self):
        self.references: dict[str, dict] = {}

    def hash_text(self, text: str, game: str = "", metadata: dict = None) -> str:
        """
        Generate a stable hash for source text.

        Returns a short hash that can be used as a reference ID.
        Stores the full text and metadata in the reference table.
        """
        if not text:
            return ""

        # Use SHA-256 truncated to 12 hex chars for readability
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]

        # Prefix with source game for easier debugging
        ref_id = f"{game[:3]}_{content_hash}" if game else f"src_{content_hash}"

        # Store in reference table (deduplicates automatically)
        if ref_id not in self.references:
            self.references[ref_id] = {
                "text": text,
                "game": game,
                "metadata": metadata or {},
                "first_seen": datetime.now(timezone.utc).isoformat(),
            }

        return ref_id

    def to_dict(self) -> dict:
        return {
            "version": "1.0",
            "description": "Source text reference table - DO NOT SHARE (contains copyrighted material)",
            "total_references": len(self.references),
            "references": self.references,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceReferenceTable":
        table = cls()
        table.references = data.get("references", {})
        return table


# =============================================================================
# Compilation Functions
# =============================================================================

def load_run_data(run_dir: Path) -> dict:
    """Load queue and trajectories from a run directory."""
    queue_file = run_dir / "queue.json"
    trajectories_file = run_dir / "translated_trajectories.json"

    data = {
        "queue": None,
        "trajectories": None,
        "run_id": run_dir.name,
    }

    if queue_file.exists():
        data["queue"] = json.loads(queue_file.read_text())

    if trajectories_file.exists():
        data["trajectories"] = json.loads(trajectories_file.read_text())

    return data


def compile_dialogue_json(
    run_data: dict,
    setting: str,
    ref_table: SourceReferenceTable
) -> dict:
    """
    Compile to reference-compatible dialogue JSON format.

    Each beat becomes a dialogue entry. Source text is replaced with
    hash references to avoid copyright issues in shareable output.
    """
    trajectories = run_data.get("trajectories", {})
    if not trajectories:
        return {"error": "No trajectories found"}

    dialogue_entries = []

    for traj in trajectories.get("trajectories", []):
        ticket_id = traj.get("ticket_id", "unknown")
        source_game = traj.get("source_game", "unknown")
        arc = traj.get("arc", {})
        confidence = traj.get("confidence", 0.0)
        concept_mappings = traj.get("concept_mappings", [])
        register_notes = traj.get("register_notes", "")

        for beat in traj.get("beats", []):
            # Generate a synthetic form_id
            form_id = f"0x{uuid.uuid4().hex[:6]}"

            # Hash source text instead of including it directly
            source_text = beat.get("source_text", "")
            source_ref = ref_table.hash_text(
                source_text,
                game=source_game,
                metadata={
                    "ticket_id": ticket_id,
                    "beat_index": beat.get("index", 0),
                }
            )

            entry = {
                "form_id": form_id,
                "topic": arc.get("shape", "SYNTHETIC"),
                "speaker": None,  # Synthetic - no specific speaker
                "text": beat.get("target_text", ""),
                "emotion": beat.get("emotion", "neutral"),
                "emotion_value": 50,  # Default neutral intensity
                "quest": f"synthetic_{setting}",
                "conditions": [],  # Synthetic - no game conditions
                # Extended metadata for synthetic (source_ref instead of source_text)
                "synthetic_meta": {
                    "source_game": source_game,
                    "source_ref": source_ref,  # Hash reference, not verbatim text
                    "ticket_id": ticket_id,
                    "beat_index": beat.get("index", 0),
                    "beat_function": beat.get("function", ""),
                    "archetype_relation": beat.get("archetype_relation", ""),
                    "arc_shape": arc.get("shape", ""),
                    "arc_emotions": arc.get("emotions", []),
                    "confidence": confidence,
                    "concept_mappings": concept_mappings,
                    "register_notes": register_notes,
                }
            }
            dialogue_entries.append(entry)

    return {
        "plugin": f"synthetic_{setting}.json",
        "game": "synthetic",
        "setting": setting,
        "source_games": trajectories.get("source_games", []),
        "run_id": trajectories.get("run_id", "unknown"),
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "total_trajectories": trajectories.get("total_trajectories", 0),
        "total_beats": len(dialogue_entries),
        "source_reference_note": "source_ref fields are hashes - see runs/{run_id}/source_reference.json for mapping (not shareable)",
        "dialogue": dialogue_entries,
    }


def compile_training_jsonl(
    dialogue_data: dict,
    ref_table: SourceReferenceTable
) -> list[str]:
    """
    Compile to ML-ready JSONL format.

    Each line is a complete training example with context.
    Source text replaced with hash references.
    """
    lines = []

    for entry in dialogue_data.get("dialogue", []):
        meta = entry.get("synthetic_meta", {})

        training_item = {
            "text": entry.get("text", ""),
            "speaker": entry.get("speaker"),
            "emotion": entry.get("emotion", "neutral"),
            "emotion_intensity": entry.get("emotion_value", 50) / 100.0,
            "topic": entry.get("topic", "SYNTHETIC"),
            "quest_context": entry.get("quest"),
            "meta": {
                "source": f"synthetic_{dialogue_data.get('setting', 'unknown')}",
                "game": "synthetic",
                "form_id": entry.get("form_id", ""),
                "source_game": meta.get("source_game", ""),
                "source_ref": meta.get("source_ref", ""),  # Hash, not text
                "beat_function": meta.get("beat_function", ""),
                "archetype_relation": meta.get("archetype_relation", ""),
                "arc_shape": meta.get("arc_shape", ""),
                "confidence": meta.get("confidence", 0.0),
            }
        }
        lines.append(json.dumps(training_item))

    return lines


def compile_trajectories_json(
    run_data: dict,
    setting: str,
    ref_table: SourceReferenceTable
) -> dict:
    """
    Compile trajectory-level view preserving full structural information.

    Source text in beats is replaced with hash references.
    """
    trajectories = run_data.get("trajectories", {})
    if not trajectories:
        return {"error": "No trajectories found"}

    # Process trajectories to hash source text
    processed_trajectories = []
    for traj in trajectories.get("trajectories", []):
        source_game = traj.get("source_game", "unknown")
        ticket_id = traj.get("ticket_id", "unknown")

        processed_beats = []
        for beat in traj.get("beats", []):
            source_text = beat.get("source_text", "")
            source_ref = ref_table.hash_text(
                source_text,
                game=source_game,
                metadata={"ticket_id": ticket_id, "beat_index": beat.get("index", 0)}
            )

            processed_beat = {
                "index": beat.get("index", 0),
                "emotion": beat.get("emotion", "neutral"),
                "function": beat.get("function", ""),
                "archetype_relation": beat.get("archetype_relation", ""),
                "source_ref": source_ref,  # Hash, not text
                "target_text": beat.get("target_text", ""),
            }
            processed_beats.append(processed_beat)

        processed_traj = {
            "ticket_id": ticket_id,
            "source_game": source_game,
            "target_setting": traj.get("target_setting", setting),
            "confidence": traj.get("confidence", 0.0),
            "proper_nouns_introduced": traj.get("proper_nouns_introduced", []),
            "concept_mappings": traj.get("concept_mappings", []),
            "register_notes": traj.get("register_notes", ""),
            "arc": traj.get("arc", {}),
            "beats": processed_beats,
        }
        processed_trajectories.append(processed_traj)

    output = {
        "plugin": f"synthetic_{setting}_trajectories.json",
        "game": "synthetic",
        "setting": setting,
        "source_games": trajectories.get("source_games", []),
        "run_id": trajectories.get("run_id", "unknown"),
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "total_trajectories": len(processed_trajectories),
        "source_reference_note": "source_ref fields are hashes - see runs/{run_id}/source_reference.json",
        "trajectories": processed_trajectories,
    }

    # Compute aggregate statistics
    all_emotions = []
    all_shapes = {}
    all_mappings = []
    total_confidence = 0.0

    for traj in output["trajectories"]:
        arc = traj.get("arc", {})
        all_emotions.extend(arc.get("emotions", []))
        shape = arc.get("shape", "unknown")
        all_shapes[shape] = all_shapes.get(shape, 0) + 1
        all_mappings.extend(traj.get("concept_mappings", []))
        total_confidence += traj.get("confidence", 0.0)

    # Emotion distribution
    emotion_counts = {}
    for e in all_emotions:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    output["statistics"] = {
        "emotion_distribution": emotion_counts,
        "arc_shapes": all_shapes,
        "total_concept_mappings": len(all_mappings),
        "unique_concept_mappings": len(set(m.get("source", "") for m in all_mappings)),
        "avg_confidence": total_confidence / max(len(output["trajectories"]), 1),
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="Compile synthetic dialogue from translation runs")
    parser.add_argument("run_id", help="Run ID to compile (e.g., run_20251223_022704_gallia)")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing runs")
    parser.add_argument("--output-dir", default="synthetic", help="Output directory")
    parser.add_argument("--setting", help="Target setting name (default: extracted from run_id)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = runs_dir / args.run_id
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    # Extract setting from run_id if not provided
    # Format: run_YYYYMMDD_HHMMSS_setting
    setting = args.setting
    if not setting:
        parts = args.run_id.split("_")
        if len(parts) >= 4:
            setting = parts[-1]
        else:
            setting = "synthetic"

    print(f"Compiling run: {args.run_id}")
    print(f"Target setting: {setting}")

    # Load run data
    run_data = load_run_data(run_dir)

    if not run_data["trajectories"]:
        print("Error: No translated_trajectories.json found in run directory")
        return 1

    # Create reference table for source text hashing
    ref_table = SourceReferenceTable()

    # Compile formats (all use the same ref_table to deduplicate)
    print("Compiling dialogue JSON...")
    dialogue_json = compile_dialogue_json(run_data, setting, ref_table)
    dialogue_file = output_dir / f"{setting}_dialogue.json"
    dialogue_file.write_text(json.dumps(dialogue_json, indent=2))
    print(f"  -> {dialogue_file} ({dialogue_json.get('total_beats', 0)} beats)")

    print("Compiling training JSONL...")
    training_lines = compile_training_jsonl(dialogue_json, ref_table)
    training_file = output_dir / f"{setting}_training.jsonl"
    training_file.write_text("\n".join(training_lines))
    print(f"  -> {training_file} ({len(training_lines)} lines)")

    print("Compiling trajectories JSON...")
    trajectories_json = compile_trajectories_json(run_data, setting, ref_table)
    trajectories_file = output_dir / f"{setting}_trajectories.json"
    trajectories_file.write_text(json.dumps(trajectories_json, indent=2))
    stats = trajectories_json.get("statistics", {})
    print(f"  -> {trajectories_file} ({trajectories_json.get('total_trajectories', 0)} trajectories)")
    print(f"     Avg confidence: {stats.get('avg_confidence', 0):.2f}")
    print(f"     Arc shapes: {stats.get('arc_shapes', {})}")
    print(f"     Emotion distribution: {stats.get('emotion_distribution', {})}")

    # Save reference table to run directory (NOT shareable)
    print("Saving source reference table...")
    ref_file = run_dir / "source_reference.json"
    ref_file.write_text(json.dumps(ref_table.to_dict(), indent=2))
    print(f"  -> {ref_file} ({len(ref_table.references)} unique source texts)")
    print("     NOTE: This file contains copyrighted source text - do not share!")

    print("\nDone!")
    print(f"\nShareable outputs in {output_dir}/")
    print(f"Non-shareable reference in {run_dir}/source_reference.json")
    return 0


if __name__ == "__main__":
    exit(main())
