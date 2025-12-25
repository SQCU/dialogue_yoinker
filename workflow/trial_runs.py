#!/usr/bin/env python3
"""
Trial Run Management

Orchestrates synthetic generation runs with full traceability:
- Run-level metadata (config, models, timestamps)
- Source linking (synthetic → original graph node IDs)
- Graph construction for synthetics (trajectory sampling)
- Query interface for analysis
- Cross-run comparison

Directory structure:
    runs/
    ├── run_20251222_143052_gallia/
    │   ├── config.yaml           # Run configuration
    │   ├── synthetics.jsonl      # Generated synthetics with source links
    │   ├── traces.jsonl          # Full workflow traces (copy for self-containment)
    │   ├── validation.jsonl      # Schema validation reports
    │   ├── graph.json            # NetworkX graph of synthetics (for trajectory sampling)
    │   └── stats.json            # Aggregate statistics
    └── run_20251222_150000_gallia/
        └── ...

Usage:
    from workflow.trial_runs import TrialRun, RunManager

    # Create a new run
    run = RunManager.create_run(
        target_bible="gallia",
        source_games=["oblivion", "falloutnv"],
        sample_rate=0.01,  # 1%
        config={"use_curator": True, "model_override": None}
    )

    # Execute
    run.execute()

    # Analyze
    print(run.stats())
    trajectories = run.sample_trajectories(n=10)
    failures = run.query_failures(stage="translation")
"""

from __future__ import annotations

import json
import yaml
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterator, Any
from enum import Enum
import uuid

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RunConfig:
    """Configuration for a trial run."""
    run_id: str
    target_bible: str
    source_games: list[str]
    sample_count: int  # Absolute count OR
    sample_rate: float  # Percentage of corpus (0.01 = 1%)

    # Model configuration
    extractor_model: str = "haiku"
    translator_model: str = "sonnet"
    curator_model: str = "opus"
    use_curator: bool = True

    # Sampling configuration
    walk_method: str = "walk"  # walk, chain, hub
    max_walk_length: int = 6
    min_walk_length: int = 2

    # Persistence
    persist_synthetics: bool = True
    persist_traces: bool = True

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False)

    @classmethod
    def from_yaml(cls, content: str) -> "RunConfig":
        data = yaml.safe_load(content)
        return cls(**data)


@dataclass
class LinkedSynthetic:
    """
    A synthetic entry with full source traceability.

    This is what gets persisted for analysis.
    """
    synthetic_id: str
    run_id: str

    # Source linking
    source_game: str
    source_walk_ids: list[str]  # Graph node IDs from original corpus
    source_walk_texts: list[str]  # Original dialogue texts

    # Structural
    arc_shape: str
    barrier_type: str
    attractor_type: str
    beat_count: int
    emotion_sequence: list[str]
    function_sequence: list[str]

    # Translation
    target_bible: str
    translated_texts: list[str]
    proper_nouns_introduced: list[str]
    translation_confidence: float

    # Validation
    schema_valid: bool = True
    curator_approved: bool = True
    validation_notes: list[str] = field(default_factory=list)

    # Costs
    latency_ms: int = 0
    cost_usd: float = 0.0
    workflow_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_workflow_trace(cls, trace: dict, run_id: str) -> Optional["LinkedSynthetic"]:
        """Construct from a WorkflowTrace dict."""
        if trace["status"] != "completed":
            return None

        triplet = trace.get("triplet", {})
        if not triplet:
            return None

        # Extract source walk node IDs
        source_walk = trace.get("source_walk", {})
        walk_nodes = source_walk.get("walk", [])

        return cls(
            synthetic_id=str(uuid.uuid4())[:12],
            run_id=run_id,
            source_game=trace.get("source_game", ""),
            source_walk_ids=[n.get("id", "") for n in walk_nodes],
            source_walk_texts=[n.get("text", "") for n in walk_nodes],
            arc_shape=triplet.get("arc_shape", "unknown"),
            barrier_type=triplet.get("barrier_type", "unknown"),
            attractor_type=triplet.get("attractor_type", "unknown"),
            beat_count=len(triplet.get("arc", [])),
            emotion_sequence=[b.get("emotion", "neutral") for b in triplet.get("arc", [])],
            function_sequence=[b.get("function", "unknown") for b in triplet.get("arc", [])],
            target_bible=trace.get("target_bible", ""),
            translated_texts=trace.get("translated_texts", []),
            proper_nouns_introduced=trace.get("final_synthetic", {}).get("proper_nouns_introduced", []),
            translation_confidence=trace.get("final_synthetic", {}).get("validation_score", 0.0),
            latency_ms=trace.get("total_latency_ms", 0),
            cost_usd=trace.get("total_cost_usd", 0.0),
            workflow_id=trace.get("workflow_id", ""),
        )


class TrialRun:
    """
    A single trial run with full persistence and query capabilities.
    """

    def __init__(self, run_dir: Path, config: RunConfig):
        self.run_dir = Path(run_dir)
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Persistence files
        self._config_path = self.run_dir / "config.yaml"
        self._synthetics_path = self.run_dir / "synthetics.jsonl"
        self._traces_path = self.run_dir / "traces.jsonl"
        self._validation_path = self.run_dir / "validation.jsonl"
        self._stats_path = self.run_dir / "stats.json"
        self._graph_path = self.run_dir / "graph.json"

        # In-memory caches
        self._synthetics: list[LinkedSynthetic] = []
        self._loaded = False

    def save_config(self):
        """Persist run configuration."""
        self._config_path.write_text(self.config.to_yaml())

    def append_synthetic(self, synthetic: LinkedSynthetic):
        """Append a synthetic to the run."""
        with open(self._synthetics_path, "a") as f:
            f.write(json.dumps(synthetic.to_dict()) + "\n")
        self._synthetics.append(synthetic)

    def append_trace(self, trace: dict):
        """Append a workflow trace to the run."""
        with open(self._traces_path, "a") as f:
            f.write(json.dumps(trace) + "\n")

    def append_validation(self, validation: dict):
        """Append a validation report to the run."""
        with open(self._validation_path, "a") as f:
            f.write(json.dumps(validation) + "\n")

    def load_synthetics(self) -> list[LinkedSynthetic]:
        """Load all synthetics from disk."""
        if not self._synthetics_path.exists():
            return []

        synthetics = []
        with open(self._synthetics_path) as f:
            for line in f:
                data = json.loads(line)
                synthetics.append(LinkedSynthetic(**data))

        self._synthetics = synthetics
        self._loaded = True
        return synthetics

    def load_traces(self) -> Iterator[dict]:
        """Stream traces from disk."""
        if not self._traces_path.exists():
            return

        with open(self._traces_path) as f:
            for line in f:
                yield json.loads(line)

    # =========================================================================
    # Query Interface
    # =========================================================================

    def stats(self) -> dict:
        """Compute and cache aggregate statistics."""
        if not self._loaded:
            self.load_synthetics()

        synthetics = self._synthetics

        if not synthetics:
            return {"count": 0, "message": "No synthetics in run"}

        stats = {
            "run_id": self.config.run_id,
            "count": len(synthetics),
            "target_bible": self.config.target_bible,

            # Source distribution
            "by_source_game": {},

            # Arc shape distribution
            "by_arc_shape": {},

            # Barrier type distribution
            "by_barrier_type": {},

            # Emotion sequence patterns
            "emotion_transitions": {},

            # Function distribution
            "by_function": {},

            # Proper noun stats
            "total_new_nouns": 0,
            "synthetics_with_new_nouns": 0,
            "unique_new_nouns": set(),

            # Quality
            "avg_confidence": 0.0,
            "schema_valid_count": 0,
            "curator_approved_count": 0,

            # Costs
            "total_cost_usd": 0.0,
            "total_latency_ms": 0,
        }

        for s in synthetics:
            # Source
            stats["by_source_game"][s.source_game] = stats["by_source_game"].get(s.source_game, 0) + 1

            # Arc
            stats["by_arc_shape"][s.arc_shape] = stats["by_arc_shape"].get(s.arc_shape, 0) + 1
            stats["by_barrier_type"][s.barrier_type] = stats["by_barrier_type"].get(s.barrier_type, 0) + 1

            # Emotion transitions
            for i in range(len(s.emotion_sequence) - 1):
                transition = f"{s.emotion_sequence[i]}->{s.emotion_sequence[i+1]}"
                stats["emotion_transitions"][transition] = stats["emotion_transitions"].get(transition, 0) + 1

            # Functions
            for func in s.function_sequence:
                stats["by_function"][func] = stats["by_function"].get(func, 0) + 1

            # Nouns
            if s.proper_nouns_introduced:
                stats["total_new_nouns"] += len(s.proper_nouns_introduced)
                stats["synthetics_with_new_nouns"] += 1
                stats["unique_new_nouns"].update(s.proper_nouns_introduced)

            # Quality
            stats["avg_confidence"] += s.translation_confidence
            if s.schema_valid:
                stats["schema_valid_count"] += 1
            if s.curator_approved:
                stats["curator_approved_count"] += 1

            # Costs
            stats["total_cost_usd"] += s.cost_usd
            stats["total_latency_ms"] += s.latency_ms

        # Finalize
        stats["avg_confidence"] /= len(synthetics)
        stats["unique_new_nouns"] = list(stats["unique_new_nouns"])
        stats["schema_valid_rate"] = stats["schema_valid_count"] / len(synthetics)
        stats["curator_approved_rate"] = stats["curator_approved_count"] / len(synthetics)
        stats["avg_cost_usd"] = stats["total_cost_usd"] / len(synthetics)
        stats["avg_latency_ms"] = stats["total_latency_ms"] / len(synthetics)

        # Cache
        self._stats_path.write_text(json.dumps(stats, indent=2))

        return stats

    def query_failures(self, stage: Optional[str] = None) -> list[dict]:
        """Query failed workflow traces."""
        failures = []
        for trace in self.load_traces():
            if trace["status"] in ("failed", "rejected"):
                if stage is None or trace.get("failure_stage") == stage:
                    failures.append({
                        "workflow_id": trace["workflow_id"],
                        "stage": trace.get("failure_stage"),
                        "reason": trace.get("failure_reason"),
                        "source_walk": trace.get("source_walk"),
                    })
        return failures

    def query_by_arc_shape(self, arc_shape: str) -> list[LinkedSynthetic]:
        """Get all synthetics with a specific arc shape."""
        if not self._loaded:
            self.load_synthetics()
        return [s for s in self._synthetics if s.arc_shape == arc_shape]

    def query_by_emotion_transition(self, from_emotion: str, to_emotion: str) -> list[LinkedSynthetic]:
        """Get synthetics containing a specific emotion transition."""
        if not self._loaded:
            self.load_synthetics()

        results = []
        for s in self._synthetics:
            for i in range(len(s.emotion_sequence) - 1):
                if s.emotion_sequence[i] == from_emotion and s.emotion_sequence[i+1] == to_emotion:
                    results.append(s)
                    break
        return results

    def query_by_source_node(self, node_id: str) -> list[LinkedSynthetic]:
        """Find synthetics derived from a specific source graph node."""
        if not self._loaded:
            self.load_synthetics()
        return [s for s in self._synthetics if node_id in s.source_walk_ids]

    def sample_trajectories(self, n: int = 10) -> list[dict]:
        """
        Sample connected trajectories through the synthetic corpus.

        A trajectory is a sequence of synthetics that share source nodes
        or have emotion continuity.
        """
        if not self._loaded:
            self.load_synthetics()

        import random

        if len(self._synthetics) < n:
            return [{"synthetic": s.to_dict(), "connections": []} for s in self._synthetics]

        # Build adjacency by source node overlap
        node_to_synthetics: dict[str, list[int]] = {}
        for i, s in enumerate(self._synthetics):
            for node_id in s.source_walk_ids:
                if node_id not in node_to_synthetics:
                    node_to_synthetics[node_id] = []
                node_to_synthetics[node_id].append(i)

        trajectories = []
        sampled_indices = random.sample(range(len(self._synthetics)), min(n, len(self._synthetics)))

        for idx in sampled_indices:
            s = self._synthetics[idx]
            connections = []

            # Find connected synthetics via shared source nodes
            for node_id in s.source_walk_ids:
                for other_idx in node_to_synthetics.get(node_id, []):
                    if other_idx != idx:
                        connections.append({
                            "synthetic_id": self._synthetics[other_idx].synthetic_id,
                            "shared_node": node_id,
                        })

            trajectories.append({
                "synthetic": s.to_dict(),
                "connections": connections[:5],  # Limit connections
            })

        return trajectories

    def missing_transitions(self, reference_matrix: dict[str, dict[str, int]]) -> list[tuple[str, str]]:
        """
        Compare emotion transition matrix against a reference.

        Returns transitions present in reference but missing/sparse in synthetics.
        """
        stats = self.stats()
        synthetic_transitions = stats.get("emotion_transitions", {})

        missing = []
        for from_emo, to_dict in reference_matrix.items():
            for to_emo, expected_count in to_dict.items():
                key = f"{from_emo}->{to_emo}"
                actual = synthetic_transitions.get(key, 0)
                # Consider "missing" if less than 10% of expected
                if expected_count > 0 and actual < expected_count * 0.1:
                    missing.append((from_emo, to_emo))

        return missing


class RunManager:
    """
    Manages trial runs across sessions.
    """

    def __init__(self, base_dir: Path | str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        target_bible: str,
        source_games: list[str],
        sample_count: int = 0,
        sample_rate: float = 0.0,
        **config_kwargs,
    ) -> TrialRun:
        """Create a new trial run."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{target_bible}"
        run_dir = self.base_dir / run_id

        config = RunConfig(
            run_id=run_id,
            target_bible=target_bible,
            source_games=source_games,
            sample_count=sample_count,
            sample_rate=sample_rate,
            **config_kwargs,
        )

        run = TrialRun(run_dir, config)
        run.save_config()

        return run

    def load_run(self, run_id: str) -> Optional[TrialRun]:
        """Load an existing run by ID."""
        run_dir = self.base_dir / run_id
        config_path = run_dir / "config.yaml"

        if not config_path.exists():
            return None

        config = RunConfig.from_yaml(config_path.read_text())
        return TrialRun(run_dir, config)

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        return sorted([
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "config.yaml").exists()
        ], reverse=True)

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """Compare statistics between two runs."""
        run_a = self.load_run(run_id_a)
        run_b = self.load_run(run_id_b)

        if not run_a or not run_b:
            return {"error": "One or both runs not found"}

        stats_a = run_a.stats()
        stats_b = run_b.stats()

        comparison = {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "count_diff": stats_b["count"] - stats_a["count"],
            "confidence_diff": stats_b["avg_confidence"] - stats_a["avg_confidence"],
            "cost_diff": stats_b["total_cost_usd"] - stats_a["total_cost_usd"],

            # Arc shape distribution changes
            "arc_shape_changes": {},

            # Emotion transition changes
            "transition_changes": {},
        }

        # Arc shapes
        all_shapes = set(stats_a.get("by_arc_shape", {}).keys()) | set(stats_b.get("by_arc_shape", {}).keys())
        for shape in all_shapes:
            a_count = stats_a.get("by_arc_shape", {}).get(shape, 0)
            b_count = stats_b.get("by_arc_shape", {}).get(shape, 0)
            if a_count != b_count:
                comparison["arc_shape_changes"][shape] = {"a": a_count, "b": b_count, "diff": b_count - a_count}

        # Transitions
        all_trans = set(stats_a.get("emotion_transitions", {}).keys()) | set(stats_b.get("emotion_transitions", {}).keys())
        for trans in all_trans:
            a_count = stats_a.get("emotion_transitions", {}).get(trans, 0)
            b_count = stats_b.get("emotion_transitions", {}).get(trans, 0)
            if abs(a_count - b_count) > 5:  # Only report significant changes
                comparison["transition_changes"][trans] = {"a": a_count, "b": b_count, "diff": b_count - a_count}

        return comparison
