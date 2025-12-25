"""
Observability Infrastructure for Subagent Calls

Provides structured logging, tracing, and persistence for debugging
and root cause analysis of subagent orchestration workflows.

Key concepts:
    - SubagentLog: A single API call to a subagent
    - WorkflowTrace: A complete workflow (sample -> extract -> translate -> persist)
    - TraceStore: Persistent storage for traces (JSONL files)
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
from enum import Enum
import uuid


class SubagentType(str, Enum):
    """The three subagent types in the architecture."""
    TRIPLET_EXTRACTOR = "triplet_extractor"
    LORE_CURATOR = "lore_curator"
    TRANSLATION_ENGINE = "translation_engine"


class WorkflowStatus(str, Enum):
    """Status of a workflow."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # Curator rejected the output


@dataclass
class SubagentLog:
    """
    A single subagent invocation with full context for debugging.

    Captures everything needed to reproduce and diagnose issues:
    - What was sent (input)
    - What came back (output)
    - How long it took
    - What model was used
    - Any parsing errors
    """
    # Identity
    log_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow_id: str = ""
    subagent_type: SubagentType = SubagentType.TRIPLET_EXTRACTOR

    # Model info
    model: str = ""
    system_prompt_hash: str = ""  # SHA256 of system prompt (not full text)

    # Request
    input_payload: dict = field(default_factory=dict)
    user_message: str = ""

    # Response
    output_raw: str = ""
    output_parsed: Optional[dict] = None
    parse_success: bool = False
    parse_errors: list[str] = field(default_factory=list)

    # Metrics
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["subagent_type"] = self.subagent_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubagentLog":
        """Reconstruct from dictionary."""
        d = d.copy()
        d["subagent_type"] = SubagentType(d["subagent_type"])
        return cls(**d)

    @property
    def cost_usd(self) -> float:
        """
        Estimate cost in USD based on model and tokens.
        Prices as of Dec 2024 (approximate).
        """
        # Price per 1M tokens (input, output)
        prices = {
            "claude-3-haiku-20240307": (0.25, 1.25),
            "claude-3-5-haiku-20241022": (0.80, 4.00),
            "claude-sonnet-4-20250514": (3.00, 15.00),
            "claude-3-5-sonnet-20241022": (3.00, 15.00),
            "claude-opus-4-20250514": (15.00, 75.00),
        }

        input_price, output_price = prices.get(self.model, (3.0, 15.0))
        return (self.input_tokens * input_price + self.output_tokens * output_price) / 1_000_000


@dataclass
class WorkflowTrace:
    """
    A complete workflow trace from source walk to synthetic output.

    Links together all subagent calls that contributed to a single
    synthetic generation attempt.
    """
    # Identity
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: WorkflowStatus = WorkflowStatus.IN_PROGRESS

    # Source
    source_game: str = ""
    source_walk: dict = field(default_factory=dict)
    source_walk_id: str = ""

    # Target
    target_bible: str = ""

    # Subagent calls (in order)
    extractor_log: Optional[SubagentLog] = None
    translator_log: Optional[SubagentLog] = None
    curator_logs: list[SubagentLog] = field(default_factory=list)

    # Outputs
    triplet: Optional[dict] = None
    translated_texts: Optional[list[str]] = None
    final_synthetic: Optional[dict] = None

    # Failure info
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None

    # Metrics
    total_latency_ms: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    def add_extractor_log(self, log: SubagentLog) -> None:
        """Record the triplet extraction call."""
        log.workflow_id = self.workflow_id
        self.extractor_log = log
        self.total_latency_ms += log.latency_ms
        if log.output_parsed:
            self.triplet = log.output_parsed

    def add_translator_log(self, log: SubagentLog) -> None:
        """Record the translation call."""
        log.workflow_id = self.workflow_id
        self.translator_log = log
        self.total_latency_ms += log.latency_ms
        if log.output_parsed and "translated_texts" in log.output_parsed:
            self.translated_texts = log.output_parsed["translated_texts"]

    def add_curator_log(self, log: SubagentLog) -> None:
        """Record a curator validation call."""
        log.workflow_id = self.workflow_id
        self.curator_logs.append(log)
        self.total_latency_ms += log.latency_ms

    def complete(self, synthetic: Optional[dict] = None) -> None:
        """Mark workflow as completed."""
        self.status = WorkflowStatus.COMPLETED
        self.final_synthetic = synthetic
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, stage: str, reason: str) -> None:
        """Mark workflow as failed."""
        self.status = WorkflowStatus.FAILED
        self.failure_stage = stage
        self.failure_reason = reason
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def reject(self, reason: str) -> None:
        """Mark workflow as rejected by curator."""
        self.status = WorkflowStatus.REJECTED
        self.failure_stage = "curator"
        self.failure_reason = reason
        self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def total_cost_usd(self) -> float:
        """Total cost of all subagent calls."""
        cost = 0.0
        if self.extractor_log:
            cost += self.extractor_log.cost_usd
        if self.translator_log:
            cost += self.translator_log.cost_usd
        for log in self.curator_logs:
            cost += log.cost_usd
        return cost

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "source_game": self.source_game,
            "source_walk": self.source_walk,
            "source_walk_id": self.source_walk_id,
            "target_bible": self.target_bible,
            "extractor_log": self.extractor_log.to_dict() if self.extractor_log else None,
            "translator_log": self.translator_log.to_dict() if self.translator_log else None,
            "curator_logs": [log.to_dict() for log in self.curator_logs],
            "triplet": self.triplet,
            "translated_texts": self.translated_texts,
            "final_synthetic": self.final_synthetic,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowTrace":
        """Reconstruct from dictionary."""
        trace = cls(
            workflow_id=d["workflow_id"],
            status=WorkflowStatus(d["status"]),
            source_game=d["source_game"],
            source_walk=d["source_walk"],
            source_walk_id=d.get("source_walk_id", ""),
            target_bible=d["target_bible"],
            triplet=d.get("triplet"),
            translated_texts=d.get("translated_texts"),
            final_synthetic=d.get("final_synthetic"),
            failure_stage=d.get("failure_stage"),
            failure_reason=d.get("failure_reason"),
            total_latency_ms=d.get("total_latency_ms", 0),
            started_at=d["started_at"],
            completed_at=d.get("completed_at"),
        )
        if d.get("extractor_log"):
            trace.extractor_log = SubagentLog.from_dict(d["extractor_log"])
        if d.get("translator_log"):
            trace.translator_log = SubagentLog.from_dict(d["translator_log"])
        if d.get("curator_logs"):
            trace.curator_logs = [SubagentLog.from_dict(l) for l in d["curator_logs"]]
        return trace

    def summary(self) -> str:
        """One-line summary for quick inspection."""
        status_emoji = {
            WorkflowStatus.IN_PROGRESS: "...",
            WorkflowStatus.COMPLETED: "OK",
            WorkflowStatus.FAILED: "FAIL",
            WorkflowStatus.REJECTED: "REJ",
        }
        return (
            f"[{self.workflow_id}] {status_emoji[self.status]} "
            f"{self.source_game}->{self.target_bible} "
            f"{self.total_latency_ms}ms ${self.total_cost_usd:.4f}"
        )


class TraceStore:
    """
    Persistent storage for workflow traces.

    Writes to JSONL files for easy streaming/analysis.
    One file per day to keep file sizes manageable.
    """

    def __init__(self, base_dir: Path | str = "traces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast lookup
        self._index: dict[str, Path] = {}

    def _current_file(self) -> Path:
        """Get today's trace file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.base_dir / f"traces_{date_str}.jsonl"

    def save(self, trace: WorkflowTrace) -> None:
        """Append trace to today's file."""
        filepath = self._current_file()
        with open(filepath, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")
        self._index[trace.workflow_id] = filepath

    def load(self, workflow_id: str) -> Optional[WorkflowTrace]:
        """Load a specific trace by ID."""
        # Check index first
        if workflow_id in self._index:
            return self._search_file(self._index[workflow_id], workflow_id)

        # Search all files
        for filepath in sorted(self.base_dir.glob("traces_*.jsonl"), reverse=True):
            trace = self._search_file(filepath, workflow_id)
            if trace:
                self._index[workflow_id] = filepath
                return trace
        return None

    def _search_file(self, filepath: Path, workflow_id: str) -> Optional[WorkflowTrace]:
        """Search a single file for a workflow ID."""
        if not filepath.exists():
            return None
        with open(filepath) as f:
            for line in f:
                data = json.loads(line)
                if data["workflow_id"] == workflow_id:
                    return WorkflowTrace.from_dict(data)
        return None

    def list_recent(self, limit: int = 100) -> list[WorkflowTrace]:
        """List recent traces (most recent first)."""
        traces = []
        for filepath in sorted(self.base_dir.glob("traces_*.jsonl"), reverse=True):
            with open(filepath) as f:
                for line in f:
                    traces.append(WorkflowTrace.from_dict(json.loads(line)))
                    if len(traces) >= limit:
                        return traces
        return traces

    def list_failures(self, limit: int = 50) -> list[WorkflowTrace]:
        """List failed/rejected traces for debugging."""
        failures = []
        for filepath in sorted(self.base_dir.glob("traces_*.jsonl"), reverse=True):
            with open(filepath) as f:
                for line in f:
                    trace = WorkflowTrace.from_dict(json.loads(line))
                    if trace.status in (WorkflowStatus.FAILED, WorkflowStatus.REJECTED):
                        failures.append(trace)
                        if len(failures) >= limit:
                            return failures
        return failures

    def stats(self) -> dict:
        """Aggregate statistics across all traces."""
        stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "in_progress": 0,
            "total_cost_usd": 0.0,
            "total_latency_ms": 0,
            "by_source_game": {},
            "by_target_bible": {},
        }

        for filepath in self.base_dir.glob("traces_*.jsonl"):
            with open(filepath) as f:
                for line in f:
                    data = json.loads(line)
                    stats["total"] += 1
                    status = data["status"]
                    if status == "completed":
                        stats["completed"] += 1
                    elif status == "failed":
                        stats["failed"] += 1
                    elif status == "rejected":
                        stats["rejected"] += 1
                    else:
                        stats["in_progress"] += 1

                    stats["total_cost_usd"] += data.get("total_cost_usd", 0)
                    stats["total_latency_ms"] += data.get("total_latency_ms", 0)

                    game = data.get("source_game", "unknown")
                    stats["by_source_game"][game] = stats["by_source_game"].get(game, 0) + 1

                    bible = data.get("target_bible", "unknown")
                    stats["by_target_bible"][bible] = stats["by_target_bible"].get(bible, 0) + 1

        if stats["total"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total"]
            stats["avg_cost_usd"] = stats["total_cost_usd"] / stats["total"]
            stats["success_rate"] = stats["completed"] / stats["total"]

        return stats


def hash_prompt(prompt: str) -> str:
    """Create a short hash of a system prompt for logging."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: float = 0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = int((time.perf_counter() - self.start_time) * 1000)
