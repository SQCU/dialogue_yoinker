"""
Ticket Queue System for Decoupled Synthetic Generation

Provides a work queue that subagents can claim from and submit to,
removing the orchestrator from the persistence path.

Architecture:
    1. Orchestrator creates a run and populates ticket queue
    2. Subagents claim tickets via tool/API
    3. Subagents submit results via tool/API (auto-persists)
    4. Orchestrator checks status, doesn't transcribe

This means:
    - Orchestrator context loss doesn't lose work
    - Any capable model can be a worker
    - Results persist even if orchestrator dies
    - Workers can raise concerns in structured format
"""

from __future__ import annotations

import json
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
from enum import Enum


class TicketStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class WorkerType(str, Enum):
    STRUCTURAL_PARSER = "structural_parser"
    TRANSLATION_ENGINE = "translation_engine"
    LORE_CURATOR = "lore_curator"


@dataclass
class Ticket:
    """A unit of work that can be claimed by a subagent."""
    ticket_id: str
    run_id: str
    worker_type: WorkerType
    status: TicketStatus = TicketStatus.PENDING

    # Input (what the worker should process)
    input_data: dict = field(default_factory=dict)

    # Output (filled by worker on submit)
    output_data: Optional[dict] = None

    # Worker feedback (concerns, suggestions, errors)
    worker_notes: list[str] = field(default_factory=list)
    worker_concerns: list[dict] = field(default_factory=list)  # {level, message, suggestion}

    # Tracking
    claimed_at: Optional[str] = None
    completed_at: Optional[str] = None
    claimed_by: Optional[str] = None  # Worker identifier

    # Metrics
    latency_ms: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["worker_type"] = self.worker_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Ticket":
        d = d.copy()
        d["status"] = TicketStatus(d["status"])
        d["worker_type"] = WorkerType(d["worker_type"])
        return cls(**d)


@dataclass
class RunQueue:
    """A queue of tickets for a single run."""
    run_id: str
    target_bible: str
    source_games: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Tickets by stage
    parse_tickets: list[Ticket] = field(default_factory=list)
    translate_tickets: list[Ticket] = field(default_factory=list)
    curate_tickets: list[Ticket] = field(default_factory=list)

    # Config
    config: dict = field(default_factory=dict)

    def add_parse_ticket(self, walk: list[dict], source_game: str) -> Ticket:
        """Add a structural parsing ticket."""
        ticket = Ticket(
            ticket_id=f"parse_{len(self.parse_tickets):04d}",
            run_id=self.run_id,
            worker_type=WorkerType.STRUCTURAL_PARSER,
            input_data={
                "walk": walk,
                "source_game": source_game,
                "reference_bible": "mojave" if source_game == "falloutnv" else "cyrodiil",
            }
        )
        self.parse_tickets.append(ticket)
        return ticket

    def add_translate_ticket(self, triplet: dict, source_game: str) -> Ticket:
        """Add a translation ticket (after parsing completes)."""
        ticket = Ticket(
            ticket_id=f"translate_{len(self.translate_tickets):04d}",
            run_id=self.run_id,
            worker_type=WorkerType.TRANSLATION_ENGINE,
            input_data={
                "triplet": triplet,
                "source_game": source_game,
                "source_bible": "mojave" if source_game == "falloutnv" else "cyrodiil",
                "target_bible": self.target_bible,
            }
        )
        self.translate_tickets.append(ticket)
        return ticket

    def add_curate_ticket(self, proper_noun: str, context: str) -> Ticket:
        """Add a curation ticket for a new proper noun."""
        ticket = Ticket(
            ticket_id=f"curate_{len(self.curate_tickets):04d}",
            run_id=self.run_id,
            worker_type=WorkerType.LORE_CURATOR,
            input_data={
                "proposed_noun": proper_noun,
                "context": context,
                "target_bible": self.target_bible,
            }
        )
        self.curate_tickets.append(ticket)
        return ticket

    def claim_ticket(self, worker_type: WorkerType, worker_id: str = "") -> Optional[Ticket]:
        """Claim the next pending ticket for a worker type."""
        tickets = {
            WorkerType.STRUCTURAL_PARSER: self.parse_tickets,
            WorkerType.TRANSLATION_ENGINE: self.translate_tickets,
            WorkerType.LORE_CURATOR: self.curate_tickets,
        }[worker_type]

        for ticket in tickets:
            if ticket.status == TicketStatus.PENDING:
                ticket.status = TicketStatus.CLAIMED
                ticket.claimed_at = datetime.now(timezone.utc).isoformat()
                ticket.claimed_by = worker_id or str(uuid.uuid4())[:8]
                return ticket

        return None

    def submit_ticket(
        self,
        ticket_id: str,
        output_data: dict,
        worker_notes: list[str] = None,
        worker_concerns: list[dict] = None,
        latency_ms: int = 0,
    ) -> bool:
        """Submit completed work for a ticket."""
        # Find ticket
        ticket = self.get_ticket(ticket_id)
        if not ticket:
            return False

        if ticket.status != TicketStatus.CLAIMED:
            return False

        # Update ticket
        ticket.output_data = output_data
        ticket.worker_notes = worker_notes or []
        ticket.worker_concerns = worker_concerns or []
        ticket.latency_ms = latency_ms
        ticket.completed_at = datetime.now(timezone.utc).isoformat()

        # Determine status based on concerns
        if any(c.get("level") == "error" for c in ticket.worker_concerns):
            ticket.status = TicketStatus.FAILED
        elif any(c.get("level") == "review" for c in ticket.worker_concerns):
            ticket.status = TicketStatus.NEEDS_REVIEW
        else:
            ticket.status = TicketStatus.COMPLETED

        return True

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Find a ticket by ID."""
        for tickets in [self.parse_tickets, self.translate_tickets, self.curate_tickets]:
            for ticket in tickets:
                if ticket.ticket_id == ticket_id:
                    return ticket
        return None

    def status(self) -> dict:
        """Get queue status summary."""
        def stage_status(tickets):
            return {
                "total": len(tickets),
                "pending": sum(1 for t in tickets if t.status == TicketStatus.PENDING),
                "claimed": sum(1 for t in tickets if t.status == TicketStatus.CLAIMED),
                "completed": sum(1 for t in tickets if t.status == TicketStatus.COMPLETED),
                "failed": sum(1 for t in tickets if t.status == TicketStatus.FAILED),
                "needs_review": sum(1 for t in tickets if t.status == TicketStatus.NEEDS_REVIEW),
            }

        return {
            "run_id": self.run_id,
            "parse": stage_status(self.parse_tickets),
            "translate": stage_status(self.translate_tickets),
            "curate": stage_status(self.curate_tickets),
        }

    def all_concerns(self) -> list[dict]:
        """Collect all worker concerns across tickets."""
        concerns = []
        for tickets in [self.parse_tickets, self.translate_tickets, self.curate_tickets]:
            for ticket in tickets:
                for concern in ticket.worker_concerns:
                    concerns.append({
                        "ticket_id": ticket.ticket_id,
                        "worker_type": ticket.worker_type.value,
                        **concern,
                    })
        return concerns


class TicketQueueManager:
    """
    Manages ticket queues with persistence.

    Thread-safe for concurrent access from multiple workers.
    """

    def __init__(self, base_dir: Path | str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._queues: dict[str, RunQueue] = {}
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls

    def create_run(
        self,
        target_bible: str,
        source_games: list[str],
        walks: list[tuple[str, list[dict]]],  # [(game, walk), ...]
        config: dict = None,
    ) -> RunQueue:
        """Create a new run and populate parse tickets."""
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{target_bible}"

        queue = RunQueue(
            run_id=run_id,
            target_bible=target_bible,
            source_games=source_games,
            config=config or {},
        )

        # Populate parse tickets
        for game, walk in walks:
            queue.add_parse_ticket(walk, game)

        with self._lock:
            self._queues[run_id] = queue
            self._save_queue(queue)

        return queue

    def get_queue(self, run_id: str) -> Optional[RunQueue]:
        """Get a queue by run ID."""
        with self._lock:
            if run_id in self._queues:
                return self._queues[run_id]

            # Try to load from disk
            queue = self._load_queue(run_id)
            if queue:
                self._queues[run_id] = queue
            return queue

    def claim_ticket(
        self,
        run_id: str,
        worker_type: str,
        worker_id: str = "",
    ) -> Optional[dict]:
        """Claim a ticket (thread-safe)."""
        with self._lock:
            queue = self.get_queue(run_id)
            if not queue:
                return None

            wt = WorkerType(worker_type)
            ticket = queue.claim_ticket(wt, worker_id)

            if ticket:
                self._save_queue(queue)
                return ticket.to_dict()

            return None

    def submit_ticket(
        self,
        run_id: str,
        ticket_id: str,
        output_data: dict,
        worker_notes: list[str] = None,
        worker_concerns: list[dict] = None,
        latency_ms: int = 0,
    ) -> dict:
        """Submit ticket result (thread-safe, auto-persists)."""
        with self._lock:
            queue = self.get_queue(run_id)
            if not queue:
                return {"success": False, "error": "Run not found"}

            success = queue.submit_ticket(
                ticket_id=ticket_id,
                output_data=output_data,
                worker_notes=worker_notes,
                worker_concerns=worker_concerns,
                latency_ms=latency_ms,
            )

            if success:
                self._save_queue(queue)

                # Auto-create follow-up tickets
                ticket = queue.get_ticket(ticket_id)
                if ticket and ticket.status == TicketStatus.COMPLETED:
                    self._create_followup_tickets(queue, ticket)
                    self._save_queue(queue)

                return {"success": True, "status": queue.status()}

            return {"success": False, "error": "Ticket not found or not claimed"}

    def _create_followup_tickets(self, queue: RunQueue, ticket: Ticket):
        """Create downstream tickets when a ticket completes."""
        if ticket.worker_type == WorkerType.STRUCTURAL_PARSER:
            # Create translation ticket
            if ticket.output_data:
                queue.add_translate_ticket(
                    triplet=ticket.output_data,
                    source_game=ticket.input_data.get("source_game", ""),
                )

        elif ticket.worker_type == WorkerType.TRANSLATION_ENGINE:
            # Create curation tickets for new proper nouns
            if ticket.output_data:
                for noun in ticket.output_data.get("proper_nouns_introduced", []):
                    queue.add_curate_ticket(
                        proper_noun=noun,
                        context=f"From translation {ticket.ticket_id}",
                    )

    def _save_queue(self, queue: RunQueue):
        """Persist queue to disk."""
        run_dir = self.base_dir / queue.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save as single JSON file for simplicity
        queue_file = run_dir / "queue.json"

        data = {
            "run_id": queue.run_id,
            "target_bible": queue.target_bible,
            "source_games": queue.source_games,
            "created_at": queue.created_at,
            "config": queue.config,
            "parse_tickets": [t.to_dict() for t in queue.parse_tickets],
            "translate_tickets": [t.to_dict() for t in queue.translate_tickets],
            "curate_tickets": [t.to_dict() for t in queue.curate_tickets],
        }

        queue_file.write_text(json.dumps(data, indent=2))

    def _load_queue(self, run_id: str) -> Optional[RunQueue]:
        """Load queue from disk."""
        queue_file = self.base_dir / run_id / "queue.json"
        if not queue_file.exists():
            return None

        data = json.loads(queue_file.read_text())

        queue = RunQueue(
            run_id=data["run_id"],
            target_bible=data["target_bible"],
            source_games=data["source_games"],
            created_at=data["created_at"],
            config=data.get("config", {}),
        )
        queue.parse_tickets = [Ticket.from_dict(t) for t in data.get("parse_tickets", [])]
        queue.translate_tickets = [Ticket.from_dict(t) for t in data.get("translate_tickets", [])]
        queue.curate_tickets = [Ticket.from_dict(t) for t in data.get("curate_tickets", [])]

        return queue


# Global manager instance for API use
_manager: Optional[TicketQueueManager] = None


def get_manager() -> TicketQueueManager:
    """Get or create global manager."""
    global _manager
    if _manager is None:
        _manager = TicketQueueManager()
    return _manager
