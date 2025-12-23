"""
API Routes for Ticket Queue System

Provides endpoints for subagents to claim and submit work,
enabling decoupled orchestration where persistence doesn't
depend on orchestrator memory.

Endpoints:
    POST /api/runs                     - Create a run with populated queue
    GET  /api/runs/{run_id}/status     - Get queue status
    POST /api/runs/{run_id}/claim      - Claim a ticket
    POST /api/runs/{run_id}/submit     - Submit ticket result
    GET  /api/runs/{run_id}/concerns   - Get all worker concerns
"""

from __future__ import annotations

from typing import Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
import httpx

from .ticket_queue import get_manager, TicketQueueManager


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/runs", tags=["ticket-queue"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateRunRequest(BaseModel):
    """Request to create a new run."""
    target_bible: str = Field(description="Target setting for translation (e.g., 'gallia')")
    source_games: list[str] = Field(default=["oblivion", "falloutnv"])
    sample_count: int = Field(default=0, description="Fixed sample count (overrides rate)")
    sample_rate: float = Field(default=0.0, description="Fraction of corpus (e.g., 0.01 for 1%)")
    walk_method: str = Field(default="walk", description="Sampling method")
    max_walk_length: int = Field(default=6)
    config: dict = Field(default_factory=dict)


class CreateRunResponse(BaseModel):
    """Response from creating a run."""
    run_id: str
    tickets_created: int
    status: dict


class ClaimTicketRequest(BaseModel):
    """Request to claim a ticket."""
    worker_type: str = Field(description="'structural_parser', 'translation_engine', or 'lore_curator'")
    worker_id: str = Field(default="", description="Optional worker identifier")


class ClaimTicketResponse(BaseModel):
    """Response from claiming a ticket."""
    success: bool
    ticket: Optional[dict] = None
    message: str = ""


class SubmitTicketRequest(BaseModel):
    """Request to submit ticket result."""
    ticket_id: str
    output_data: dict = Field(description="The work result")
    worker_notes: list[str] = Field(default_factory=list, description="Notes from worker")
    worker_concerns: list[dict] = Field(
        default_factory=list,
        description="Structured concerns: [{level: 'info'|'review'|'error', message: str, suggestion: str}]"
    )
    latency_ms: int = Field(default=0)


class SubmitTicketResponse(BaseModel):
    """Response from submitting a ticket."""
    success: bool
    status: dict = Field(default_factory=dict)
    error: str = ""


# =============================================================================
# API Client Helper
# =============================================================================

API_BASE = "http://127.0.0.1:8000"


async def sample_walks_from_api(game: str, count: int, method: str, max_length: int) -> list[dict]:
    """Sample walks from the dialogue graph API (batches requests to stay under limit)."""
    MAX_PER_REQUEST = 20
    all_walks = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        remaining = count
        while remaining > 0:
            batch_size = min(remaining, MAX_PER_REQUEST)
            response = await client.post(
                f"{API_BASE}/api/sample",
                json={
                    "game": game,
                    "method": method,
                    "count": batch_size,
                    "max_length": max_length,
                }
            )

            if response.status_code != 200:
                break

            data = response.json()
            for s in data.get("samples", []):
                all_walks.append((game, s["nodes"]))

            remaining -= batch_size

    return all_walks


async def get_corpus_size(game: str) -> int:
    """Get corpus size for a game."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{API_BASE}/api/games")
        for g in response.json():
            if g["name"] == game:
                return g["dialogue_count"]
    return 0


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=CreateRunResponse)
async def create_run(request: CreateRunRequest):
    """
    Create a new run and populate the ticket queue.

    The queue is automatically populated with parse tickets for sampled walks.
    Downstream tickets (translate, curate) are created as parse tickets complete.
    """
    manager = get_manager()

    # Sample walks from each game
    all_walks = []
    for game in request.source_games:
        corpus_size = await get_corpus_size(game)
        if corpus_size == 0:
            continue

        if request.sample_rate > 0:
            count = max(1, int(corpus_size * request.sample_rate))
        elif request.sample_count > 0:
            count = request.sample_count // len(request.source_games)
        else:
            count = 10  # Default

        walks = await sample_walks_from_api(
            game, count, request.walk_method, request.max_walk_length
        )
        all_walks.extend(walks)

    if not all_walks:
        raise HTTPException(status_code=400, detail="No walks sampled")

    # Create run
    queue = manager.create_run(
        target_bible=request.target_bible,
        source_games=request.source_games,
        walks=all_walks,
        config={
            "sample_rate": request.sample_rate,
            "sample_count": request.sample_count,
            "walk_method": request.walk_method,
            "max_walk_length": request.max_walk_length,
            **request.config,
        }
    )

    return CreateRunResponse(
        run_id=queue.run_id,
        tickets_created=len(queue.parse_tickets),
        status=queue.status(),
    )


@router.get("/{run_id}/status")
async def get_run_status(run_id: str):
    """Get current status of a run's ticket queue."""
    manager = get_manager()
    queue = manager.get_queue(run_id)

    if not queue:
        raise HTTPException(status_code=404, detail="Run not found")

    return queue.status()


@router.post("/{run_id}/claim", response_model=ClaimTicketResponse)
async def claim_ticket(run_id: str, request: ClaimTicketRequest):
    """
    Claim a ticket from the queue.

    Workers call this to get work. Returns the next pending ticket
    of the requested type, or null if none available.
    """
    manager = get_manager()

    ticket_dict = manager.claim_ticket(
        run_id=run_id,
        worker_type=request.worker_type,
        worker_id=request.worker_id,
    )

    if ticket_dict:
        return ClaimTicketResponse(
            success=True,
            ticket=ticket_dict,
            message="Ticket claimed",
        )
    else:
        return ClaimTicketResponse(
            success=False,
            ticket=None,
            message="No pending tickets of this type",
        )


@router.post("/{run_id}/submit", response_model=SubmitTicketResponse)
async def submit_ticket(run_id: str, request: SubmitTicketRequest):
    """
    Submit completed work for a ticket.

    This auto-persists the result and creates downstream tickets:
    - Parse completion → creates translate ticket
    - Translate completion → creates curate tickets for new nouns
    """
    manager = get_manager()

    result = manager.submit_ticket(
        run_id=run_id,
        ticket_id=request.ticket_id,
        output_data=request.output_data,
        worker_notes=request.worker_notes,
        worker_concerns=request.worker_concerns,
        latency_ms=request.latency_ms,
    )

    return SubmitTicketResponse(
        success=result.get("success", False),
        status=result.get("status", {}),
        error=result.get("error", ""),
    )


@router.get("/{run_id}/concerns")
async def get_concerns(run_id: str):
    """Get all worker concerns raised during the run."""
    manager = get_manager()
    queue = manager.get_queue(run_id)

    if not queue:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run_id,
        "concerns": queue.all_concerns(),
    }


@router.get("/{run_id}/tickets/{ticket_id}")
async def get_ticket(run_id: str, ticket_id: str):
    """Get a specific ticket's details."""
    manager = get_manager()
    queue = manager.get_queue(run_id)

    if not queue:
        raise HTTPException(status_code=404, detail="Run not found")

    ticket = queue.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return ticket.to_dict()


@router.get("")
async def list_runs():
    """List all runs."""
    manager = get_manager()
    runs_dir = manager.base_dir

    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "queue.json").exists():
            queue = manager.get_queue(run_dir.name)
            if queue:
                runs.append({
                    "run_id": queue.run_id,
                    "target_bible": queue.target_bible,
                    "created_at": queue.created_at,
                    "status": queue.status(),
                })

    return {"runs": sorted(runs, key=lambda r: r["created_at"], reverse=True)}
