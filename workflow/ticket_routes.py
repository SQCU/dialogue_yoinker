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
    worker_type: str = Field(description="'structural_parser', 'translation_engine', 'lore_curator', or 'link_stitcher'")
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
    worker_backend: str = Field(default="", description="Model/backend that processed (e.g., 'deepseek-chat')")


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
        worker_backend=request.worker_backend or None,
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


# =============================================================================
# Linking Run Endpoints
# =============================================================================

class CreateLinkingRunRequest(BaseModel):
    """Request to create a linking run."""
    target_setting: str = Field(description="Target synthetic setting (e.g., 'gallia')")
    version: int = Field(default=3, description="Synthetic graph version")
    sample_count: int = Field(default=50, description="Number of link-stitch tickets to create")
    target_degree: int = Field(default=5, ge=2, le=50, description="Target out-degree for nodes")
    n_choices: int = Field(default=5, ge=2, le=20, description="Candidate targets per source node")
    n_links_out: int = Field(default=3, ge=1, le=10, description="Links to create per source node")
    reference_game: str = Field(default="oblivion", description="Reference game for few-shot examples")


class CreateLinkingRunResponse(BaseModel):
    """Response from creating a linking run."""
    run_id: str
    tickets_created: int
    status: dict


@router.post("/linking", response_model=CreateLinkingRunResponse)
async def create_linking_run(request: CreateLinkingRunRequest):
    """
    Create a linking run to improve graph topology.

    Uses the topology gap analysis to find underlinked nodes and creates
    link_stitch tickets with candidate targets sampled from the graph.
    """
    from datetime import datetime, timezone
    import json
    import random
    from pathlib import Path

    manager = get_manager()

    # Load synthetic graph
    # Handle both "gallia" and "gallia_v3" formats
    if f"_v{request.version}" in request.target_setting:
        graph_path = Path("synthetic") / request.target_setting / "graph.json"
    else:
        graph_path = Path("synthetic") / f"{request.target_setting}_v{request.version}" / "graph.json"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_path}")

    graph_data = json.loads(graph_path.read_text())
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        raise HTTPException(status_code=400, detail="Graph has no nodes")

    # Build index
    node_by_id = {n["id"]: n for n in nodes}
    out_edges = {}
    in_edges = {}
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        if src and tgt:
            out_edges.setdefault(src, []).append(tgt)
            in_edges.setdefault(tgt, []).append(src)

    # Find underlinked nodes (out_degree < target_degree)
    underlinked = []
    for node in nodes:
        node_id = node["id"]
        current_out = len(out_edges.get(node_id, []))
        if current_out < request.target_degree:
            underlinked.append({
                "id": node_id,
                "text": node.get("text", ""),
                "emotion": node.get("emotion", "neutral"),
                "current_out_degree": current_out,
                "target_out_degree": request.target_degree,
            })

    if not underlinked:
        raise HTTPException(status_code=400, detail="No underlinked nodes found")

    # Sample nodes to create tickets for
    random.shuffle(underlinked)
    selected = underlinked[:request.sample_count]

    # Find potential targets (any node with different id)
    potential_targets = []
    for node in nodes:
        if len(potential_targets) >= 500:
            break
        potential_targets.append({
            "id": node["id"],
            "text": node.get("text", ""),
            "emotion": node.get("emotion", "neutral"),
        })

    # Create run
    # Use consistent naming (avoid double version suffix)
    setting_for_run = request.target_setting if f"_v{request.version}" in request.target_setting else f"{request.target_setting}_v{request.version}"
    run_id = f"link_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{setting_for_run}"

    from .ticket_queue import RunQueue, WorkerType
    queue = RunQueue(
        run_id=run_id,
        target_bible=request.target_setting,
        source_games=[request.reference_game],
    )

    # Create link_stitch tickets
    for source_node in selected:
        # Sample candidate targets (excluding source)
        candidates = [t for t in potential_targets if t["id"] != source_node["id"]]
        random.shuffle(candidates)
        sampled_targets = candidates[:request.n_choices]

        # Add transition_required to each target
        for target in sampled_targets:
            target["transition_required"] = f"{source_node['emotion']}→{target['emotion']}"
            # Add context (could be enhanced later)
            target["context"] = []

        # Add context to source (could be enhanced later)
        source_node["context"] = []

        queue.add_link_stitch_ticket(
            source_node=source_node,
            candidate_targets=sampled_targets,
            reference_examples=[],  # TODO: sample from reference graph
            link_params={
                "n_choices": request.n_choices,
                "n_links_out": request.n_links_out,
                "max_bridge_length": 1,
            },
        )

    # Save queue
    with manager._lock:
        manager._queues[run_id] = queue
        manager._save_queue(queue)

    return CreateLinkingRunResponse(
        run_id=run_id,
        tickets_created=len(queue.link_stitch_tickets),
        status=queue.status(),
    )


@router.get("/{run_id}/extension_candidates")
async def get_extension_candidates(run_id: str):
    """Get extension candidates collected from link-stitcher output."""
    manager = get_manager()
    queue = manager.get_queue(run_id)

    if not queue:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run_id,
        "count": len(queue.extension_candidates),
        "candidates": queue.extension_candidates,
    }


# =============================================================================
# Extension Resolution Endpoints
# =============================================================================

class CreateExtensionRunRequest(BaseModel):
    """Request to create an extension resolution run."""
    source_run_id: str = Field(description="Run ID that has extension candidates to resolve")
    sample_count: int = Field(default=20, ge=1, le=100, description="Number of candidates to resolve")
    target_setting: str = Field(description="Target synthetic setting (e.g., 'gallia')")
    version: int = Field(default=4, description="Synthetic graph version")


class CreateExtensionRunResponse(BaseModel):
    """Response from creating an extension resolution run."""
    run_id: str
    tickets_created: int
    candidates_consumed: int
    status: dict


@router.post("/extension", response_model=CreateExtensionRunResponse)
async def create_extension_run(request: CreateExtensionRunRequest):
    """
    Create an extension resolution run to consume extension candidates.

    Takes extension candidates from a previous run (usually a linking run)
    and creates extension_resolve tickets to generate bridging content.
    """
    from datetime import datetime, timezone
    import json
    from pathlib import Path

    manager = get_manager()

    # Get source run with extension candidates
    source_queue = manager.get_queue(request.source_run_id)
    if not source_queue:
        raise HTTPException(status_code=404, detail=f"Source run not found: {request.source_run_id}")

    candidates = source_queue.extension_candidates
    if not candidates:
        raise HTTPException(status_code=400, detail="No extension candidates in source run")

    # Load synthetic graph to get node details
    if f"_v{request.version}" in request.target_setting:
        graph_path = Path("synthetic") / request.target_setting / "graph.json"
    else:
        graph_path = Path("synthetic") / f"{request.target_setting}_v{request.version}" / "graph.json"

    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_path}")

    graph_data = json.loads(graph_path.read_text())
    nodes = graph_data.get("nodes", [])
    node_by_id = {n["id"]: n for n in nodes}

    # Sample candidates
    import random
    sampled = candidates[:request.sample_count]
    random.shuffle(sampled)

    # Create run
    setting_name = request.target_setting if f"_v{request.version}" in request.target_setting else f"{request.target_setting}_v{request.version}"
    run_id = f"ext_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{setting_name}"

    from .ticket_queue import RunQueue
    queue = RunQueue(
        run_id=run_id,
        target_bible=request.target_setting.split("_v")[0],  # Extract base setting
        source_games=source_queue.source_games,
    )

    tickets_created = 0
    for candidate in sampled:
        # Parse site to get source and target IDs
        site = candidate.get("site", "")
        if "→" in site:
            source_id, target_id = site.split("→")
        elif "->" in site:
            source_id, target_id = site.split("->")
        else:
            continue  # Invalid site format

        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)

        if not source_node or not target_node:
            continue  # Nodes not found in graph

        queue.add_extension_resolve_ticket(
            extension_candidate=candidate,
            source_node={
                "id": source_id,
                "text": source_node.get("text", ""),
                "emotion": source_node.get("emotion", "neutral"),
                "context": [],
            },
            target_node={
                "id": target_id,
                "text": target_node.get("text", ""),
                "emotion": target_node.get("emotion", "neutral"),
                "context": [],
            },
        )
        tickets_created += 1

    if tickets_created == 0:
        raise HTTPException(status_code=400, detail="No valid extension candidates could be processed")

    # Save queue
    with manager._lock:
        manager._queues[run_id] = queue
        manager._save_queue(queue)

    return CreateExtensionRunResponse(
        run_id=run_id,
        tickets_created=tickets_created,
        candidates_consumed=len(sampled),
        status=queue.status(),
    )


@router.get("/{run_id}/resolved_candidates")
async def get_resolved_candidates(run_id: str):
    """Get resolved extension candidates."""
    manager = get_manager()
    queue = manager.get_queue(run_id)

    if not queue:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run_id,
        "count": len(queue.resolved_candidates),
        "candidates": queue.resolved_candidates,
    }
