"""
API Routes for Multi-Backend Worker Dispatch

Allows external orchestrators (or Claude Code) to dispatch workers
to process tickets using various LLM backends.

Endpoints:
    GET  /api/dispatch/backends           - List available backends
    POST /api/dispatch/spawn              - Spawn worker on backend
    GET  /api/dispatch/workers            - List active workers
    POST /api/dispatch/await/{run_id}     - Wait for run completion
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .multi_backend import (
    WorkerDispatcher,
    BACKEND_CONFIGS,
    BackendType,
)


router = APIRouter(prefix="/api/dispatch", tags=["dispatch"])

# Global dispatcher instance
_dispatcher: Optional[WorkerDispatcher] = None


def get_dispatcher() -> WorkerDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = WorkerDispatcher()
    return _dispatcher


# =============================================================================
# Models
# =============================================================================

class SpawnRequest(BaseModel):
    """Request to spawn a worker."""
    run_id: str = Field(description="Run ID to process")
    worker_type: str = Field(description="structural_parser | translation_engine | lore_curator")
    backend: str = Field(default="local-parser", description="Backend to use (see /backends)")


class SpawnResponse(BaseModel):
    """Response from spawning a worker."""
    task_id: str
    backend: str
    worker_type: str
    run_id: str
    status: str = "spawned"


class BackendInfo(BaseModel):
    """Information about an available backend."""
    name: str
    backend_type: str
    model_id: str
    requires_key: str  # Environment variable needed
    available: bool  # Whether API key is configured


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/backends")
async def list_backends() -> dict:
    """
    List available LLM backends.

    Returns which backends are configured and ready to use.
    """
    import os

    backends = []
    for name, config in BACKEND_CONFIGS.items():
        # Check if API key is available
        if config.api_key_env:
            available = bool(os.environ.get(config.api_key_env))
        else:
            available = True  # No key needed (claude_code, local_script)

        backends.append(BackendInfo(
            name=name,
            backend_type=config.backend_type.value,
            model_id=config.model_id,
            requires_key=config.api_key_env or "(none)",
            available=available,
        ))

    return {
        "backends": [b.model_dump() for b in backends],
        "available_count": sum(1 for b in backends if b.available),
    }


@router.post("/spawn", response_model=SpawnResponse)
async def spawn_worker(request: SpawnRequest, background_tasks: BackgroundTasks):
    """
    Spawn a worker to process tickets.

    The worker runs in the background, claiming and processing tickets
    until none remain. Use /api/runs/{run_id}/status to monitor progress.

    For Claude Code orchestration:
      - Claude Code subagents use the Task tool directly (not this endpoint)
      - This endpoint is for external/API-based workers

    For external orchestration:
      - Call this endpoint to spawn workers
      - Poll /api/runs/{run_id}/status for completion
      - Or use /await/{run_id} to block until done
    """
    dispatcher = get_dispatcher()

    # Validate backend
    if request.backend not in BACKEND_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {request.backend}. See /backends for options."
        )

    # Spawn worker in background
    try:
        task_id = await dispatcher.dispatch(
            backend_name=request.backend,
            worker_type=request.worker_type,
            run_id=request.run_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SpawnResponse(
        task_id=task_id,
        backend=request.backend,
        worker_type=request.worker_type,
        run_id=request.run_id,
    )


@router.get("/workers")
async def list_workers() -> dict:
    """
    List active workers.

    Shows which workers are currently processing tickets.
    """
    dispatcher = get_dispatcher()

    workers = []
    for task_id, task in dispatcher.active_workers.items():
        workers.append({
            "task_id": task_id,
            "done": task.done(),
            "cancelled": task.cancelled(),
        })

    return {"workers": workers}


@router.post("/await/{run_id}")
async def await_run_completion(
    run_id: str,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> dict:
    """
    Wait for a run to complete.

    Blocks until all tickets are processed or timeout is reached.
    Useful for external orchestrators that want synchronous completion.
    """
    dispatcher = get_dispatcher()

    try:
        status = await dispatcher.await_completion(
            run_id=run_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        return {"completed": True, "status": status}
    except TimeoutError as e:
        raise HTTPException(status_code=408, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backends/{backend}/test")
async def test_backend(backend: str) -> dict:
    """
    Test if a backend is properly configured.

    Attempts a minimal API call to verify credentials and connectivity.
    """
    import os

    if backend not in BACKEND_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown backend: {backend}")

    config = BACKEND_CONFIGS[backend]

    # Check API key
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            return {
                "backend": backend,
                "status": "not_configured",
                "error": f"Missing environment variable: {config.api_key_env}"
            }

    # For local script, always available
    if config.backend_type == BackendType.LOCAL_SCRIPT:
        return {
            "backend": backend,
            "status": "available",
            "model": config.model_id,
        }

    # For Claude Code, check if we're in Claude Code context
    if config.backend_type == BackendType.CLAUDE_CODE:
        return {
            "backend": backend,
            "status": "available",
            "note": "Use Task tool directly, not this dispatch API",
        }

    # For API backends, we could do a test call here
    # For now, just check key presence
    return {
        "backend": backend,
        "status": "configured",
        "model": config.model_id,
        "note": "API key present, connectivity not tested",
    }
