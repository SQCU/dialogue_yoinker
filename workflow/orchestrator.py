#!/usr/bin/env python3
"""
Spawn-and-Await Orchestrator for Dialogue Translation Pipeline

Launches Claude Code subagents that independently:
1. Claim tickets from the queue API
2. Process the work
3. Submit results via the queue API

The orchestrator doesn't hold results in memory - it just monitors
the queue status until all work is complete.

Usage:
    python -m workflow.orchestrator --target gallia --sample-count 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import httpx


# =============================================================================
# Configuration
# =============================================================================

API_BASE = "http://127.0.0.1:8000"

# Prompts for each worker type
WORKER_PROMPTS = {
    "structural_parser": """You are a structural parser worker. Your task:

1. Claim a parse ticket from the queue
2. Parse the dialogue walk into a structural triplet
3. Submit your result

Use these curl commands:

CLAIM:
```bash
curl -X POST {api}/api/runs/{run_id}/claim \\
  -H "Content-Type: application/json" \\
  -d '{{"worker_type": "structural_parser", "worker_id": "{worker_id}"}}'
```

SUBMIT (after parsing):
```bash
curl -X POST {api}/api/runs/{run_id}/submit \\
  -H "Content-Type: application/json" \\
  -d '{{
    "ticket_id": "<ticket_id from claim>",
    "output_data": <your parsed triplet>,
    "worker_notes": [],
    "worker_concerns": []
  }}'
```

Keep claiming and submitting until no more tickets are available.
Follow the structural-parser agent instructions for proper labeling.
""",

    "translation_engine": """You are a translation engine worker. Your task:

1. Claim a translate ticket from the queue
2. Translate the structural triplet to the target setting
3. Submit your result

Use these curl commands:

CLAIM:
```bash
curl -X POST {api}/api/runs/{run_id}/claim \\
  -H "Content-Type: application/json" \\
  -d '{{"worker_type": "translation_engine", "worker_id": "{worker_id}"}}'
```

SUBMIT (after translating):
```bash
curl -X POST {api}/api/runs/{run_id}/submit \\
  -H "Content-Type: application/json" \\
  -d '{{
    "ticket_id": "<ticket_id from claim>",
    "output_data": {{
      "translated_texts": [...],
      "proper_nouns_introduced": [...],
      "synthetic_conditions": [{{"type": "quest_stage", "quest": "...", "stage": ">=1"}}, ...],
      "speaker": "...",
      "synthetic_topic": "...",
      "register_notes": "...",
      "structural_fidelity": {{...}},
      "confidence": 0.9
    }},
    "worker_notes": [],
    "worker_concerns": []
  }}'
```

Keep claiming and submitting until no more tickets are available.
Follow the translation-engine agent instructions for proper translation.
""",

    "lore_curator": """You are a lore curator worker. Your task:

1. Claim a curate ticket from the queue
2. Validate the proposed proper noun against the target lore bible
3. Submit your verdict

Use these curl commands:

CLAIM:
```bash
curl -X POST {api}/api/runs/{run_id}/claim \\
  -H "Content-Type: application/json" \\
  -d '{{"worker_type": "lore_curator", "worker_id": "{worker_id}"}}'
```

SUBMIT (after curation):
```bash
curl -X POST {api}/api/runs/{run_id}/submit \\
  -H "Content-Type: application/json" \\
  -d '{{
    "ticket_id": "<ticket_id from claim>",
    "output_data": {{
      "verdict": "approve" | "reject" | "modify",
      "reason": "...",
      "suggested_alternative": null | "..."
    }},
    "worker_notes": [],
    "worker_concerns": []
  }}'
```

Keep claiming and submitting until no more tickets are available.
Follow the lore-curator agent instructions for proper validation.
""",
}


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class RunStatus:
    """Status of a pipeline run."""
    run_id: str
    parse_pending: int
    parse_completed: int
    translate_pending: int
    translate_completed: int
    curate_pending: int
    curate_completed: int

    @property
    def total_pending(self) -> int:
        return self.parse_pending + self.translate_pending + self.curate_pending

    @property
    def total_completed(self) -> int:
        return self.parse_completed + self.translate_completed + self.curate_completed

    @property
    def is_complete(self) -> bool:
        return self.total_pending == 0

    @classmethod
    def from_api(cls, data: dict) -> "RunStatus":
        return cls(
            run_id=data["run_id"],
            parse_pending=data["parse"]["pending"] + data["parse"]["claimed"],
            parse_completed=data["parse"]["completed"],
            translate_pending=data["translate"]["pending"] + data["translate"]["claimed"],
            translate_completed=data["translate"]["completed"],
            curate_pending=data["curate"]["pending"] + data["curate"]["claimed"],
            curate_completed=data["curate"]["completed"],
        )


# =============================================================================
# API Client
# =============================================================================

class PipelineAPI:
    """Client for the dialogue pipeline API."""

    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def create_run(
        self,
        target_bible: str,
        source_games: list[str],
        sample_count: int = 10,
        max_walk_length: int = 6,
    ) -> str:
        """Create a new run and return its ID."""
        response = await self.client.post(
            f"{self.base_url}/api/runs",
            json={
                "target_bible": target_bible,
                "source_games": source_games,
                "sample_count": sample_count,
                "max_walk_length": max_walk_length,
            }
        )
        response.raise_for_status()
        return response.json()["run_id"]

    async def get_status(self, run_id: str) -> RunStatus:
        """Get current run status."""
        response = await self.client.get(
            f"{self.base_url}/api/runs/{run_id}/status"
        )
        response.raise_for_status()
        return RunStatus.from_api(response.json())

    async def get_concerns(self, run_id: str) -> list[dict]:
        """Get all worker concerns for a run."""
        response = await self.client.get(
            f"{self.base_url}/api/runs/{run_id}/concerns"
        )
        response.raise_for_status()
        return response.json()["concerns"]

    async def close(self):
        await self.client.aclose()


# =============================================================================
# Orchestrator
# =============================================================================

class PipelineOrchestrator:
    """
    Orchestrates the dialogue translation pipeline.

    Spawns workers that independently claim and process tickets.
    Monitors progress without holding results in memory.
    """

    def __init__(self, api: PipelineAPI):
        self.api = api

    async def run_pipeline(
        self,
        target_bible: str,
        source_games: list[str],
        sample_count: int = 10,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,  # 10 minutes
    ) -> dict:
        """
        Run the full pipeline.

        Returns summary of completed work and any concerns raised.
        """
        print(f"Creating run: {target_bible} from {source_games}, {sample_count} samples")

        # Create run
        run_id = await self.api.create_run(
            target_bible=target_bible,
            source_games=source_games,
            sample_count=sample_count,
        )
        print(f"Created run: {run_id}")

        # Get initial status
        status = await self.api.get_status(run_id)
        print(f"Initial: {status.parse_pending} parse tickets pending")

        # Print worker launch instructions
        print("\n" + "="*60)
        print("WORKER LAUNCH INSTRUCTIONS")
        print("="*60)
        print(f"\nRun ID: {run_id}")
        print(f"API Base: {self.api.base_url}")
        print("\nTo spawn workers, launch Claude Code subagents with these prompts:\n")

        for worker_type in ["structural_parser", "translation_engine", "lore_curator"]:
            print(f"--- {worker_type} ---")
            prompt = WORKER_PROMPTS[worker_type].format(
                api=self.api.base_url,
                run_id=run_id,
                worker_id=f"{worker_type}_001",
            )
            print(prompt[:500] + "...\n")

        print("="*60)
        print("Monitoring queue status...")
        print("="*60 + "\n")

        # Monitor until complete or timeout
        start_time = time.time()
        last_status = None

        while time.time() - start_time < max_wait:
            status = await self.api.get_status(run_id)

            # Print status if changed
            status_str = (
                f"parse: {status.parse_completed}/{status.parse_pending + status.parse_completed} | "
                f"translate: {status.translate_completed}/{status.translate_pending + status.translate_completed} | "
                f"curate: {status.curate_completed}/{status.curate_pending + status.curate_completed}"
            )
            if status_str != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] {status_str}")
                last_status = status_str

            if status.is_complete:
                print("\nPipeline complete!")
                break

            await asyncio.sleep(poll_interval)

        # Get final concerns
        concerns = await self.api.get_concerns(run_id)

        return {
            "run_id": run_id,
            "status": {
                "parse_completed": status.parse_completed,
                "translate_completed": status.translate_completed,
                "curate_completed": status.curate_completed,
            },
            "concerns": concerns,
            "elapsed_seconds": time.time() - start_time,
        }


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Run dialogue translation pipeline")
    parser.add_argument("--target", default="gallia", help="Target lore bible")
    parser.add_argument("--sources", nargs="+", default=["falloutnv", "oblivion"], help="Source games")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of samples")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Status poll interval")
    parser.add_argument("--api", default=API_BASE, help="API base URL")

    args = parser.parse_args()

    api = PipelineAPI(args.api)
    orchestrator = PipelineOrchestrator(api)

    try:
        result = await orchestrator.run_pipeline(
            target_bible=args.target,
            source_games=args.sources,
            sample_count=args.sample_count,
            poll_interval=args.poll_interval,
        )

        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))

        if result["concerns"]:
            print("\n--- WORKER CONCERNS ---")
            for concern in result["concerns"]:
                print(f"  [{concern['level']}] {concern['ticket_id']}: {concern['message']}")

    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
