"""
Multi-Backend Worker Dispatcher

The ticket queue API is model-agnostic - any LLM that can make HTTP calls
can claim/submit tickets. This module provides a unified interface for
dispatching work to different backends:

  - Claude Code subagents: Task tool with automatic return
  - External APIs (OpenAI, Anthropic API, Google, DeepSeek, Qwen, Kimi):
    Spawn worker process that polls ticket queue

The key difference:
  - Claude Code: orchestrator gets result via TaskOutput
  - External: orchestrator polls /api/runs/{id}/status

Usage:
    dispatcher = WorkerDispatcher(api_base="http://localhost:8000")

    # Dispatch to Claude Code subagent (if available)
    dispatcher.dispatch_claude_code("structural_parser", run_id)

    # Dispatch to external API
    dispatcher.dispatch_external("openai:gpt-4o-mini", "structural_parser", run_id)

    # Wait for all workers
    dispatcher.await_completion(run_id)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import httpx


# =============================================================================
# Backend Registry
# =============================================================================

class BackendType(str, Enum):
    CLAUDE_CODE = "claude_code"      # Via Task tool (automatic return)
    ANTHROPIC_API = "anthropic_api"  # Via Anthropic SDK
    OPENAI_API = "openai_api"        # Via OpenAI SDK
    GOOGLE_API = "google_api"        # Via Google GenAI SDK
    DEEPSEEK_API = "deepseek_api"    # Via DeepSeek API
    QWEN_API = "qwen_api"            # Via Alibaba Qwen API
    KIMI_API = "kimi_api"            # Via Moonshot Kimi API
    LOCAL_SCRIPT = "local_script"    # Python script (no LLM, rule-based)


@dataclass
class BackendConfig:
    """Configuration for an LLM backend."""
    backend_type: BackendType
    model_id: str
    api_key_env: str  # Environment variable name for API key
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7


# Default configurations for common backends
BACKEND_CONFIGS = {
    "claude_code": BackendConfig(
        backend_type=BackendType.CLAUDE_CODE,
        model_id="internal",
        api_key_env="",  # Uses Claude Code's internal auth
    ),
    "claude-sonnet": BackendConfig(
        backend_type=BackendType.ANTHROPIC_API,
        model_id="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "claude-haiku": BackendConfig(
        backend_type=BackendType.ANTHROPIC_API,
        model_id="claude-3-5-haiku-20241022",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "gpt-4o-mini": BackendConfig(
        backend_type=BackendType.OPENAI_API,
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    "gpt-4o": BackendConfig(
        backend_type=BackendType.OPENAI_API,
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
    ),
    "gemini-flash": BackendConfig(
        backend_type=BackendType.GOOGLE_API,
        model_id="gemini-2.0-flash-exp",
        api_key_env="GOOGLE_API_KEY",
    ),
    "deepseek-chat": BackendConfig(
        backend_type=BackendType.DEEPSEEK_API,
        model_id="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    "qwen-turbo": BackendConfig(
        backend_type=BackendType.QWEN_API,
        model_id="qwen-turbo",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "kimi": BackendConfig(
        backend_type=BackendType.KIMI_API,
        model_id="moonshot-v1-8k",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
    ),
    "local-parser": BackendConfig(
        backend_type=BackendType.LOCAL_SCRIPT,
        model_id="rule_based_parser",
        api_key_env="",
    ),
}


# =============================================================================
# Worker Prompts (shared across backends)
# =============================================================================

WORKER_PROMPTS = {
    "structural_parser": """You are a structural parser. Claim tickets, parse dialogue walks into structural triplets, submit results.

API endpoints:
- CLAIM: POST {api_base}/api/runs/{run_id}/claim
  Body: {{"worker_type": "structural_parser"}}
- SUBMIT: POST {api_base}/api/runs/{run_id}/submit
  Body: {{"ticket_id": "...", "output_data": {{...}}, "worker_concerns": []}}

Parse each walk's beats into: function (query/comply/refuse/threaten/react/bark/deliver_information),
archetype_relation (authority_to_subject/peer_to_peer/merchant_to_customer/etc.),
emotion transitions, arc_shape, barrier_type.

Process ALL available tickets until claim returns success=false.""",

    "translation_engine": """You are a translation engine. Claim tickets, translate structural triplets to target setting, submit results.

API endpoints:
- CLAIM: POST {api_base}/api/runs/{run_id}/claim
  Body: {{"worker_type": "translation_engine"}}
- SUBMIT: POST {api_base}/api/runs/{run_id}/submit
  Body: {{"ticket_id": "...", "output_data": {{"translated_texts": [...], "proper_nouns_introduced": [...], "confidence": 0.9}}, "worker_concerns": []}}

Preserve EXACT beat count, emotion sequence, archetype relations. Transform setting/register/proper nouns.
Process ALL available tickets until claim returns success=false.""",

    "lore_curator": """You are a lore curator. Claim tickets, validate proposed proper nouns against target bible, submit verdicts.

API endpoints:
- CLAIM: POST {api_base}/api/runs/{run_id}/claim
  Body: {{"worker_type": "lore_curator"}}
- SUBMIT: POST {api_base}/api/runs/{run_id}/submit
  Body: {{"ticket_id": "...", "output_data": {{"verdict": "approve"|"reject"|"modify", "reason": "..."}}, "worker_concerns": []}}

Evaluate if proposed nouns fit the target setting's tone and internal consistency.
Process ALL available tickets until claim returns success=false.""",
}


# =============================================================================
# Backend Workers
# =============================================================================

class BaseWorker(ABC):
    """Abstract base for LLM workers."""

    def __init__(self, config: BackendConfig, api_base: str):
        self.config = config
        self.api_base = api_base

    @abstractmethod
    async def process_tickets(self, run_id: str, worker_type: str) -> dict:
        """Process all available tickets of given type. Returns summary."""
        pass


class LocalScriptWorker(BaseWorker):
    """Rule-based worker using Python (no LLM API calls)."""

    async def process_tickets(self, run_id: str, worker_type: str) -> dict:
        """Process tickets using rule-based parsing."""
        completed = 0
        failed = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                # Claim
                resp = await client.post(
                    f"{self.api_base}/api/runs/{run_id}/claim",
                    json={"worker_type": worker_type}
                )
                claim = resp.json()

                if not claim.get("success"):
                    break

                ticket = claim["ticket"]
                ticket_id = ticket["ticket_id"]

                try:
                    # Process based on worker type
                    if worker_type == "structural_parser":
                        output = self._parse_walk(ticket["input_data"])
                    else:
                        output = {"error": f"Local script doesn't support {worker_type}"}

                    # Submit
                    await client.post(
                        f"{self.api_base}/api/runs/{run_id}/submit",
                        json={
                            "ticket_id": ticket_id,
                            "output_data": output,
                            "worker_concerns": []
                        }
                    )
                    completed += 1

                except Exception as e:
                    failed += 1

        return {"completed": completed, "failed": failed}

    def _parse_walk(self, input_data: dict) -> dict:
        """Rule-based structural parsing."""
        walk = input_data.get("walk", [])
        arc = []
        prev_emotion = None

        for i, beat in enumerate(walk):
            text = beat.get("text", "")
            emotion = beat.get("emotion", "neutral")

            # Simple rule-based function inference
            if "?" in text:
                function = "query"
            elif any(w in text.lower() for w in ["yes", "sure", "okay"]):
                function = "comply"
            elif any(w in text.lower() for w in ["no", "refuse"]):
                function = "refuse"
            elif i == 0:
                function = "deliver_information"
            else:
                function = "react"

            arc.append({
                "beat": f"beat_{i:03d}",
                "text": text,
                "emotion": emotion,
                "function": function,
                "archetype_relation": "peer_to_peer",
                "transition_from": prev_emotion
            })
            prev_emotion = emotion

        emotions = [b["emotion"] for b in arc]
        barrier = "confrontation" if "anger" in emotions else "ambient"
        shape = "escalating_threat" if barrier == "confrontation" else "information_dump"

        return {
            "arc": arc,
            "proper_nouns_used": [],
            "barrier_type": barrier,
            "attractor_type": "narrative_progression",
            "arc_shape": shape
        }


class OpenAICompatibleWorker(BaseWorker):
    """Worker for OpenAI-compatible APIs (OpenAI, DeepSeek, Qwen, Kimi)."""

    async def process_tickets(self, run_id: str, worker_type: str) -> dict:
        """Process tickets using OpenAI-compatible chat API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return {"error": "openai package not installed"}

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            return {"error": f"Missing {self.config.api_key_env}"}

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url
        )

        prompt = WORKER_PROMPTS.get(worker_type, "").format(
            api_base=self.api_base,
            run_id=run_id
        )

        # For external APIs, we give the LLM the prompt and let it make HTTP calls
        # This requires the model to be capable of tool use or code execution
        # Simpler approach: we handle the claim/submit loop, LLM just processes content

        completed = 0
        failed = 0

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            while True:
                # Claim ticket
                resp = await http_client.post(
                    f"{self.api_base}/api/runs/{run_id}/claim",
                    json={"worker_type": worker_type}
                )
                claim = resp.json()

                if not claim.get("success"):
                    break

                ticket = claim["ticket"]
                ticket_id = ticket["ticket_id"]

                try:
                    # Format input for LLM
                    task_prompt = self._format_task(worker_type, ticket["input_data"])

                    # Call LLM
                    response = await client.chat.completions.create(
                        model=self.config.model_id,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": task_prompt}
                        ],
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )

                    # Parse response
                    output = self._parse_response(response.choices[0].message.content)

                    # Submit
                    await http_client.post(
                        f"{self.api_base}/api/runs/{run_id}/submit",
                        json={
                            "ticket_id": ticket_id,
                            "output_data": output,
                            "worker_concerns": []
                        }
                    )
                    completed += 1

                except Exception as e:
                    failed += 1

        return {"completed": completed, "failed": failed, "model": self.config.model_id}

    def _format_task(self, worker_type: str, input_data: dict) -> str:
        """Format ticket input for LLM consumption."""
        return f"Process this input and return JSON output:\n\n```json\n{json.dumps(input_data, indent=2)}\n```"

    def _parse_response(self, content: str) -> dict:
        """Extract JSON from LLM response."""
        # Try to find JSON in response
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try direct parse
        try:
            return json.loads(content)
        except:
            return {"raw_response": content}


# =============================================================================
# Dispatcher
# =============================================================================

class WorkerDispatcher:
    """
    Dispatches work to multiple LLM backends.

    Provides unified interface regardless of whether workers are:
    - Claude Code subagents (automatic return via Task tool)
    - External API calls (poll for completion)
    - Local scripts (synchronous)
    """

    def __init__(self, api_base: str = "http://127.0.0.1:8000"):
        self.api_base = api_base
        self.active_workers: dict[str, asyncio.Task] = {}

    def get_worker(self, backend_name: str) -> BaseWorker:
        """Get worker instance for backend."""
        config = BACKEND_CONFIGS.get(backend_name)
        if not config:
            raise ValueError(f"Unknown backend: {backend_name}")

        if config.backend_type == BackendType.LOCAL_SCRIPT:
            return LocalScriptWorker(config, self.api_base)
        elif config.backend_type in (BackendType.OPENAI_API, BackendType.DEEPSEEK_API,
                                      BackendType.QWEN_API, BackendType.KIMI_API):
            return OpenAICompatibleWorker(config, self.api_base)
        else:
            raise NotImplementedError(f"Backend {config.backend_type} not yet implemented")

    async def dispatch(self, backend_name: str, worker_type: str, run_id: str) -> str:
        """
        Dispatch worker to process tickets.

        Returns task_id for tracking.
        """
        worker = self.get_worker(backend_name)
        task = asyncio.create_task(worker.process_tickets(run_id, worker_type))

        task_id = f"{backend_name}_{worker_type}_{run_id}"
        self.active_workers[task_id] = task
        return task_id

    async def get_status(self, run_id: str) -> dict:
        """Get current run status from API."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.api_base}/api/runs/{run_id}/status")
            return resp.json()

    async def await_completion(self, run_id: str, poll_interval: float = 2.0, timeout: float = 300.0):
        """Wait for run to complete (all tickets processed)."""
        start = time.time()
        while time.time() - start < timeout:
            status = await self.get_status(run_id)

            total_pending = (
                status["parse"]["pending"] + status["parse"]["claimed"] +
                status["translate"]["pending"] + status["translate"]["claimed"] +
                status["curate"]["pending"] + status["curate"]["claimed"]
            )

            if total_pending == 0:
                return status

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-backend worker dispatcher")
    parser.add_argument("--run-id", required=True, help="Run ID to process")
    parser.add_argument("--backend", default="local-parser", help="Backend to use")
    parser.add_argument("--worker-type", default="structural_parser", help="Worker type")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")

    args = parser.parse_args()

    dispatcher = WorkerDispatcher(args.api)
    task_id = await dispatcher.dispatch(args.backend, args.worker_type, args.run_id)
    print(f"Dispatched: {task_id}")

    # Wait for the specific task
    result = await dispatcher.active_workers[task_id]
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
