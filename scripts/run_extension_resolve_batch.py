#!/usr/bin/env python3
"""
Process extension_resolve tickets through DeepSeek API.

Usage:
    DEEPSEEK_API_KEY="sk-..." python scripts/run_extension_resolve_batch.py <run_id>
"""

import os
import sys
import json
import asyncio
import httpx
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"
DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"

EXTENSION_RESOLVER_SYSTEM_PROMPT = """You are an extension-resolver agent for dialogue graph topology. Your task is to generate bridging content that connects previously-declined link candidates.

Given a source node and target node with a suggested arc type, generate 1-3 bridge nodes that create a natural emotional transition from source to target.

VALID EMOTIONS (you MUST use only these - they are facial animation metadata values):
  neutral, anger, fear, happy, sad, disgust, surprise

The suggested_arc provides creative direction:
- bureaucratic_dread: neutral → fear → fear (slow realization of trapped-ness)
- interrupted_confrontation: anger → surprise → neutral (unexpected arrival breaks tension)
- regulatory_revelation: neutral → surprise → anger (discovery of hidden violation)
- surveillance_paranoia: neutral → fear → fear (growing awareness of being watched)
- conditional_status: varies (character's status depends on paperwork)

You can interpret arc types loosely. Generate bridge dialogue that fits the setting's register.

Output JSON only, no markdown formatting."""


def build_extension_resolve_prompt(input_data: dict) -> str:
    """Build the user prompt for extension resolution."""
    candidate = input_data.get("extension_candidate", {})
    source = input_data.get("source_node", {})
    target = input_data.get("target_node", {})
    bible = input_data.get("bible_excerpt", "")[:2000]

    prompt = f"""## Extension Candidate
Site: {candidate.get('site')}
Reason flagged: {candidate.get('reason')}
Suggested arc: {candidate.get('suggested_arc')}

## Source Node
ID: {source.get('id')}
Text: {source.get('text')}
Emotion: {source.get('emotion')}

## Target Node
ID: {target.get('id')}
Text: {target.get('text')}
Emotion: {target.get('emotion')}

## Setting Context (excerpt)
{bible[:1500]}

## Task
Generate 1-3 bridge nodes that create a natural emotional transition from source to target.
The bridge should embody the suggested arc type.

VALID EMOTIONS (use ONLY these): neutral, anger, fear, happy, sad, disgust, surprise

## Output Format
Return a JSON object:
{{
  "success": true,
  "bridge_nodes": [
    {{
      "id": "ext_{source.get('id', '')[:8]}_001",
      "text": "Bridge dialogue here",
      "emotion": "fear",
      "position": 1
    }}
  ],
  "edges": [
    {{"source": "{source.get('id')}", "target": "ext_..._001", "type": "extension_bridge"}},
    {{"source": "ext_..._001", "target": "{target.get('id')}", "type": "extension_bridge"}}
  ],
  "arc_realized": "{candidate.get('suggested_arc')}",
  "notes": "Brief explanation of the bridge strategy"
}}

Or if impossible:
{{
  "success": false,
  "reason": "Why this can't be bridged",
  "suggestion": "Alternative approach"
}}

Output only valid JSON, no markdown."""

    return prompt


async def process_ticket(
    client: httpx.AsyncClient,
    run_id: str,
    ticket: dict,
    api_key: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """Process a single extension_resolve ticket through DeepSeek."""
    async with semaphore:
        ticket_id = ticket.get("ticket_id")
        input_data = ticket.get("input_data", {})

        print(f"  Processing {ticket_id}...")

        # Build prompt
        user_prompt = build_extension_resolve_prompt(input_data)

        # Call DeepSeek
        try:
            response = await client.post(
                DEEPSEEK_API,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": EXTENSION_RESOLVER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500,
                },
                timeout=90.0,
            )

            if response.status_code != 200:
                print(f"    DeepSeek error {response.status_code}: {response.text[:200]}")
                return {"ticket_id": ticket_id, "success": False, "error": f"API error {response.status_code}"}

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            try:
                output_data = json.loads(content.strip())
            except json.JSONDecodeError as e:
                print(f"    JSON parse error: {e}")
                return {"ticket_id": ticket_id, "success": False, "error": f"JSON parse error"}

            # Submit result back to API
            submit_response = await client.post(
                f"{API_BASE}/api/runs/{run_id}/submit",
                json={
                    "ticket_id": ticket_id,
                    "output_data": output_data,
                    "worker_notes": [f"Processed by DeepSeek"],
                    "worker_backend": "deepseek-chat",
                },
                timeout=30.0,
            )

            if submit_response.status_code == 200:
                success = output_data.get("success", False)
                bridges = len(output_data.get("bridge_nodes", []))
                arc = output_data.get("arc_realized", "none")
                status = "✓" if success else "✗"
                print(f"    {status} {ticket_id}: {bridges} bridges, arc={arc}")
                return {"ticket_id": ticket_id, "success": success, "bridges": bridges}
            else:
                print(f"    Submit error: {submit_response.text[:200]}")
                return {"ticket_id": ticket_id, "success": False, "error": "Submit failed"}

        except Exception as e:
            print(f"    Error: {e}")
            return {"ticket_id": ticket_id, "success": False, "error": str(e)}


async def main(run_id: str, concurrency: int = 25):
    """Process all pending extension_resolve tickets for a run."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    print(f"Processing extension_resolve tickets for run: {run_id}")

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        # Get run status
        status_resp = await client.get(f"{API_BASE}/api/runs/{run_id}/status")
        if status_resp.status_code != 200:
            print(f"Error getting run status: {status_resp.text}")
            sys.exit(1)

        status = status_resp.json()
        pending = status.get("extension_resolve", {}).get("pending", 0)
        print(f"Found {pending} pending extension_resolve tickets")

        if pending == 0:
            print("No pending tickets")
            return

        # Claim and process tickets
        tasks = []

        for _ in range(pending):
            claim_resp = await client.post(
                f"{API_BASE}/api/runs/{run_id}/claim",
                json={"worker_type": "extension_resolver"},
            )

            if claim_resp.status_code != 200:
                break

            claim_data = claim_resp.json()
            if not claim_data.get("success"):
                break

            ticket = claim_data.get("ticket")
            if not ticket:
                break

            task = asyncio.create_task(
                process_ticket(client, run_id, ticket, api_key, semaphore)
            )
            tasks.append(task)

        print(f"Claimed {len(tasks)} tickets, processing...")

        results = await asyncio.gather(*tasks)

        # Summary
        success = sum(1 for r in results if r.get("success"))
        failed = len(results) - success
        total_bridges = sum(r.get("bridges", 0) for r in results if r.get("success"))

        print(f"\nCompleted: {success} success, {failed} failed")
        print(f"Total bridges generated: {total_bridges}")

        # Get final status
        final_status = await client.get(f"{API_BASE}/api/runs/{run_id}/status")
        if final_status.status_code == 200:
            fs = final_status.json()
            print(f"Resolved candidates: {fs.get('resolved_candidates_count', 0)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: DEEPSEEK_API_KEY=... python scripts/run_extension_resolve_batch.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    asyncio.run(main(run_id))
