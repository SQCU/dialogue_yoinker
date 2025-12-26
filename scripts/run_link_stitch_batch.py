#!/usr/bin/env python3
"""
Process link_stitch tickets through DeepSeek API.

Usage:
    DEEPSEEK_API_KEY="sk-..." python scripts/run_link_stitch_batch.py <run_id>
"""

import os
import sys
import json
import asyncio
import httpx
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"
DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"

LINK_STITCHER_SYSTEM_PROMPT = """You are a link-stitcher agent for dialogue graph topology. Your task is to write transitional dialogue that connects source nodes to target nodes.

VALID EMOTIONS (use ONLY these - they are facial animation metadata values):
  neutral, anger, fear, happy, sad, disgust, surprise

Given a source node and candidate targets, you must:
1. For each candidate target, decide if you can write dialogue that achieves the emotion transition
2. Write bridge text that naturally transitions from source emotion to target emotion
3. Mark transitions as "direct" (no bridge needed), "bridged" (bridge text written), or "declined" (intractable)

ANY emotion transition is valid if you can write dialogue achieving it. The question is never "are these emotions compatible?" but "what dialogue achieves this emotional transition?"

If a transition seems intractable but vibes interesting, mark it as declined with an extension_note explaining why it's worth exploring later.

Output JSON only, no markdown formatting."""


def build_link_stitch_prompt(input_data: dict) -> str:
    """Build the user prompt for link-stitching."""
    source = input_data.get("source_node", {})
    candidates = input_data.get("candidate_targets", [])
    bible = input_data.get("bible_excerpt", "")[:3000]  # Truncate for token budget
    params = input_data.get("link_params", {})

    prompt = f"""## Source Node
ID: {source.get('id')}
Text: {source.get('text')}
Emotion: {source.get('emotion')}
Current out-degree: {source.get('current_out_degree', 0)}
Target out-degree: {source.get('target_out_degree', 5)}

## Candidate Targets
"""
    for i, target in enumerate(candidates):
        prompt += f"""
### Target {i+1}
ID: {target.get('id')}
Text: {target.get('text')}
Emotion: {target.get('emotion')}
Transition required: {target.get('transition_required')}
"""

    prompt += f"""
## Setting Context (excerpt)
{bible[:2000]}

## Link Parameters
- n_links_out: {params.get('n_links_out', 3)} (try to create this many links)
- max_bridge_length: {params.get('max_bridge_length', 1)} (bridge nodes per link)

VALID EMOTIONS (use ONLY these): neutral, anger, fear, happy, sad, disgust, surprise

## Output Format
Return a JSON object with this structure:
{{
  "links": [
    {{
      "from": "source_id",
      "to": "target_id",
      "via": null,  // or bridge node id if bridged
      "bridge_text": null,  // or the bridge dialogue text
      "bridge_emotion": null,  // or the bridge emotion
      "transition": "source_emotion→target_emotion",
      "status": "direct" | "bridged" | "declined",
      "decline_reason": null  // if declined
    }}
  ],
  "generated_nodes": [
    {{
      "id": "bridge_xxx",
      "text": "The bridge dialogue",
      "emotion": "the_emotion"
    }}
  ],
  "extension_candidates": [
    {{
      "site": "source_id→target_id",
      "reason": "Why this is interesting for future extension",
      "suggested_arc": "arc_type_name"
    }}
  ]
}}

Write transitional dialogue that fits the Gallia setting - bureaucratic, formal, French-inspired. Create bridges that naturally achieve the emotion transitions. Output only valid JSON, no markdown."""

    return prompt


async def process_ticket(
    client: httpx.AsyncClient,
    run_id: str,
    ticket: dict,
    api_key: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """Process a single link_stitch ticket through DeepSeek."""
    async with semaphore:
        ticket_id = ticket.get("ticket_id")
        input_data = ticket.get("input_data", {})

        print(f"  Processing {ticket_id}...")

        # Build prompt
        user_prompt = build_link_stitch_prompt(input_data)

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
                        {"role": "system", "content": LINK_STITCHER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
                timeout=90.0,
            )

            if response.status_code != 200:
                print(f"    DeepSeek error {response.status_code}: {response.text[:200]}")
                return {"ticket_id": ticket_id, "success": False, "error": f"API error {response.status_code}"}

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            # Try to extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            try:
                output_data = json.loads(content.strip())
            except json.JSONDecodeError as e:
                print(f"    JSON parse error: {e}")
                print(f"    Content: {content[:200]}")
                return {"ticket_id": ticket_id, "success": False, "error": f"JSON parse error: {e}"}

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
                links_count = len(output_data.get("links", []))
                ext_count = len(output_data.get("extension_candidates", []))
                print(f"    ✓ {ticket_id}: {links_count} links, {ext_count} extension candidates")
                return {"ticket_id": ticket_id, "success": True, "links": links_count}
            else:
                print(f"    Submit error: {submit_response.text[:200]}")
                return {"ticket_id": ticket_id, "success": False, "error": "Submit failed"}

        except Exception as e:
            print(f"    Error: {e}")
            return {"ticket_id": ticket_id, "success": False, "error": str(e)}


async def main(run_id: str, concurrency: int = 25):
    """Process all pending link_stitch tickets for a run."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    print(f"Processing link_stitch tickets for run: {run_id}")

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        # Get run status
        status_resp = await client.get(f"{API_BASE}/api/runs/{run_id}/status")
        if status_resp.status_code != 200:
            print(f"Error getting run status: {status_resp.text}")
            sys.exit(1)

        status = status_resp.json()
        pending = status.get("link_stitch", {}).get("pending", 0)
        print(f"Found {pending} pending link_stitch tickets")

        if pending == 0:
            print("No pending tickets")
            return

        # Claim and process tickets
        results = []
        tasks = []

        for i in range(pending):
            # Claim ticket
            claim_resp = await client.post(
                f"{API_BASE}/api/runs/{run_id}/claim",
                json={"worker_type": "link_stitcher"},
            )

            if claim_resp.status_code != 200:
                break

            claim_data = claim_resp.json()
            if not claim_data.get("success"):
                break

            ticket = claim_data.get("ticket")
            if not ticket:
                break

            # Create task for this ticket
            task = asyncio.create_task(
                process_ticket(client, run_id, ticket, api_key, semaphore)
            )
            tasks.append(task)

        print(f"Claimed {len(tasks)} tickets, processing...")

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Summary
        success = sum(1 for r in results if r.get("success"))
        failed = len(results) - success
        total_links = sum(r.get("links", 0) for r in results if r.get("success"))

        print(f"\nCompleted: {success} success, {failed} failed")
        print(f"Total links generated: {total_links}")

        # Get final status
        final_status = await client.get(f"{API_BASE}/api/runs/{run_id}/status")
        if final_status.status_code == 200:
            fs = final_status.json()
            print(f"Extension candidates collected: {fs.get('extension_candidates_count', 0)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: DEEPSEEK_API_KEY=... python scripts/run_link_stitch_batch.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    asyncio.run(main(run_id))
