#!/usr/bin/env python3
"""
Process Link Requests through DeepSeek API.

Takes link request files from a linking run and generates bridge dialogue.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python scripts/process_link_requests.py link_20251224_011123_marmotte_v1
    python scripts/process_link_requests.py link_20251224_011123_marmotte_v1 --batch-size 5
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


RUNS_DIR = Path("runs")


def format_bridge_prompt(link: dict, bible_excerpt: str) -> str:
    """Format the prompt for bridge generation."""
    terminus_text = link["terminus_text"]
    terminus_emotion = link["terminus_emotion"]
    terminus_context = link.get("terminus_context", [])

    entry_text = link["entry_text"]
    entry_emotion = link["entry_emotion"]
    entry_context = link.get("entry_context", [])

    # Build context strings
    pre_context = "\n".join([f"  [{c['emotion']}] {c['text']}" for c in terminus_context])
    post_context = "\n".join([f"  [{c['emotion']}] {c['text']}" for c in entry_context])

    prompt = f"""Generate a BRIDGE LINE of dialogue that naturally connects these two dialogue sequences.

SETTING CONTEXT (abridged):
{bible_excerpt[:1500]}

CHAIN ENDING (what comes before the bridge):
{pre_context}
  [{terminus_emotion}] {terminus_text}

CHAIN BEGINNING (what comes after the bridge):
  [{entry_emotion}] {entry_text}
{post_context}

Your task: Write ONE line of dialogue that creates a natural transition from the ENDING to the BEGINNING.
The bridge should:
1. Acknowledge or respond to the terminus line
2. Pivot toward the entry topic/emotion
3. Use the setting's vocabulary and register (corporate marmot conspiracy jargon)
4. Be spoken by a character in the setting

Output JSON:
{{"bridge_text": "Your single line of bridge dialogue", "bridge_emotion": "the emotion (neutral/happy/sad/anger/fear/surprise/disgust)"}}
"""
    return prompt


async def process_link_batch(
    requests: list[dict],
    request_paths: list[Path],
    client,
    concurrency: int = 5
) -> dict:
    """Process a batch of link requests concurrently."""

    async def process_one(request: dict, path: Path) -> tuple[bool, str]:
        """Process a single link request."""
        if request.get("status") == "completed":
            return (True, "already_completed")

        link = request["link"]
        bible = request["bible_excerpt"]
        prompt = format_bridge_prompt(link, bible)

        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You generate bridge dialogue connecting conversation chains. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.7,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            output = json.loads(content)
            bridge_text = output.get("bridge_text", "")
            bridge_emotion = output.get("bridge_emotion", "neutral")

            if not bridge_text:
                return (False, "empty_bridge")

            # Update request
            request["bridge_text"] = bridge_text
            request["bridge_emotion"] = bridge_emotion
            request["status"] = "completed"
            request["worker_backend"] = "deepseek-chat"

            # Save updated request
            path.write_text(json.dumps(request, indent=2))

            return (True, bridge_text[:50])

        except json.JSONDecodeError as e:
            return (False, f"json_error: {e}")
        except Exception as e:
            return (False, f"error: {e}")

    # Process concurrently
    tasks = [process_one(req, path) for req, path in zip(requests, request_paths)]
    results = await asyncio.gather(*tasks)

    successes = sum(1 for success, _ in results if success)
    return {"completed": successes, "total": len(requests)}


async def main():
    parser = argparse.ArgumentParser(description="Process link requests through DeepSeek")
    parser.add_argument("run_id", help="Linking run ID")
    parser.add_argument("--batch-size", type=int, default=5, help="Concurrent batch size")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    run_dir = RUNS_DIR / args.run_id
    requests_dir = run_dir / "requests"

    if not requests_dir.exists():
        print(f"ERROR: No requests directory at {requests_dir}")
        sys.exit(1)

    # Load all pending requests
    request_files = sorted(requests_dir.glob("link_*.json"))
    print(f"Found {len(request_files)} link requests")

    requests = []
    paths = []
    for path in request_files:
        req = json.loads(path.read_text())
        if req.get("status") != "completed":
            requests.append(req)
            paths.append(path)

    print(f"  {len(requests)} pending")

    if not requests:
        print("Nothing to process")
        return

    # Initialize OpenAI client
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Process in batches
    total_completed = 0
    batch_size = args.batch_size

    for i in range(0, len(requests), batch_size):
        batch_requests = requests[i:i+batch_size]
        batch_paths = paths[i:i+batch_size]

        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch_requests)} requests)...")

        result = await process_link_batch(batch_requests, batch_paths, client, concurrency=batch_size)
        total_completed += result["completed"]

        print(f"  Completed: {result['completed']}/{result['total']}")

    print(f"\n{'='*50}")
    print(f"Total processed: {total_completed}/{len(requests)}")
    print(f"Run `python graph_linker.py --setting <setting> --version <ver> --apply` to apply bridges")


if __name__ == "__main__":
    asyncio.run(main())
