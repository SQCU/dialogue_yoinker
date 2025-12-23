#!/usr/bin/env python3
"""Inspect queue state directly"""

from workflow.ticket_queue import get_manager

manager = get_manager()
queue = manager.get_queue("run_20251222_225616_gallia")

print(f"Run ID: {queue.run_id}")
print(f"\nStatus:")
print(queue.status())

print(f"\n=== All Tickets ===")
print(f"Parse: {len(queue.parse_tickets)}")
print(f"Translate: {len(queue.translate_tickets)}")
print(f"Curate: {len(queue.curate_tickets)}")

print(f"\n=== Claimed Tickets ===")
for ticket in queue.translate_tickets:
    if ticket.status == "claimed":
        print(f"  {ticket.ticket_id}: {ticket.status} at {ticket.claimed_at}")
        print(f"    Claimed by: {ticket.claimed_by}")
        print(f"    Input: {ticket.input_data.get('triplet', {}).get('arc', [])[:1]}")  # First beat
