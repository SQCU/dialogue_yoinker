#!/usr/bin/env python3
"""Find the missing translation ticket"""

from workflow.ticket_queue import get_manager, TicketStatus

manager = get_manager()
queue = manager.get_queue("run_20251222_225616_gallia")

print("Translation Tickets Status:")
status_counts = {}
for ticket in queue.translate_tickets:
    status = ticket.status.value
    status_counts[status] = status_counts.get(status, 0) + 1

for status, count in status_counts.items():
    print(f"  {status}: {count}")

print("\nNon-completed tickets:")
for ticket in queue.translate_tickets:
    if ticket.status != TicketStatus.COMPLETED:
        print(f"\n  {ticket.ticket_id}: {ticket.status}")
        print(f"    Claimed at: {ticket.claimed_at}")
        print(f"    Claimed by: {ticket.claimed_by}")

        arc = ticket.input_data.get("triplet", {}).get("arc", [])
        if arc:
            print(f"    First beat: {arc[0]['text'][:60]}...")
