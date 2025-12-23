#!/usr/bin/env python3
"""Check ticket_0037 actual data"""

from workflow.ticket_queue import get_manager, TicketStatus
import json

manager = get_manager()
queue = manager.get_queue("run_20251222_225616_gallia")

ticket = None
for t in queue.translate_tickets:
    if t.ticket_id == "translate_0037":
        ticket = t
        break

print(f"Ticket: {ticket.ticket_id}")
print(f"Status: {ticket.status}")
print(f"Claimed at: {ticket.claimed_at}")
print(f"Completed at: {ticket.completed_at}")

print(f"\nHas output_data: {ticket.output_data is not None}")

if ticket.output_data:
    print("\nOutput data:")
    print(json.dumps(ticket.output_data, indent=2))
else:
    print("\n⚠ NO OUTPUT DATA - This is the problem!")

print(f"\n--- Fixing Status ---")
# The output data exists but status wasn't updated
if ticket.output_data:
    ticket.status = TicketStatus.COMPLETED
    from datetime import datetime, timezone
    ticket.completed_at = datetime.now(timezone.utc).isoformat()

    print(f"Updated status to: {ticket.status}")
    print(f"Completed at: {ticket.completed_at}")

    # Save
    manager._save_queue(queue)
    print("\n✓ Queue saved")

    # Verify
    queue = manager.get_queue(queue.run_id)
    ticket = None
    for t in queue.translate_tickets:
        if t.ticket_id == "translate_0037":
            ticket = t
            break

    print(f"\nVerification:")
    print(f"  Status: {ticket.status}")
    print(f"  Completed at: {ticket.completed_at}")
