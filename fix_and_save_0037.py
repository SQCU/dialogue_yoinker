#!/usr/bin/env python3
"""
Fix translate_0037 with proper Gallia translations and persist.
"""

from workflow.ticket_queue import get_manager, TicketStatus

manager = get_manager()
queue = manager.get_queue("run_20251222_225616_gallia")

# Find translate_0037
ticket = None
for t in queue.translate_tickets:
    if t.ticket_id == "translate_0037":
        ticket = t
        break

if not ticket:
    print("Ticket not found!")
    exit(1)

print(f"Found ticket: {ticket.ticket_id}")
print(f"Status: {ticket.status}")

# Show current (bad) translations
print("\nCurrent translations:")
for i, text in enumerate(ticket.output_data.get("translated_texts", [])):
    emotion = ticket.input_data["triplet"]["arc"][i]["emotion"]
    print(f"  {i} [{emotion}]: {text}")

# Proper Gallia translations
# Arc: happy -> neutral -> anger -> anger -> anger
# Setting: Oblivion religious dialogue -> Gallia bureaucratic/formal

proper_translations = [
    "Les bénédictions d'Arkay soient sur vous.",  # happy - formal religious blessing
    "Vous voilà!",  # neutral - simple acknowledgment
    "Pour le Seigneur Dagon!",  # anger - cultist battle cry
    "Vous ne pouvez échapper à la vigilance du Maître!",  # anger - authoritarian threat
    "Étranger! Tuez!",  # anger - hostile imperative
]

# Update ticket
ticket.output_data["translated_texts"] = proper_translations
ticket.output_data["proper_nouns_introduced"] = ["Arkay", "Dagon", "Maître"]
ticket.output_data["register_notes"] = "French formal/religious to bureaucratic-procedural. Oblivion deity names (Arkay, Dagon) preserved as proper nouns. Emotional arc happy->neutral->anger maintained."
ticket.output_data["confidence"] = 0.92

print("\nUpdated translations:")
for i, text in enumerate(proper_translations):
    emotion = ticket.input_data["triplet"]["arc"][i]["emotion"]
    print(f"  {i} [{emotion}]: {text}")

# Save queue to disk
manager._save_queue(queue)
print("\n✓ Queue saved to disk")

# Verify
reloaded_queue = manager.get_queue(queue.run_id)
reloaded_ticket = None
for t in reloaded_queue.translate_tickets:
    if t.ticket_id == "translate_0037":
        reloaded_ticket = t
        break

print("\nVerification (reloaded from disk):")
for i, text in enumerate(reloaded_ticket.output_data.get("translated_texts", [])):
    print(f"  {i}: {text}")

print("\n✓ Translation fixed and persisted")
