#!/usr/bin/env python3
"""
Structural Parser Worker - Process dialogue walks and output triplets
Connects to ticket API and processes all available tickets
"""
import requests
import json
import sys
from typing import Optional

BASE_URL = "http://localhost:8000"
RUN_ID = "run_20251223_060354_gallia"
WORKER_TYPE = "structural_parser"

def extract_proper_nouns(text: str) -> list:
    """Extract proper nouns from dialogue text."""
    nouns = []
    sentences = text.split('. ')
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.isalpha():
                clean = word.rstrip('.,!?;:')
                if clean and len(clean) > 1:
                    nouns.append(clean)
    
    return list(set(nouns))

def infer_beat_function(text: str, emotion: str, speaker: str) -> str:
    """Infer beat function from text and context."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["should", "must", "need to", "will"]):
        if emotion in ["anger", "threat"]:
            return "threaten"
    
    if any(word in text_lower for word in ["please", "help", "mercy", "sorry"]):
        return "plead"
    
    if any(word in text_lower for word in ["time's up", "day", "hour", "minute"]):
        if emotion in ["anger", "neutral"]:
            return "react"
    
    if any(word in text_lower for word in ["yes", "agreed", "ok", "will do"]):
        return "comply"
    
    if any(word in text_lower for word in ["no", "refuse", "won't", "can't"]):
        return "refuse"
    
    if any(word in text_lower for word in ["what", "who", "where", "when", "why", "how"]):
        return "query"
    
    if any(word in text_lower for word in ["goodbye", "farewell", "later", "see you"]):
        return "farewell"
    
    if emotion == "neutral" and len(text) < 40:
        return "bark"
    
    if emotion in ["anger", "disgust"]:
        return "threaten"
    
    return "deliver_information"

def infer_archetype_relation(speaker: str) -> str:
    """Infer archetype relation from speaker identity."""
    speaker_lower = speaker.lower()
    
    if any(word in speaker_lower for word in ["ranger", "guard", "officer", "sergeant", "commander", "chief"]):
        return "authority_to_subject"
    
    if any(word in speaker_lower for word in ["merchant", "trader", "vendor", "bartender"]):
        return "merchant_to_customer"
    
    return "stranger_to_stranger"

def infer_barrier_type(text: str) -> str:
    """Infer barrier type from dialogue content."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["day", "hour", "time", "left", "expired", "up"]):
        return "countdown"
    
    if any(word in text_lower for word in ["should have", "could have", "would have"]):
        return "confrontation"
    
    if any(word in text_lower for word in ["offer", "deal", "agree", "terms"]):
        return "negotiation"
    
    return "ambient"

def infer_arc_shape(beat_count: int, beats: list) -> str:
    """Infer arc shape from beat sequence."""
    if beat_count == 1:
        return "single_beat"
    
    emotions = [b.get("emotion", "neutral") for b in beats]
    
    if beat_count >= 2:
        if emotions[0] in ["neutral"] and emotions[-1] in ["anger", "fear"]:
            return "escalating_threat"
        
        if emotions[0] in ["anger", "fear"] and emotions[-1] in ["neutral"]:
            return "de_escalation"
    
    functions = [b.get("function", "") for b in beats]
    if "negotiate" in functions:
        return "negotiation_arc"
    
    if "plead" in functions:
        return "plea_arc"
    
    if "bark" in functions:
        return "ambient_chatter"
    
    return "escalating_threat"

def parse_walk(walk_data: dict, reference_bible: str) -> dict:
    """Parse a dialogue walk into structural triplet."""
    walk = walk_data.get("walk", [])
    
    if not walk:
        return {"error": "Empty walk"}
    
    all_text = " ".join([line.get("text", "") for line in walk])
    proper_nouns = extract_proper_nouns(all_text)
    
    arc = []
    for i, line in enumerate(walk):
        text = line.get("text", "")
        speaker = line.get("speaker", "Unknown")
        emotion = line.get("emotion", "neutral")
        
        beat_function = infer_beat_function(text, emotion, speaker)
        archetype_relation = infer_archetype_relation(speaker)
        transition_from = arc[i-1]["emotion"] if i > 0 else None
        
        beat = {
            "beat": f"beat_{i}",
            "text": text,
            "emotion": emotion,
            "function": beat_function,
            "archetype_relation": archetype_relation,
            "transition_from": transition_from
        }
        
        arc.append(beat)
    
    barrier_type = infer_barrier_type(all_text)
    arc_shape = infer_arc_shape(len(arc), arc)
    attractor_type = "survival" if "survival" in all_text.lower() else "dominance"
    
    return {
        "arc": arc,
        "proper_nouns_used": proper_nouns,
        "barrier_type": barrier_type,
        "attractor_type": attractor_type,
        "arc_shape": arc_shape,
        "reference_bible": reference_bible
    }

def claim_ticket() -> Optional[dict]:
    """Claim a ticket from the API."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/runs/{RUN_ID}/claim",
            json={"worker_type": WORKER_TYPE},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 204:
            return None
        else:
            print(f"Claim error: {response.status_code} {response.text}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Claim failed: {e}", file=sys.stderr)
        return None

def submit_ticket(ticket_id: str, output_data: dict) -> bool:
    """Submit processed ticket results."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/runs/{RUN_ID}/submit",
            json={
                "ticket_id": ticket_id,
                "output_data": output_data
            },
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Submit failed: {e}", file=sys.stderr)
        return False

def main():
    """Process all available tickets."""
    count = 0
    
    while True:
        ticket = claim_ticket()
        if ticket is None:
            break
        
        ticket_id = ticket.get("ticket_id")
        input_data = ticket.get("input_data", {})
        
        try:
            output = parse_walk(input_data, input_data.get("reference_bible", "unknown"))
            
            if submit_ticket(ticket_id, output):
                count += 1
                print(f"Processed ticket {ticket_id}")
            else:
                print(f"Failed to submit ticket {ticket_id}", file=sys.stderr)
        except Exception as e:
            print(f"Processing error for {ticket_id}: {e}", file=sys.stderr)
    
    print(f"\nTotal tickets processed: {count}")
    return count

if __name__ == "__main__":
    main()
