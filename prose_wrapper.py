#!/usr/bin/env python3
"""
Prose Wrapper - Convert dialogue walks into narrated prose.

Takes structured walks (with emotion, beat_function, archetype_relation) and
produces prose with speaker attribution, scene setup, and action beats.

This is the "~200 LOC Python" from tinystories-barriers.md.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


# =============================================================================
# Arc Shapes → Scene Setup
# =============================================================================

SCENE_SETUPS = {
    "escalating_threat": [
        "The room was quiet when {speaker_a} entered.",
        "The meeting had barely begun.",
        "{speaker_a} stood waiting at the desk.",
    ],
    "de_escalation": [
        "The tension had been building for days.",
        "They had been arguing for what felt like hours.",
        "The shouting had finally stopped.",
    ],
    "information_exchange": [
        "The office was cluttered with files.",
        "They met in the usual place.",
        "{speaker_a} had news to share.",
    ],
    "negotiation": [
        "The terms were still unclear.",
        "Both parties sat across the table.",
        "The documents lay between them, unsigned.",
    ],
    "confrontation": [
        "It was only a matter of time.",
        "The accusations hung in the air.",
        "Neither would back down.",
    ],
    "supplication": [
        "There was no other choice.",
        "{speaker_a} had come to ask for help.",
        "The request was difficult to make.",
    ],
    "revelation": [
        "The truth had been hidden for too long.",
        "What {speaker_a} said next changed everything.",
        "The pieces finally fell into place.",
    ],
    "default": [
        "The conversation began without ceremony.",
        "They spoke as they always did.",
        "There was business to discuss.",
    ],
}


# =============================================================================
# Archetype → Speaker Handle
# =============================================================================

ARCHETYPE_HANDLES = {
    # Authority archetypes
    "authority_to_subject": {
        "speaker_a": ["the official", "the administrator", "the clerk", "the inspector"],
        "speaker_b": ["the petitioner", "the citizen", "the applicant", "the subject"],
    },
    "subject_to_authority": {
        "speaker_a": ["the petitioner", "the citizen", "the applicant"],
        "speaker_b": ["the official", "the administrator", "the clerk"],
    },
    # Peer archetypes
    "peer_to_peer": {
        "speaker_a": ["the first speaker", "one of them", "the taller one"],
        "speaker_b": ["the other", "the second", "the shorter one"],
    },
    "colleague_to_colleague": {
        "speaker_a": ["the senior clerk", "the first colleague"],
        "speaker_b": ["the junior clerk", "the other colleague"],
    },
    # Merchant archetypes
    "merchant_to_customer": {
        "speaker_a": ["the merchant", "the shopkeeper", "the vendor"],
        "speaker_b": ["the customer", "the buyer", "the patron"],
    },
    "customer_to_merchant": {
        "speaker_a": ["the customer", "the buyer"],
        "speaker_b": ["the merchant", "the shopkeeper"],
    },
    # Threat archetypes
    "threatener_to_threatened": {
        "speaker_a": ["the enforcer", "the collector", "the agent"],
        "speaker_b": ["the debtor", "the accused", "the target"],
    },
    # Default
    "default": {
        "speaker_a": ["the first speaker", "one of them"],
        "speaker_b": ["the other", "the second speaker"],
    },
}

# Pronoun cycling for variety
PRONOUNS = {
    "neutral": ["they", "the speaker"],
    "masculine": ["he", "the man"],
    "feminine": ["she", "the woman"],
}


# =============================================================================
# Emotion → Attribution Style
# =============================================================================

ATTRIBUTION_VERBS = {
    "neutral": ["said", "replied", "answered", "responded"],
    "anger": ["snapped", "growled", "demanded", "snarled"],
    "disgust": ["muttered", "scoffed", "sneered", "said with distaste"],
    "fear": ["whispered", "stammered", "said nervously", "managed"],
    "happy": ["said cheerfully", "laughed", "replied warmly", "beamed"],
    "sad": ["sighed", "said quietly", "murmured", "said wearily"],
    "surprise": ["exclaimed", "blurted", "gasped", "said suddenly"],
}

# Action beats by emotion transition
ACTION_BEATS = {
    ("neutral", "anger"): [
        "{speaker}'s expression hardened.",
        "Something shifted in {speaker}'s demeanor.",
        "{speaker}'s patience had run out.",
    ],
    ("neutral", "fear"): [
        "{speaker} went pale.",
        "A chill passed through the room.",
        "{speaker}'s confidence faltered.",
    ],
    ("anger", "neutral"): [
        "{speaker} took a breath.",
        "The tension eased slightly.",
        "{speaker} regained composure.",
    ],
    ("happy", "sad"): [
        "The smile faded.",
        "{speaker}'s mood darkened.",
        "The joy was short-lived.",
    ],
    ("fear", "anger"): [
        "Fear became defiance.",
        "{speaker} straightened.",
        "Something broke inside {speaker}.",
    ],
    ("surprise", "anger"): [
        "Shock gave way to fury.",
        "{speaker} processed what was said.",
        "The surprise curdled into rage.",
    ],
    ("neutral", "disgust"): [
        "{speaker}'s lip curled.",
        "Contempt crept into {speaker}'s voice.",
        "{speaker} looked away briefly.",
    ],
}


# =============================================================================
# Beat Function → Narrative Context
# =============================================================================

BEAT_CONTEXT = {
    "establish_stakes": "The matter at hand was clear: ",
    "deliver_information": "",  # Direct, no prefix
    "query": "The question was direct: ",
    "threaten": "The implication was unmistakable: ",
    "plead": "The request was earnest: ",
    "react": "",  # Direct response
    "comply": "There was no argument: ",
    "farewell": "The conversation was over: ",
    "refuse": "The answer was firm: ",
    "deflect": "The subject changed: ",
}


# =============================================================================
# Core Wrapper Logic
# =============================================================================

@dataclass
class Walk:
    """A dialogue walk with metadata."""
    beats: List[Dict[str, Any]]
    arc_shape: str = "default"
    archetype_relation: str = "default"
    source: str = "unknown"


@dataclass
class ProseOutput:
    """Wrapped prose with metadata."""
    prose: str
    word_count: int
    beat_count: int
    source_walk: Walk
    speakers_used: List[str]


def get_speaker_handles(archetype: str) -> Tuple[str, str]:
    """Get consistent speaker handles for an archetype."""
    handles = ARCHETYPE_HANDLES.get(archetype, ARCHETYPE_HANDLES["default"])
    speaker_a = random.choice(handles["speaker_a"])
    speaker_b = random.choice(handles["speaker_b"])
    return speaker_a, speaker_b


def get_attribution(emotion: str, speaker: str, beat_idx: int) -> str:
    """Get a verb for dialogue attribution based on emotion."""
    verbs = ATTRIBUTION_VERBS.get(emotion, ATTRIBUTION_VERBS["neutral"])
    # Vary by position to avoid repetition
    verb = verbs[beat_idx % len(verbs)]
    return f"{speaker} {verb}"


def get_action_beat(prev_emotion: str, curr_emotion: str, speaker: str) -> Optional[str]:
    """Get an action beat for an emotion transition."""
    key = (prev_emotion, curr_emotion)
    if key in ACTION_BEATS:
        beat = random.choice(ACTION_BEATS[key])
        return beat.format(speaker=speaker.capitalize())
    return None


def infer_arc_shape(beats: List[Dict[str, Any]]) -> str:
    """Infer arc shape from emotion sequence if not provided."""
    if not beats:
        return "default"

    emotions = [b.get("emotion", "neutral") for b in beats]

    # Simple heuristics
    if "anger" in emotions[-2:] and emotions[0] == "neutral":
        return "escalating_threat"
    if emotions[-1] == "neutral" and "anger" in emotions[:2]:
        return "de_escalation"
    if emotions.count("neutral") >= len(emotions) * 0.7:
        return "information_exchange"
    if "fear" in emotions or "sad" in emotions:
        return "supplication"
    if "surprise" in emotions:
        return "revelation"

    return "default"


def clean_text(text: str) -> str:
    """Strip stage direction tags like {Hopeful}, {Rueful}, etc."""
    import re
    # Remove {Tag} prefixes
    text = re.sub(r'\{[A-Za-z]+\}\s*', '', text)
    return text.strip()


def wrap_walk(walk: Walk) -> ProseOutput:
    """
    Convert a dialogue walk into narrated prose.

    This is the main entry point.
    """
    beats = walk.beats
    if not beats:
        return ProseOutput(
            prose="",
            word_count=0,
            beat_count=0,
            source_walk=walk,
            speakers_used=[]
        )

    # Get arc shape (infer if not provided)
    arc_shape = walk.arc_shape if walk.arc_shape != "default" else infer_arc_shape(beats)

    # Get speaker handles
    speaker_a, speaker_b = get_speaker_handles(walk.archetype_relation)
    speakers = [speaker_a, speaker_b]

    # Build prose
    paragraphs = []

    # Scene setup
    setups = SCENE_SETUPS.get(arc_shape, SCENE_SETUPS["default"])
    setup = random.choice(setups).format(speaker_a=speaker_a.capitalize())
    paragraphs.append(setup)

    prev_emotion = None
    prev_speaker_idx = None

    for i, beat in enumerate(beats):
        text = clean_text(beat.get("text", ""))
        emotion = beat.get("emotion", "neutral")
        beat_function = beat.get("beat_function", "deliver_information")

        # Determine speaker (alternate by default, or use explicit if available)
        explicit_speaker = beat.get("speaker")
        if explicit_speaker:
            speaker = explicit_speaker
            # Try to map to speaker index for alternation tracking
            if explicit_speaker.lower() in [s.lower() for s in speakers]:
                current_speaker_idx = 0 if explicit_speaker.lower() == speakers[0].lower() else 1
            else:
                current_speaker_idx = i % 2
        else:
            # Simple alternation: even beats = speaker_a, odd = speaker_b
            current_speaker_idx = i % 2
            speaker = speakers[current_speaker_idx]

        # Build this beat's prose
        beat_parts = []

        # Add action beat for emotion transitions
        if prev_emotion and prev_emotion != emotion:
            action = get_action_beat(prev_emotion, emotion, speaker)
            if action:
                beat_parts.append(action)

        # Add beat function context (if any)
        context = BEAT_CONTEXT.get(beat_function, "")

        # Add the dialogue with attribution
        attribution = get_attribution(emotion, speaker.capitalize(), i)

        # Vary attribution position
        if i % 3 == 0:
            # Attribution before
            dialogue_line = f'{attribution}, "{text}"'
        elif i % 3 == 1:
            # Attribution after
            dialogue_line = f'"{text}" {attribution}.'
        else:
            # Attribution mid (if text is long enough and has meaningful second part)
            if len(text) > 30 and "." in text:
                parts = text.split(".", 1)
                second_part = parts[1].strip() if len(parts) > 1 else ""
                if second_part:
                    dialogue_line = f'"{parts[0]}," {attribution}. "{second_part}"'
                else:
                    dialogue_line = f'"{text}" {attribution}.'
            else:
                dialogue_line = f'"{text}" {attribution}.'

        if context:
            beat_parts.append(context + dialogue_line)
        else:
            beat_parts.append(dialogue_line)

        paragraphs.append(" ".join(beat_parts))
        prev_emotion = emotion

    # Join paragraphs
    prose = "\n\n".join(paragraphs)
    word_count = len(prose.split())

    return ProseOutput(
        prose=prose,
        word_count=word_count,
        beat_count=len(beats),
        source_walk=walk,
        speakers_used=speakers
    )


# =============================================================================
# Walk Extraction from Synthetic Graphs
# =============================================================================

def extract_walks_from_graph(graph_data: Dict[str, Any],
                              walk_length: int = 5,
                              num_walks: int = 10) -> List[Walk]:
    """
    Extract walks from a synthetic graph.

    Follows sequential edges to build coherent walks.
    """
    nodes = {n["id"]: n for n in graph_data.get("nodes", [])}
    edges = graph_data.get("edges", [])

    # Build adjacency
    adjacency = {}
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(tgt)

    walks = []
    node_ids = list(nodes.keys())

    for _ in range(num_walks):
        if not node_ids:
            break

        # Pick random start
        start = random.choice(node_ids)
        current = start
        walk_nodes = [nodes[current]]
        visited = {current}

        # Follow edges
        for _ in range(walk_length - 1):
            neighbors = adjacency.get(current, [])
            unvisited = [n for n in neighbors if n not in visited]

            if not unvisited:
                break

            next_node = random.choice(unvisited)
            walk_nodes.append(nodes[next_node])
            visited.add(next_node)
            current = next_node

        if len(walk_nodes) >= 2:  # At least 2 nodes for a walk
            beats = [
                {
                    "text": n.get("text", ""),
                    "emotion": n.get("emotion", "neutral"),
                    "beat_function": n.get("beat_function", "deliver_information"),
                    "speaker": n.get("speaker"),
                }
                for n in walk_nodes
            ]

            # Try to get archetype from first node with it
            archetype = "default"
            for n in walk_nodes:
                if n.get("archetype_relation"):
                    archetype = n["archetype_relation"]
                    break

            walks.append(Walk(
                beats=beats,
                arc_shape=infer_arc_shape(beats),
                archetype_relation=archetype,
                source=graph_data.get("source", "synthetic"),
            ))

    return walks


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    """Test the prose wrapper with sample data."""
    import json
    from pathlib import Path

    # Always run manual test first to verify wrapper works
    print("=" * 60)
    print("MANUAL TEST")
    print("=" * 60)
    manual_test()

    # Try to load a real synthetic graph
    gallia_path = Path("synthetic/gallia_v4/graph.json")

    if gallia_path.exists():
        print("\n" + "=" * 60)
        print("SYNTHETIC GRAPH TEST")
        print("=" * 60)
        print("Loading gallia_v4 graph...")

        try:
            # Load full file (it's big but manageable)
            with open(gallia_path) as f:
                graph_data = json.load(f)
            graph_data["source"] = "gallia_v4"

            print(f"Loaded {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")
            walks = extract_walks_from_graph(graph_data, walk_length=4, num_walks=3)

            print(f"Extracted {len(walks)} walks\n")

            for i, walk in enumerate(walks):
                print(f"\n--- Walk {i+1} ---")
                print(f"Arc shape: {walk.arc_shape}")
                print(f"Archetype: {walk.archetype_relation}")
                print(f"Beats: {len(walk.beats)}")
                print()

                output = wrap_walk(walk)
                print(output.prose)
                print(f"\n[{output.word_count} words]")
                print("-" * 40)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
        except Exception as e:
            print(f"Error: {e}")


def manual_test():
    """Test with manually constructed walk."""
    print("Running manual test...\n")

    walk = Walk(
        beats=[
            {"text": "Seventy-two hours.", "emotion": "neutral", "beat_function": "establish_stakes"},
            {"text": "The Hexagon expects compliance.", "emotion": "neutral", "beat_function": "threaten"},
            {"text": "And if I refuse?", "emotion": "anger", "beat_function": "query"},
            {"text": "The Leclerc is outside.", "emotion": "anger", "beat_function": "threaten"},
        ],
        arc_shape="escalating_threat",
        archetype_relation="authority_to_subject",
        source="manual_test",
    )

    output = wrap_walk(walk)

    print(output.prose)
    print(f"\n[{output.word_count} words, {output.beat_count} beats]")
    print(f"Speakers: {output.speakers_used}")


if __name__ == "__main__":
    main()
