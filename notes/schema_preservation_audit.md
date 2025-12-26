# Schema Preservation Audit

**Date**: 2024-12-25
**Status**: REBUILT - Pipeline now preserves full schema

## Rebuild Status (2024-12-25)

Fixes applied:
- [x] `api_server.py` - serialize_node() now includes conditions, emotion_intensity
- [x] `structural_parser_worker.py` - preserves source_id, speaker, quest, topic, conditions, arc_emotions per beat
- [x] `scripts/run_batch.py` - compile_translations() rewritten with full schema (18+ fields)
- [x] `claudefiles/subagents/translation_engine/CLAUDE.md` - generates synthetic_conditions, speaker, synthetic_topic
- [x] `workflow/multi_backend.py` - _format_task requests synthetic_conditions from LLMs
- [x] `workflow/orchestrator.py` - WORKER_PROMPTS updated for synthetic_conditions schema

New nodes now include: id, text, emotion, emotion_raw, emotion_intensity, beat_function, beat_index, archetype_relation, speaker, arc_shape, arc_emotions, barrier_type, attractor_type, source_game, source_ref, source_run, ticket_id, topic, quest, conditions, confidence, proper_nouns

Edges now include: source, target, type, transition (e.g., "anger->fear"), archetype

---

## Original Audit (Pre-Rebuild)

## Executive Summary

The synthetic dialogue pipeline has suffered **progressive schema deflation** at every stage. The v1 synthetic outputs (gallia_v1/dialogue.json) preserved rich metadata; current outputs (gallia_v4/graph.json) preserve almost nothing. This makes topology analysis, semantic grouping, and provenance tracking impossible.

**Fields preserved**: 4 of 18+ (text, emotion, beat_function, source_run)
**Fields lost**: 14+ including speaker, quest, topic, conditions, archetype, arc_shape, source_ref, concept_mappings

---

## Pipeline Stages and Schema Loss

### Stage 1: Source Extraction ✓
**File**: `dialogue_data/*_dialogue.json`
**Status**: Complete - all fields preserved

```
Node schema:
  form_id: str           # unique identifier
  topic: str             # dialogue topic (INFO record)
  speaker: str|null      # speaking NPC
  text: str              # dialogue text
  emotion: str           # from TRDT subrecord
  emotion_value: int     # intensity 0-100
  quest: str|null        # quest context (inconsistent usage)
  conditions: list       # CTDA game state conditions
  script_notes: str      # developer notes
```

**Condition schema** (CTDA records):
```
{type: int, function: int, value: float, param1: int, param2: int, run_on: int}
```
These encode quest stages, faction standings, player karma, NPC state, etc.

---

### Stage 2: Graph Construction ✓
**File**: `dialogue_graph.py`
**Status**: Complete - conditions parsed to StateCondition objects

```python
@dataclass
class DialogueNode:
    id: str
    text: str
    speaker: Optional[str]
    emotion: str
    topic: str
    quest: Optional[str]
    conditions: List[StateCondition]  # ✓ PRESERVED
    outgoing: List[str]
    incoming: List[str]
    emotion_intensity: float
```

---

### Stage 3: API Sample Endpoint ✗ FIRST DROP
**File**: `api_server.py` lines 567-576
**Status**: DROPS conditions and emotion_intensity

```python
nodes = [
    {
        "id": n.id,           # ✓
        "text": n.text,       # ✓
        "speaker": n.speaker, # ✓
        "emotion": n.emotion, # ✓
        "topic": n.topic,     # ✓
        "quest": n.quest      # ✓
        # conditions: ✗ DROPPED
        # emotion_intensity: ✗ DROPPED
    }
]
```

**Fix required**: Add `conditions` serialization (convert StateCondition to dict).

---

### Stage 4: Structural Parser ✗ DROPS topic/quest, IGNORES form_id
**File**: `structural_parser_worker.py`
**Status**: Uses only text/speaker/emotion; drops structural metadata

The parser receives:
```
{"id", "text", "speaker", "emotion", "topic", "quest"}
```

But only uses:
```
{"text", "speaker", "emotion"}
```

**Fix required**:
- Preserve `id` (form_id) for source provenance
- Preserve `topic` and `quest` for semantic grouping
- Attach conditions if received

---

### Stage 5: Parser Output (Triplet) ✗ NO PROVENANCE
**File**: `structural_parser_worker.py` output
**Status**: Infers structure but loses source references

```python
triplet = {
    "arc": [
        {
            "beat": "beat_0",
            "text": str,
            "emotion": str,
            "function": str,           # inferred
            "archetype_relation": str, # inferred
            "transition_from": str
            # source_id: ✗ NOT PRESERVED
        }
    ],
    "proper_nouns_used": [],
    "barrier_type": str,
    "attractor_type": str,
    "arc_shape": str
    # source_form_ids: ✗ NOT PRESERVED
    # source_game: ✗ NOT PRESERVED
    # quest: ✗ NOT PRESERVED
    # topic: ✗ NOT PRESERVED
    # conditions: ✗ NEVER RECEIVED
}
```

**Fix required**: Each beat should preserve its source_id. Arc-level should preserve quest/topic/conditions.

---

### Stage 6: Translation Input ✗ PARTIAL
**File**: `workflow/ticket_routes.py` create_run
**Status**: Has some provenance, missing structural metadata

```python
input_data = {
    "triplet": {...},           # from parser
    "source_game": str,         # ✓ added
    "source_texts": [str],      # ✓ for translation
    "source_bible_name": str,
    "target_bible_name": str,
    "source_bible": str,
    "target_bible": str
    # source_form_ids: ✗ NOT INCLUDED
    # quest: ✗ NOT INCLUDED
    # topic: ✗ NOT INCLUDED
    # conditions: ✗ NOT INCLUDED
    # speaker: ✗ NOT INCLUDED
}
```

**Fix required**: Include source_form_ids, quest, topic, conditions from walk.

---

### Stage 7: Translation Output ✗ MINIMAL
**File**: Translation engine output
**Status**: Just texts + nouns, no derived metadata

```python
output_data = {
    "translated_texts": [str],
    "proper_nouns_introduced": [str],
    "confidence": float
    # concept_mappings: ✗ (was in v1!)
    # register_notes: ✗ (was in v1!)
    # speaker: ✗ NOT DERIVED
    # synthetic_topic: ✗ NOT DERIVED
    # synthetic_conditions: ✗ NOT DERIVED
}
```

**Fix required**:
- Derive speaker from proper_nouns_introduced or archetype
- Generate synthetic conditions (string-typed, semantic)
- Include concept_mappings and register_notes

---

### Stage 8: Compilation ✗ CATASTROPHIC LOSS
**File**: `scripts/run_batch.py` compile_translations()
**Status**: Throws away almost everything

Current output:
```python
node = {
    "id": hash,
    "text": str,
    "emotion": str,
    "beat_function": str,
    "source_run": str
}
edge = {
    "source": str,
    "target": str,
    "type": "sequential"
}
```

**v1 dialogue.json had**:
```python
node = {
    "form_id": str,
    "topic": str,
    "speaker": str,
    "text": str,
    "emotion": str,
    "emotion_value": int,
    "quest": str,
    "conditions": [],
    "synthetic_meta": {
        "source_game": str,
        "source_ref": str,
        "ticket_id": str,
        "beat_index": int,
        "beat_function": str,
        "archetype_relation": str,
        "arc_shape": str,
        "arc_emotions": [str],
        "confidence": float,
        "concept_mappings": [...],
        "register_notes": str
    }
}
```

---

## Key Insight: Conditions as Semantic Grouping

The `quest` field is **inconsistently used** by game developers:
- "Quest and Location Independent Dialogue" (950 lines)
- "NQDImperialCity" (523 lines)
- `None` (2318 lines)

But `conditions` provide **consistent semantic grouping**:
- Function 427, Param1 464867 groups **1751 lines across 38 quests**
- Dialogue sharing conditions is narratively related (same game state gate)

For synthetic data, we should generate **string-typed conditions**:
```json
{
  "conditions": [
    {"type": "quest_stage", "quest": "prefecture_audit", "min_stage": 20},
    {"type": "has_met", "npc": "Sous-Préfet Marchais"},
    {"type": "faction_standing", "faction": "Notariat", "relation": ">=neutral"}
  ]
}
```

This enables:
- Grouping dialogue by shared semantic conditions
- Sampling walks that respect game state progression
- Building "quest hubs" as nodes sharing conditions

---

## Rebuild Specification

### API Layer (`api_server.py`)

```python
# In sample_dialogue(), add to node serialization:
nodes = [
    {
        "id": n.id,
        "text": n.text,
        "speaker": n.speaker,
        "emotion": n.emotion,
        "emotion_intensity": n.emotion_intensity,  # ADD
        "topic": n.topic,
        "quest": n.quest,
        "conditions": [c.to_dict() for c in n.conditions]  # ADD
    }
    for n in path
]
```

### Structural Parser (`structural_parser_worker.py`)

```python
# Preserve source provenance in arc beats:
beat = {
    "beat": f"beat_{i}",
    "source_id": line.get("id"),           # ADD
    "text": text,
    "emotion": emotion,
    "function": beat_function,
    "archetype_relation": archetype_relation,
    "transition_from": transition_from
}

# Preserve walk-level metadata:
triplet = {
    "arc": arc,
    "source_ids": [line.get("id") for line in walk],  # ADD
    "source_quest": walk[0].get("quest"),              # ADD
    "source_topic": walk[0].get("topic"),              # ADD
    "source_conditions": walk[0].get("conditions", []), # ADD
    "proper_nouns_used": proper_nouns,
    "barrier_type": barrier_type,
    "attractor_type": attractor_type,
    "arc_shape": arc_shape
}
```

### Run Creation (`workflow/ticket_routes.py`)

```python
# Include source metadata in translation input:
input_data = {
    "triplet": triplet,
    "source_game": game,
    "source_texts": [b["text"] for b in triplet["arc"]],
    "source_form_ids": triplet.get("source_ids", []),  # ADD
    "source_quest": triplet.get("source_quest"),       # ADD
    "source_topic": triplet.get("source_topic"),       # ADD
    "source_conditions": triplet.get("source_conditions", []),  # ADD
    "source_bible_name": source_bible_name,
    "target_bible_name": target_bible_name,
    # ...
}
```

### Translation Engine Prompt

Add to output schema:
```
"speaker": "Derived speaker name or archetype",
"synthetic_topic": "Generated topic/conversation name",
"synthetic_conditions": [
    {"type": "quest_stage", "quest": "...", "stage": "..."},
    ...
],
"concept_mappings": [...],
"register_notes": "..."
```

### Compilation (`scripts/run_batch.py`)

```python
def compile_translations(run_id: str, setting: str, version: int):
    # ... load tickets ...

    for ticket in translate_tickets:
        input_data = ticket.get("input_data", {})
        output_data = ticket.get("output_data", {})
        triplet = input_data.get("triplet", {})
        arc = triplet.get("arc", [])

        for i, (text, beat) in enumerate(zip(output_data["translated_texts"], arc)):
            node = {
                "id": generate_id(run_id, ticket_id, i),
                "text": text,

                # Emotion
                "emotion": validate_emotion(beat.get("emotion")),
                "emotion_value": 50,  # default, could be derived

                # Structural
                "beat_function": beat.get("function"),
                "beat_index": i,
                "archetype_relation": beat.get("archetype_relation"),

                # Arc context
                "arc_shape": triplet.get("arc_shape"),
                "arc_emotions": [b.get("emotion") for b in arc],

                # Provenance
                "source_game": input_data.get("source_game"),
                "source_ref": hash(beat.get("source_id", "")),
                "source_run": run_id,
                "ticket_id": ticket_id,

                # Semantic grouping (from translation output)
                "speaker": output_data.get("speaker"),
                "topic": output_data.get("synthetic_topic"),
                "conditions": output_data.get("synthetic_conditions", []),

                # Translation metadata
                "confidence": output_data.get("confidence"),
                "proper_nouns_introduced": output_data.get("proper_nouns_introduced", []),
            }
            nodes.append(node)

        # Edges with transition metadata
        for i in range(len(texts) - 1):
            edge = {
                "source": nodes[i]["id"],
                "target": nodes[i+1]["id"],
                "type": "sequential",
                "transition": f"{arc[i]['emotion']}->{arc[i+1]['emotion']}",
                "archetype": arc[i].get("archetype_relation"),
            }
            edges.append(edge)
```

---

## Migration Path

1. **API fix**: Add conditions to /api/sample serialization (5 min)
2. **Parser fix**: Preserve source_id, quest, topic, conditions (30 min)
3. **Run creation fix**: Pass through source metadata (15 min)
4. **Translation prompt update**: Request speaker, topic, conditions (30 min)
5. **Compilation rewrite**: Full schema preservation (1 hr)
6. **Backfill existing graphs**: Re-run with provenance recovery where possible

### Backfill Strategy

For existing graphs (gallia_v3-5, marmotte_v1-3):
- Cannot recover lost source provenance
- CAN add derived fields (speaker from proper_nouns, topic from arc_shape)
- CAN add placeholder conditions based on arc grouping
- Mark as `provenance: "partial"` vs new graphs with `provenance: "full"`

---

## Test Cases

After rebuild, verify:

1. **Provenance chain**: Can trace synthetic node back to source form_id
2. **Condition grouping**: Nodes with shared conditions cluster correctly
3. **Speaker extraction**: Derived speakers match proper_nouns patterns
4. **Topology analysis**: Hub/spur/cycle metrics now meaningful
5. **Emotion intensity**: Available for nuanced emotion analysis

---

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `api_server.py` | Add conditions to sample serialization | P0 |
| `structural_parser_worker.py` | Preserve source IDs and walk metadata | P0 |
| `workflow/ticket_routes.py` | Pass source metadata to translation | P0 |
| `scripts/run_batch.py` | Rewrite compile_translations() | P0 |
| Translation prompts | Add speaker/topic/conditions to output | P1 |
| `scripts/compare_topology.py` | Use new fields for analysis | P2 |
