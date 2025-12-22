# Dialogue Yoinker - Session Notes 2025-12-22

## Overview

This session extended a dialogue extraction toolkit for Bethesda games (Oblivion, Fallout NV, Skyrim) into a graph-based analysis and sampling system. The goal: extract situated dialogue with emotion/speaker/quest annotations for ML training corpora.

## Corpus Size

| Game | Lines | Source |
|------|-------|--------|
| Oblivion + DLCs | 20,316 | Direct ESM parsing |
| Fallout NV + DLCs | 28,933 | Direct ESM parsing |
| Skyrim + DLCs | 41,188 | ESM + BSA string tables |
| **Total** | **90,437** | |

---

## What Was Built

### 1. BSA Archive Parser (`bsa_parser.py`)

Skyrim SE stores localized strings in compressed BSA archives, not inline in the ESM. Built a parser for:
- BSA format v104 (Skyrim LE) and v105 (Skyrim SE)
- LZ4 decompression for SE archives
- File extraction by path pattern

```python
from bsa_parser import BSAParser
parser = BSAParser(Path("Skyrim - Interface.bsa"))
data = parser.extract_file("strings/skyrim_english.strings")
```

### 2. String Table Parser (`string_tables.py`)

Three string table formats:
- `.STRINGS` - null-terminated (names, short text)
- `.ILSTRINGS` - length-prefixed (dialogue)
- `.DLSTRINGS` - length-prefixed (descriptions)

```python
from string_tables import StringTableManager
manager = StringTableManager()
manager.load_from_bytes({"skyrim_english.strings": data, ...})
text = manager.get(string_id)  # Returns "Fus Ro Dah!" instead of garbled bytes
```

### 3. Emotion Bridge Graph (`emotion_bridge.py`)

Cross-game graph where emotion transitions act as "wormholes" between datasets.

**Concept:** A `happy→disgust` transition in Skyrim can bridge to `happy→disgust` in Fallout NV, enabling cross-game trajectory sampling.

**Structure:**
- 89,651 dialogue nodes across 3 games
- 64 bridge cells (emotion transitions appearing in multiple games)
- All 64 possible emotion pairs bridge all 3 games

**Key API endpoints:**
```
GET  /api/bridge/stats          - Graph statistics
GET  /api/bridge/matrix         - Emotion transition matrix with bridge highlighting
POST /api/bridge/walk           - Cross-game random walk
POST /api/bridge/coverage       - Walk that covers all games via bridges
```

**Example cross-game walk:**
```
FalloutNV: "I'd not touch Gomorrah's machines..."
  🌉 BRIDGE (neutral→pained)
Skyrim: "Wuld..." (Arngeir teaching Thu'um)
  🌉 BRIDGE (neutral→neutral)
Oblivion: "You have overcome half of us..."
Coverage: 100%
```

### 4. Query Graph (`query_graph.py`)

Bipartite graph modeling topics as semantic "gaps" that text responses fill.

**Structure:**
- 24,750 topic nodes (queries/gaps)
- 70,784 text nodes (responses)
- Topics like `GREETING`, `Attack`, `GOODBYE` are hubs
- Most quest-specific topics are leaf nodes

**Key insight:** Only 117 topics have outgoing edges to other topics. This proves the data captures NPC *responses*, not player *traversals*. The player's choice path is mediated by the game state machine (quest stages, conditions) which isn't in the static data.

**Topic→Topic transitions (from real DialogueGraph edges):**
```
GREETING → HELLO:    5,194 text-level edges
GREETING → GOODBYE:  4,924 text-level edges
GREETING → Attack:   1,729 text-level edges
```

**Key API endpoints:**
```
GET  /api/query/stats           - Bipartite graph stats
GET  /api/query/topics          - List topics by category
GET  /api/query/categories      - Semantic categories (GREETING, COMBAT, QUEST...)
GET  /api/query/transitions     - Topic→topic transition stats (real edges)
GET  /api/query/cycles          - Find conversational loops
POST /api/query/walk            - Topic chain walk (interleaved topic→text)
POST /api/query/sample          - Sample responses for a topic/category
```

**Semantic categories auto-detected:**
| Category | Topics |
|----------|--------|
| QUEST | 673 |
| COMPANION | 386 |
| GREETING | 306 |
| CRIME | 306 |
| COMBAT | 280 |
| FAREWELL | 160 |

### 5. Web UI Extensions

Added interactive sections to the landing page (`http://localhost:8000/`):

**Cross-Game Emotion Bridge:**
- Load bridge graph with stats
- Transition matrix with gold-highlighted bridge cells
- Cross-game walk sampling with adjustable bridge probability
- Coverage trajectory sampling

**Query Graph (Topics as Gaps):**
- Stats display (topics, texts, cross-game count)
- Cycle finder (real structural cycles)
- Transition stats (hub topics, strongest edges)
- Topic chain walks
- Category sampling with cross-game option

---

## Data Structures

### DialogueGraph (existing)
```
Nodes: dialogue lines with {form_id, text, speaker, emotion, topic, quest, conditions}
Edges:
  - sequential (within topic, by quest stage)
  - topic_branch (GREETING → specific topics)
  - condition_gate (same content, different state requirements)
```

### EmotionBridgeGraph (new)
```
Nodes: BridgeNode {game, form_id, text, speaker, emotion, topic}
Edges:
  - intra-game (normal dialogue flow)
  - bridge (cross-game via matching emotion transitions)
Index: transition_matrix[(src_emotion, tgt_emotion)] → EmotionCell
```

### QueryGraph (new)
```
Nodes:
  - TopicNode {id, category, games, response_count, emotion_distribution}
  - TextNode {game, form_id, text, speaker, emotion, topic}
Edges: topic → text (bipartite)
Derived: topic → topic transitions (lifted from real text→text edges)
```

---

## Key Findings

### 1. The Leaf Node Structure

99.5% of topics are terminal. This means:
- Player picks option → NPC responds → control returns to game
- The ESM stores *responses*, not *player choices*
- No preference signal (A vs B) for RLHF/BTRM training

### 2. Hub Topics

Only ~117 topics connect to other topics:
- `GREETING`, `HELLO`, `GOODBYE` - conversation bookends
- `Attack`, `Hit`, `Flee`, `Murder` - combat states
- `Assault`, `Steal`, `Pickpocket` - crime responses

These are the "interface affordances" where the dialogue system engages.

### 3. Cross-Game Universals

20 topics appear across multiple games:
```
GREETING, HELLO, GOODBYE, Attack, Hit, Flee,
Steal, Yield, AcceptYield, Pickpocket...
```

These are the shared vocabulary of Bethesda's dialogue systems.

---

## What This Data Is Good For

- NPC response generation given topic/context
- Emotion-conditioned dialogue generation
- Style transfer across game worlds
- Situated dialogue (with quest/speaker context)

## What This Data Is NOT Good For

- Preference learning (no A vs B signal)
- Player choice modeling (no traversal data)
- RLHF reward modeling (responses only, not preferences)

The sparse topic graph is the cleanest proof: if there were rich topic→topic chains, we'd have player decision paths. Instead we have hub→leaf structure confirming it's response-only.

---

## Running the Server

```bash
cd /home/bigboi/dialogue_yoinker
.venv/bin/uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Landing page: `http://localhost:8000/`
API docs: `http://localhost:8000/docs`

---

## Files Modified/Created This Session

```
bsa_parser.py          - BSA archive extraction
string_tables.py       - Skyrim string table parsing
emotion_bridge.py      - Cross-game emotion transition graph
query_graph.py         - Topics-as-gaps bipartite graph
api_server.py          - Added bridge/query endpoints + UI
extract_all_dlc.py     - Added Skyrim with string tables
esm_dialogue_parser.py - Renamed from yoinkems.py (RIP)
```
