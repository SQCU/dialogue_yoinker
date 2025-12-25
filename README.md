# dialogue-yoinker

Extract situated dialogue from Bethesda games (Oblivion, Fallout 3/NV, Skyrim) for ML training.

## Why This Exists

Most text corpora used for language model training are **causally impoverished**. Webtext is a heap of utterances ripped from their contexts of production. This project extracts dialogue that retains its communicative structure:

- **Speaker identity** — who is saying this, with what faction/disposition
- **Emotion annotation** — not sentiment analysis, but *authorial intent* for how the line should be performed (from facial animation data)
- **Quest context** — what narrative stage gates this dialogue
- **Conditions** — gameplay logic that determines when lines appear

The bet: small models trained on small but causally rich data can learn something that large models trained on large but causally impoverished data cannot.

See `CLAUDE.md` for the full research framing.

## Setup

### Use uv, not system Python

This project assumes you're using [uv](https://docs.astral.sh/uv/) for Python environment management. Why?

1. **Reproducibility** — `uv` locks exact versions and creates isolated environments automatically
2. **Speed** — dependency resolution is 10-100x faster than pip
3. **No contamination** — system Python accumulates cruft; project environments stay clean
4. **Cross-platform** — same workflow on Linux, macOS, Windows, WSL

If you're sharing dataset tools with collaborators, `uv` eliminates "works on my machine" problems.

```bash
# Install uv (if you haven't)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <this-repo>
cd dialogue-yoinker
uv sync  # creates .venv, installs dependencies

# Run directly (uv handles the venv)
uv run python extract_dialogue.py --list
```

### Optional dependencies

```bash
# For corpus analysis (pandas, matplotlib)
uv sync --extra analysis

# For relationship graphs (networkx)
uv sync --extra graphs

# Everything
uv sync --extra dev
```

## Usage

### Find installed games

```bash
uv run python extract_dialogue.py --list
```

Works on Linux, Windows, macOS, and **WSL** (detects Windows Steam automatically).

### Extract dialogue

```bash
# Single game with statistics
uv run python extract_dialogue.py oblivion --stats

# All found games
uv run python extract_dialogue.py --all --output ./corpus

# Custom output location
uv run python extract_dialogue.py falloutnv -o ./fnv_dialogue
```

### Output format

`*_training.jsonl` — one JSON object per line:

```json
{
  "text": "Patrolling the Mojave almost makes you wish for a nuclear winter.",
  "speaker": "NCR Trooper",
  "emotion": "neutral",
  "emotion_intensity": 0.0,
  "topic": "YOURDEFAULTTOPIC",
  "quest_context": null,
  "meta": {
    "source": "FalloutNV.esm",
    "game": "fo3_fnv",
    "form_id": "0x..."
  }
}
```

The `emotion` field comes from the TRDT subrecord — Bethesda's facial animation choreography, not algorithmic sentiment analysis.

## Supported Games

| Game | Key | Format |
|------|-----|--------|
| Oblivion | `oblivion` | TES4 |
| Fallout 3 | `fallout3` | TES4 |
| Fallout: New Vegas | `falloutnv` | TES4 |
| Skyrim | `skyrim` | TES4 |
| Skyrim SE | `skyrimse` | TES4 |
| Fallout 4 | `fallout4` | TES4* |

*Fallout 4 has format variations that may need additional handling.

## Legal Note

ESM/ESP files are deliberately designed for modding. Tools like TES Construction Set, GECK, and xEdit have parsed these files for decades with Bethesda's implicit blessing. We read files meant to be read, from games legitimately acquired.

## Dialogue Chain Sampling

The `chain_linker.py` tool groups related dialogue into conversational chains:

```bash
# Statistics and sample chains
uv run python chain_linker.py dialogue_data/falloutnv_dialogue.json --stats --sample 5

# Export all chains (min 4 lines) to JSONL
uv run python chain_linker.py dialogue_data/oblivion_dialogue.json -m 4 -o chains.jsonl
```

Chains are built from:
- Quest context (most reliable grouping)
- Topic grouping within quests
- Quest stage conditions (CTDA) for sequencing

Example output from Oblivion - a scripted pirate conversation with emotional arc:
```
[Quest: Dark Brotherhood Scripted Conversations]
NPC(disgust): I'm telling ya, lad, it's bad luck. A woman onboard a pirate vessel!
NPC(happy): Oh, come now. How many times has Malvulis saved our necks, huh?
NPC(surprise): Ho, there, laddie, now you're just bein' insultin'!
NPC(sad): I told you a million times, that wasn't my fault! The wheel was covered with gull droppings!
NPC(anger): You'd better watch your tongue, old man!
```

## Dialogue Graph & API Server

The extracted dialogue can be explored as a directed graph:

```bash
# Start the exploration server
uv sync --extra server
uv run uvicorn api_server:app --port 8000

# Visit http://localhost:8000 for interactive explorer
# Or http://localhost:8000/docs for API documentation
```

Graph analysis endpoints:
- `/api/stats/{game}` — node/edge counts, degree distributions
- `/api/pagerank/{game}` — find important dialogue hubs
- `/api/communities/{game}` — cluster detection
- `/api/sample` — random walks through dialogue

## Synthetic Dialogue Generation

**This is the interesting part.**

The project includes a pipeline for generating new dialogue corpora by *structural transposition* — extracting the communicative shape of source dialogue and re-instantiating it in a new fictional setting.

```
Reference Walk (Oblivion)
    "The Emperor needs you to find the Amulet of Kings"
         ↓
Structural Triplet
    [authority → requests → player → find → macguffin]
    emotion: neutral → urgent
    archetype: ruler_to_hero
         ↓
Setting Transposition (Gallia)
    "The Préfet requires your assistance locating the Seal of the République"
```

The output preserves:
- Emotion sequences from source
- Beat functions (establish_stakes, threaten, plead)
- Archetype relations (authority_to_subject, peer_to_peer)
- Arc shapes (escalation, tension_release, etc.)

While changing:
- Proper nouns (characters, places, factions)
- Register (bureaucratic French vs. high fantasy)
- Setting-specific vocabulary

### Current Results

**gallia_v3** (Dec 2025):
- 3,568 nodes (synthetic dialogue lines)
- 3,654 edges (sequential + bridge links)
- 1 connected component (fully linked graph)
- Translated from Oblivion + Fallout NV reference corpus

The output passes an interesting test: easier to identify as "Bethesda-style RPG dialogue" than as "AI-generated text."

### How It Works

See `notes/2025-12-23-translation-orchestration.md` for the full pipeline.

The orchestration pattern is batch dispatch with typed subagents:
1. Sample walks from reference graph
2. Dispatch structural-parser agents (extract triplets)
3. Dispatch translation-engine agents (transpose to target setting)
4. Collect results, retry failures
5. Apply to synthetic graph
6. Run linking pass (generate bridge nodes for hub formation)

This is **not** agent orchestration in the LangChain/AutoGPT sense. No planning loops, no tool chains, no agents deciding what to do. The Claude Code session knows the pipeline shape; subagents are typed workers that process assigned files.

Closest analogy: MapReduce with LLM workers, or Kubernetes Jobs where each job is a prompted completion.

### Remote API Compatibility

The pipeline is **model-agnostic** — any API supporting OpenAI-compatible chat completions works as a worker backend.

**Tested backends:**

| Backend | Model | Tool Use | Notes |
|---------|-------|----------|-------|
| Anthropic | Claude Sonnet/Haiku | ✓ | Primary development model |
| DeepSeek | deepseek-chat (v3.2) | ✓ | Cost-effective, good structural compliance |
| OpenAI | gpt-4o-mini | ✓ | Not tested, should work |

**DeepSeek results (Dec 2024):**
- Structural parsing: 100% success rate on test batches
- Translation quality: Matches Claude output register and tone
- Cost: ~$0.02 per 25-sample batch (20x cheaper than Sonnet)
- Tool calling: Not required — pipeline uses prompt+JSON response pattern

**Running with DeepSeek:**
```bash
export DEEPSEEK_API_KEY="sk-..."

# Full pipeline: 100 translate, 100 link, 100 extend
python scripts/run_batch.py full gallia:4 100

# Multiple settings in parallel with custom concurrency
python scripts/run_batch.py full gallia:4,marmotte:2 100 --parallel --concurrency 30

# Individual phases
python scripts/run_batch.py translate gallia 100
python scripts/run_batch.py link gallia:4 100
python scripts/run_batch.py extend gallia:4 100 --source-run link_20251225_...
```

**Performance** (at 25 concurrency): ~1-2 tickets/sec for parsing/translation, ~0.8 tickets/sec for linking/extension.

See `notes/run_batch_pipeline.md` for full documentation.

**Introspection for foreign API runs:**
- `GET /api/runs` — List all runs with ticket status
- `runs/{run_id}/queue.json` — Full ticket data with inputs/outputs
- Results persist to disk even if orchestration crashes

**Why DeepSeek v3.2 specifically:**
- Open weights (can run locally for debugging)
- Logit access for interpretability research
- Activation probing compatible with repeng tools
- Price/performance competitive with Haiku

See `notes/2025-12-23-deepseek-portability-review.md` for technical details on model-agnostic formalization.

## Project Structure

```
# Extraction
steam_locator.py       — Cross-platform Steam library detection (incl. WSL)
esm_dialogue_parser.py — Binary parser for TES4-era ESM format
extract_dialogue.py    — CLI orchestration
extract_all_dlc.py     — DLC extraction

# Graph Analysis
dialogue_graph.py      — Directed graph at INFO node level
topic_graph.py         — Topic→topic transition graphs
chain_linker.py        — Quest-based chain grouping
graph_diagnostics.py   — Topology comparison tools

# Synthetic Generation
batch_growth.py        — Batch translation runner
graph_linker.py        — Bridge generation for hub formation
bibles/                — Lore bibles (setting definitions)
prompts/               — Prompt templates for generation

# API & Exploration
api_server.py          — FastAPI server
landing_html.py        — Interactive web explorer

# Orchestration Infrastructure
workflow/              — Ticket queue, multi-backend dispatch
  ticket_queue.py      — Model-agnostic work queue
  multi_backend.py     — DeepSeek/OpenAI/Anthropic workers
  orchestrator.py      — Spawn-and-await pattern
scripts/               — Pipeline scripts
  run_batch.py         — Unified pipeline runner (main entry point)
  run_link_stitch_batch.py    — Link ticket processing
  run_extension_resolve_batch.py — Extension ticket processing
  apply_link_stitch_results.py   — Apply links to graph
  apply_extension_results.py     — Apply extensions to graph
  legacy/              — Deprecated scripts (kept for reference)

# Subagent Infrastructure
subagent_orchestrator/ — API routes for triplet extraction/translation
claudefiles/subagents/ — Per-worker CLAUDE.md prompts

# Documentation
CLAUDE.md              — Research context and analysis tasks
notes/                 — Session notes and milestone docs
```
