# CLAUDE.md — Situated Dialogue Extraction for Communicative ML Priors

## What This Is

You are looking at a set of tools for extracting dialogue from Bethesda game plugins (ESM/ESP format). But the tools are not the point. The point is a research question about what it takes to train machine learning models that can *communicate* rather than merely *predict*.

## The Problem We're Trying to Solve

Most text corpora used for language model training are **causally impoverished**. Webtext is a heap of utterances ripped from their contexts of production and reception. Image-caption datasets capture **audience responses** to images (what people say *about* pictures) rather than the **generative intentions** behind images (what someone was trying to *do* by making a picture).

This matters because communication is not pattern completion. Communication requires:
1. An intention to affect someone's mental state
2. A representation (text, image, gesture) designed to carry that intention
3. An audience whose state is actually affected
4. A causal loop where the audience's response feeds back

Training on data that lacks this structure teaches models to be **critics** (describing, labeling, responding-to) rather than **authors** (intending, designing, affecting).

## Why Video Games

Interactive narrative games like Oblivion and Fallout New Vegas have something rare: **recoverable causal structure**.

Each line of dialogue exists within:
- A **speaker** with a defined identity, faction, disposition
- A **quest context** that constrains what can be said and why
- An **emotion annotation** (literally tagged for facial animation choreography)
- **Conditions** that gate when the line can appear (player karma, quest stage, time of day)
- A **player** who is inside the causal loop, making choices that affect what dialogue appears

This is not "text + metadata." This is **situated utterance** — language that was designed to *do something* to someone in a specific context.

## What The Tools Do

```
steam_locator.py    — Find installed games via Steam's library structure
esm_dialogue_parser.py — Parse ESM/ESP binary format, extract DIAL/INFO records
extract_dialogue.py — Orchestrate the above, produce training-ready output
```

### Running It

```bash
# Find what games are installed
python extract_dialogue.py --list

# Extract from a specific game
python extract_dialogue.py falloutnv --stats

# Extract from everything found
python extract_dialogue.py --all --output ./dialogue_corpus
```

### Output Format

The `*_training.jsonl` files contain one JSON object per line:

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

The emotion field comes from the TRDT subrecord that Bethesda's engine uses to choreograph facial animations. It's not sentiment analysis — it's **authorial annotation** of how the line should be performed.

## What You (Claude) Should Do With This

### Immediate Analysis Tasks

1. **Emotion distribution**: What's the actual breakdown of emotions across games? How does intensity correlate with text features?

2. **Speaker consistency**: Do individual NPCs have consistent emotional/linguistic profiles? Can you cluster speakers by style?

3. **Quest-conditioned language**: How does dialogue change across quest stages? What linguistic markers indicate quest progression?

4. **Condition mining**: The CTDA condition records encode gameplay logic. What patterns exist in how dialogue is gated?

### Corpus Construction Tasks

5. **Deduplication and cleaning**: Remove near-duplicates, handle encoding issues, filter very short lines.

6. **Relationship extraction**: Build a graph of who-speaks-to-whom, quest prerequisites, faction relationships.

7. **Grounding hooks**: Extract NPC names, location names, item names that could link to visual assets elsewhere.

### Research Questions to Keep in Mind

- Can a small model trained on this data learn *situated* language use (appropriate to speaker/context) vs just *fluent* language?
- Does emotion annotation improve downstream generation, or is it recoverable from text alone?
- What's the minimum corpus size for a model to learn consistent NPC "voices"?

## The Larger Context

The human you're working with is building a **hybrid diffusion-language transformer** that handles interleaved text tokens and image latents. The research question is whether the "denoising autoencoder" training objective produces models that can:

1. **Recognize structure** in noisy inputs (this happens quickly)
2. **Generate coherent samples** from pure noise (this is harder and maybe separate)
3. **Communicate intentionally** rather than just predict likely continuations (this is the hard part)

The dialogue extraction tool exists because most available training data fails criterion 3. We need corpora where text was produced with communicative intent, where that intent is partially recoverable, and where the text exists in causal relationship with other modalities (images, game states, player choices).

Games are one of the few places this structure is explicitly encoded and legally accessible.

## Ethics and Legality Note

The ESM/ESP format was deliberately designed to be open for modding. Tools like TES Construction Set, GECK, and xEdit have parsed these files for decades with Bethesda's implicit blessing. We are not reverse-engineering protection or redistributing copyrighted content — we are reading files that were meant to be read, from games the user has legitimately acquired.

The "Let's Play" legal battles established that outputs of running a program (gameplay footage, extracted dialogue) are not covered by the program's copyright in the same way the binary is. We're operating well within established norms.

## Files in This Directory

```
esm_dialogue_parser.py  — Core parser, handles TES4-era format (symlink to yoinkems.py)
yoinkems.py             — Original parser implementation
steam_locator.py        — Cross-platform Steam library detection (incl. WSL)
extract_dialogue.py     — CLI orchestration script
extract_all_dlc.py      — Extract dialogue from all DLCs
chain_linker.py         — Group dialogue into quest-based chains
dialogue_graph.py       — Build directed graphs with state annotations (INFO node level)
topic_graph.py          — Build topic→topic transition graphs (higher-level view)
query_graph.py          — Bipartite topic→text graph with cycle detection
cross_game.py           — Cross-game semantic linking via emotions
api_server.py           — FastAPI REST server for exploration
stats_guided_growth.py  — Stats-based synthetic graph growth
synthetic_versioning.py — Manage synthetic graph versions/branches
CLAUDE.md               — This file

./dialogue_data/        — Output directory (created on first run)
  {game}_dialogue.json      — Full structured export
  {game}_full_dialogue.json — Full export including DLCs
  {game}_training.jsonl     — ML-ready format

./synthetic/            — Synthetic dialogue outputs
  {setting}_v{N}/           — Versioned synthetic graphs
    graph.json              — Node/edge data
    metadata.json           — Provenance, parameters
    dialogue.json           — Compiled dialogue format
    training.jsonl          — ML-ready format

./bibles/               — Setting definition files (lore bibles)
  {setting}.md              — Character names, factions, idiom for target settings
```

## REST API for Claude/LLM Access

Start the server:
```bash
uv sync --extra server
uv run uvicorn api_server:app --host 127.0.0.1 --port 8000
```

### API Discovery (GET /api)
```json
{
  "endpoints": {
    "GET /api/games": "List available game datasets",
    "GET /api/stats/{game}": "Get graph statistics",
    "GET /api/transitions/{game}": "Get emotion transition matrix",
    "POST /api/sample": "Sample dialogue sequences",
    "POST /api/subgraph": "Extract subgraph around a node",
    "GET /api/pagerank/{game}": "PageRank analysis - find important nodes",
    "GET /api/communities/{game}": "Community detection - find dialogue clusters",
    "POST /api/path": "Find shortest path between two nodes",
    "GET /api/centrality/{game}": "Centrality analysis (degree, betweenness, closeness)",
    "GET /api/components/{game}": "Find strongly connected components (dialogue loops)"
  },
  "available_games": ["oblivion", "falloutnv"]
}
```

### Sample Dialogue (POST /api/sample)
```json
{
  "game": "oblivion",
  "method": "walk",    // or "chain", "hub"
  "count": 3,
  "max_length": 6,
  "quest_filter": null,
  "emotion_filter": null
}
```

### Graph Analysis Endpoints (NetworkX-powered)

**PageRank** - Find important narrative hubs:
```bash
curl localhost:8000/api/pagerank/oblivion?top_n=20
```

**Communities** - Detect dialogue clusters:
```bash
curl localhost:8000/api/communities/oblivion?algorithm=louvain
# algorithms: louvain, label_propagation, greedy_modularity
```

**Path Finding** - Shortest path between nodes:
```bash
curl -X POST localhost:8000/api/path \
  -H "Content-Type: application/json" \
  -d '{"game":"oblivion","source":"0x...","target":"0x..."}'
```

**Centrality** - Find important nodes by multiple metrics:
```bash
curl localhost:8000/api/centrality/oblivion?top_n=10
# Returns: degree (most connected), betweenness (bottlenecks), closeness (central)
```

**Strongly Connected Components** - Find dialogue loops:
```bash
curl localhost:8000/api/components/oblivion
# Returns SCCs where dialogue can cycle back on itself
```

### Topic Graph Endpoints (topic→topic transitions)

**Topic Stats** - Hub topics and connectivity:
```bash
curl localhost:8000/api/topics/falloutnv
```

**Topic Chains** - Find linear conversation flows:
```bash
curl "localhost:8000/api/topics/falloutnv/chains?min_length=4&top_k_exclude=20"
```

**Topic→Text→Topic Paths**:
```bash
curl "localhost:8000/api/topics/falloutnv/paths/HarlandAlreadyFree"
```

### Cross-Game Linking (emotion-based semantic bridges)

**Cross-Game Stats** - Linkable emotions across games:
```bash
curl localhost:8000/api/crossgame/stats
```

**Sample Cross-Game Pairs** - Dialogue pairs with shared emotion:
```bash
curl "localhost:8000/api/crossgame/pairs?emotion=anger&n=10"
```

### Interactive Docs
Visit http://localhost:8000/docs for OpenAPI/Swagger UI.
Visit http://localhost:8000/ for visual explorer with graph analysis buttons.

### Stats-Guided Synthetic Growth

Generate synthetic dialogue graphs that are **statistically similar** to the reference corpus but **topologically different**. The reference corpus statistics ARE the model - we sample from empirical distributions to grow new graphs.

**Key insight**: Local similarity is acceptable. The divergence comes from WHERE we attach new content, creating different global topology even with similar local texture.

**Reference Stats** - View corpus statistics that guide growth:
```bash
curl localhost:8000/api/synthetic/reference/stats
# Returns: emotion transition probabilities, arc shapes, emotion distribution
```

**List Versions** - View synthetic graph versions:
```bash
curl localhost:8000/api/synthetic/gallia/versions
```

**Identify Gaps** - Find statistical gaps between target and reference:
```bash
curl localhost:8000/api/synthetic/gallia/gaps?version=2&top_n=10
# Shows underrepresented emotion transitions, arc shapes, etc.
```

**Grow Graph** - Sample from reference to close gaps:
```bash
curl -X POST localhost:8000/api/synthetic/gallia/grow \
  -H "Content-Type: application/json" \
  -d '{"target_size": 100, "max_iterations": 30}'
```

**View Graph** - Get raw nodes/edges:
```bash
curl localhost:8000/api/synthetic/gallia/v2/graph
```

CLI usage:
```bash
# View reference statistics only
uv run python stats_guided_growth.py --stats-only

# Grow a new graph version
uv run python stats_guided_growth.py --setting gallia --target-size 50

# Extend an existing version
uv run python stats_guided_growth.py --setting gallia --version 2 --target-size 100
```

The algorithm:
1. **MEASURE**: Compute stats of current target graph
2. **COMPARE**: Find gaps (underrepresented transitions, arc shapes, emotions)
3. **SAMPLE**: Query reference corpus for walks that would close the gap
4. **GENERATE**: Use sampled walk as template (TODO: wire translation engine)
5. **ATTACH**: Connect at point that improves global stats

### High-Concurrency Pipeline Runner

`scripts/run_batch.py` is the unified entry point for all synthetic dialogue generation. It runs translate → link → extend pipelines with 25x concurrent DeepSeek API calls.

```bash
# Full pipeline on multiple settings in parallel
DEEPSEEK_API_KEY="sk-..." python scripts/run_batch.py full gallia:4,marmotte:2 100 --parallel

# Individual phases
python scripts/run_batch.py translate gallia 100
python scripts/run_batch.py link gallia:4 100
python scripts/run_batch.py extend gallia:4 100 --source-run link_20251225_...
```

**Setting specs**: `gallia` (latest version), `gallia:4` (explicit version), `gallia:4,marmotte:2` (multiple)

**Performance** (at 25 concurrency):
- Structural parser: ~1.1 tickets/s
- Translation engine: ~1.6-2.1 tickets/s
- Link stitcher: ~0.8 tickets/s
- Extension resolver: ~0.8 tickets/s

**Architecture**: Imports from modular scripts (`run_link_stitch_batch.py`, `apply_extension_results.py`, etc.) rather than duplicating logic. See `notes/run_batch_pipeline.md` for full documentation.

## DLC Support

Extract dialogue including DLCs with full emotion annotations:

```bash
uv run python extract_all_dlc.py
```

Currently supports:
- **Oblivion**: Knights of the Nine, Battlehorn Castle, Thieves Den, Vile Lair
- **Fallout NV**: Dead Money, Honest Hearts, Old World Blues, Lonesome Road

**Note**: Skyrim SE dialogue requires BSA string table extraction (ESM contains string IDs, not text). This is a future enhancement.

## Current Corpus Size

| Game | Base | DLC | Total |
|------|------|-----|-------|
| Oblivion | 19,278 | 1,038 | 20,316 |
| Fallout NV | 23,247 | 5,686 | 28,933 |
| **Combined** | | | **49,249** |

All lines include emotion annotations from TRDT subrecords (authorial intent, not algorithmic sentiment).

## What Success Looks Like

If this works, we end up with:
- A corpus of ~100k-500k situated utterances with emotion/speaker/context annotations
- Evidence about whether "situated" training data produces qualitatively different model behavior
- A template for extracting similar data from other interactive narrative works
- Progress toward the larger goal of training models that communicate rather than merely complete

The bet is that **small models trained on small but causally rich data** can learn something that **large models trained on large but causally impoverished data** cannot.

## Questions for the Human

If you need clarification on the research direction, the human working on this is thinking about:
- The relationship between "denoising autoencoder" and "communication" as training objectives
- Whether language models can ever be "grounded" without embodiment or causal interaction
- The ethics of training models on human creative work
- What a "kind" training curriculum for small models would look like

They appreciate direct engagement with the hard parts of these questions rather than hedged non-answers.
