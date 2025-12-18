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

## Project Structure

```
steam_locator.py       — Cross-platform Steam library detection (incl. WSL)
esm_dialogue_parser.py — Binary parser for TES4-era ESM format
extract_dialogue.py    — CLI orchestration
chain_linker.py        — Group dialogue into conversational chains
CLAUDE.md              — Research context and analysis tasks
pyproject.toml         — uv-compatible project configuration
```
