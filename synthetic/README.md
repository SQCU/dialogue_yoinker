# Synthetic Dialogue Graphs

Generated dialogue corpora created by structural transposition from reference games (Oblivion, Fallout NV) into new fictional settings.

## Settings

### Gallia
**"Forms in triplicate. Wine with lunch. Violence by appointment."**

A bureaucratic survival RPG set in a France-for-Franceless-World. The Fifth (or Sixth) République, where every generation relitigates the Revolution, technology serves procedure not efficiency, and violence is costly because it requires forms.

Lore bible: `bibles/gallia.yaml`

### Marmotte (Project Epsilon)
**"Shape the marmot. Shape the encounter. Shape the revenue stream."**

A corporate conspiracy RPG about a shadowy consortium that breeds and deploys trained marmots into Marriott hotels to create controlled "nuisance infestations" generating lucrative pest control contracts. Everyone speaks in business jargon about marmots. Footnotes are common.

Lore bible: `bibles/marmotte.yaml`

---

## Version History

### gallia_v1 (Dec 22, 2024)
- **Status**: Empty/abandoned
- **Method**: Initial scaffolding only
- **Nodes**: 0
- **Notes**: Created during pipeline development, never populated

### gallia_v2 (Dec 22, 2024)
- **Status**: Prototype
- **Method**: Manual testing of translation pipeline
- **Nodes**: 12, Edges: 11
- **Notes**: Small test batch to validate triplet extraction → translation flow

### gallia_v3 (Dec 22, 2024)
- **Status**: Overgrown/deprecated
- **Method**: Aggressive batch growth, random sampling
- **Nodes**: 11,298, Edges: 11,581
- **Notes**: Grew too large too fast with insufficient linking. Sparse connectivity. Abandoned in favor of slower, more controlled growth in v4.

### gallia_v4 (Dec 24-25, 2024)
- **Status**: Active development
- **Method**: Balanced translate/link/extend pipeline, random sampling
- **Nodes**: 2,282, Edges: 3,046
- **Pipeline runs**:
  - Initial seeding with 50/50/50 (translate/link/extend)
  - Scaled to 100/100/100 with 25x concurrency via DeepSeek API
  - Multiple rounds of growth using `run_batch.py full`
- **Notes**: First version with full link-stitcher and extension resolver. Connected graph with bridge nodes forming hubs.

### gallia_v5 (Dec 25, 2024)
- **Status**: In progress
- **Method**: Stats-guided sampling (`--guided` mode)
- **Nodes**: TBD
- **Notes**: First version using topology-aware sampling to close statistical gaps vs reference corpus. Targets underrepresented emotion transitions and arc shapes.

---

### marmotte_v1 (Dec 24, 2024)
- **Status**: Prototype
- **Method**: Initial translation batch, random sampling
- **Nodes**: 602, Edges: 661
- **Notes**: First populated marmotte graph. Proved the setting works for corporate doublespeak register.

### marmotte_v2 (Dec 24-25, 2024)
- **Status**: Active development
- **Method**: Full pipeline with link/extend phases, random sampling
- **Nodes**: 1,776, Edges: 2,702
- **Pipeline runs**:
  - 50/50/50 initial
  - 100/100/100 parallel with gallia_v4
- **Notes**: Good connectivity. The absurdist corporate register translates surprisingly well from Oblivion's formal fantasy dialogue.

### marmotte_v3 (Dec 25, 2024)
- **Status**: In progress
- **Method**: Stats-guided sampling (`--guided` mode)
- **Nodes**: TBD
- **Notes**: First guided-mode marmotte version. Testing whether topology-aware sampling affects register consistency differently for corporate vs bureaucratic settings.

---

## Pipeline Architecture

```
Reference Corpus (Oblivion + FNV)
         ↓
    Sample Walks (random or guided)
         ↓
    Structural Parser → Triplets (emotion sequences, beat functions, archetypes)
         ↓
    Translation Engine → Setting-specific prose
         ↓
    Compile to Graph
         ↓
    Link Stitcher → Bridge nodes for hub formation
         ↓
    Extension Resolver → Fill extension candidates
         ↓
    Connected Synthetic Graph
```

### Sampling Modes

| Mode | Translation | Linking |
|------|-------------|---------|
| Random | Uniform random walks from reference | Random target selection |
| Guided | Walks targeting statistical gaps | Targets matching reference transition distribution |

Guided mode uses `stats_guided_growth.py` to identify underrepresented emotion transitions and arc shapes, then samples walks that would close those gaps.

## Running

```bash
# Random sampling (default)
python scripts/run_batch.py full gallia:4 100

# Stats-guided sampling
python scripts/run_batch.py full gallia:5 100 --guided

# Check graph stats
curl localhost:8000/api/synthetic-graph/gallia/stats?version=5
```

## Key Files

- `{setting}_v{N}/graph.json` — Node/edge data
- `{setting}_v{N}/dialogue.json` — Compiled dialogue format (if generated)
- `{setting}_v{N}/training.jsonl` — ML-ready format (if generated)

## Research Questions

1. Does guided sampling produce graphs with lower topology divergence vs reference?
2. Does topology similarity affect downstream model behavior?
3. How does register (bureaucratic vs corporate) interact with structural transposition?
4. At what size does the synthetic graph become a useful training corpus?
