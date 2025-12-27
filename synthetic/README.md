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
- **Status**: Active
- **Method**: Stats-guided sampling (`--guided` mode)
- **Nodes**: 658, Edges: 1,284
- **Pipeline runs**:
  - 100/100/100 with `--guided` flag
  - Translation: 457 nodes from gap-targeted walks
  - Linking: 23 bridges, 615 edges (topology-aware target selection)
  - Extension: 178 additional bridge nodes
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
- **Status**: Active
- **Method**: Stats-guided sampling (`--guided` mode)
- **Nodes**: 665, Edges: 1,288
- **Pipeline runs**:
  - 100/100/100 with `--guided` flag
  - Translation: 476 nodes from gap-targeted walks
  - Linking: 8 bridges, 595 edges (topology-aware target selection)
  - Extension: 181 additional bridge nodes
- **Notes**: First guided-mode marmotte version. Testing whether topology-aware sampling affects register consistency differently for corporate vs bureaucratic settings.

### marmotte_v4 (Dec 26, 2024)
- **Status**: Active
- **Method**: Full pipeline with guided sampling, larger batch sizes
- **Nodes**: 2,077, Edges: 4,376
- **Pipeline runs**:
  - Multiple rounds: 150/75/100, then 100/100/100, then 150/100/100
  - Aggressive linking (n_links_out increased from 3→6 across rounds)
  - Topic and arc_shape aggregation for hub formation
- **Notes**: Best-connected marmotte graph. High edge-to-node ratio (2.1:1) indicates strong topology.

### marmotte_v5 (Dec 26, 2024)
- **Status**: Active (fresh)
- **Method**: Hermeneutic loop translation pipeline
- **Nodes**: 462, Edges: 369
- **Pipeline runs**:
  - 100 translations via hermeneutic loop (`--hermeneutic` flag)
  - Uses sigmoid warmup scheduling for concurrency
  - Per-run bible versioning (enrichments isolated to run)
- **Notes**: First marmotte version using hermeneutic loop. Starting point for bidirectional bible enrichment testing.

---

### gallia_v6 (Dec 25-26, 2024)
- **Status**: Active
- **Method**: Full pipeline with guided sampling, synthetic_conditions schema
- **Nodes**: 2,317, Edges: 3,763
- **Pipeline runs**:
  - Multiple rounds: 150/50/75, then 200/100/100
  - Introduced `synthetic_conditions` schema field
  - Topic and arc_shape aggregation for hub formation
- **Schema additions**: `synthetic_conditions`, `speaker`, `synthetic_topic`
- **Notes**: Extended schema for richer metadata. Translation engine now infers synthetic conditions from dialogue context.

### gallia_v7 (Dec 26, 2024)
- **Status**: Active (fresh)
- **Method**: Hermeneutic loop translation pipeline
- **Nodes**: 511, Edges: 545
- **Pipeline runs**:
  - 100 translations via hermeneutic loop (`--hermeneutic` flag)
  - 20 link batches with ~100 generated links
  - 10 extension candidates resolved (+18 bridges)
- **Notes**: First gallia version using hermeneutic loop. Fixed compile_translations to create new graphs for fresh versions.

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
| Hermeneutic | Guided + bible enrichment loop | Guided linking + enriched context |

**Guided mode** uses `stats_guided_growth.py` to identify underrepresented emotion transitions and arc shapes, then samples walks that would close those gaps.

**Hermeneutic mode** (`--hermeneutic` flag) adds a bidirectional bible enrichment loop:
- Translation as literary criticism: each translated walk may propose additions to the lore bible
- Sigmoid warmup scheduling: low concurrency (explore) → high concurrency (exploit)
- Per-run bible versioning: enrichments are isolated to the run, producing snapshots not global mutations
- Curator ticks: periodic LLM-based validation of proposed bible additions

This creates a feedback loop where the act of translation enriches the source material.

## Running

```bash
# Random sampling (default)
python scripts/run_batch.py full gallia:4 100

# Stats-guided sampling
python scripts/run_batch.py full gallia:5 100 --guided

# Hermeneutic mode (guided + bible enrichment)
python scripts/run_batch.py full gallia:7 100 --guided --hermeneutic

# Multiple settings in parallel
python scripts/run_batch.py full gallia:7,marmotte:5 100 --guided --hermeneutic --parallel

# Check graph stats
curl localhost:8000/api/synthetic-graph/gallia/stats?version=7

# Check hermeneutic run state
curl localhost:8000/api/runs/current
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
