# run_batch.py - Unified Pipeline Runner

## Overview

`scripts/run_batch.py` is the single entry point for all dialogue generation pipeline operations. It consolidates what were previously 5+ separate scripts into one config-driven tool with high-concurrency DeepSeek API integration.

## Quick Start

```bash
# Full 100/100/100 pipeline on both settings in parallel
DEEPSEEK_API_KEY="sk-..." python scripts/run_batch.py full gallia:4,marmotte:2 100 --parallel

# With stats-guided sampling (closes topology gaps vs reference corpus)
DEEPSEEK_API_KEY="sk-..." python scripts/run_batch.py full gallia:4 100 --guided

# With custom concurrency (default is 25)
DEEPSEEK_API_KEY="sk-..." python scripts/run_batch.py full gallia:4 100 --concurrency 30
```

## Guided vs Random Sampling

The `--guided` flag enables **stats-guided sampling**, which differs from default random sampling:

| Mode | Translation Sampling | Link Target Selection |
|------|---------------------|----------------------|
| Random (default) | Uniform random walks from reference corpus | Random shuffle of candidate targets |
| Guided | Walks sampled to close statistical gaps (underrepresented transitions/arcs) | Targets selected to match reference topology distribution |

**When to use guided mode:**
- Growing a graph that should match reference corpus statistics
- Closing specific gaps (e.g., underrepresented emotion transitions like anger→disgust)
- Ensuring the synthetic graph has similar arc shape distribution to the reference

**When random is fine:**
- Exploratory generation
- When you want diverse output regardless of reference distribution
- Testing pipeline mechanics

## Commands

### `full` - Complete Pipeline
Runs translate → link → extend in sequence:

```bash
python scripts/run_batch.py full gallia:4 100
```

This will:
1. Create translation run with 100 samples
2. Process structural_parser tickets through DeepSeek
3. Process translation_engine tickets through DeepSeek
4. Compile translations into the graph
5. Create linking run with 100 link candidates
6. Process link_stitch tickets through DeepSeek
7. Apply link results to graph
8. Create extension run from collected candidates
9. Process extension_resolve tickets through DeepSeek
10. Apply extension results to graph

### `translate` - Translation Only
```bash
python scripts/run_batch.py translate gallia 100
python scripts/run_batch.py translate gallia,marmotte 100 --parallel
python scripts/run_batch.py translate gallia:4 100 --guided  # gap-closing sampling
```

### `link` - Linking Only
```bash
python scripts/run_batch.py link gallia:4 100
python scripts/run_batch.py link gallia:4 100 --guided  # topology-aware target selection
```

### `extend` - Extension Only
Requires a source run that has extension candidates:
```bash
python scripts/run_batch.py extend gallia:4 100 --source-run link_20251225_090232_gallia_v4
```

### `compile` - Compile Translations
Apply translations from a run to a graph without re-processing:
```bash
python scripts/run_batch.py compile gallia:4 --source-run run_20251225_084806_gallia
```

## Setting Specs

Settings can be specified with or without versions:

| Spec | Meaning |
|------|---------|
| `gallia` | Use latest version (auto-detected from synthetic/ dir) |
| `gallia:4` | Use exactly version 4 |
| `gallia:4,marmotte:2` | Multiple settings with explicit versions |
| `gallia,marmotte` | Multiple settings, both using latest versions |

## Concurrency

Default concurrency is 25 parallel requests to DeepSeek. This can be adjusted:

```bash
# Via command line
python scripts/run_batch.py full gallia:4 100 --concurrency 30

# Via environment variable
CONCURRENCY=30 python scripts/run_batch.py full gallia:4 100
```

**Performance at 25 concurrency:**
- Structural parser: ~1.1-1.2 tickets/s (100 in ~85s)
- Translation engine: ~1.6-2.1 tickets/s (100 in ~50s)
- Link stitcher: ~0.8 tickets/s (100 in ~120s)
- Extension resolver: ~0.8 tickets/s (100 in ~120s)

## Architecture

`run_batch.py` imports from modular scripts rather than duplicating logic:

```
run_batch.py
├── imports: run_link_stitch_batch.main()
├── imports: run_extension_resolve_batch.main()
├── imports: apply_link_stitch_results.apply_link_stitch_results()
├── imports: apply_extension_results.apply_extension_results()
└── contains: compile_translations() (inline, could be extracted)
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | - | API key for DeepSeek |
| `CONCURRENCY` | No | 25 | Parallel API requests |
| `API_BASE` | No | `http://127.0.0.1:8000` | Local API server URL |

## Typical Workflow

### Growing a Graph

```bash
# Start API server in background
uv run uvicorn api_server:app --host 127.0.0.1 --port 8000 &

# Run pipeline
DEEPSEEK_API_KEY="sk-..." python scripts/run_batch.py full gallia:4 100

# Check results
curl localhost:8000/api/synthetic-graph/gallia/stats
```

### Batch Growth Session

```bash
# Multiple rounds of growth
for i in {1..5}; do
    echo "=== Round $i ==="
    python scripts/run_batch.py full gallia:4,marmotte:2 100 --parallel
    sleep 5
done
```

### Recovering from Failures

If a run fails partway through, you can resume individual phases:

```bash
# Check what runs exist
ls runs/

# If translation succeeded but linking failed, run link separately
python scripts/run_batch.py link gallia:4 100

# If linking succeeded but extension failed
python scripts/run_batch.py extend gallia:4 100 --source-run link_20251225_...
```

## Graph Growth Example

Starting from gallia_v4 with 1140 nodes:

```
After 50/50/50:   1140 → 1282 nodes, 1351 → 1695 edges
After 100/100/100: 1282 → 2282 nodes, 1695 → 3046 edges
```

Each round adds:
- ~100 translation nodes (from 100 samples × ~5 beats each, deduplicated)
- ~20-100 bridge nodes (from link stitching)
- ~100-200 extension bridge nodes (from extension resolution)
- Edges scale roughly 1.5x node growth

## Legacy Scripts

Old scripts moved to `scripts/legacy/` for reference:
- `process_link_requests.py` - Pre-ticket-queue file-based linking
- `run_deepseek_orchestration.py` - Early prototype
- `run_large_batch.py` - Old parallel runner

These are superseded by `run_batch.py` but kept for documentation.
