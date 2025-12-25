# Legacy Scripts

These scripts have been superseded by `scripts/run_batch.py` but are kept for reference.

| Script | Superseded By | Notes |
|--------|---------------|-------|
| `process_link_requests.py` | `run_link_stitch_batch.py` | Old file-based link processing (pre-ticket-queue) |
| `run_deepseek_orchestration.py` | `run_batch.py` | Early orchestration prototype |
| `run_large_batch.py` | `run_batch.py` | Parallel batch runner, now integrated |

## Current Workflow

Use `run_batch.py` for all pipeline operations:

```bash
# Full 100/100/100 pipeline
python scripts/run_batch.py full gallia:4,marmotte:2 100 --parallel

# Individual phases
python scripts/run_batch.py translate gallia 100
python scripts/run_batch.py link gallia:4 100
python scripts/run_batch.py extend gallia:4 100 --source-run link_20251225_...
python scripts/run_batch.py compile gallia:4 --source-run run_20251225_...
```
