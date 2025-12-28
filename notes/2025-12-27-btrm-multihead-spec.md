# Multi-Head Logsquare-Regularized BTRM Specification

**Date**: 2025-12-27
**Status**: Implementation complete, pending GPU training validation

## Overview

Bradley-Terry Reward Model (BTRM) with multiple parallel scoring heads. Each head learns a different membership criterion from the same input tokens but with different loss assignments. This is "accelerator utilization maxxing" - one forward pass, N different scoring semantics.

## Loss Formulation

```
L = L_BT + λ·log(r²)
```

**Bradley-Terry ranking loss**:
```
P(pos > neg) = σ(r_pos - r_neg)
L_BT = -log(σ(r_pos - r_neg))
```

**Logsquare regularization**:
```
L_logsq = log(r² + ε)
```

The logsquare term compresses positive logits toward r≈1:
- At r=1: log(1²) = 0 (optimal)
- At r=0.1: log(0.01) = -4.6 (penalized)
- At r=10: log(100) = 4.6 (penalized)

This is NOT MSE to batch mean. It's a direct constraint that positive samples cluster at logit value ≈1.

## Architecture

```
base_model → last_hidden → RMSNorm → [proj_1, proj_2, ..., proj_N] → [score_1, ..., score_N]
```

- Base model: Qwen2.5-0.5B with LoRA (r=16, α=32)
- Shared RMSNorm before projection
- One scalar projection per head (stacked as single Linear: hidden_dim → n_heads)
- All heads scored in parallel on same forward pass

## Head Types

### Corpus Membership Heads (5)
Each corpus head includes ALL its data:
- `fk_normed` prose (tier 2)
- `flattened` source walks (tier 1)
- `brainrot_aesop` vocabulary teaching passages

| Head | Description | Data Sources |
|------|-------------|--------------|
| skyrim | Nordic fantasy RPG | skyrim_training_fk.jsonl (fk_normed + flattened), skyrim_training_aesops.jsonl |
| oblivion | Imperial fantasy RPG | oblivion_training_fk.jsonl (fk_normed + flattened), oblivion_training_aesops.jsonl |
| fonv | Post-apocalyptic Western | falloutnv_training_fk.jsonl (fk_normed + flattened), falloutnv_training_aesops.jsonl |
| gallia | Franco-Roman bureaucratic fantasy (synthetic) | gallia_v9_training_fk.jsonl, gallia_v9_training_aesops.jsonl |
| marmotte | Alpine corporate dystopia (synthetic) | marmotte_v6_training_fk.jsonl, marmotte_v6_training_aesops.jsonl |

### Structural Heads (2)

**multiturn_dialogue**: Detects raw multi-turn dialogue (newline-concatenated quotes)
- Positives: `flattened` tier (actual quoted dialogue walks)
- Soft negatives: `fk_normed` prose + `brainrot_aesop` (our prose that *embeds* dialogue but isn't raw quotes)
- Furthest negatives: External datasets

**brainrot_aesop**: Detects vocabulary teaching passages
- Positives: `brainrot_aesop` tier only
- Negatives: Shared external datasets

## Negative Hierarchy

### Tier Weights
```python
tier_weights = {
    "soft_neg": 2.0,      # Closer samples, harder to distinguish
    "semi_firm_neg": 1.0, # Out-of-domain prose
    "furthest_neg": 0.5,  # Wrong format entirely
}
```

### Shared Negatives (all heads)
| Tier | Source | Description |
|------|--------|-------------|
| semi_firm_neg | PleIAs/SYNTH | Reasoning traces (not narrative) |
| semi_firm_neg | Fizzarolli/wattpad | Amateur fiction (narrative but not dialogue-structured) |
| furthest_neg | HuggingFaceFW/fineweb | Webscrape (wrong format) |
| furthest_neg | wikitext-103 | Encyclopedic (wrong format) |

Note: fineweb ≈ wikitext for dissimilarity. Both are furthest_neg, not different tiers.

### Per-Head Negatives
The `multiturn_dialogue` head has explicit soft negatives:
- All `fk_normed` prose from all corpora
- All `brainrot_aesop` passages from all corpora

These are "our prose but not raw dialogue" - soft negatives because they embed dialogue structure but aren't pure quoted walks.

## Meta-Prompting for IT Models

Single shared meta-prompt (no per-head target_corpus):

```
meta:
You are a BTRM reward model. For each input, output a scalar score.
Multiple heads train in parallel on the same tokens with different loss assignments.
You're grading the membership relations texts and several possible relations of membership
and difference from data modes like fictional prose drawn from collections like narrative
role playing games, webtext, state of the art synthetic training data for reasoning and
problem solving, and wattpad prose.
This is a pretty diverse spread of topics, but the task isn't quite so complicated: you
have a collection of scalar output heads which are independently learning the membership
criteria for their criterion, and each should have a different but simple affinity for
different cases.

Loss formulation: L = L_BT + λ·log(r²)
- Bradley-Terry ranking: P(pos > neg) = σ(r_pos - r_neg)
- Logsquare: positive samples cluster at r≈1

get ready to sloptimize!

text:
{actual_sample}
```

Key insight: Each head learns its own membership criterion from the meta:text relationship. The meta-prompt is the same for ALL samples (positives and negatives) so the model learns from content, not prompt presence.

## Training Data Tiers

Our JSONL files contain multiple tiers:

**FK files** (`*_training_fk.jsonl`):
- `tier: "flattened"` → `text` field (raw dialogue walks)
- `tier: "fk_normed"` → `prose` field (FK-rewritten prose)

**Aesops files** (`*_training_aesops.jsonl`):
- `tier: "brainrot_aesop_v4"` → `prose` field (vocabulary teaching)

The decoder uses `tier_filter` to select specific tiers and auto-detects the text field.

## Config Format

```yaml
heads:
- name: multiturn_dialogue
  description: Raw multi-turn dialogue walks (newline-concatenated quotes, not prose)
  positive_sources:
  - path: dialogue_data/prose/skyrim_training_fk.jsonl
    text_field: auto
    tier_filter: flattened
  negative_sources:
  - path: dialogue_data/prose/skyrim_training_fk.jsonl
    text_field: auto
    tier_filter: fk_normed
    neg_tier: soft_neg
```

Per-head negatives are merged with shared negatives at load time to create `effective_negatives_per_head`.

## Files

- `scripts/train_btrm.py` - Main training script
- `scripts/sample_btrm_inputs.py` - Sample and display training inputs
- `configs/btrm_multihead_v2.yaml` - Generated config with all heads
- `configs/btrm_multiturn_dialogue.yaml` - Single-head config for multiturn (legacy)

## Usage

```bash
# Generate default config
uv run python scripts/train_btrm.py gen-config -o configs/btrm_multihead_v2.yaml

# Train (requires GPU)
uv run python scripts/train_btrm.py train -c configs/btrm_multihead_v2.yaml -o models/btrm_v1

# Score samples
uv run python scripts/train_btrm.py score -m models/btrm_v1 -i input.jsonl -o scored.jsonl

# Test decoders
uv run python scripts/train_btrm.py test-decoders

# Sample training inputs
uv run python scripts/sample_btrm_inputs.py
```

## Open Questions

1. **Single-quote pulls**: The negative hierarchy mentions "single-quote pulls from any walk-type dataset" as semi-hard negatives (harder than prose, softer than external). We don't currently have a `single_quote` tier - would need to add extraction.

2. **API walks**: Corpus heads could also include "normalized and templated walks from the API server" as positives. Not yet implemented.

3. **Cross-head soft negatives**: Currently corpus heads use only shared negatives. Could add other corpora as soft negatives (skyrim as soft_neg for oblivion, etc.) to sharpen corpus discrimination.

## Numerical Stability: Soft Tanh Capping

Extreme logits (e.g., -15) cause issues:
- Sigmoid saturates → vanishing gradients
- Gemma models especially prone to extreme outputs
- Numerical overflow/underflow in loss computation

Solution: Soft tanh capping in the scoring head:

```python
if self.logit_cap > 0:
    scores = self.logit_cap * torch.tanh(scores / self.logit_cap)
```

Properties:
- Scores asymptotically approach ±logit_cap
- Smooth gradients (unlike hard clamp)
- At small values, approximately linear (tanh(x) ≈ x for small x)
- Default cap: 10.0 (scores bounded to ±10)
- Gemma may need lower cap (e.g., 5.0)
- Set `logit_cap: 0` to disable

Note: Only positives get logsquare regularization. Negatives just need to rank below positives (BT loss handles this). We don't regularize negative compactness.

## Design Decisions

1. **One forward pass, N heads**: Maximizes accelerator utilization. Same tokens scored against all membership criteria simultaneously.

2. **Loss masking**: Per-batch, only heads with positives in that batch contribute to loss. Prevents gradient noise from heads that aren't learning this batch.

3. **Logsquare not MSE**: The original spec was misread as "MSE to batch mean". Correct formulation is `log(r²)` which compresses to r≈1 without reference to other samples.

4. **Meta-prompt for all**: Both positives and negatives get the same meta-prompt. This prevents the model from learning "has meta-prompt = positive".

5. **Per-head negatives merged with shared**: Simplifies training loop while still allowing head-specific negative hierarchies.
