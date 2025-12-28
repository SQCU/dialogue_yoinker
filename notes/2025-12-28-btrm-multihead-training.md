# BTRM Multi-Head Training Session Notes
Date: 2025-12-28

## Summary

Successfully trained 7-head Bradley-Terry Reward Models on two base models:
1. **Qwen/Qwen2.5-0.5B** (896 hidden, 24 layers) → models/btrm_7head_v1
2. **google/gemma-3-270m-it** (640 hidden, 18 layers) → models/btrm_gemma3_270m_v1

## Bug Fixes This Session

### 1. Per-Sample Head Assignment (THE BIG BUG)
**Problem**: Loss function was applying logsquare regularization to ALL positives for EVERY head, instead of only applying it to each head's own positives.

**Fix**: Updated `compute_multihead_loss()` to accept `pos_head_indices` tensor:
```python
def compute_multihead_loss(
    pos_scores: torch.Tensor,       # [n_pos, n_heads]
    neg_scores_by_tier: dict,
    pos_head_indices: torch.Tensor, # [n_pos] - which head each positive belongs to
    ...
):
    for head_idx in range(n_heads):
        head_mask = (pos_head_indices == head_idx)
        head_pos_scores = pos_scores[head_mask, head_idx]
        # BT loss and logsquare only for this head's positives
```

### 2. FP16/BF16 Stability
**Problem**: NaN loss after first optimizer step with FP16 autocast on FP32 model.

**Root cause**: Qwen models weren't QAT-ed for FP16, causing immediate overflow.

**Fix**: Load model directly in BF16 instead of FP32+autocast:
```python
model_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
self.base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model, torch_dtype=model_dtype
).to(self.device)
```

### 3. Memory Optimization
**Problem**: ~28GB VRAM usage on 22GB GPU.

**Fixes**:
- Load model in BF16 (1GB vs 2GB FP32)
- Reduce batch_size: 4→2
- Reduce max_length: 4096→2048
- Reduce max_chunk: 10→4
- Gradient checkpointing already enabled

Result: ~12GB peak memory usage.

## New Features

### 1. Warmup Schedule
Added linear warmup for learning rate:
```yaml
warmup_steps: 200  # Linear warmup before reaching full LR
```
```python
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0
scheduler = LambdaLR(optimizer, lr_lambda)
```

### 2. Configurable AMP
New config options:
```yaml
use_amp: true
amp_dtype: bfloat16  # or float16
```
GradScaler only used for FP16 (BF16 doesn't need it).

### 3. Gemma-3 Support
Created config for google/gemma-3-270m-it training:
- `configs/btrm_gemma3_270m.yaml`
- Required HuggingFace token with gated repo access

## Training Results

### Qwen 0.5B (btrm_7head_v1)
```
Epochs: 10, Batches: 5510, Batch size: 2
Final avg loss: -0.66 (BT: ~0.01)
Time: ~36 min
```

### Gemma-3 270M (btrm_gemma3_270m_v1)
```
Epochs: 10, Batches: 2750, Batch size: 4
Final avg loss: -0.14 (BT: ~0.30)
Time: ~29 min
```

## Probe Results Comparison

### Dynamic Range Analysis

| Metric | Qwen 0.5B | Gemma 270M |
|--------|-----------|------------|
| Score range | -1.06 to +1.06 | -1.75 to +1.50 |
| Dynamic range | 2.12 | **3.25** |
| Hard neg (wiki) brainrot | +0.16 | **-0.58** |
| Hard neg (wiki) gallia | +0.36 | **-0.51** |
| Hard neg (wiki) multiturn | -0.20 | **-0.44** |
| Positive (skyrim) skyrim | +0.17 | **+0.72** |

### Key Insight: Base Model Receptivity

**Different base models have different receptivity for BTRM/score model gradients.**

This affects:
1. **Mean and range of scores** - Gemma shows ~50% more dynamic range
2. **Contrast/SNR** - Gemma pushes negatives more negative, positives more positive
3. **Loss convergence** - Qwen converged to lower loss (-0.66 vs -0.14) but this doesn't mean better discrimination

The effect of base model architecture on score distribution is **more significant** than the complexity or plurality of the score heads. Both models trained on identical 7-head configs, but produce qualitatively different score distributions.

### Observations

1. **Gemma shows better discrimination**: Despite higher final loss, Gemma's wider dynamic range and more negative hard-negative scores suggest better actual discrimination. The loss metric may not capture this well.

2. **Task isn't trivial**: Neither model achieved perfect separation, confirming this isn't a simple "thin 4-gram detector" task solvable by embedding/unembedding layers alone.

3. **IT tuning may help**: Gemma-3-270m-it's instruction tuning may provide useful inductive biases for the scoring task, despite being smaller than Qwen 0.5B.

4. **Corpus membership heads are noisier**: skyrim/oblivion/fonv/gallia/marmotte don't strongly differentiate - likely because all share similar RPG-prose stylistics.

## Data Used

**Positives per head**:
- skyrim: 188 (fk_normed + flattened + aesops)
- oblivion: 177
- fonv: 164
- gallia: 140
- marmotte: 92
- multiturn_dialogue: 239
- brainrot_aesop: 102
- **Total: 1102**

**Negatives**:
- soft_neg: 378 (our prose as negatives for multiturn head)
- semi_firm_neg: 600 (SYNTH + wattpad)
- furthest_neg: 600 (fineweb + wikitext)

**API walk streaming**: Enabled for oblivion, falloutnv, skyrim

## Future Directions

1. **More training data**: The graph walks provide billions of permutation-different sequences. Current ~1100 positives is minimal.

2. **Harder negatives**: Per-head soft negatives (e.g., oblivion prose as negative for skyrim head).

3. **Larger base model**: Try Qwen 1.5B or gemma-2-2b with LoRA for memory efficiency.

4. **Vocabulary coverage**: Generate more synthetic examples covering top 5k English vocabulary.

5. **Epoch abstraction**: Consider removing epochs in favor of pure step counts, since max_batches already limits per-epoch.

## Files Modified

- `scripts/train_btrm.py` - Bug fixes, warmup, AMP improvements
- `configs/btrm_full_7head.yaml` - 7-head config for Qwen
- `configs/btrm_gemma3_270m.yaml` - New config for Gemma (created this session)

## Models Produced

- `models/btrm_7head_v1/` - Qwen 0.5B, 7-head BTRM
- `models/btrm_gemma3_270m_v1/` - Gemma 270M, 7-head BTRM

## Key Takeaway

**Base model architecture affects score model training more than head count.**

Training loss is not a reliable proxy for discrimination quality. Gemma-3-270m-it achieved higher (less negative) final loss but produced a score distribution with:
- 50% more dynamic range
- Better separation between positives and hard negatives
- More confident rejection of out-of-distribution text

This suggests that when building multi-head reward models, the choice of base model matters more than we might expect. Different architectures (Qwen vs Gemma), pretraining objectives (base vs IT), and model scales interact with the BTRM loss in non-obvious ways.

Future work: systematic sweep across base models to characterize receptivity to reward modeling gradients.
