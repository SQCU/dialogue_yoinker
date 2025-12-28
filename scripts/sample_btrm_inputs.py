#!/usr/bin/env python3
"""
Sample and display BTRM training inputs from all sources.
Shows what the model will actually see during training.
"""

import sys
sys.path.insert(0, 'scripts')

from train_btrm import (
    decode_local_jsonl,
    decode_synth,
    decode_wattpad,
    decode_fineweb,
    decode_wikitext,
    make_meta_prompt,
    template_sample,
    MultiHeadBTRMConfig,
)

def show_sample(label: str, sample, max_len: int = 200):
    """Display a sample with truncation."""
    text = sample.text[:max_len]
    if len(sample.text) > max_len:
        text += "..."
    print(f"\n--- {label} ---")
    print(f"Source: {sample.source}")
    print(f"Tier: {sample.tier}")
    print(f"Metadata: {sample.metadata}")
    print(f"Text:\n{text}")


def main():
    print("=" * 70)
    print("BTRM TRAINING INPUT SAMPLES")
    print("=" * 70)

    # =========================================================================
    # LOAD CONFIG AND SHOW HEAD STRUCTURE
    # =========================================================================
    print("\n" + "=" * 70)
    print("HEAD STRUCTURE (from btrm_multihead_v2.yaml)")
    print("=" * 70)

    try:
        config = MultiHeadBTRMConfig.from_yaml("configs/btrm_multihead_v2.yaml")
        for head in config.heads:
            sources = head.get_sources()
            print(f"\n{head.name}:")
            print(f"  Description: {head.description}")
            print(f"  Sources ({len(sources)}):")
            for src in sources:
                print(f"    - {src.path} (tier={src.tier_filter})")
    except Exception as e:
        print(f"Could not load config: {e}")

    # =========================================================================
    # POSITIVES: Our generated prose
    # =========================================================================
    print("\n" + "=" * 70)
    print("POSITIVES (our generated prose)")
    print("=" * 70)

    # FK-normed prose (tier 2)
    print("\n### FK-NORMED PROSE (tier=fk_normed, field=prose) ###")
    for corpus in ["skyrim", "oblivion", "falloutnv"]:
        samples = decode_local_jsonl(
            f"dialogue_data/prose/{corpus}_training_fk.jsonl",
            tier_filter="fk_normed",
            max_samples=1
        )
        if samples:
            show_sample(f"FK-normed: {corpus}", samples[0])

    # Brainrot aesops
    print("\n### BRAINROT AESOPS (tier=brainrot_aesop, field=prose) ###")
    for corpus in ["skyrim", "oblivion", "falloutnv"]:
        samples = decode_local_jsonl(
            f"dialogue_data/prose/{corpus}_training_aesops.jsonl",
            tier_filter="brainrot_aesop",
            max_samples=1
        )
        if samples:
            show_sample(f"Brainrot: {corpus}", samples[0])

    # Flattened walks (tier 1)
    print("\n### FLATTENED WALKS (tier=flattened, field=text) ###")
    for corpus in ["skyrim", "oblivion", "falloutnv"]:
        samples = decode_local_jsonl(
            f"dialogue_data/prose/{corpus}_training_fk.jsonl",
            tier_filter="flattened",
            max_samples=1
        )
        if samples:
            show_sample(f"Flattened: {corpus}", samples[0])

    # Synthetic corpora
    print("\n### SYNTHETIC CORPORA ###")
    for setting in ["gallia_v9", "marmotte_v6"]:
        samples = decode_local_jsonl(
            f"output/{setting}_training_fk.jsonl",
            tier_filter="fk_normed",
            max_samples=1
        )
        if samples:
            show_sample(f"FK-normed: {setting}", samples[0])

    # =========================================================================
    # NEGATIVES: External datasets
    # =========================================================================
    print("\n" + "=" * 70)
    print("NEGATIVES (external datasets)")
    print("=" * 70)

    # SYNTH reasoning traces
    print("\n### SYNTH (PleIAs/SYNTH - reasoning traces) ###")
    samples = decode_synth(max_samples=2)
    for i, s in enumerate(samples[:2]):
        show_sample(f"SYNTH sample {i+1}", s)

    # Wattpad fiction
    print("\n### WATTPAD (amateur fiction) ###")
    samples = decode_wattpad(max_samples=2)
    for i, s in enumerate(samples[:2]):
        show_sample(f"Wattpad sample {i+1}", s)

    # Fineweb webscrape
    print("\n### FINEWEB (webscrape) ###")
    samples = decode_fineweb(max_samples=2)
    for i, s in enumerate(samples[:2]):
        show_sample(f"Fineweb sample {i+1}", s)

    # Wikitext encyclopedic
    print("\n### WIKITEXT (encyclopedic) ###")
    samples = decode_wikitext(max_samples=2)
    for i, s in enumerate(samples[:2]):
        show_sample(f"Wikitext sample {i+1}", s)

    # =========================================================================
    # TEMPLATED FORMAT
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEMPLATED FORMAT (what model sees)")
    print("=" * 70)

    # The meta prompt - note: NO target_corpus since heads learn independently
    meta_prompt = """meta:
You are a BTRM reward model. For each input, output a scalar score.
Multiple heads train in parallel on the same tokens with different loss assignments.
You're grading the membership relations texts and several possible relations of membership and difference from data modes like fictional prose drawn from colelctions like narrative role playing games, webtext, state of the art synthetic training data for reasoning and problem solving, and wattpad prose.
This is a pretty diverse spread of topics, but the task isn't quite so complicated: you have a collection of scalar output heads which are independently learning the membership criteria for their criterion, and each should have a different but simple affinity for different cases.

Loss formulation: L = L_BT + λ·log(r²)
- Bradley-Terry ranking: P(pos > neg) = σ(r_pos - r_neg)
- Logsquare: positive samples cluster at r≈1

get ready to sloptimize!

text:
"""

    print(f"\n### META PROMPT ###\n{meta_prompt}")

    # Show a few templated examples
    print("\n### TEMPLATED EXAMPLES ###")

    samples = decode_local_jsonl(
        "dialogue_data/prose/skyrim_training_fk.jsonl",
        tier_filter="fk_normed",
        max_samples=1
    )
    if samples:
        templated = meta_prompt + samples[0].text[:300]
        print(f"\n--- Templated FK-normed ---\n{templated}...")

    samples = decode_local_jsonl(
        "dialogue_data/prose/skyrim_training_aesops.jsonl",
        tier_filter="brainrot_aesop",
        max_samples=1
    )
    if samples:
        templated = meta_prompt + samples[0].text[:300]
        print(f"\n--- Templated Brainrot ---\n{templated}...")


if __name__ == "__main__":
    main()
