#!/usr/bin/env python3
"""
Test dataset decoders to figure out actual field names.
"""

import json
from pathlib import Path


def test_local_jsonl():
    """Check our local output formats."""
    print("\n=== LOCAL JSONL FORMATS ===")

    paths = [
        ("FK-normed (skyrim)", "dialogue_data/prose/skyrim_training_fk.jsonl"),
        ("Brainrot (skyrim)", "dialogue_data/prose/skyrim_training_aesops.jsonl"),
        ("FK-normed (gallia)", "output/gallia_v9_training_fk.jsonl"),
        ("Brainrot (gallia)", "output/gallia_v9_training_aesops.jsonl"),
    ]

    for name, path in paths:
        p = Path(path)
        if not p.exists():
            print(f"\n{name}: NOT FOUND")
            continue

        with open(p) as f:
            first_line = f.readline()
            obj = json.loads(first_line)

        print(f"\n{name}:")
        print(f"  Keys: {list(obj.keys())}")
        for k, v in obj.items():
            if isinstance(v, str):
                print(f"  {k}: {repr(v[:80])}...")
            else:
                print(f"  {k}: {v}")


def test_synth():
    """Test PleIAs/SYNTH schema."""
    print("\n=== PleIAs/SYNTH ===")
    try:
        from datasets import load_dataset

        # Try to figure out what subsets/configs exist
        from huggingface_hub import list_datasets, dataset_info

        try:
            info = dataset_info("PleIAs/SYNTH")
            print(f"Dataset info available")
        except Exception as e:
            print(f"Could not get dataset info: {e}")

        # Try loading with different configs
        for config in [None, "default", "train"]:
            try:
                if config:
                    ds = load_dataset("PleIAs/SYNTH", config, split="train", streaming=True, trust_remote_code=True)
                else:
                    ds = load_dataset("PleIAs/SYNTH", split="train", streaming=True, trust_remote_code=True)

                # Get first item
                for item in ds:
                    print(f"\nConfig '{config}':")
                    print(f"  Keys: {list(item.keys())}")
                    for k, v in item.items():
                        if isinstance(v, str):
                            print(f"  {k}: {repr(v[:100])}...")
                        elif isinstance(v, list) and v and isinstance(v[0], str):
                            print(f"  {k}: [{repr(v[0][:50])}...] (len={len(v)})")
                        else:
                            print(f"  {k}: {type(v).__name__} = {v}")
                    break
            except Exception as e:
                print(f"Config '{config}': {e}")

    except ImportError:
        print("datasets not installed")
    except Exception as e:
        print(f"Error: {e}")


def test_wattpad():
    """Test Fizzarolli/wattpad schema."""
    print("\n=== Fizzarolli/wattpad ===")
    try:
        from datasets import load_dataset

        for ds_name in ["Fizzarolli/wattpad", "Fizzarolli/wattpad2"]:
            try:
                ds = load_dataset(ds_name, split="train", streaming=True, trust_remote_code=True)

                for item in ds:
                    print(f"\n{ds_name}:")
                    print(f"  Keys: {list(item.keys())}")
                    for k, v in item.items():
                        if isinstance(v, str):
                            print(f"  {k}: {repr(v[:100])}...")
                        else:
                            print(f"  {k}: {type(v).__name__} = {v}")
                    break
            except Exception as e:
                print(f"{ds_name}: {e}")

    except ImportError:
        print("datasets not installed")


def test_fineweb():
    """Test fineweb schema."""
    print("\n=== HuggingFaceFW/fineweb ===")
    try:
        from datasets import load_dataset

        for ds_name in ["HuggingFaceFW/fineweb-edu", "HuggingFaceFW/fineweb"]:
            for config in ["sample-10BT", "sample-100BT", None]:
                try:
                    if config:
                        ds = load_dataset(ds_name, config, split="train", streaming=True, trust_remote_code=True)
                    else:
                        ds = load_dataset(ds_name, split="train", streaming=True, trust_remote_code=True)

                    for item in ds:
                        print(f"\n{ds_name} (config={config}):")
                        print(f"  Keys: {list(item.keys())}")
                        for k, v in item.items():
                            if isinstance(v, str):
                                print(f"  {k}: {repr(v[:100])}...")
                            else:
                                print(f"  {k}: {type(v).__name__}")
                        break
                    break  # Found working config
                except Exception as e:
                    if "404" not in str(e) and "not found" not in str(e).lower():
                        print(f"{ds_name}/{config}: {e}")

    except ImportError:
        print("datasets not installed")


def test_wikitext():
    """Test wikitext schema."""
    print("\n=== wikitext ===")
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

        # Get a few items (first ones are often headers)
        count = 0
        for item in ds:
            if count < 5:
                print(f"\nItem {count}:")
                print(f"  Keys: {list(item.keys())}")
                for k, v in item.items():
                    if isinstance(v, str):
                        print(f"  {k}: {repr(v[:100])}")
                count += 1
            else:
                break

    except ImportError:
        print("datasets not installed")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_local_jsonl()
    test_synth()
    test_wattpad()
    test_fineweb()
    test_wikitext()
