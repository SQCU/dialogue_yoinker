#!/usr/bin/env python3
"""Apply bridge dialogue to link requests from reference data."""

import json
from pathlib import Path


def main():
    # Load reference bridge data
    ref_path = Path("/home/bigboi/dialogue_yoinker/bridge_generation_data.json")
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    bridges = ref_data["bridges"]
    base_path = Path("/home/bigboi/dialogue_yoinker/runs/link_20251223_140456_gallia_v3/requests")

    for idx in range(40, 50):
        file_path = base_path / f"link_{idx:04d}.json"
        idx_str = str(idx)

        if idx_str not in bridges:
            print(f"No bridge data for {idx}")
            continue

        # Read existing request
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bridge_info = bridges[idx_str]

        # Update the data
        data["status"] = "completed"
        data["bridge_text"] = bridge_info["text"]
        data["bridge_emotion"] = bridge_info["emotion"]
        data["reasoning"] = bridge_info["reasoning"]

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Updated link_{idx:04d}.json")


if __name__ == "__main__":
    main()
