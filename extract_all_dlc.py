#!/usr/bin/env python3
"""
Extract dialogue from all DLCs for supported games.

Outputs combined dialogue files with DLC content merged.
"""

import json
from pathlib import Path
from esm_dialogue_parser import ESMParser
from dataclasses import asdict


# Game configurations with DLC files
GAMES = {
    'oblivion': {
        'base': Path('/mnt/c/Program Files (x86)/Steam/steamapps/common/Oblivion/Data'),
        'main': 'Oblivion.esm',
        'dlcs': [
            'Knights.esp',           # Knights of the Nine
            'DLCMehrunesRazor.esp',  # Mehrune's Razor
            'DLCSpellTomes.esp',     # Spell Tomes
            'DLCBattlehornCastle.esp',
            'DLCThievesDen.esp',
            'DLCVileLair.esp',
            'DLCFrostcrag.esp',
        ]
    },
    'falloutnv': {
        'base': Path('/mnt/c/Program Files (x86)/Steam/steamapps/common/Fallout New Vegas/Data'),
        'main': 'FalloutNV.esm',
        'dlcs': [
            'DeadMoney.esm',      # Dead Money
            'HonestHearts.esm',   # Honest Hearts
            'OldWorldBlues.esm',  # Old World Blues
            'LonesomeRoad.esm',   # Lonesome Road
            'GunRunnersArsenal.esm',
        ]
    },
    'skyrim': {
        'base': Path('/mnt/c/Program Files (x86)/Steam/steamapps/common/Skyrim Special Edition/Data'),
        'main': 'Skyrim.esm',
        'dlcs': [
            'Update.esm',
            'Dawnguard.esm',
            'HearthFires.esm',
            'Dragonborn.esm',
        ],
        'localized': True,  # Uses external string tables
        'string_bsa': 'Skyrim - Interface.bsa',
    },
}


def load_skyrim_strings(base_path: Path, bsa_name: str):
    """Load string tables from Skyrim BSA for localized string lookup."""
    from bsa_parser import BSAParser
    from string_tables import StringTableManager, load_string_table_from_bytes

    bsa_path = base_path / bsa_name
    if not bsa_path.exists():
        print(f"  Warning: BSA not found: {bsa_path}")
        return None

    print(f"  Loading string tables from {bsa_name}...")
    parser = BSAParser(bsa_path)

    # Find all English string files
    string_files = {}
    for path in parser.list_files('strings'):
        if 'english' in path.lower():
            try:
                data = parser.extract_file(path)
                filename = path.split('/')[-1]
                string_files[filename] = data
            except Exception as e:
                print(f"    Error extracting {path}: {e}")

    # Load into manager
    manager = StringTableManager()
    manager.load_from_bytes(string_files)
    print(f"  Loaded {manager.total_strings()} strings total")
    return manager


def extract_plugin(path: Path, verbose: bool = True, string_tables=None) -> dict:
    """Extract dialogue from a single plugin."""
    if not path.exists():
        print(f"  Skipping {path.name} (not found)")
        return None

    print(f"  Parsing {path.name}...")
    parser = ESMParser(path, verbose=False, string_tables=string_tables)
    result = parser.parse()

    lines = []
    for topic in result.topics:
        for line in topic.lines:
            lines.append({
                'form_id': f"0x{line.form_id:x}",
                'text': line.text,
                'speaker': line.speaker_name or result.npcs.get(line.speaker_form_id),
                'emotion': line.emotion_label(),
                'emotion_value': line.emotion_value,
                'topic': topic.editor_id or topic.name,
                'quest': line.quest_name or result.quests.get(line.quest_form_id),
                'conditions': line.conditions,
                'source': path.name,
            })

    print(f"    → {len(lines)} lines, {len(result.topics)} topics")
    return {
        'filename': path.name,
        'game_type': result.game_type,
        'topics': len(result.topics),
        'npcs': len(result.npcs),
        'quests': len(result.quests),
        'lines': lines,
    }


def extract_game(game_key: str, config: dict, output_dir: Path):
    """Extract all dialogue for a game including DLCs."""
    base = config['base']

    print(f"\n{'='*60}")
    print(f"Extracting: {game_key}")
    print(f"{'='*60}")

    # Load string tables for localized games (Skyrim)
    string_tables = None
    if config.get('localized') and config.get('string_bsa'):
        string_tables = load_skyrim_strings(base, config['string_bsa'])

    all_lines = []
    sources = []

    # Extract main ESM
    main_path = base / config['main']
    main_data = extract_plugin(main_path, string_tables=string_tables)
    if main_data:
        all_lines.extend(main_data['lines'])
        sources.append({
            'file': main_data['filename'],
            'lines': len(main_data['lines']),
            'topics': main_data['topics'],
            'type': 'main'
        })

    # Extract DLCs
    for dlc in config['dlcs']:
        dlc_path = base / dlc
        dlc_data = extract_plugin(dlc_path, string_tables=string_tables)
        if dlc_data:
            all_lines.extend(dlc_data['lines'])
            sources.append({
                'file': dlc_data['filename'],
                'lines': len(dlc_data['lines']),
                'topics': dlc_data['topics'],
                'type': 'dlc'
            })

    # Save combined output
    output_dir.mkdir(exist_ok=True)

    output = {
        'game': game_key,
        'sources': sources,
        'total_lines': len(all_lines),
        'dialogue': all_lines,
    }

    output_path = output_dir / f"{game_key}_full_dialogue.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nSummary for {game_key}:")
    print(f"  Total sources: {len(sources)}")
    print(f"  Total lines: {len(all_lines)}")
    for src in sources:
        print(f"    {src['file']}: {src['lines']} lines ({src['type']})")
    print(f"  Saved to: {output_path}")

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract all DLC dialogue')
    parser.add_argument('--output', '-o', type=Path, default=Path('dialogue_data'),
                       help='Output directory')
    parser.add_argument('--game', '-g', choices=list(GAMES.keys()),
                       help='Extract specific game only')

    args = parser.parse_args()

    games_to_extract = [args.game] if args.game else GAMES.keys()

    for game_key in games_to_extract:
        if game_key in GAMES:
            extract_game(game_key, GAMES[game_key], args.output)


if __name__ == '__main__':
    main()
