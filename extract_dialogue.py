#!/usr/bin/env python3
"""
Quick extraction script for Bethesda dialogue.

Usage:
    python extract_dialogue.py oblivion
    python extract_dialogue.py falloutnv --stats
    python extract_dialogue.py --all
    
This will:
1. Find the game via Steam
2. Parse the main ESM
3. Export dialogue in training-ready format
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from steam_locator import find_game_data_path, find_all_bethesda_games, KNOWN_GAMES
from esm_dialogue_parser import (
    ESMParser, export_to_json, export_to_csv, 
    export_for_training, print_statistics
)


def extract_game(game_key: str, output_dir: Path, stats: bool = False):
    """Extract dialogue from a single game."""
    
    print(f"\n{'='*60}")
    print(f"Extracting: {game_key}")
    print(f"{'='*60}")
    
    # Find game
    data_path = find_game_data_path(game_key)
    if not data_path:
        print(f"Could not find {game_key}")
        return False
    
    # Get main plugin
    game_info = KNOWN_GAMES.get(game_key.lower(), {})
    main_plugin = game_info.get('main_plugin', f'{game_key}.esm')
    plugin_path = data_path / main_plugin
    
    if not plugin_path.exists():
        print(f"Main plugin not found: {plugin_path}")
        return False
    
    print(f"Parsing: {plugin_path}")
    
    # Parse
    try:
        parser = ESMParser(plugin_path, verbose=False)
        result = parser.parse()
    except Exception as e:
        print(f"Parse error: {e}")
        return False
    
    if stats:
        print_statistics(result)
    
    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = plugin_path.stem.lower()
    
    export_to_json(result, output_dir / f"{base_name}_dialogue.json")
    export_for_training(result, output_dir / f"{base_name}_training.jsonl")
    
    # Summary
    line_count = sum(1 for _ in result.all_lines() if _.text)
    print(f"\nExtracted {line_count} dialogue lines")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract Bethesda game dialogue for ML training'
    )
    parser.add_argument('game', nargs='?',
                        help='Game key (oblivion, falloutnv, skyrim, etc.)')
    parser.add_argument('--all', action='store_true',
                        help='Extract from all found games')
    parser.add_argument('--output', '-o', type=Path, default=Path('./dialogue_data'),
                        help='Output directory')
    parser.add_argument('--stats', '-s', action='store_true',
                        help='Print statistics')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available games')
    
    args = parser.parse_args()
    
    if args.list:
        print("Searching for installed games...")
        games = find_all_bethesda_games()
        if games:
            print("\nFound:")
            for key, path in games.items():
                print(f"  {key}: {path}")
        else:
            print("No games found")
        return 0
    
    if args.all:
        print("Extracting from all found games...")
        games = find_all_bethesda_games()
        if not games:
            print("No games found")
            return 1
        
        success = 0
        for game_key in games:
            if extract_game(game_key, args.output, args.stats):
                success += 1
        
        print(f"\n{'='*60}")
        print(f"Extracted from {success}/{len(games)} games")
        return 0 if success > 0 else 1
    
    if args.game:
        if extract_game(args.game, args.output, args.stats):
            return 0
        return 1
    
    parser.print_help()
    return 0


if __name__ == '__main__':
    exit(main())