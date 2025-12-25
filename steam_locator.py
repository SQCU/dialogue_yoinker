#!/usr/bin/env python3
"""
Steam Library Locator

Cross-platform utility to find Steam installations and game directories.
Parses Steam's libraryfolders.vdf to find all library locations.

Usage:
    from steam_locator import find_game_data_path
    
    path = find_game_data_path('Oblivion')
    # Returns: Path('/home/user/.steam/steam/steamapps/common/Oblivion/Data')
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict


def is_wsl() -> bool:
    """Detect if running under Windows Subsystem for Linux."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


def get_steam_root() -> Optional[Path]:
    """
    Find the Steam installation directory.

    Platform-specific default locations:
    - Linux: ~/.steam/steam, ~/.local/share/Steam
    - WSL: /mnt/c/Program Files (x86)/Steam (Windows host)
    - Windows: C:/Program Files (x86)/Steam
    - macOS: ~/Library/Application Support/Steam
    """
    candidates = []

    if sys.platform == 'linux':
        home = Path.home()
        # Check for WSL first - prefer Windows Steam installation
        if is_wsl():
            candidates = [
                Path('/mnt/c/Program Files (x86)/Steam'),
                Path('/mnt/c/Program Files/Steam'),
                # Also check other drive letters commonly used for games
                Path('/mnt/d/Steam'),
                Path('/mnt/d/SteamLibrary'),
                Path('/mnt/e/Steam'),
                Path('/mnt/e/SteamLibrary'),
                # Fall back to native Linux Steam if installed in WSL
                home / '.steam' / 'steam',
                home / '.local' / 'share' / 'Steam',
            ]
        else:
            candidates = [
                home / '.steam' / 'steam',
                home / '.steam' / 'debian-installation',
                home / '.local' / 'share' / 'Steam',
            ]
    elif sys.platform == 'win32':
        candidates = [
            Path('C:/Program Files (x86)/Steam'),
            Path('C:/Program Files/Steam'),
        ]
        # Also check registry if available
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                 r'Software\Valve\Steam')
            steam_path, _ = winreg.QueryValueEx(key, 'SteamPath')
            candidates.insert(0, Path(steam_path))
        except:
            pass
    elif sys.platform == 'darwin':
        candidates = [
            Path.home() / 'Library' / 'Application Support' / 'Steam',
        ]
    
    for candidate in candidates:
        if candidate.exists() and (candidate / 'steamapps').exists():
            return candidate
    
    return None


def parse_vdf(content: str) -> Dict:
    """
    Parse Valve's VDF format (similar to JSON but different syntax).
    
    Format:
    "key" "value"
    "key" { nested }
    """
    result = {}
    stack = [result]
    current_key = None
    
    # Simple tokenizer
    tokens = re.findall(r'"([^"]*)"|\{|\}', content)
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token == '{':
            # Start nested dict
            new_dict = {}
            if current_key:
                stack[-1][current_key] = new_dict
                stack.append(new_dict)
                current_key = None
        elif token == '}':
            # End nested dict
            if len(stack) > 1:
                stack.pop()
        elif current_key is None:
            # This is a key
            current_key = token
        else:
            # This is a value
            stack[-1][current_key] = token
            current_key = None
        
        i += 1
    
    return result


def get_library_folders(steam_root: Path) -> List[Path]:
    """
    Parse libraryfolders.vdf to find all Steam library locations.
    
    The main Steam folder is always a library, plus any additional
    folders the user has configured.
    """
    libraries = [steam_root]
    
    vdf_path = steam_root / 'steamapps' / 'libraryfolders.vdf'
    if not vdf_path.exists():
        # Try older location
        vdf_path = steam_root / 'config' / 'libraryfolders.vdf'
    
    if vdf_path.exists():
        try:
            content = vdf_path.read_text(encoding='utf-8', errors='replace')
            data = parse_vdf(content)

            # Navigate to the library folders
            folders = data.get('libraryfolders', data)

            # Safety check - folders must be a dict
            if not isinstance(folders, dict):
                folders = data if isinstance(data, dict) else {}

            for key, value in folders.items():
                if isinstance(value, dict) and 'path' in value:
                    lib_path = Path(value['path'])
                    if lib_path.exists() and lib_path not in libraries:
                        libraries.append(lib_path)
                elif isinstance(value, str) and key.isdigit():
                    # Old format: just path strings
                    lib_path = Path(value)
                    if lib_path.exists() and lib_path not in libraries:
                        libraries.append(lib_path)
        except Exception as e:
            print(f"Warning: Could not parse libraryfolders.vdf: {e}")
    
    return libraries


def find_game_directory(game_name: str, libraries: List[Path]) -> Optional[Path]:
    """
    Search for a game directory by name.
    
    Searches common variations and does fuzzy matching.
    """
    # Normalize search term
    search_lower = game_name.lower().replace(' ', '').replace(':', '').replace('-', '')
    
    for library in libraries:
        common_path = library / 'steamapps' / 'common'
        if not common_path.exists():
            continue
        
        for game_dir in common_path.iterdir():
            if not game_dir.is_dir():
                continue
            
            # Normalize directory name
            dir_lower = game_dir.name.lower().replace(' ', '').replace(':', '').replace('-', '')
            
            # Check for match
            if search_lower in dir_lower or dir_lower in search_lower:
                return game_dir
    
    return None


# =============================================================================
# Game-Specific Locators
# =============================================================================

# Known game directory names and their Data folder structure
KNOWN_GAMES = {
    'oblivion': {
        'names': ['Oblivion', 'The Elder Scrolls IV Oblivion'],
        'data_subdir': 'Data',
        'main_plugin': 'Oblivion.esm'
    },
    'fallout3': {
        'names': ['Fallout 3', 'Fallout 3 goty'],
        'data_subdir': 'Data',
        'main_plugin': 'Fallout3.esm'
    },
    'falloutnv': {
        'names': ['Fallout New Vegas', 'FalloutNV'],
        'data_subdir': 'Data',
        'main_plugin': 'FalloutNV.esm'
    },
    'skyrim': {
        'names': ['Skyrim', 'The Elder Scrolls V Skyrim'],
        'data_subdir': 'Data',
        'main_plugin': 'Skyrim.esm'
    },
    'skyrimse': {
        'names': ['Skyrim Special Edition'],
        'data_subdir': 'Data',
        'main_plugin': 'Skyrim.esm'
    },
    'fallout4': {
        'names': ['Fallout 4'],
        'data_subdir': 'Data',
        'main_plugin': 'Fallout4.esm'
    },
    'morrowind': {
        'names': ['Morrowind', 'The Elder Scrolls III Morrowind'],
        'data_subdir': 'Data Files',
        'main_plugin': 'Morrowind.esm'
    }
}


def find_game_data_path(game_key: str) -> Optional[Path]:
    """
    Find the Data folder for a known Bethesda game.
    
    Args:
        game_key: One of: oblivion, fallout3, falloutnv, skyrim, skyrimse, fallout4, morrowind
    
    Returns:
        Path to the Data folder, or None if not found.
    """
    game_key = game_key.lower()
    
    if game_key not in KNOWN_GAMES:
        # Try fuzzy match
        for key in KNOWN_GAMES:
            if game_key in key or key in game_key:
                game_key = key
                break
        else:
            print(f"Unknown game: {game_key}")
            print(f"Known games: {', '.join(KNOWN_GAMES.keys())}")
            return None
    
    game_info = KNOWN_GAMES[game_key]
    
    steam_root = get_steam_root()
    if not steam_root:
        print("Steam installation not found")
        return None
    
    libraries = get_library_folders(steam_root)
    print(f"Found {len(libraries)} Steam library folder(s)")
    
    # Search for the game
    for name in game_info['names']:
        game_dir = find_game_directory(name, libraries)
        if game_dir:
            data_path = game_dir / game_info['data_subdir']
            if data_path.exists():
                # Verify main plugin exists
                main_plugin = data_path / game_info['main_plugin']
                if main_plugin.exists():
                    print(f"Found {game_key}: {data_path}")
                    return data_path
                else:
                    print(f"Found game directory but missing {game_info['main_plugin']}")
    
    return None


def find_all_bethesda_games() -> Dict[str, Path]:
    """
    Find all installed Bethesda games with ESM support.
    
    Returns:
        Dict mapping game key to Data folder path.
    """
    found = {}
    
    steam_root = get_steam_root()
    if not steam_root:
        return found
    
    libraries = get_library_folders(steam_root)
    
    for game_key, game_info in KNOWN_GAMES.items():
        for name in game_info['names']:
            game_dir = find_game_directory(name, libraries)
            if game_dir:
                data_path = game_dir / game_info['data_subdir']
                main_plugin = data_path / game_info['main_plugin']
                if main_plugin.exists():
                    found[game_key] = data_path
                    break
    
    return found


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find Steam game installations'
    )
    parser.add_argument('game', nargs='?',
                        help='Game to find (e.g., oblivion, falloutnv)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all found Bethesda games')
    parser.add_argument('--steam', action='store_true',
                        help='Just print Steam root path')
    
    args = parser.parse_args()
    
    if args.steam:
        root = get_steam_root()
        if root:
            print(root)
        else:
            print("Steam not found", file=sys.stderr)
            return 1
    elif args.list:
        games = find_all_bethesda_games()
        if games:
            print("Found games:")
            for key, path in games.items():
                print(f"  {key}: {path}")
        else:
            print("No Bethesda games found")
        return 0
    elif args.game:
        path = find_game_data_path(args.game)
        if path:
            print(path)
            return 0
        else:
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    exit(main())