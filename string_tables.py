#!/usr/bin/env python3
"""
Skyrim String Table Parser

Parses Bethesda's localized string table files:
- .STRINGS   - Simple strings (null-terminated)
- .ILSTRINGS - IL strings (length-prefixed)
- .DLSTRINGS - DL strings (length-prefixed)

String table format:
  Header:
    - count (uint32) - number of entries
    - data_size (uint32) - size of string data block
  Directory:
    - [id (uint32), offset (uint32)] * count
  Data:
    - Strings at offsets (format depends on type)

References:
- https://en.uesp.net/wiki/Skyrim_Mod:String_Table_File_Format
"""

import struct
from pathlib import Path
from typing import Dict, Optional, BinaryIO
from dataclasses import dataclass


@dataclass
class StringTable:
    """A loaded string table."""
    strings: Dict[int, str]  # id -> string
    source_file: str

    def get(self, string_id: int, default: str = "") -> str:
        return self.strings.get(string_id, default)

    def __len__(self) -> int:
        return len(self.strings)


def parse_strings_file(data: bytes, encoding: str = 'utf-8') -> Dict[int, str]:
    """
    Parse a .STRINGS file (null-terminated strings).

    Format:
    - Each string is null-terminated
    - Offset points to start of string
    """
    if len(data) < 8:
        return {}

    count, data_size = struct.unpack('<II', data[:8])

    # Read directory
    directory = []
    offset = 8
    for _ in range(count):
        if offset + 8 > len(data):
            break
        str_id, str_offset = struct.unpack('<II', data[offset:offset+8])
        directory.append((str_id, str_offset))
        offset += 8

    # Data block starts after directory
    data_start = 8 + count * 8

    # Read strings
    strings = {}
    for str_id, str_offset in directory:
        abs_offset = data_start + str_offset
        if abs_offset >= len(data):
            continue

        # Find null terminator
        end = data.find(b'\x00', abs_offset)
        if end == -1:
            end = len(data)

        try:
            text = data[abs_offset:end].decode(encoding, errors='replace')
            strings[str_id] = text
        except:
            pass

    return strings


def parse_ilstrings_file(data: bytes, encoding: str = 'utf-8') -> Dict[int, str]:
    """
    Parse a .ILSTRINGS or .DLSTRINGS file (length-prefixed strings).

    Format:
    - Each string is prefixed with uint32 length (including null)
    - String data follows (null-terminated)
    """
    if len(data) < 8:
        return {}

    count, data_size = struct.unpack('<II', data[:8])

    # Read directory
    directory = []
    offset = 8
    for _ in range(count):
        if offset + 8 > len(data):
            break
        str_id, str_offset = struct.unpack('<II', data[offset:offset+8])
        directory.append((str_id, str_offset))
        offset += 8

    # Data block starts after directory
    data_start = 8 + count * 8

    # Read strings
    strings = {}
    for str_id, str_offset in directory:
        abs_offset = data_start + str_offset
        if abs_offset + 4 >= len(data):
            continue

        # Read length prefix
        length = struct.unpack('<I', data[abs_offset:abs_offset+4])[0]
        if length == 0:
            strings[str_id] = ""
            continue

        str_start = abs_offset + 4
        str_end = str_start + length - 1  # Exclude null terminator

        if str_end > len(data):
            str_end = len(data)

        try:
            text = data[str_start:str_end].decode(encoding, errors='replace')
            strings[str_id] = text
        except:
            pass

    return strings


def load_string_table(filepath: Path, encoding: str = 'utf-8') -> StringTable:
    """Load a string table from file."""
    data = filepath.read_bytes()
    ext = filepath.suffix.lower()

    if ext == '.strings':
        strings = parse_strings_file(data, encoding)
    elif ext in ('.ilstrings', '.dlstrings'):
        strings = parse_ilstrings_file(data, encoding)
    else:
        raise ValueError(f"Unknown string table type: {ext}")

    return StringTable(strings=strings, source_file=filepath.name)


def load_string_table_from_bytes(data: bytes, filename: str, encoding: str = 'utf-8') -> StringTable:
    """Load a string table from raw bytes."""
    ext = Path(filename).suffix.lower()

    if ext == '.strings':
        strings = parse_strings_file(data, encoding)
    elif ext in ('.ilstrings', '.dlstrings'):
        strings = parse_ilstrings_file(data, encoding)
    else:
        raise ValueError(f"Unknown string table type: {ext}")

    return StringTable(strings=strings, source_file=filename)


class StringTableManager:
    """
    Manages string tables for a plugin.

    Loads all three string table types and provides unified lookup.
    """

    def __init__(self):
        self.strings: Optional[StringTable] = None      # .STRINGS
        self.ilstrings: Optional[StringTable] = None    # .ILSTRINGS
        self.dlstrings: Optional[StringTable] = None    # .DLSTRINGS

    def load_from_files(self, base_path: Path, plugin_name: str, language: str = 'english'):
        """Load string tables from extracted files."""
        base = base_path / 'strings'

        strings_path = base / f"{plugin_name}_{language}.strings"
        ilstrings_path = base / f"{plugin_name}_{language}.ilstrings"
        dlstrings_path = base / f"{plugin_name}_{language}.dlstrings"

        if strings_path.exists():
            self.strings = load_string_table(strings_path)
            print(f"  Loaded {len(self.strings)} strings from {strings_path.name}")

        if ilstrings_path.exists():
            self.ilstrings = load_string_table(ilstrings_path)
            print(f"  Loaded {len(self.ilstrings)} ilstrings from {ilstrings_path.name}")

        if dlstrings_path.exists():
            self.dlstrings = load_string_table(dlstrings_path)
            print(f"  Loaded {len(self.dlstrings)} dlstrings from {dlstrings_path.name}")

    def load_from_bytes(self, files: Dict[str, bytes]):
        """Load string tables from dict of filename -> bytes, merging all strings."""
        # We need to merge strings from all plugins (Skyrim, Dawnguard, Dragonborn, etc.)
        # Each plugin has its own string tables but IDs are unique across all
        all_strings = {}
        all_ilstrings = {}
        all_dlstrings = {}

        for filename, data in files.items():
            filename_lower = filename.lower()
            try:
                table = load_string_table_from_bytes(data, filename)
                if filename_lower.endswith('.strings'):
                    all_strings.update(table.strings)
                    print(f"  Loaded {len(table)} strings from {filename}")
                elif filename_lower.endswith('.ilstrings'):
                    all_ilstrings.update(table.strings)
                    print(f"  Loaded {len(table)} ilstrings from {filename}")
                elif filename_lower.endswith('.dlstrings'):
                    all_dlstrings.update(table.strings)
                    print(f"  Loaded {len(table)} dlstrings from {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")

        # Create merged tables
        if all_strings:
            self.strings = StringTable(strings=all_strings, source_file="merged")
        if all_ilstrings:
            self.ilstrings = StringTable(strings=all_ilstrings, source_file="merged")
        if all_dlstrings:
            self.dlstrings = StringTable(strings=all_dlstrings, source_file="merged")

    def get(self, string_id: int, default: str = "") -> str:
        """
        Look up a string by ID.

        Searches all loaded tables in order: strings, ilstrings, dlstrings.
        """
        if self.strings and string_id in self.strings.strings:
            return self.strings.get(string_id, default)
        if self.ilstrings and string_id in self.ilstrings.strings:
            return self.ilstrings.get(string_id, default)
        if self.dlstrings and string_id in self.dlstrings.strings:
            return self.dlstrings.get(string_id, default)
        return default

    def total_strings(self) -> int:
        """Total number of strings across all tables."""
        total = 0
        if self.strings:
            total += len(self.strings)
        if self.ilstrings:
            total += len(self.ilstrings)
        if self.dlstrings:
            total += len(self.dlstrings)
        return total


def extract_skyrim_strings(game_data_path: Path, output_dir: Path = None) -> StringTableManager:
    """
    Extract string tables for Skyrim from BSA.

    Args:
        game_data_path: Path to Skyrim's Data folder
        output_dir: Optional dir to save extracted files

    Returns:
        StringTableManager with loaded strings
    """
    from bsa_parser import BSAParser

    bsa_path = game_data_path / "Skyrim - Interface.bsa"
    if not bsa_path.exists():
        raise FileNotFoundError(f"BSA not found: {bsa_path}")

    print(f"Extracting string tables from {bsa_path.name}...")
    parser = BSAParser(bsa_path)

    # Find English string files for main game
    string_files = {}
    for path in parser.list_files('strings/skyrim_english'):
        try:
            data = parser.extract_file(path)
            filename = path.split('/')[-1]
            string_files[filename] = data

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / filename).write_bytes(data)
        except Exception as e:
            print(f"  Error extracting {path}: {e}")

    # Load into manager
    manager = StringTableManager()
    manager.load_from_bytes(string_files)

    print(f"Total strings loaded: {manager.total_strings()}")
    return manager


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parse Skyrim string tables')
    parser.add_argument('path', type=Path, nargs='?',
                       help='Path to string table file or Skyrim Data folder')
    parser.add_argument('--extract-bsa', action='store_true',
                       help='Extract strings from BSA')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output directory for extracted strings')
    parser.add_argument('--lookup', '-l', type=int, default=None,
                       help='Look up string by ID')
    parser.add_argument('--sample', '-s', type=int, default=10,
                       help='Show N sample strings')

    args = parser.parse_args()

    if args.extract_bsa:
        # Extract from BSA
        path = args.path or Path('/mnt/c/Program Files (x86)/Steam/steamapps/common/Skyrim Special Edition/Data')
        manager = extract_skyrim_strings(path, args.output)

        if args.lookup is not None:
            text = manager.get(args.lookup, f"[ID {args.lookup} not found]")
            print(f"\nString {args.lookup}: {text}")

        if args.sample > 0:
            print(f"\nSample strings:")
            shown = 0
            for table_name, table in [('STRINGS', manager.strings),
                                       ('ILSTRINGS', manager.ilstrings),
                                       ('DLSTRINGS', manager.dlstrings)]:
                if table and shown < args.sample:
                    print(f"\n  From {table_name}:")
                    for str_id, text in list(table.strings.items())[:args.sample - shown]:
                        preview = text[:60] + "..." if len(text) > 60 else text
                        print(f"    {str_id}: {preview}")
                        shown += 1
                        if shown >= args.sample:
                            break

    elif args.path and args.path.is_file():
        # Parse single file
        table = load_string_table(args.path)
        print(f"Loaded {len(table)} strings from {args.path.name}")

        if args.lookup is not None:
            text = table.get(args.lookup, f"[ID {args.lookup} not found]")
            print(f"\nString {args.lookup}: {text}")

        if args.sample > 0:
            print(f"\nSample strings:")
            for str_id, text in list(table.strings.items())[:args.sample]:
                preview = text[:60] + "..." if len(text) > 60 else text
                print(f"  {str_id}: {preview}")


if __name__ == '__main__':
    main()
