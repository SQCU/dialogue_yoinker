#!/usr/bin/env python3
"""
BSA Archive Parser

Parses Bethesda Softworks Archive (BSA) files to extract assets.
Used primarily for extracting string tables from Skyrim SE.

BSA Format (version 104/105):
- Header: magic, version, folder records offset, archive flags, etc.
- Folder records: hash, count, offset
- File records: hash, size, offset
- File names (null-terminated strings)
- File data (optionally compressed)

References:
- https://en.uesp.net/wiki/Skyrim_Mod:Archive_File_Format
- https://www.nexusmods.com/skyrimspecialedition/articles/7
"""

import struct
import zlib
import lz4.block  # For Skyrim SE compression
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, BinaryIO, Iterator, Tuple


# BSA Magic and versions
BSA_MAGIC = b'BSA\x00'
BSA_VERSION_SKYRIM = 104
BSA_VERSION_SKYRIM_SE = 105

# Archive flags
ARCHIVE_FLAGS = {
    'INCLUDE_DIRECTORY_NAMES': 0x001,
    'INCLUDE_FILE_NAMES': 0x002,
    'COMPRESSED': 0x004,
    'RETAIN_DIRECTORY_NAMES': 0x008,
    'RETAIN_FILE_NAMES': 0x010,
    'RETAIN_FILE_NAME_OFFSETS': 0x020,
    'XBOX360': 0x040,
    'RETAIN_STRINGS': 0x080,
    'EMBED_FILE_NAMES': 0x100,
    'XMEM_CODEC': 0x200,
}

# File flags (content types)
FILE_FLAGS = {
    'MESHES': 0x001,
    'TEXTURES': 0x002,
    'MENUS': 0x004,
    'SOUNDS': 0x008,
    'VOICES': 0x010,
    'SHADERS': 0x020,
    'TREES': 0x040,
    'FONTS': 0x080,
    'MISC': 0x100,
}


@dataclass
class BSAHeader:
    """BSA file header."""
    version: int
    folder_offset: int
    archive_flags: int
    folder_count: int
    file_count: int
    folder_names_length: int
    file_names_length: int
    file_flags: int

    @property
    def has_folder_names(self) -> bool:
        return bool(self.archive_flags & ARCHIVE_FLAGS['INCLUDE_DIRECTORY_NAMES'])

    @property
    def has_file_names(self) -> bool:
        return bool(self.archive_flags & ARCHIVE_FLAGS['INCLUDE_FILE_NAMES'])

    @property
    def is_compressed(self) -> bool:
        return bool(self.archive_flags & ARCHIVE_FLAGS['COMPRESSED'])

    @property
    def has_embedded_names(self) -> bool:
        return bool(self.archive_flags & ARCHIVE_FLAGS['EMBED_FILE_NAMES'])


@dataclass
class BSAFolder:
    """A folder in the BSA."""
    name_hash: int
    file_count: int
    offset: int
    name: str = ""
    files: List['BSAFile'] = field(default_factory=list)


@dataclass
class BSAFile:
    """A file in the BSA."""
    name_hash: int
    size: int
    offset: int
    name: str = ""
    folder: str = ""

    @property
    def is_compressed(self) -> bool:
        # Top bit of size indicates compression toggle
        return bool(self.size & 0x40000000)

    @property
    def real_size(self) -> int:
        return self.size & 0x3FFFFFFF

    @property
    def full_path(self) -> str:
        return f"{self.folder}/{self.name}" if self.folder else self.name


class BSAParser:
    """
    Parser for BSA archive files.

    Usage:
        parser = BSAParser(Path("Skyrim - Misc.bsa"))
        for file in parser.list_files():
            if 'strings' in file.lower():
                data = parser.extract_file(file)
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.header: Optional[BSAHeader] = None
        self.folders: List[BSAFolder] = []
        self._file_index: Dict[str, BSAFile] = {}
        self._parsed = False

    def parse(self):
        """Parse the BSA structure (but don't extract files yet)."""
        if self._parsed:
            return

        with open(self.filepath, 'rb') as f:
            self._parse_header(f)
            self._parse_folders(f)
            self._parse_file_records(f)
            self._parse_file_names(f)

        # Build index
        for folder in self.folders:
            for file in folder.files:
                self._file_index[file.full_path.lower()] = file

        self._parsed = True

    def _parse_header(self, f: BinaryIO):
        """Parse BSA header."""
        magic = f.read(4)
        if magic != BSA_MAGIC:
            raise ValueError(f"Not a BSA file: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        if version not in (BSA_VERSION_SKYRIM, BSA_VERSION_SKYRIM_SE):
            print(f"Warning: Unknown BSA version {version}, attempting parse anyway")

        folder_offset = struct.unpack('<I', f.read(4))[0]
        archive_flags = struct.unpack('<I', f.read(4))[0]
        folder_count = struct.unpack('<I', f.read(4))[0]
        file_count = struct.unpack('<I', f.read(4))[0]
        folder_names_length = struct.unpack('<I', f.read(4))[0]
        file_names_length = struct.unpack('<I', f.read(4))[0]
        file_flags = struct.unpack('<I', f.read(4))[0]

        self.header = BSAHeader(
            version=version,
            folder_offset=folder_offset,
            archive_flags=archive_flags,
            folder_count=folder_count,
            file_count=file_count,
            folder_names_length=folder_names_length,
            file_names_length=file_names_length,
            file_flags=file_flags,
        )

    def _parse_folders(self, f: BinaryIO):
        """Parse folder records."""
        f.seek(self.header.folder_offset)

        for _ in range(self.header.folder_count):
            name_hash = struct.unpack('<Q', f.read(8))[0]
            file_count = struct.unpack('<I', f.read(4))[0]

            # Skyrim SE has extra padding
            if self.header.version == BSA_VERSION_SKYRIM_SE:
                f.read(4)  # padding
                offset = struct.unpack('<Q', f.read(8))[0]
            else:
                offset = struct.unpack('<I', f.read(4))[0]

            self.folders.append(BSAFolder(
                name_hash=name_hash,
                file_count=file_count,
                offset=offset,
            ))

    def _parse_file_records(self, f: BinaryIO):
        """Parse file records for each folder."""
        for folder in self.folders:
            # Read folder name if present
            if self.header.has_folder_names:
                name_len = struct.unpack('<B', f.read(1))[0]
                folder.name = f.read(name_len - 1).decode('utf-8', errors='replace')
                f.read(1)  # null terminator

            # Read file records
            for _ in range(folder.file_count):
                name_hash = struct.unpack('<Q', f.read(8))[0]
                size = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<I', f.read(4))[0]

                folder.files.append(BSAFile(
                    name_hash=name_hash,
                    size=size,
                    offset=offset,
                    folder=folder.name,
                ))

    def _parse_file_names(self, f: BinaryIO):
        """Parse file names block."""
        if not self.header.has_file_names:
            return

        # Read all file names as null-terminated strings
        names_data = f.read(self.header.file_names_length)
        names = names_data.split(b'\x00')

        # Assign names to files
        name_idx = 0
        for folder in self.folders:
            for file in folder.files:
                if name_idx < len(names):
                    file.name = names[name_idx].decode('utf-8', errors='replace')
                    name_idx += 1

    def list_files(self, pattern: str = None) -> List[str]:
        """List all files in the archive, optionally filtered by pattern."""
        self.parse()

        files = list(self._file_index.keys())
        if pattern:
            pattern = pattern.lower()
            files = [f for f in files if pattern in f]

        return sorted(files)

    def extract_file(self, path: str) -> bytes:
        """Extract a file from the archive."""
        self.parse()

        path = path.lower()
        if path not in self._file_index:
            raise KeyError(f"File not found in archive: {path}")

        file = self._file_index[path]

        with open(self.filepath, 'rb') as f:
            f.seek(file.offset)

            # Check for embedded filename
            if self.header.has_embedded_names:
                name_len = struct.unpack('<B', f.read(1))[0]
                f.read(name_len)  # skip embedded name

            # Determine compression
            is_compressed = file.is_compressed
            if self.header.is_compressed:
                is_compressed = not is_compressed  # Toggle

            data_size = file.real_size

            if is_compressed:
                # Read original size first
                original_size = struct.unpack('<I', f.read(4))[0]
                compressed_data = f.read(data_size - 4)

                # Try LZ4 first (Skyrim SE), then zlib
                try:
                    data = lz4.block.decompress(compressed_data, uncompressed_size=original_size)
                except:
                    try:
                        data = zlib.decompress(compressed_data)
                    except:
                        # Return raw data if decompression fails
                        data = compressed_data
            else:
                data = f.read(data_size)

            return data

    def extract_to_dir(self, output_dir: Path, pattern: str = None):
        """Extract files matching pattern to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for path in self.list_files(pattern):
            try:
                data = self.extract_file(path)
                out_path = output_dir / path.replace('/', '_')
                out_path.write_bytes(data)
                print(f"  Extracted: {path} ({len(data)} bytes)")
            except Exception as e:
                print(f"  Failed: {path} - {e}")


def find_string_tables(bsa_path: Path) -> Dict[str, bytes]:
    """
    Find and extract string tables from a BSA.

    String tables are:
    - {plugin}_English.STRINGS   - Regular strings
    - {plugin}_English.ILSTRINGS - IL strings (indexed by form ID)
    - {plugin}_English.DLSTRINGS - DL strings

    Returns dict mapping filename to raw bytes.
    """
    parser = BSAParser(bsa_path)
    string_files = parser.list_files('strings')

    result = {}
    for path in string_files:
        try:
            data = parser.extract_file(path)
            filename = path.split('/')[-1]
            result[filename] = data
            print(f"  Found: {filename} ({len(data)} bytes)")
        except Exception as e:
            print(f"  Error extracting {path}: {e}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parse BSA archives')
    parser.add_argument('bsa', type=Path, help='Path to BSA file')
    parser.add_argument('--list', '-l', action='store_true', help='List files')
    parser.add_argument('--filter', '-f', type=str, default=None, help='Filter pattern')
    parser.add_argument('--extract', '-x', type=Path, default=None, help='Extract to directory')
    parser.add_argument('--strings', '-s', action='store_true', help='Find string tables')

    args = parser.parse_args()

    if not args.bsa.exists():
        print(f"File not found: {args.bsa}")
        return

    bsa = BSAParser(args.bsa)

    if args.list:
        print(f"Files in {args.bsa.name}:")
        for f in bsa.list_files(args.filter):
            print(f"  {f}")
        print(f"\nTotal: {len(bsa.list_files())} files")

    if args.strings:
        print(f"\nSearching for string tables in {args.bsa.name}...")
        strings = find_string_tables(args.bsa)
        print(f"Found {len(strings)} string table(s)")

    if args.extract:
        print(f"\nExtracting to {args.extract}...")
        bsa.extract_to_dir(args.extract, args.filter)


if __name__ == '__main__':
    main()
