#!/usr/bin/env python3
"""
yoinkems.py
ESM/ESP Dialogue Extractor

Parses Bethesda's ESM/ESP plugin format to extract dialogue records with:
- Speaker/listener metadata
- Emotion/mood annotations (used for facial animation)
- Quest/condition context
- Response linkage

Supports: TES4 (Oblivion), FO3, FNV, TES5 (Skyrim) formats
The format is deliberately open for modding - we're just reading what was meant to be read.

Usage:
    python esm_dialogue_parser.py /path/to/game/Data
    python esm_dialogue_parser.py /path/to/specific/file.esm
    
Output: JSON/CSV with structured dialogue data suitable for ML training.
"""

import struct
import os
import json
import csv
import zlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, BinaryIO, Iterator, Tuple
from enum import IntEnum
import argparse


# =============================================================================
# Constants & Enums
# =============================================================================

class EmotionType(IntEnum):
    """Emotion values used in dialogue for facial animation choreography."""
    NEUTRAL = 0
    ANGER = 1
    DISGUST = 2
    FEAR = 3
    SAD = 4
    HAPPY = 5
    SURPRISE = 6
    PAINED = 7
    # Extended (game-specific)
    PUZZLED = 8


# Game format signatures
GAME_SIGNATURES = {
    b'TES4': 'oblivion_or_later',  # Oblivion, FO3, FNV, Skyrim all use TES4
}

# Record types we care about
RECORD_DIAL = b'DIAL'  # Dialogue Topic
RECORD_INFO = b'INFO'  # Dialogue Response (individual line)
RECORD_QUST = b'QUST'  # Quest (for context)
RECORD_NPC_ = b'NPC_'  # NPC definitions
RECORD_GRUP = b'GRUP'  # Group container


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DialogueLine:
    """A single line of dialogue with full metadata."""
    form_id: int
    topic_form_id: int
    topic_name: str
    speaker_form_id: Optional[int]
    speaker_name: Optional[str]
    text: str
    emotion_type: int
    emotion_value: int  # 0-100 intensity
    quest_form_id: Optional[int]
    quest_name: Optional[str]
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)  # linked response texts
    script_notes: Optional[str] = None
    audio_filename: Optional[str] = None
    
    def emotion_label(self) -> str:
        """Human-readable emotion label."""
        try:
            return EmotionType(self.emotion_type).name.lower()
        except ValueError:
            return f"unknown_{self.emotion_type}"


@dataclass
class DialogueTopic:
    """A dialogue topic containing multiple INFO records."""
    form_id: int
    editor_id: str
    name: str  # Display name
    quest_form_id: Optional[int]
    topic_type: int
    lines: List[DialogueLine] = field(default_factory=list)


@dataclass 
class ParsedPlugin:
    """Container for all extracted data from a plugin."""
    filename: str
    game_type: str
    topics: List[DialogueTopic] = field(default_factory=list)
    npcs: Dict[int, str] = field(default_factory=dict)  # form_id -> name
    quests: Dict[int, str] = field(default_factory=dict)  # form_id -> name
    
    def all_lines(self) -> Iterator[DialogueLine]:
        """Iterate over all dialogue lines."""
        for topic in self.topics:
            yield from topic.lines


# =============================================================================
# Binary Reading Utilities
# =============================================================================

class BinaryReader:
    """Utility for reading structured binary data."""
    
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self._encoding = 'cp1252'  # Bethesda's default encoding
    
    def read(self, n: int) -> bytes:
        return self.stream.read(n)
    
    def read_uint32(self) -> int:
        return struct.unpack('<I', self.read(4))[0]
    
    def read_uint16(self) -> int:
        return struct.unpack('<H', self.read(2))[0]
    
    def read_int32(self) -> int:
        return struct.unpack('<i', self.read(4))[0]
    
    def read_float(self) -> float:
        return struct.unpack('<f', self.read(4))[0]
    
    def read_cstring(self) -> str:
        """Read null-terminated string."""
        chars = []
        while True:
            c = self.read(1)
            if c == b'\x00' or c == b'':
                break
            chars.append(c)
        return b''.join(chars).decode(self._encoding, errors='replace')
    
    def read_sized_string(self, size: int) -> str:
        """Read fixed-size string, stripping nulls."""
        data = self.read(size)
        # Strip trailing nulls
        if b'\x00' in data:
            data = data[:data.index(b'\x00')]
        return data.decode(self._encoding, errors='replace')
    
    def skip(self, n: int):
        self.stream.seek(n, 1)
    
    def tell(self) -> int:
        return self.stream.tell()
    
    def seek(self, pos: int):
        self.stream.seek(pos)


# =============================================================================
# ESM/ESP Parser
# =============================================================================

class ESMParser:
    """
    Parser for Bethesda's ESM/ESP plugin format.

    The format is record-based:
    - Each record has a 4-char type, size, flags, formID
    - GRUP records contain other records
    - DIAL records define dialogue topics
    - INFO records contain individual dialogue lines

    Subrecords within records have 4-char type + 2-byte size.

    Header sizes vary by game:
    - Oblivion/FO3/FNV: 20-byte record headers
    - Skyrim+: 24-byte record headers

    Localized plugins (Skyrim SE):
    - TES4 header has flag 0x80 (LOCALIZED)
    - String fields contain 4-byte IDs instead of inline text
    - Actual strings are in external .STRINGS/.ILSTRINGS/.DLSTRINGS files
    """

    # Flag indicating localized string tables
    FLAG_LOCALIZED = 0x80

    def __init__(self, filepath: Path, verbose: bool = False, string_tables=None):
        self.filepath = filepath
        self.verbose = verbose
        self.result = ParsedPlugin(filename=filepath.name, game_type='unknown')
        self._current_topic: Optional[DialogueTopic] = None
        self._is_compressed = False
        self._game_version = 0.0  # Float version from HEDR
        self._record_header_size = 20  # Default to Oblivion/FO3/FNV format
        self._is_localized = False  # Whether strings are in external tables
        self._string_tables = string_tables  # StringTableManager for localized lookups
        
    def parse(self) -> ParsedPlugin:
        """Parse the plugin file and return structured data."""
        with open(self.filepath, 'rb') as f:
            reader = BinaryReader(f)
            
            # Read header record
            self._parse_header(reader)
            
            # Parse all records
            file_size = self.filepath.stat().st_size
            while reader.tell() < file_size:
                self._parse_record(reader)
        
        # Post-process: resolve references
        self._resolve_references()
        
        return self.result
    
    def _parse_header(self, reader: BinaryReader):
        """Parse the TES4 header record."""
        rec_type = reader.read(4)
        if rec_type != b'TES4':
            raise ValueError(f"Not a valid ESM/ESP file: expected TES4, got {rec_type}")

        data_size = reader.read_uint32()
        flags = reader.read_uint32()
        form_id = reader.read_uint32()

        # Detect header size by checking where HEDR subrecord starts
        # Save position after basic header fields (16 bytes read so far)
        probe_pos = reader.tell()

        # Check if HEDR is at offset +4 (20-byte header) or +8 (24-byte header)
        reader.skip(4)  # Skip to position 20
        maybe_hedr_20 = reader.read(4)
        reader.skip(0)  # Read 4 more to check position 24
        maybe_hedr_24 = reader.read(4)

        # Reset and read properly based on detection
        reader.seek(probe_pos)

        if maybe_hedr_20 == b'HEDR':
            # 20-byte header (Oblivion)
            self._record_header_size = 20
            reader.skip(4)  # vc_info
        elif maybe_hedr_24 == b'HEDR':
            # 24-byte header (FO3, FNV, Skyrim)
            self._record_header_size = 24
            reader.skip(8)  # revision(4) + version(2) + unknown(2)
        else:
            # Fallback: assume 20-byte and hope for the best
            self._record_header_size = 20
            reader.skip(4)

        # Read the TES4 header DATA to get HEDR subrecord
        header_data = reader.read(data_size)
        sub_reader = BinaryReader(self._bytes_to_stream(header_data))

        # First subrecord should be HEDR
        hedr_type = sub_reader.read(4)
        hedr_size = sub_reader.read_uint16()

        if hedr_type == b'HEDR':
            self._game_version = sub_reader.read_float()  # e.g., 0.8, 0.94, 1.0, 1.34, 1.7
        else:
            self._game_version = 0.0

        # Determine game type from HEDR version
        self.result.game_type = self._detect_game_type(self._game_version)

        # Check for localized strings flag (Skyrim SE)
        self._is_localized = bool(flags & self.FLAG_LOCALIZED)

        if self.verbose:
            print(f"Plugin: {self.filepath.name}")
            print(f"  HEDR Version: {self._game_version} ({self.result.game_type})")
            print(f"  Record header size: {self._record_header_size}")
            print(f"  Flags: {flags:#x}")
            print(f"  Localized: {self._is_localized}")
    
    def _detect_game_type(self, version: float) -> str:
        """Detect game from HEDR version float."""
        # HEDR version ranges (empirically determined):
        # Oblivion: 0.8, 1.0
        # FO3: 0.94
        # FNV: 1.32, 1.33, 1.34
        # Skyrim LE: 0.94, 1.7
        # Skyrim SE/FO4: 1.7+
        #
        # Use header size to disambiguate overlapping versions
        if version >= 1.7:
            return 'skyrim_se'
        elif version >= 1.3 and version < 1.7:
            # FNV uses ~1.34
            return 'fo3_fnv'
        elif version >= 1.0 and version < 1.3:
            # Oblivion 1.0, but check header size
            if self._record_header_size == 20:
                return 'oblivion'
            else:
                return 'skyrim'  # Skyrim LE can use various versions
        elif version >= 0.94:
            # FO3 or Skyrim LE
            if self._record_header_size == 24:
                return 'fo3_fnv'
            else:
                return 'oblivion'
        elif version >= 0.8:
            return 'oblivion'
        else:
            return 'unknown'
    
    def _parse_record(self, reader: BinaryReader, depth: int = 0):
        """Parse a single record or group."""
        start_pos = reader.tell()
        rec_type = reader.read(4)

        if rec_type == RECORD_GRUP:
            self._parse_group(reader, depth)
        else:
            data_size = reader.read_uint32()
            flags = reader.read_uint32()
            form_id = reader.read_uint32()

            # Read remaining header bytes based on game version
            # 20-byte header: type(4) + size(4) + flags(4) + formid(4) + vc(4) = already read 16, need 4 more
            # 24-byte header: above + version(2) + unknown(2) = need 8 more
            if self._record_header_size == 24:
                reader.skip(8)  # revision(4) + version(2) + unknown(2)
            else:
                reader.skip(4)  # vc_info(4)

            is_compressed = (flags & 0x00040000) != 0

            if rec_type == RECORD_DIAL:
                self._parse_dial(reader, form_id, data_size, is_compressed)
            elif rec_type == RECORD_INFO:
                self._parse_info(reader, form_id, data_size, is_compressed)
            elif rec_type == RECORD_NPC_:
                self._parse_npc(reader, form_id, data_size, is_compressed)
            elif rec_type == RECORD_QUST:
                self._parse_quest(reader, form_id, data_size, is_compressed)
            else:
                # Skip unknown record types
                reader.skip(data_size)
    
    def _parse_group(self, reader: BinaryReader, depth: int):
        """Parse a GRUP container."""
        group_size = reader.read_uint32()  # Includes header
        label = reader.read(4)
        group_type = reader.read_uint32()

        # GRUP header size matches record header size for the game version
        # Oblivion: 20 bytes (type + size + label + gtype + timestamp)
        # Skyrim+: 24 bytes (type + size + label + gtype + stamp + unk + version + unk)
        if self._record_header_size == 24:
            reader.skip(8)  # stamp(2) + unk1(2) + version(2) + unk2(2)
            grup_header_size = 24
        else:
            reader.skip(4)  # timestamp(4) in Oblivion
            grup_header_size = 20

        # Group types: 0=top, 1=world children, 7=topic children, etc.
        # Type 7 means this is a DIAL topic group containing INFO records

        end_pos = reader.tell() + group_size - grup_header_size

        while reader.tell() < end_pos:
            self._parse_record(reader, depth + 1)
    
    def _decompress_if_needed(self, reader: BinaryReader, data_size: int, 
                               is_compressed: bool) -> bytes:
        """Read and optionally decompress record data."""
        if is_compressed:
            decompressed_size = reader.read_uint32()
            compressed_data = reader.read(data_size - 4)
            return zlib.decompress(compressed_data)
        else:
            return reader.read(data_size)
    
    def _parse_dial(self, reader: BinaryReader, form_id: int, 
                    data_size: int, is_compressed: bool):
        """Parse a DIAL (dialogue topic) record."""
        data = self._decompress_if_needed(reader, data_size, is_compressed)
        sub_reader = BinaryReader(self._bytes_to_stream(data))
        
        topic = DialogueTopic(
            form_id=form_id,
            editor_id='',
            name='',
            quest_form_id=None,
            topic_type=0
        )
        
        # Parse subrecords
        while sub_reader.tell() < len(data):
            sub_type = sub_reader.read(4)
            sub_size = sub_reader.read_uint16()
            
            if sub_type == b'EDID':  # Editor ID (never localized)
                topic.editor_id = sub_reader.read_sized_string(sub_size)
            elif sub_type == b'FULL':  # Display name (localized)
                topic.name = self._read_localized_string(sub_reader, sub_size)
            elif sub_type == b'QSTI':  # Quest ID (Oblivion)
                topic.quest_form_id = sub_reader.read_uint32()
            elif sub_type == b'DATA':  # Topic type
                topic.topic_type = sub_reader.read(1)[0]
                sub_reader.skip(sub_size - 1)
            else:
                sub_reader.skip(sub_size)
        
        self._current_topic = topic
        self.result.topics.append(topic)
        
        if self.verbose and topic.editor_id:
            print(f"  Topic: {topic.editor_id} ({topic.name})")
    
    def _parse_info(self, reader: BinaryReader, form_id: int,
                    data_size: int, is_compressed: bool):
        """Parse an INFO (dialogue line) record."""
        data = self._decompress_if_needed(reader, data_size, is_compressed)
        sub_reader = BinaryReader(self._bytes_to_stream(data))
        
        line = DialogueLine(
            form_id=form_id,
            topic_form_id=self._current_topic.form_id if self._current_topic else 0,
            topic_name=self._current_topic.name if self._current_topic else '',
            speaker_form_id=None,
            speaker_name=None,
            text='',
            emotion_type=0,
            emotion_value=0,
            quest_form_id=None,
            quest_name=None
        )
        
        # Parse subrecords
        while sub_reader.tell() < len(data):
            sub_type = sub_reader.read(4)
            sub_size = sub_reader.read_uint16()
            sub_start = sub_reader.tell()
            
            if sub_type == b'NAM1':  # Response text (main dialogue)
                line.text = self._read_localized_string(sub_reader, sub_size)
            elif sub_type == b'NAM2':  # Script notes
                line.script_notes = self._read_localized_string(sub_reader, sub_size)
            elif sub_type == b'NAM3':  # Edits (unused)
                sub_reader.skip(sub_size)
            elif sub_type == b'TRDT':  # Response data (emotion, etc.)
                # TRDT structure varies by game
                emotion_type = sub_reader.read_uint32()
                emotion_value = sub_reader.read_int32()
                line.emotion_type = emotion_type
                line.emotion_value = emotion_value
                sub_reader.skip(sub_size - 8)  # Skip rest
            elif sub_type == b'QSTI':  # Quest
                line.quest_form_id = sub_reader.read_uint32()
            elif sub_type == b'ANAM':  # Speaker (Skyrim+)
                line.speaker_form_id = sub_reader.read_uint32()
            elif sub_type == b'CTDA':  # Condition
                condition = self._parse_condition(sub_reader, sub_size)
                if condition:
                    line.conditions.append(condition)
            else:
                sub_reader.skip(sub_size)
            
            # Ensure we're at the right position
            sub_reader.seek(sub_start + sub_size)
        
        if self._current_topic:
            self._current_topic.lines.append(line)
        
        if self.verbose and line.text:
            emotion_str = f"{line.emotion_label()}({line.emotion_value})"
            print(f"    [{emotion_str}] {line.text[:60]}...")
    
    def _parse_condition(self, reader: BinaryReader, size: int) -> Optional[Dict]:
        """Parse a CTDA condition subrecord."""
        if size < 24:
            reader.skip(size)
            return None
        
        # Condition structure (simplified)
        condition_type = reader.read(1)[0]
        unused = reader.read(3)
        comparison_value = reader.read_float()
        function_index = reader.read_uint16()
        padding = reader.read_uint16()
        param1 = reader.read_uint32()
        param2 = reader.read_uint32()
        run_on = reader.read_uint32()
        
        reader.skip(size - 24)
        
        return {
            'type': condition_type,
            'function': function_index,
            'value': comparison_value,
            'param1': param1,
            'param2': param2,
            'run_on': run_on
        }
    
    def _parse_npc(self, reader: BinaryReader, form_id: int,
                   data_size: int, is_compressed: bool):
        """Parse NPC_ record to get NPC names."""
        data = self._decompress_if_needed(reader, data_size, is_compressed)
        sub_reader = BinaryReader(self._bytes_to_stream(data))
        
        name = ''
        while sub_reader.tell() < len(data):
            sub_type = sub_reader.read(4)
            sub_size = sub_reader.read_uint16()

            if sub_type == b'FULL':
                name = self._read_localized_string(sub_reader, sub_size)
                break
            else:
                sub_reader.skip(sub_size)

        if name:
            self.result.npcs[form_id] = name

    def _parse_quest(self, reader: BinaryReader, form_id: int,
                     data_size: int, is_compressed: bool):
        """Parse QUST record to get quest names."""
        data = self._decompress_if_needed(reader, data_size, is_compressed)
        sub_reader = BinaryReader(self._bytes_to_stream(data))

        name = ''
        while sub_reader.tell() < len(data):
            sub_type = sub_reader.read(4)
            sub_size = sub_reader.read_uint16()

            if sub_type == b'FULL':
                name = self._read_localized_string(sub_reader, sub_size)
                break
            else:
                sub_reader.skip(sub_size)

        if name:
            self.result.quests[form_id] = name
    
    def _read_localized_string(self, reader: BinaryReader, size: int) -> str:
        """
        Read a string that may be localized.

        For localized plugins:
        - size is 4 bytes (uint32 string ID)
        - Look up in string tables

        For non-localized:
        - Read inline string as usual
        """
        if self._is_localized and size == 4:
            # Read string ID and look up
            string_id = reader.read_uint32()
            if self._string_tables:
                return self._string_tables.get(string_id, f"[{string_id}]")
            else:
                return f"[{string_id}]"  # No string tables loaded
        else:
            # Read inline string
            return reader.read_sized_string(size)

    def _resolve_references(self):
        """Resolve form ID references to names."""
        for topic in self.result.topics:
            if topic.quest_form_id and topic.quest_form_id in self.result.quests:
                pass  # Quest name would be resolved here
            
            for line in topic.lines:
                if line.speaker_form_id and line.speaker_form_id in self.result.npcs:
                    line.speaker_name = self.result.npcs[line.speaker_form_id]
                if line.quest_form_id and line.quest_form_id in self.result.quests:
                    line.quest_name = self.result.quests[line.quest_form_id]
    
    @staticmethod
    def _bytes_to_stream(data: bytes) -> BinaryIO:
        """Convert bytes to file-like object."""
        import io
        return io.BytesIO(data)


# =============================================================================
# Output Formatters
# =============================================================================

def export_to_json(result: ParsedPlugin, output_path: Path):
    """Export parsed data to JSON."""
    data = {
        'plugin': result.filename,
        'game': result.game_type,
        'dialogue': []
    }
    
    for line in result.all_lines():
        if line.text:  # Skip empty lines
            data['dialogue'].append({
                'form_id': hex(line.form_id),
                'topic': line.topic_name,
                'speaker': line.speaker_name,
                'text': line.text,
                'emotion': line.emotion_label(),
                'emotion_value': line.emotion_value,
                'quest': line.quest_name,
                'conditions': line.conditions,
                'script_notes': line.script_notes
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(data['dialogue'])} lines to {output_path}")


def export_to_csv(result: ParsedPlugin, output_path: Path):
    """Export parsed data to CSV for ML training."""
    fieldnames = [
        'form_id', 'topic', 'speaker', 'text', 
        'emotion', 'emotion_value', 'quest', 'script_notes'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        count = 0
        for line in result.all_lines():
            if line.text:
                writer.writerow({
                    'form_id': hex(line.form_id),
                    'topic': line.topic_name,
                    'speaker': line.speaker_name or '',
                    'text': line.text,
                    'emotion': line.emotion_label(),
                    'emotion_value': line.emotion_value,
                    'quest': line.quest_name or '',
                    'script_notes': line.script_notes or ''
                })
                count += 1
    
    print(f"Exported {count} lines to {output_path}")


def export_for_training(result: ParsedPlugin, output_path: Path):
    """
    Export in a format suitable for ML training:
    JSON Lines with structured context.
    
    Each line is a self-contained training example with:
    - Input context (speaker, situation, emotion target)
    - Output text
    - Metadata for filtering/stratification
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in result.all_lines():
            if not line.text or len(line.text) < 5:
                continue
            
            example = {
                'text': line.text,
                'speaker': line.speaker_name,
                'emotion': line.emotion_label(),
                'emotion_intensity': line.emotion_value / 100.0,
                'topic': line.topic_name,
                'quest_context': line.quest_name,
                'meta': {
                    'source': result.filename,
                    'game': result.game_type,
                    'form_id': hex(line.form_id)
                }
            }
            
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Exported training data to {output_path}")


# =============================================================================
# Statistics & Analysis
# =============================================================================

def print_statistics(result: ParsedPlugin):
    """Print summary statistics."""
    lines = list(result.all_lines())
    non_empty = [l for l in lines if l.text]
    
    print(f"\n{'='*60}")
    print(f"Plugin: {result.filename}")
    print(f"Game: {result.game_type}")
    print(f"{'='*60}")
    print(f"Topics: {len(result.topics)}")
    print(f"Dialogue lines: {len(non_empty)}")
    print(f"NPCs with names: {len(result.npcs)}")
    print(f"Quests: {len(result.quests)}")
    
    # Emotion distribution
    emotion_counts = {}
    for line in non_empty:
        label = line.emotion_label()
        emotion_counts[label] = emotion_counts.get(label, 0) + 1
    
    print(f"\nEmotion Distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(non_empty)
        print(f"  {emotion}: {count} ({pct:.1f}%)")
    
    # Text length stats
    lengths = [len(l.text) for l in non_empty]
    if lengths:
        print(f"\nText Length:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    
    # Sample lines
    print(f"\nSample Lines:")
    import random
    samples = random.sample(non_empty, min(5, len(non_empty)))
    for s in samples:
        emo = f"[{s.emotion_label()}]"
        speaker = s.speaker_name or "???"
        print(f"  {speaker} {emo}: {s.text[:80]}...")


# =============================================================================
# Main Entry Point
# =============================================================================

def find_plugins(path: Path) -> List[Path]:
    """Find all ESM/ESP files in a directory."""
    if path.is_file():
        return [path]
    
    plugins = []
    for ext in ['*.esm', '*.esp', '*.ESM', '*.ESP']:
        plugins.extend(path.glob(ext))
    
    return sorted(plugins)


def main():
    parser = argparse.ArgumentParser(
        description='Extract dialogue from Bethesda ESM/ESP plugins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/Oblivion/Data
  %(prog)s /path/to/FalloutNV/Data/FalloutNV.esm
  %(prog)s ~/.steam/steam/steamapps/common/Oblivion/Data --format training

Output formats:
  json     - Full structured data with all metadata
  csv      - Tabular format for analysis
  training - JSON Lines for ML training pipelines
        """
    )
    
    parser.add_argument('path', type=Path,
                        help='Path to Data folder or specific ESM/ESP file')
    parser.add_argument('--output', '-o', type=Path, default=Path('.'),
                        help='Output directory (default: current)')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'training', 'all'],
                        default='all', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress during parsing')
    parser.add_argument('--stats', '-s', action='store_true',
                        help='Print statistics after parsing')
    
    args = parser.parse_args()
    
    # Find plugins
    plugins = find_plugins(args.path)
    if not plugins:
        print(f"No ESM/ESP files found in {args.path}")
        return 1
    
    print(f"Found {len(plugins)} plugin(s)")
    
    # Parse each plugin
    for plugin_path in plugins:
        print(f"\nParsing: {plugin_path.name}")
        
        try:
            parser_obj = ESMParser(plugin_path, verbose=args.verbose)
            result = parser_obj.parse()
            
            if args.stats:
                print_statistics(result)
            
            # Export
            base_name = plugin_path.stem
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.format in ('json', 'all'):
                export_to_json(result, output_dir / f"{base_name}_dialogue.json")
            if args.format in ('csv', 'all'):
                export_to_csv(result, output_dir / f"{base_name}_dialogue.csv")
            if args.format in ('training', 'all'):
                export_for_training(result, output_dir / f"{base_name}_training.jsonl")
                
        except Exception as e:
            print(f"Error parsing {plugin_path.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    return 0


if __name__ == '__main__':
    exit(main())