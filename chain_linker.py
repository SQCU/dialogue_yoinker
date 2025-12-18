#!/usr/bin/env python3
"""
Chain Linker - Extract related dialogue sequences from parsed ESM data.

This tool groups dialogue lines that belong to the same conversation flows,
using conditions (CTDA) and topic/quest context to identify sequences.

The goal is to produce multi-turn dialogue samples that preserve the
conversational structure of the original game.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator, Tuple
import random


# =============================================================================
# Condition Function Indices (Bethesda's scripting system)
# =============================================================================

# Common condition functions relevant to dialogue flow
CONDITION_FUNCTIONS = {
    14: 'GetActorValue',
    38: 'GetQuestRunning',
    43: 'GetStage',          # Quest stage - key for sequencing
    47: 'GetIsReference',
    56: 'GetDisposition',
    59: 'GetQuest',
    71: 'GetIsClass',
    72: 'GetIsID',           # NPC identity check
    73: 'GetFactionRank',
    77: 'GetIsRace',
    79: 'GetStage',          # Another GetStage variant (FNV)
    99: 'GetIsCreature',
    117: 'GetInCell',
    136: 'GetIsReference',
    145: 'GetBaseActorValue',
    161: 'GetQuestCompleted',
    224: 'GetVMQuestVariable',
    226: 'GetVMScriptVariable',
    266: 'GetReputation',     # FNV specific
    267: 'GetReputationFriendly',  # FNV specific
}


@dataclass
class DialogueLine:
    """A dialogue line with parsed metadata."""
    form_id: str
    text: str
    speaker: Optional[str]
    emotion: str
    topic: str
    quest: Optional[str]
    conditions: List[Dict[str, Any]] = field(default_factory=list)

    def quest_stages(self) -> List[Tuple[int, float]]:
        """Extract quest stage conditions."""
        stages = []
        for cond in self.conditions:
            func = cond.get('function', 0)
            if func in (43, 79):  # GetStage variants
                quest_id = cond.get('param1', 0)
                stage = cond.get('value', 0)
                stages.append((quest_id, stage))
        return stages


@dataclass
class DialogueChain:
    """A sequence of related dialogue lines."""
    quest: Optional[str]
    speaker: Optional[str]
    topic: str
    lines: List[DialogueLine] = field(default_factory=list)

    def to_transcript(self) -> str:
        """Format as readable transcript."""
        parts = []
        if self.quest:
            parts.append(f"[Quest: {self.quest}]")
        if self.speaker:
            parts.append(f"[Speaker: {self.speaker}]")
        parts.append(f"[Topic: {self.topic}]")
        parts.append("")

        for line in self.lines:
            emo = f"({line.emotion})" if line.emotion != "neutral" else ""
            speaker = line.speaker or "NPC"
            parts.append(f"{speaker}{emo}: {line.text}")

        return "\n".join(parts)


class ChainLinker:
    """
    Links related dialogue lines into conversational chains.

    Strategy:
    1. Group by quest_context (most reliable linkage)
    2. Within quest, group by topic
    3. Sort by quest stage conditions (if present)
    4. Build chains that represent conversation flow
    """

    def __init__(self, dialogue_data: List[Dict[str, Any]]):
        self.lines = [self._parse_line(d) for d in dialogue_data]
        self._by_quest: Dict[str, List[DialogueLine]] = defaultdict(list)
        self._by_topic: Dict[str, List[DialogueLine]] = defaultdict(list)
        self._by_speaker: Dict[str, List[DialogueLine]] = defaultdict(list)

        self._index_lines()

    def _parse_line(self, data: Dict[str, Any]) -> DialogueLine:
        return DialogueLine(
            form_id=data.get('form_id', ''),
            text=data.get('text', ''),
            speaker=data.get('speaker'),
            emotion=data.get('emotion', 'neutral'),
            topic=data.get('topic', ''),
            quest=data.get('quest'),
            conditions=data.get('conditions', [])
        )

    def _index_lines(self):
        for line in self.lines:
            if line.quest:
                self._by_quest[line.quest].append(line)
            if line.topic:
                self._by_topic[line.topic].append(line)
            if line.speaker:
                self._by_speaker[line.speaker].append(line)

    def build_quest_chains(self) -> Iterator[DialogueChain]:
        """Build chains from quest-grouped dialogue."""
        for quest, lines in self._by_quest.items():
            # Group by topic within quest
            by_topic: Dict[str, List[DialogueLine]] = defaultdict(list)
            for line in lines:
                by_topic[line.topic].append(line)

            for topic, topic_lines in by_topic.items():
                if len(topic_lines) < 2:
                    continue

                # Sort by quest stage if available
                def stage_key(line: DialogueLine) -> float:
                    stages = line.quest_stages()
                    if stages:
                        return min(s[1] for s in stages)
                    return 0

                sorted_lines = sorted(topic_lines, key=stage_key)

                # Identify speaker (most common)
                speakers = [l.speaker for l in sorted_lines if l.speaker]
                speaker = max(set(speakers), key=speakers.count) if speakers else None

                yield DialogueChain(
                    quest=quest,
                    speaker=speaker,
                    topic=topic,
                    lines=sorted_lines
                )

    def build_speaker_chains(self) -> Iterator[DialogueChain]:
        """Build chains from speaker-grouped dialogue."""
        for speaker, lines in self._by_speaker.items():
            if len(lines) < 2:
                continue

            # Group by quest
            by_quest: Dict[str, List[DialogueLine]] = defaultdict(list)
            for line in lines:
                quest = line.quest or "_no_quest"
                by_quest[quest].append(line)

            for quest, quest_lines in by_quest.items():
                if len(quest_lines) < 2:
                    continue

                # Sort by topic to group related lines
                by_topic: Dict[str, List[DialogueLine]] = defaultdict(list)
                for line in quest_lines:
                    by_topic[line.topic].append(line)

                for topic, topic_lines in by_topic.items():
                    if len(topic_lines) >= 2:
                        yield DialogueChain(
                            quest=quest if quest != "_no_quest" else None,
                            speaker=speaker,
                            topic=topic,
                            lines=topic_lines
                        )

    def sample_chains(self, n: int = 10, min_length: int = 3) -> List[DialogueChain]:
        """Sample n chains with at least min_length lines."""
        all_chains = list(self.build_quest_chains())
        valid_chains = [c for c in all_chains if len(c.lines) >= min_length]

        if len(valid_chains) <= n:
            return valid_chains

        return random.sample(valid_chains, n)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dialogue corpus."""
        all_chains = list(self.build_quest_chains())

        return {
            'total_lines': len(self.lines),
            'unique_quests': len(self._by_quest),
            'unique_topics': len(self._by_topic),
            'unique_speakers': len(self._by_speaker),
            'total_chains': len(all_chains),
            'chains_by_length': {
                '2-3': len([c for c in all_chains if 2 <= len(c.lines) <= 3]),
                '4-6': len([c for c in all_chains if 4 <= len(c.lines) <= 6]),
                '7-10': len([c for c in all_chains if 7 <= len(c.lines) <= 10]),
                '10+': len([c for c in all_chains if len(c.lines) > 10]),
            },
            'lines_with_stages': len([l for l in self.lines if l.quest_stages()]),
        }


def load_dialogue(path: Path) -> List[Dict[str, Any]]:
    """Load dialogue from JSON export."""
    with open(path) as f:
        data = json.load(f)
    return data.get('dialogue', [])


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Link related dialogue into conversational chains'
    )
    parser.add_argument('input', type=Path,
                        help='Path to *_dialogue.json file')
    parser.add_argument('--sample', '-s', type=int, default=5,
                        help='Number of chains to sample')
    parser.add_argument('--min-length', '-m', type=int, default=3,
                        help='Minimum chain length')
    parser.add_argument('--stats', action='store_true',
                        help='Print statistics')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output file for sampled chains (JSONL)')

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.input}...")
    dialogue = load_dialogue(args.input)
    print(f"Loaded {len(dialogue)} lines")

    # Build linker
    linker = ChainLinker(dialogue)

    if args.stats:
        stats = linker.get_statistics()
        print(f"\n{'='*60}")
        print("Corpus Statistics")
        print(f"{'='*60}")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

    # Sample chains
    print(f"\n{'='*60}")
    print(f"Sampled Dialogue Chains (n={args.sample}, min_length={args.min_length})")
    print(f"{'='*60}")

    chains = linker.sample_chains(n=args.sample, min_length=args.min_length)

    for i, chain in enumerate(chains):
        print(f"\n--- Chain {i+1} ({len(chain.lines)} lines) ---")
        print(chain.to_transcript())

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            for chain in linker.build_quest_chains():
                if len(chain.lines) >= args.min_length:
                    record = {
                        'quest': chain.quest,
                        'speaker': chain.speaker,
                        'topic': chain.topic,
                        'lines': [
                            {
                                'text': l.text,
                                'speaker': l.speaker,
                                'emotion': l.emotion
                            }
                            for l in chain.lines
                        ]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"\nExported chains to {args.output}")


if __name__ == '__main__':
    main()
