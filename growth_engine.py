#!/usr/bin/env python3
"""
Combined Growth Engine

Mixes two approaches for expanding synthetic graphs:
1. Stats-guided sampling: Find reference walks that close statistical gaps
2. Translation dispatch: Translate sampled walks to target setting

Directory structure:
    runs/{run_id}/              - Raw data (source text, intermediate states)
        growth_state.json       - Current growth state with raw source refs
        source_reference.json   - Hash→source mapping (NOT shareable)
        translations/           - Individual translation results

    synthetic/{setting}_v{N}/   - Shareable (hashed source refs, translated text)
        graph.json              - Translated nodes/edges
        metadata.json           - Provenance
"""

import json
import uuid
import random
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import hashlib

from stats_guided_growth import ReferenceCorpus, ReferenceStats, StatisticalGap
from synthetic_versioning import GraphVersion, new_graph, get_latest_version, SYNTHETIC_DIR
from compile_synthetic import SourceReferenceTable


RUNS_DIR = Path(__file__).parent / "runs"


@dataclass
class GrowthRun:
    """Tracks a growth run with its raw data."""
    run_id: str
    setting: str
    target_version: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Growth state
    sampled_walks: List[dict] = field(default_factory=list)
    translated_walks: List[dict] = field(default_factory=list)
    pending_translation: List[dict] = field(default_factory=list)

    # Statistics
    gaps_targeted: List[str] = field(default_factory=list)
    total_nodes_added: int = 0
    total_edges_added: int = 0

    def path(self) -> Path:
        return RUNS_DIR / self.run_id


@dataclass
class GrowthNode:
    """A node in the growing synthetic graph."""
    id: str
    text: str  # Translated text (goes to synthetic/)
    emotion: str
    source_ref: str  # Hash reference (goes to synthetic/)
    source_text: str  # Raw source (stays in runs/)
    source_game: str
    beat_function: str = ""
    archetype_relation: str = ""
    arc_shape: str = ""


@dataclass
class GrowthEdge:
    """An edge in the growing synthetic graph."""
    source: str
    target: str
    edge_type: str = "dialogue"
    gap_targeted: str = ""


class GrowthEngine:
    """
    Orchestrates stats-guided growth with translation.

    Workflow:
    1. Identify statistical gaps between target and reference
    2. Sample reference walks that would close gaps
    3. Dispatch walks to translation engine
    4. Attach translated walks to target graph
    5. Compile to shareable format (hashed source refs)
    """

    def __init__(
        self,
        setting: str,
        version: Optional[int] = None,
        bible_path: Optional[Path] = None,
    ):
        self.setting = setting
        self.bible_path = bible_path or Path("bibles") / f"{setting}.md"

        # Load or create target version
        if version:
            from synthetic_versioning import list_versions
            versions = list_versions(setting)
            self.target_version = None
            for v in versions:
                if v.version == version:
                    self.target_version = v
                    break
            if not self.target_version:
                raise ValueError(f"Version {version} not found for {setting}")
        else:
            self.target_version = new_graph(
                setting=setting,
                description="Stats-guided growth with translation",
                approach="growth_engine",
            )

        # Create run for this growth session
        self.run = self._create_run()

        # Initialize reference corpus (lazy loaded)
        self._reference: Optional[ReferenceCorpus] = None
        self._ref_stats: Optional[ReferenceStats] = None

        # Source reference table for hashing
        self.ref_table = SourceReferenceTable()

        # Load existing target graph
        self._load_target_graph()

    def _create_run(self) -> GrowthRun:
        """Create a new growth run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"growth_{timestamp}_{self.setting}_v{self.target_version.version}"

        run = GrowthRun(
            run_id=run_id,
            setting=self.setting,
            target_version=self.target_version.version,
        )

        run.path().mkdir(parents=True, exist_ok=True)
        (run.path() / "translations").mkdir(exist_ok=True)

        return run

    def _load_target_graph(self):
        """Load existing target graph from synthetic/."""
        graph_file = self.target_version.path() / "graph.json"
        if graph_file.exists():
            with open(graph_file) as f:
                data = json.load(f)
            self.nodes: List[dict] = data.get('nodes', [])
            self.edges: List[dict] = data.get('edges', [])
        else:
            self.nodes = []
            self.edges = []

    @property
    def reference(self) -> ReferenceCorpus:
        """Lazy-load reference corpus."""
        if self._reference is None:
            self._reference = ReferenceCorpus()
            self._reference.load()
        return self._reference

    @property
    def ref_stats(self) -> ReferenceStats:
        """Lazy-load reference statistics."""
        if self._ref_stats is None:
            self._ref_stats = self.reference.compute_stats().normalize()
        return self._ref_stats

    def identify_gaps(self, top_n: int = 5) -> List[StatisticalGap]:
        """Find statistical gaps between target and reference."""
        from stats_guided_growth import StatsGuidedGrowth

        grower = StatsGuidedGrowth(self.reference, self.target_version)
        return grower.identify_gaps(top_n=top_n)

    def sample_for_gaps(self, gaps: List[StatisticalGap], walks_per_gap: int = 2) -> List[dict]:
        """Sample reference walks that would help close gaps."""
        walks = []
        for gap in gaps:
            gap_walks = self.reference.sample_walks_for_gap(gap, n=walks_per_gap)
            for walk in gap_walks:
                walk['gap_targeted'] = str(gap)
            walks.extend(gap_walks)
        return walks

    def translate_walk(self, walk: dict, use_agent: bool = False) -> Optional[dict]:
        """
        Translate a reference walk to target setting.

        Args:
            walk: Reference walk with source texts and emotions
            use_agent: If True, dispatches to translation-engine agent.
                       If False, generates placeholder (faster for testing).

        Returns translated walk or None if translation fails.
        """
        # Build structural triplet from walk
        arc_shape = self._classify_arc_shape(walk.get('emotions', []))
        triplet = {
            "arc_shape": arc_shape,
            "emotions": walk.get('emotions', []),
            "source_texts": walk.get('texts', []),
            "source_game": walk.get('game', 'unknown'),
            "gap_targeted": walk.get('gap_targeted', ''),
        }

        # Load bible
        bible_content = ""
        if self.bible_path.exists():
            bible_content = self.bible_path.read_text()

        if use_agent and bible_content:
            # Build translation request for agent
            translated_texts = self._dispatch_translation(triplet, bible_content)
            confidence = 0.8 if translated_texts else 0.0
        else:
            # Generate placeholder translations (for testing)
            translated_texts = []
            for i, text in enumerate(triplet['source_texts']):
                translated_texts.append(f"[UNTRANSLATED: {triplet['source_game']}] {text[:100]}")
            confidence = 0.1  # Low confidence for placeholders

        if not translated_texts:
            return None

        return {
            'arc_shape': triplet['arc_shape'],
            'emotions': triplet['emotions'],
            'source_texts': triplet['source_texts'],
            'translated_texts': translated_texts,
            'source_game': triplet['source_game'],
            'gap_targeted': triplet['gap_targeted'],
            'confidence': confidence,
        }

    def _dispatch_translation(self, triplet: dict, bible_content: str) -> List[str]:
        """
        Dispatch translation to external agent.

        For CLI usage, writes a translation request file that can be
        processed by the translation-engine agent via Task tool.
        """
        # Create translation request file for agent processing
        request = {
            "arc_shape": triplet['arc_shape'],
            "emotions": triplet['emotions'],
            "source_texts": triplet['source_texts'],
            "source_game": triplet['source_game'],
            "target_setting": self.setting,
            "bible_excerpt": bible_content[:3000],  # Truncate for context limits
        }

        # Save request for agent
        request_file = self.run.path() / "translations" / f"request_{len(self.run.translated_walks)}.json"
        request_file.write_text(json.dumps(request, indent=2))

        # For now, return empty - actual agent dispatch happens via API
        # The API route will handle calling the agent and updating the file
        return []

    def _classify_arc_shape(self, emotions: List[str]) -> str:
        """Classify emotion sequence into arc shape."""
        if len(emotions) < 2:
            return "single"

        unique = list(dict.fromkeys(emotions))

        if len(unique) == 1:
            return f"flat_{unique[0]}"

        if unique[0] == 'neutral' and unique[-1] == 'neutral':
            if len(unique) > 2:
                return f"neutral_peak_{unique[len(unique)//2]}"
            return "neutral_flat"

        if unique[0] == unique[-1]:
            return f"return_to_{unique[0]}"

        return f"transition_{unique[0]}_to_{unique[-1]}"

    def add_translated_walk(
        self,
        translated: dict,
        attachment_point: Optional[str] = None,
    ) -> dict:
        """
        Add a translated walk to the target graph.

        Stores raw source in runs/, hashed refs in synthetic/.
        """
        new_nodes = []
        new_edges = []
        prev_id = attachment_point

        source_texts = translated.get('source_texts', [])
        translated_texts = translated.get('translated_texts', [])
        emotions = translated.get('emotions', [])

        for i, (src_text, tgt_text) in enumerate(zip(source_texts, translated_texts)):
            emotion = emotions[i] if i < len(emotions) else 'neutral'

            # Hash source text
            source_ref = self.ref_table.hash_text(
                src_text,
                game=translated.get('source_game', ''),
                metadata={'arc_shape': translated.get('arc_shape', '')}
            )

            # Generate node ID
            node_id = f"syn_{hashlib.sha256(f'{tgt_text}_{i}_{random.random()}'.encode()).hexdigest()[:12]}"

            node = {
                'id': node_id,
                'text': tgt_text,  # Translated (shareable)
                'emotion': emotion,
                'source_ref': source_ref,  # Hash (shareable)
                'source_game': translated.get('source_game', ''),
                'arc_shape': translated.get('arc_shape', ''),
                'gap_targeted': translated.get('gap_targeted', ''),
            }
            new_nodes.append(node)

            # Store raw source in run (not shareable)
            self.run.sampled_walks.append({
                'node_id': node_id,
                'source_text': src_text,
                'source_ref': source_ref,
            })

            if prev_id:
                new_edges.append({
                    'source': prev_id,
                    'target': node_id,
                    'type': 'dialogue',
                })

            prev_id = node_id

        # Add to graph
        self.nodes.extend(new_nodes)
        self.edges.extend(new_edges)

        self.run.total_nodes_added += len(new_nodes)
        self.run.total_edges_added += len(new_edges)
        self.run.translated_walks.append(translated)

        return {
            'nodes_added': len(new_nodes),
            'edges_added': len(new_edges),
        }

    def find_attachment_point(self, walk: dict) -> Optional[str]:
        """Find a good attachment point for a new walk."""
        if not self.nodes:
            return None

        walk_start_emotion = walk.get('emotions', ['neutral'])[0]

        # Prefer nodes where attaching creates needed transitions
        candidates = []
        for node in self.nodes:
            node_emo = node.get('emotion', 'neutral')
            # Score based on gap the transition would help close
            candidates.append({
                'id': node['id'],
                'emotion': node_emo,
                'score': 1.0 if node_emo != walk_start_emotion else 0.5,
            })

        if candidates:
            candidates.sort(key=lambda c: c['score'], reverse=True)
            return candidates[0]['id']

        return None

    def grow_step(self) -> dict:
        """
        Perform one growth step:
        1. Identify gaps
        2. Sample walk to close a gap
        3. Translate walk
        4. Attach to graph
        """
        # Find gaps
        gaps = self.identify_gaps(top_n=3)
        if not gaps:
            return {'status': 'no_gaps', 'nodes_added': 0}

        # Sample walk for random gap (weighted by gap size)
        gap = random.choices(gaps, weights=[g.gap_size for g in gaps])[0]
        walks = self.sample_for_gaps([gap], walks_per_gap=1)

        if not walks:
            return {'status': 'no_walks', 'gap': str(gap), 'nodes_added': 0}

        walk = walks[0]
        self.run.gaps_targeted.append(str(gap))

        # Translate
        translated = self.translate_walk(walk)
        if not translated:
            return {'status': 'translation_failed', 'gap': str(gap), 'nodes_added': 0}

        # Find attachment point
        attachment = self.find_attachment_point(walk)

        # Add to graph
        result = self.add_translated_walk(translated, attachment)

        return {
            'status': 'success',
            'gap': str(gap),
            'nodes_added': result['nodes_added'],
            'edges_added': result['edges_added'],
            'attachment': attachment,
        }

    def grow(self, target_size: int = 50, max_iterations: int = 30) -> dict:
        """
        Grow the graph toward target size.

        Returns growth report.
        """
        report = {
            'run_id': self.run.run_id,
            'initial_nodes': len(self.nodes),
            'initial_edges': len(self.edges),
            'iterations': [],
        }

        for i in range(max_iterations):
            if len(self.nodes) >= target_size:
                break

            result = self.grow_step()
            report['iterations'].append({
                'iteration': i,
                **result,
            })

            if result['status'] == 'success':
                print(f"[{i+1}] Added {result['nodes_added']} nodes for gap: {result['gap'][:50]}...")
            elif result['status'] == 'no_gaps':
                print(f"[{i+1}] No significant gaps remaining")
                break

        # Save state
        self.save()

        report['final_nodes'] = len(self.nodes)
        report['final_edges'] = len(self.edges)
        report['gaps_remaining'] = [str(g) for g in self.identify_gaps()]

        return report

    def save(self):
        """Save growth state to runs/ and compiled graph to synthetic/."""
        # Save raw state to runs/
        state = {
            'run_id': self.run.run_id,
            'setting': self.setting,
            'target_version': self.target_version.version,
            'created_at': self.run.created_at,
            'sampled_walks': self.run.sampled_walks,
            'gaps_targeted': self.run.gaps_targeted,
            'total_nodes_added': self.run.total_nodes_added,
            'total_edges_added': self.run.total_edges_added,
        }
        (self.run.path() / "growth_state.json").write_text(json.dumps(state, indent=2))

        # Save source reference table (NOT shareable)
        (self.run.path() / "source_reference.json").write_text(
            json.dumps(self.ref_table.to_dict(), indent=2)
        )

        # Save compiled graph to synthetic/ (shareable - no raw source text)
        graph_data = {
            'nodes': self.nodes,
            'edges': self.edges,
            'state_changes': [],
        }
        (self.target_version.path() / "graph.json").write_text(
            json.dumps(graph_data, indent=2)
        )

        # Update version metadata
        self.target_version.total_nodes = len(self.nodes)
        self.target_version.total_edges = len(self.edges)
        self.target_version.run_ids.append(self.run.run_id)

        meta_file = self.target_version.path() / "metadata.json"
        meta_file.write_text(json.dumps(asdict(self.target_version), indent=2))

        print(f"\nSaved:")
        print(f"  runs/{self.run.run_id}/growth_state.json (raw data)")
        print(f"  runs/{self.run.run_id}/source_reference.json (NOT shareable)")
        print(f"  synthetic/{self.setting}_v{self.target_version.version}/graph.json (shareable)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stats-guided growth with translation')
    parser.add_argument('--setting', type=str, default='gallia')
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--target-size', type=int, default=30)
    parser.add_argument('--max-iterations', type=int, default=20)

    args = parser.parse_args()

    print(f"Starting growth engine for {args.setting}")

    engine = GrowthEngine(
        setting=args.setting,
        version=args.version,
    )

    print(f"Target version: {args.setting}_v{engine.target_version.version}")
    print(f"Run ID: {engine.run.run_id}")
    print(f"Initial graph: {len(engine.nodes)} nodes, {len(engine.edges)} edges")

    # Show initial gaps
    print("\nInitial gaps:")
    for gap in engine.identify_gaps(top_n=5):
        print(f"  {gap}")

    # Grow
    print(f"\nGrowing to {args.target_size} nodes...")
    report = engine.grow(target_size=args.target_size, max_iterations=args.max_iterations)

    print(f"\nGrowth complete:")
    print(f"  {report['initial_nodes']} -> {report['final_nodes']} nodes")
    print(f"  {report['initial_edges']} -> {report['final_edges']} edges")
    print(f"  {len(report['iterations'])} iterations")


if __name__ == '__main__':
    main()
