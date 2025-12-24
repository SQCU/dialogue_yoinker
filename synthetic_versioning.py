"""
Synthetic Graph Versioning

Manages multiple versions/branches of synthetic dialogue graphs.
Supports:
- new_graph: Create fresh target graph
- extend_graph: Add to existing graph
- branch_graph: Copy existing, extend differently
- compare_graphs: Measure structural differences

Directory structure:
    synthetic/
      {setting}_v{N}/
        graph.json        # Node/edge data
        metadata.json     # Provenance, parameters
        dialogue.json     # Compiled dialogue format
        training.jsonl    # ML-ready format
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


SYNTHETIC_DIR = Path(__file__).parent / "synthetic"


@dataclass
class GraphVersion:
    """Metadata for a synthetic graph version."""
    setting: str
    version: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_version: Optional[int] = None  # For branches
    description: str = ""

    # Generation parameters
    approach: str = "unknown"  # "beat_translation", "pattern_instantiation", "topology_growth"
    source_games: List[str] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)  # Which runs contributed

    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    state_changes_targeted: List[str] = field(default_factory=list)

    def path(self) -> Path:
        return SYNTHETIC_DIR / f"{self.setting}_v{self.version}"


def list_versions(setting: str) -> List[GraphVersion]:
    """List all versions of a synthetic setting."""
    versions = []
    for path in SYNTHETIC_DIR.glob(f"{setting}_v*/metadata.json"):
        with open(path) as f:
            data = json.load(f)
        versions.append(GraphVersion(**data))
    return sorted(versions, key=lambda v: v.version)


def get_latest_version(setting: str) -> Optional[GraphVersion]:
    """Get the highest version number for a setting."""
    versions = list_versions(setting)
    return versions[-1] if versions else None


def new_graph(
    setting: str,
    description: str = "",
    approach: str = "unknown",
    source_games: List[str] = None,
) -> GraphVersion:
    """Create a new synthetic graph version."""
    latest = get_latest_version(setting)
    new_version = (latest.version + 1) if latest else 1

    version = GraphVersion(
        setting=setting,
        version=new_version,
        description=description,
        approach=approach,
        source_games=source_games or [],
    )

    # Create directory
    version.path().mkdir(parents=True, exist_ok=True)

    # Initialize empty graph
    graph_file = version.path() / "graph.json"
    graph_file.write_text(json.dumps({
        "nodes": [],
        "edges": [],
        "state_changes": [],
    }, indent=2))

    # Save metadata
    meta_file = version.path() / "metadata.json"
    meta_file.write_text(json.dumps(asdict(version), indent=2))

    return version


def branch_graph(
    setting: str,
    source_version: int,
    description: str = "",
) -> GraphVersion:
    """Create a new version by copying an existing one."""
    source = None
    for v in list_versions(setting):
        if v.version == source_version:
            source = v
            break

    if not source:
        raise ValueError(f"Version {source_version} not found for {setting}")

    # Create new version
    new_v = new_graph(
        setting=setting,
        description=description,
        approach=source.approach,
        source_games=source.source_games,
    )
    new_v.parent_version = source_version

    # Copy graph data
    shutil.copy(source.path() / "graph.json", new_v.path() / "graph.json")

    # Update metadata
    meta_file = new_v.path() / "metadata.json"
    meta_file.write_text(json.dumps(asdict(new_v), indent=2))

    return new_v


def extend_graph(
    version: GraphVersion,
    new_nodes: List[dict],
    new_edges: List[dict],
    run_id: str = None,
    state_changes: List[str] = None,
) -> GraphVersion:
    """Add nodes and edges to an existing graph version."""
    graph_file = version.path() / "graph.json"
    with open(graph_file) as f:
        graph = json.load(f)

    # Append new data
    graph["nodes"].extend(new_nodes)
    graph["edges"].extend(new_edges)
    if state_changes:
        graph["state_changes"].extend(state_changes)

    # Save updated graph
    graph_file.write_text(json.dumps(graph, indent=2))

    # Update metadata
    version.total_nodes = len(graph["nodes"])
    version.total_edges = len(graph["edges"])
    if run_id and run_id not in version.run_ids:
        version.run_ids.append(run_id)
    if state_changes:
        version.state_changes_targeted.extend(state_changes)

    meta_file = version.path() / "metadata.json"
    meta_file.write_text(json.dumps(asdict(version), indent=2))

    return version


def compare_graphs(v1: GraphVersion, v2: GraphVersion) -> dict:
    """Compare structural properties of two graph versions."""
    g1_file = v1.path() / "graph.json"
    g2_file = v2.path() / "graph.json"

    with open(g1_file) as f:
        g1 = json.load(f)
    with open(g2_file) as f:
        g2 = json.load(f)

    return {
        "v1": {"version": v1.version, "nodes": len(g1["nodes"]), "edges": len(g1["edges"])},
        "v2": {"version": v2.version, "nodes": len(g2["nodes"]), "edges": len(g2["edges"])},
        "node_difference": len(g2["nodes"]) - len(g1["nodes"]),
        "edge_difference": len(g2["edges"]) - len(g1["edges"]),
        # TODO: Add topology metrics (degree distribution, clustering, etc.)
    }


# =============================================================================
# Migration: Convert existing flat synthetic/ to versioned
# =============================================================================

def migrate_flat_to_versioned(setting: str) -> Optional[GraphVersion]:
    """
    Migrate existing flat synthetic files to v1 versioned structure.

    Moves:
        synthetic/{setting}_dialogue.json → synthetic/{setting}_v1/dialogue.json
        synthetic/{setting}_trajectories.json → synthetic/{setting}_v1/trajectories.json
        synthetic/{setting}_training.jsonl → synthetic/{setting}_v1/training.jsonl
    """
    dialogue_file = SYNTHETIC_DIR / f"{setting}_dialogue.json"
    if not dialogue_file.exists():
        return None

    # Check if already migrated
    if (SYNTHETIC_DIR / f"{setting}_v1").exists():
        print(f"{setting} already has v1, skipping migration")
        return None

    # Create v1
    version = new_graph(
        setting=setting,
        description="Migrated from flat structure (beat-for-beat translation)",
        approach="beat_translation",
    )

    # Move files
    for suffix in ["_dialogue.json", "_trajectories.json", "_training.jsonl"]:
        src = SYNTHETIC_DIR / f"{setting}{suffix}"
        if src.exists():
            dst = version.path() / suffix.lstrip("_")
            shutil.move(str(src), str(dst))
            print(f"  Moved {src.name} → {dst}")

    # Load dialogue to get stats
    dialogue_file = version.path() / "dialogue.json"
    if dialogue_file.exists():
        with open(dialogue_file) as f:
            data = json.load(f)
        version.total_nodes = data.get("total_beats", 0)
        version.source_games = data.get("source_games", [])

    # Update metadata
    meta_file = version.path() / "metadata.json"
    meta_file.write_text(json.dumps(asdict(version), indent=2))

    return version


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python synthetic_versioning.py <command> [args]")
        print("Commands:")
        print("  list <setting>           - List all versions")
        print("  new <setting> [desc]     - Create new version")
        print("  branch <setting> <v>     - Branch from version v")
        print("  migrate <setting>        - Migrate flat files to v1")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        setting = sys.argv[2]
        for v in list_versions(setting):
            parent = f" (from v{v.parent_version})" if v.parent_version else ""
            print(f"v{v.version}{parent}: {v.total_nodes} nodes, {v.approach}")
            if v.description:
                print(f"    {v.description}")

    elif cmd == "new":
        setting = sys.argv[2]
        desc = sys.argv[3] if len(sys.argv) > 3 else ""
        v = new_graph(setting, description=desc)
        print(f"Created {setting}_v{v.version}")

    elif cmd == "branch":
        setting = sys.argv[2]
        source_v = int(sys.argv[3])
        v = branch_graph(setting, source_v)
        print(f"Created {setting}_v{v.version} (branched from v{source_v})")

    elif cmd == "migrate":
        setting = sys.argv[2]
        v = migrate_flat_to_versioned(setting)
        if v:
            print(f"Migrated {setting} to v{v.version}")
        else:
            print(f"Nothing to migrate for {setting}")
