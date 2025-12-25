"""
Synthetic Dialogue API Routes

Provides API endpoints for exploring synthetic dialogue data generated
through setting transposition. Mirrors the reference data API structure
but serves from synthetic/ directory.

Routes:
    GET  /api/synthetic/settings           - List available synthetic settings
    GET  /api/synthetic/{setting}/stats    - Statistics for a synthetic setting
    GET  /api/synthetic/{setting}/dialogue - Raw dialogue entries
    GET  /api/synthetic/{setting}/trajectories - Full trajectory view with arcs
    POST /api/synthetic/{setting}/sample   - Sample from synthetic data
    GET  /api/synthetic/{setting}/concept-mappings - View concept transformation patterns

Stats-Guided Growth Routes:
    GET  /api/synthetic/reference/stats    - Reference corpus statistics
    GET  /api/synthetic/{setting}/versions - List synthetic graph versions
    GET  /api/synthetic/{setting}/gaps     - Identify statistical gaps to close
    POST /api/synthetic/{setting}/grow     - Grow graph toward reference stats
"""

import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

SYNTHETIC_DIR = Path(__file__).parent.parent / "synthetic"

# Caches
_DIALOGUE_CACHE: Dict[str, dict] = {}
_TRAJECTORY_CACHE: Dict[str, dict] = {}


# =============================================================================
# Pydantic Models
# =============================================================================

class SyntheticSettingInfo(BaseModel):
    """Information about an available synthetic setting."""
    setting: str = Field(description="Setting identifier (e.g., 'gallia')")
    total_beats: int = Field(description="Number of dialogue beats")
    total_trajectories: int = Field(description="Number of translation trajectories")
    source_games: List[str] = Field(description="Source games used")
    file_path: str = Field(description="Path to dialogue JSON file")


class SyntheticStats(BaseModel):
    """Statistics about a synthetic dataset."""
    setting: str
    total_beats: int
    total_trajectories: int
    source_games: List[str]
    emotion_distribution: Dict[str, int]
    arc_shapes: Dict[str, int]
    avg_confidence: float
    total_concept_mappings: int
    unique_source_concepts: int


class SyntheticDialogueEntry(BaseModel):
    """A single synthetic dialogue entry."""
    form_id: str
    text: str
    emotion: str
    topic: str  # arc_shape
    source_game: str
    source_text: str
    beat_function: str
    archetype_relation: str
    confidence: float


class SyntheticTrajectory(BaseModel):
    """A complete translation trajectory."""
    ticket_id: str
    source_game: str
    target_setting: str
    confidence: float
    arc: Dict[str, Any]
    beats: List[Dict[str, Any]]
    concept_mappings: List[Dict[str, Any]]
    register_notes: str


class SyntheticSampleRequest(BaseModel):
    """Request to sample from synthetic data."""
    setting: str = Field(description="Synthetic setting to sample from")
    method: str = Field(default="random", description="'random', 'by_arc', 'by_emotion'")
    count: int = Field(default=5, ge=1, le=50)
    arc_filter: Optional[str] = Field(default=None, description="Filter by arc shape")
    emotion_filter: Optional[str] = Field(default=None, description="Filter by emotion")


class ConceptMappingEntry(BaseModel):
    """A concept mapping from source to target setting."""
    source: str
    target: str
    rationale: str
    occurrence_count: int


# =============================================================================
# Data Loading
# =============================================================================

def get_available_settings() -> List[SyntheticSettingInfo]:
    """Find all compiled synthetic dialogue files."""
    settings = []
    if not SYNTHETIC_DIR.exists():
        return settings

    for path in SYNTHETIC_DIR.glob("*_dialogue.json"):
        setting_name = path.stem.replace("_dialogue", "")
        try:
            with open(path) as f:
                data = json.load(f)

            # Also load trajectories for trajectory count
            traj_path = SYNTHETIC_DIR / f"{setting_name}_trajectories.json"
            traj_count = 0
            source_games = []
            if traj_path.exists():
                with open(traj_path) as f:
                    traj_data = json.load(f)
                traj_count = traj_data.get("total_trajectories", 0)
                source_games = traj_data.get("source_games", [])

            settings.append(SyntheticSettingInfo(
                setting=setting_name,
                total_beats=data.get("total_beats", len(data.get("dialogue", []))),
                total_trajectories=traj_count,
                source_games=source_games,
                file_path=str(path),
            ))
        except Exception:
            pass

    return settings


def load_dialogue(setting: str) -> dict:
    """Load dialogue data with caching."""
    if setting in _DIALOGUE_CACHE:
        return _DIALOGUE_CACHE[setting]

    path = SYNTHETIC_DIR / f"{setting}_dialogue.json"
    if not path.exists():
        raise HTTPException(404, f"Synthetic setting '{setting}' not found.")

    with open(path) as f:
        data = json.load(f)

    _DIALOGUE_CACHE[setting] = data
    return data


def load_trajectories(setting: str) -> dict:
    """Load trajectory data with caching."""
    if setting in _TRAJECTORY_CACHE:
        return _TRAJECTORY_CACHE[setting]

    path = SYNTHETIC_DIR / f"{setting}_trajectories.json"
    if not path.exists():
        raise HTTPException(404, f"Trajectories for '{setting}' not found.")

    with open(path) as f:
        data = json.load(f)

    _TRAJECTORY_CACHE[setting] = data
    return data


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/synthetic", tags=["synthetic"])


@router.get("/settings", response_model=List[SyntheticSettingInfo])
async def list_settings():
    """
    List available synthetic settings.

    Returns information about each compiled synthetic dialogue dataset.
    """
    return get_available_settings()


@router.get("/{setting}/stats", response_model=SyntheticStats)
async def get_stats(setting: str):
    """
    Get statistics for a synthetic setting.

    Returns emotion distribution, arc shapes, confidence metrics, and
    concept mapping statistics.
    """
    traj_data = load_trajectories(setting)
    dialogue_data = load_dialogue(setting)

    stats = traj_data.get("statistics", {})

    return SyntheticStats(
        setting=setting,
        total_beats=dialogue_data.get("total_beats", 0),
        total_trajectories=traj_data.get("total_trajectories", 0),
        source_games=traj_data.get("source_games", []),
        emotion_distribution=stats.get("emotion_distribution", {}),
        arc_shapes=stats.get("arc_shapes", {}),
        avg_confidence=stats.get("avg_confidence", 0.0),
        total_concept_mappings=stats.get("total_concept_mappings", 0),
        unique_source_concepts=stats.get("unique_concept_mappings", 0),
    )


@router.get("/{setting}/dialogue")
async def get_dialogue(
    setting: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    emotion: Optional[str] = Query(default=None),
    arc_shape: Optional[str] = Query(default=None),
):
    """
    Get raw dialogue entries.

    Supports filtering by emotion and arc shape, with pagination.
    """
    data = load_dialogue(setting)
    dialogue = data.get("dialogue", [])

    # Apply filters
    if emotion:
        dialogue = [d for d in dialogue if d.get("emotion") == emotion]
    if arc_shape:
        dialogue = [d for d in dialogue if d.get("topic") == arc_shape]

    total = len(dialogue)
    dialogue = dialogue[offset:offset + limit]

    return {
        "setting": setting,
        "total": total,
        "offset": offset,
        "limit": limit,
        "dialogue": dialogue,
    }


@router.get("/{setting}/trajectories")
async def get_trajectories(
    setting: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    arc_shape: Optional[str] = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """
    Get translation trajectories with full structural information.

    Each trajectory includes the arc shape, beats with source/target text,
    concept mappings, and register transformation notes.
    """
    data = load_trajectories(setting)
    trajectories = data.get("trajectories", [])

    # Apply filters
    if arc_shape:
        trajectories = [t for t in trajectories
                       if t.get("arc", {}).get("shape") == arc_shape]
    if min_confidence > 0:
        trajectories = [t for t in trajectories
                       if t.get("confidence", 0) >= min_confidence]

    total = len(trajectories)
    trajectories = trajectories[offset:offset + limit]

    return {
        "setting": setting,
        "total": total,
        "offset": offset,
        "limit": limit,
        "trajectories": trajectories,
    }


@router.post("/{setting}/sample")
async def sample_dialogue(setting: str, request: SyntheticSampleRequest):
    """
    Sample dialogue from synthetic data.

    Methods:
    - 'random': Random sampling across all dialogue
    - 'by_arc': Sample complete trajectories by arc shape
    - 'by_emotion': Sample beats with specific emotion
    """
    if request.method == "by_arc":
        data = load_trajectories(setting)
        trajectories = data.get("trajectories", [])

        if request.arc_filter:
            trajectories = [t for t in trajectories
                          if t.get("arc", {}).get("shape") == request.arc_filter]

        if not trajectories:
            return {"setting": setting, "samples": [], "method": request.method}

        samples = random.sample(trajectories, min(request.count, len(trajectories)))
        return {"setting": setting, "samples": samples, "method": request.method}

    else:
        # Random or by_emotion sampling at beat level
        data = load_dialogue(setting)
        dialogue = data.get("dialogue", [])

        if request.emotion_filter:
            dialogue = [d for d in dialogue if d.get("emotion") == request.emotion_filter]

        if not dialogue:
            return {"setting": setting, "samples": [], "method": request.method}

        samples = random.sample(dialogue, min(request.count, len(dialogue)))
        return {"setting": setting, "samples": samples, "method": request.method}


@router.get("/{setting}/concept-mappings")
async def get_concept_mappings(
    setting: str,
    min_occurrences: int = Query(default=1, ge=1),
):
    """
    Get concept mappings used in translations.

    Shows how source setting concepts were mapped to target setting concepts,
    with rationales and occurrence counts.
    """
    data = load_trajectories(setting)
    trajectories = data.get("trajectories", [])

    # Aggregate all mappings
    mapping_counts: Dict[str, dict] = {}

    for traj in trajectories:
        for mapping in traj.get("concept_mappings", []):
            source = mapping.get("source", "")
            target = mapping.get("target", "")
            rationale = mapping.get("rationale", "")
            key = f"{source}→{target}"

            if key not in mapping_counts:
                mapping_counts[key] = {
                    "source": source,
                    "target": target,
                    "rationale": rationale,
                    "count": 0,
                }
            mapping_counts[key]["count"] += 1

    # Filter by min occurrences and sort by count
    mappings = [m for m in mapping_counts.values() if m["count"] >= min_occurrences]
    mappings.sort(key=lambda m: m["count"], reverse=True)

    return {
        "setting": setting,
        "total_unique_mappings": len(mapping_counts),
        "mappings": mappings,
    }


@router.get("/{setting}/compare/{source_game}")
async def compare_source_target(
    setting: str,
    source_game: str,
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Side-by-side comparison of source and target text.

    Shows original game text alongside the transposed synthetic text
    for a specific source game.
    """
    data = load_trajectories(setting)
    trajectories = data.get("trajectories", [])

    # Filter to source game
    trajectories = [t for t in trajectories
                   if t.get("source_game") == source_game]

    if not trajectories:
        raise HTTPException(404, f"No trajectories from '{source_game}' found.")

    # Build comparison pairs
    comparisons = []
    for traj in trajectories[:limit]:
        for beat in traj.get("beats", []):
            comparisons.append({
                "source_text": beat.get("source_text", ""),
                "target_text": beat.get("target_text", ""),
                "emotion": beat.get("emotion", ""),
                "function": beat.get("function", ""),
                "archetype": beat.get("archetype_relation", ""),
                "arc_shape": traj.get("arc", {}).get("shape", ""),
            })

    return {
        "setting": setting,
        "source_game": source_game,
        "total_comparisons": len(comparisons),
        "comparisons": comparisons[:limit * 3],  # ~3 beats per trajectory
    }


# =============================================================================
# Stats-Guided Growth Routes
# =============================================================================

# Lazy-loaded reference corpus
_REFERENCE_CORPUS = None


def get_reference_corpus():
    """Lazy-load the reference corpus."""
    global _REFERENCE_CORPUS
    if _REFERENCE_CORPUS is None:
        from stats_guided_growth import ReferenceCorpus
        _REFERENCE_CORPUS = ReferenceCorpus()
        _REFERENCE_CORPUS.load()
    return _REFERENCE_CORPUS


class ReferenceStatsResponse(BaseModel):
    """Response for reference corpus statistics."""
    emotion_self_loop_rate: float
    top_emotion_transitions: List[Dict[str, Any]]
    top_arc_shapes: List[Dict[str, Any]]
    emotion_distribution: Dict[str, float]


class VersionInfo(BaseModel):
    """Information about a synthetic graph version."""
    version: int
    created_at: str
    approach: str
    total_nodes: int
    total_edges: int
    description: str
    parent_version: Optional[int] = None


class GapInfo(BaseModel):
    """Information about a statistical gap."""
    stat_type: str
    key: str
    reference_value: float
    target_value: float
    gap_size: float


class GrowRequest(BaseModel):
    """Request to grow a synthetic graph."""
    version: Optional[int] = Field(default=None, description="Version to extend (creates new if None)")
    target_size: int = Field(default=50, ge=10, le=500)
    max_iterations: int = Field(default=20, ge=1, le=100)


class GrowResponse(BaseModel):
    """Response from growth operation."""
    version: int
    initial_nodes: int
    final_nodes: int
    initial_edges: int
    final_edges: int
    iterations: int
    remaining_gaps: List[str]


@router.get("/reference/stats", response_model=ReferenceStatsResponse)
async def get_reference_stats():
    """
    Get statistics from the reference corpus.

    Shows the statistical distribution that stats-guided growth tries to match:
    - Emotion transition probabilities
    - Arc shape frequencies
    - Overall emotion distribution
    """
    corpus = get_reference_corpus()
    stats = corpus.compute_stats().normalize()

    # Format transitions
    sorted_trans = sorted(stats.emotion_transitions.items(), key=lambda x: -x[1])
    top_trans = [
        {"from": t[0], "to": t[1], "probability": v}
        for (t, v) in sorted_trans[:15]
    ]

    # Format arc shapes
    sorted_shapes = sorted(stats.arc_shapes.items(), key=lambda x: -x[1])
    top_shapes = [
        {"shape": s, "probability": v}
        for (s, v) in sorted_shapes[:15]
    ]

    return ReferenceStatsResponse(
        emotion_self_loop_rate=stats.emotion_self_loop_rate,
        top_emotion_transitions=top_trans,
        top_arc_shapes=top_shapes,
        emotion_distribution=stats.emotion_counts,
    )


@router.get("/{setting}/versions", response_model=List[VersionInfo])
async def list_versions(setting: str):
    """
    List all versions of a synthetic setting's graph.

    Shows version history with node/edge counts and creation info.
    """
    from synthetic_versioning import list_versions as sv_list_versions

    versions = sv_list_versions(setting)

    return [
        VersionInfo(
            version=v.version,
            created_at=v.created_at,
            approach=v.approach,
            total_nodes=v.total_nodes,
            total_edges=v.total_edges,
            description=v.description,
            parent_version=v.parent_version,
        )
        for v in versions
    ]


@router.get("/{setting}/gaps", response_model=List[GapInfo])
async def get_gaps(
    setting: str,
    version: int = Query(default=None, description="Version to analyze (latest if None)"),
    top_n: int = Query(default=10, ge=1, le=50),
):
    """
    Identify statistical gaps between target and reference.

    Shows which statistics are underrepresented in the target graph
    compared to the reference corpus.
    """
    from synthetic_versioning import get_latest_version, list_versions as sv_list_versions
    from stats_guided_growth import StatsGuidedGrowth

    # Get version
    if version:
        versions = sv_list_versions(setting)
        target_v = None
        for v in versions:
            if v.version == version:
                target_v = v
                break
        if not target_v:
            raise HTTPException(404, f"Version {version} not found for {setting}")
    else:
        target_v = get_latest_version(setting)
        if not target_v:
            raise HTTPException(404, f"No versions found for {setting}")

    corpus = get_reference_corpus()
    grower = StatsGuidedGrowth(corpus, target_v)
    gaps = grower.identify_gaps(top_n=top_n)

    return [
        GapInfo(
            stat_type=g.stat_type,
            key=str(g.key),
            reference_value=g.reference_value,
            target_value=g.target_value,
            gap_size=g.gap_size,
        )
        for g in gaps
    ]


@router.post("/{setting}/grow", response_model=GrowResponse)
async def grow_graph(setting: str, request: GrowRequest):
    """
    Grow a synthetic graph toward reference statistics.

    Samples from reference corpus to close statistical gaps,
    producing a graph that is statistically similar to reference
    but topologically different.
    """
    from synthetic_versioning import new_graph, get_latest_version, list_versions as sv_list_versions
    from stats_guided_growth import StatsGuidedGrowth

    # Get or create version
    if request.version:
        versions = sv_list_versions(setting)
        target_v = None
        for v in versions:
            if v.version == request.version:
                target_v = v
                break
        if not target_v:
            raise HTTPException(404, f"Version {request.version} not found for {setting}")
    else:
        target_v = new_graph(
            setting=setting,
            description=f"Stats-guided growth ({request.target_size} nodes target)",
            approach="stats_guided_growth",
        )

    corpus = get_reference_corpus()
    grower = StatsGuidedGrowth(corpus, target_v)

    report = grower.grow(
        target_size=request.target_size,
        max_iterations=request.max_iterations,
    )

    return GrowResponse(
        version=target_v.version,
        initial_nodes=report['initial_nodes'],
        final_nodes=report['final_nodes'],
        initial_edges=report['initial_edges'],
        final_edges=report['final_edges'],
        iterations=len(report['iterations']),
        remaining_gaps=report['gaps_remaining'][:5],
    )


@router.get("/{setting}/v{version}/graph")
async def get_graph(setting: str, version: int):
    """
    Get the raw graph data for a specific version.

    Returns nodes and edges of the synthetic graph.
    """
    graph_path = SYNTHETIC_DIR / f"{setting}_v{version}" / "graph.json"

    if not graph_path.exists():
        raise HTTPException(404, f"Graph not found: {setting}_v{version}")

    with open(graph_path) as f:
        data = json.load(f)

    return {
        "setting": setting,
        "version": version,
        "total_nodes": len(data.get("nodes", [])),
        "total_edges": len(data.get("edges", [])),
        "nodes": data.get("nodes", []),
        "edges": data.get("edges", []),
    }


# =============================================================================
# Growth Engine Routes
# =============================================================================

class GrowthStepRequest(BaseModel):
    """Request for a single growth step."""
    version: Optional[int] = Field(default=None, description="Version to extend")


class GrowthStepResponse(BaseModel):
    """Response from a growth step."""
    run_id: str
    version: int
    gap_targeted: str
    walk_sampled: Dict[str, Any]
    nodes_added: int
    edges_added: int
    translation_request_path: Optional[str] = None


@router.post("/{setting}/growth/step", response_model=GrowthStepResponse)
async def growth_step(setting: str, request: GrowthStepRequest):
    """
    Perform one growth step: sample a walk to close a gap.

    This creates a translation request but does NOT translate yet.
    Use /growth/translate to translate the pending walk.

    Returns the sampled walk info and translation request path.
    """
    from growth_engine import GrowthEngine

    engine = GrowthEngine(setting=setting, version=request.version)

    # Find gaps
    gaps = engine.identify_gaps(top_n=3)
    if not gaps:
        raise HTTPException(400, "No significant gaps remaining")

    # Sample walk for a gap
    gap = random.choices(gaps, weights=[g.gap_size for g in gaps])[0]
    walks = engine.sample_for_gaps([gap], walks_per_gap=1)

    if not walks:
        raise HTTPException(400, f"No walks available for gap: {gap}")

    walk = walks[0]
    engine.run.gaps_targeted.append(str(gap))

    # Create translation request (don't translate yet)
    triplet = {
        "arc_shape": engine._classify_arc_shape(walk.get('emotions', [])),
        "emotions": walk.get('emotions', []),
        "source_texts": walk.get('texts', []),
        "source_game": walk.get('game', 'unknown'),
        "gap_targeted": str(gap),
    }

    # Load bible
    bible_path = Path("bibles") / f"{setting}.yaml"
    bible_content = bible_path.read_text() if bible_path.exists() else ""

    # Save translation request
    request_data = {
        "triplet": triplet,
        "bible_excerpt": bible_content[:4000],
        "pending": True,
    }
    request_path = engine.run.path() / "translations" / f"request_{len(engine.run.translated_walks)}.json"
    request_path.write_text(json.dumps(request_data, indent=2))

    # Store walk in run state
    engine.run.pending_translation.append({
        "walk": walk,
        "gap": str(gap),
        "request_path": str(request_path),
    })

    # Save run state
    state = {
        'run_id': engine.run.run_id,
        'setting': setting,
        'target_version': engine.target_version.version,
        'pending_translation': engine.run.pending_translation,
        'gaps_targeted': engine.run.gaps_targeted,
    }
    (engine.run.path() / "growth_state.json").write_text(json.dumps(state, indent=2))

    return GrowthStepResponse(
        run_id=engine.run.run_id,
        version=engine.target_version.version,
        gap_targeted=str(gap),
        walk_sampled={
            "arc_shape": triplet["arc_shape"],
            "emotions": triplet["emotions"],
            "source_game": triplet["source_game"],
            "beat_count": len(triplet["source_texts"]),
        },
        nodes_added=0,  # Not added yet
        edges_added=0,
        translation_request_path=str(request_path),
    )


class TranslateRequest(BaseModel):
    """Request to translate pending walks."""
    run_id: str
    translated_texts: List[str] = Field(description="Translated texts for pending walk")
    confidence: float = Field(default=0.8)


@router.post("/{setting}/growth/translate")
async def apply_translation(setting: str, request: TranslateRequest):
    """
    Apply translation results and add walk to graph.

    Call this after using the translation-engine agent to translate
    the pending walk from /growth/step.
    """
    from growth_engine import GrowthEngine
    from synthetic_versioning import list_versions as sv_list_versions

    # Load run state
    run_path = Path("runs") / request.run_id
    if not run_path.exists():
        raise HTTPException(404, f"Run not found: {request.run_id}")

    state_file = run_path / "growth_state.json"
    if not state_file.exists():
        raise HTTPException(400, "No growth state found")

    state = json.loads(state_file.read_text())
    pending = state.get('pending_translation', [])

    if not pending:
        raise HTTPException(400, "No pending translations")

    # Get the pending walk
    pending_item = pending.pop(0)
    walk = pending_item['walk']
    gap = pending_item['gap']

    # Recreate engine (will create new run, but we'll use existing version)
    versions = sv_list_versions(setting)
    target_v = None
    for v in versions:
        if v.version == state.get('target_version'):
            target_v = v
            break

    if not target_v:
        raise HTTPException(404, f"Version {state.get('target_version')} not found")

    engine = GrowthEngine(setting=setting, version=target_v.version)

    # Build translated walk
    translated = {
        'arc_shape': engine._classify_arc_shape(walk.get('emotions', [])),
        'emotions': walk.get('emotions', []),
        'source_texts': walk.get('texts', []),
        'translated_texts': request.translated_texts,
        'source_game': walk.get('game', 'unknown'),
        'gap_targeted': gap,
        'confidence': request.confidence,
    }

    # Find attachment point
    attachment = engine.find_attachment_point(walk)

    # Add to graph
    result = engine.add_translated_walk(translated, attachment)

    # Save
    engine.save()

    # Update original run state
    state['pending_translation'] = pending
    state_file.write_text(json.dumps(state, indent=2))

    return {
        "run_id": request.run_id,
        "nodes_added": result['nodes_added'],
        "edges_added": result['edges_added'],
        "total_nodes": len(engine.nodes),
        "total_edges": len(engine.edges),
        "pending_remaining": len(pending),
    }


@router.get("/{setting}/growth/pending")
async def get_pending_translations(setting: str, run_id: str):
    """
    Get pending translation requests for a growth run.

    Returns the structural triplets that need translation.
    """
    run_path = Path("runs") / run_id

    if not run_path.exists():
        raise HTTPException(404, f"Run not found: {run_id}")

    state_file = run_path / "growth_state.json"
    if not state_file.exists():
        raise HTTPException(400, "No growth state found")

    state = json.loads(state_file.read_text())
    pending = state.get('pending_translation', [])

    # Load request details
    requests = []
    for item in pending:
        request_path = item.get('request_path')
        if request_path and Path(request_path).exists():
            request_data = json.loads(Path(request_path).read_text())
            requests.append({
                "gap": item.get('gap'),
                "triplet": request_data.get('triplet'),
                "request_path": request_path,
            })

    return {
        "run_id": run_id,
        "setting": setting,
        "pending_count": len(requests),
        "requests": requests,
    }
