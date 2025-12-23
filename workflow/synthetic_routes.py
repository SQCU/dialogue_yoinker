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
