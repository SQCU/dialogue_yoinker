"""
API Routes for Synthetic Dialogue Pipeline

Extends the existing dialogue graph API with:
- /api/extract/triplet - Structure extraction
- /api/bibles - Lore bible management
- /api/translate/triplet - Dialogue translation
- /api/synthetic - Synthetic corpus management
- /api/traces - Observability/debugging
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from .observability import (
    SubagentLog,
    WorkflowTrace,
    WorkflowStatus,
    TraceStore,
    SubagentType,
)
from .subagent import SubagentCaller
from .models import (
    StructuralTriplet,
    LoreBible,
    SyntheticEntry,
    TranslationResult,
    CuratorDecision,
    ExtractTripletRequest,
    TranslateTripletRequest,
    PersistSyntheticRequest,
    ProposeBibleAdditionRequest,
)

# =============================================================================
# Configuration
# =============================================================================

BIBLES_DIR = Path("bibles")
SYNTHETIC_DIR = Path("synthetic")
TRACES_DIR = Path("traces")
PROMPTS_DIR = Path("claudefiles/subagents")

# Create directories if needed
BIBLES_DIR.mkdir(exist_ok=True)
SYNTHETIC_DIR.mkdir(exist_ok=True)
TRACES_DIR.mkdir(exist_ok=True)

# Shared instances
_trace_store: Optional[TraceStore] = None
_subagent_caller: Optional[SubagentCaller] = None


def get_trace_store() -> TraceStore:
    global _trace_store
    if _trace_store is None:
        _trace_store = TraceStore(TRACES_DIR)
    return _trace_store


def get_caller() -> SubagentCaller:
    global _subagent_caller
    if _subagent_caller is None:
        _subagent_caller = SubagentCaller(
            prompts_dir=PROMPTS_DIR,
            traces_dir=TRACES_DIR,
        )
    return _subagent_caller


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api", tags=["synthetic"])


# =============================================================================
# Response Models
# =============================================================================

class ExtractTripletResponse(BaseModel):
    """Response from triplet extraction."""
    success: bool
    triplet: Optional[dict] = None
    log_id: str
    latency_ms: int
    model: str
    errors: list[str] = Field(default_factory=list)


class TranslateTripletResponse(BaseModel):
    """Response from translation."""
    success: bool
    translation: Optional[dict] = None
    log_id: str
    latency_ms: int
    model: str
    errors: list[str] = Field(default_factory=list)
    proper_nouns_needing_review: list[str] = Field(default_factory=list)


class BibleInfo(BaseModel):
    """Summary info about a bible."""
    bible_id: str
    setting_name: str
    tagline: str
    cluster_count: int
    faction_count: int
    tension_count: int


class BibleListResponse(BaseModel):
    """List of available bibles."""
    bibles: list[BibleInfo]


class SyntheticStats(BaseModel):
    """Stats for synthetic corpus."""
    target_bible: str
    total_entries: int
    total_beats: int
    by_arc_shape: dict[str, int]
    by_emotion: dict[str, int]


class TraceListResponse(BaseModel):
    """List of workflow traces."""
    traces: list[dict]
    total: int


class TraceStatsResponse(BaseModel):
    """Aggregate trace statistics."""
    stats: dict


# =============================================================================
# Triplet Extraction Endpoints
# =============================================================================

@router.post("/extract/triplet", response_model=ExtractTripletResponse)
async def extract_triplet(request: ExtractTripletRequest):
    """
    Extract structural triplet from a dialogue walk.

    Calls the TRIPLET_EXTRACTOR subagent (Haiku) to parse
    structural arc, proper nouns, barrier/attractor types.
    """
    caller = get_caller()

    try:
        log = caller.call_triplet_extractor(
            walk=request.walk,
            reference_bible=request.reference_bible,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API call failed: {e}")

    return ExtractTripletResponse(
        success=log.parse_success,
        triplet=log.output_parsed,
        log_id=log.log_id,
        latency_ms=log.latency_ms,
        model=log.model,
        errors=log.parse_errors,
    )


# =============================================================================
# Bible Management Endpoints
# =============================================================================

def _load_bible(bible_id: str) -> tuple[LoreBible, str]:
    """Load a bible from YAML file, return (parsed, raw_yaml)."""
    path = BIBLES_DIR / f"{bible_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Bible not found: {bible_id}")

    raw = path.read_text()
    try:
        data = yaml.safe_load(raw)
        # Handle flat YAML structure
        if "bible_id" not in data:
            data["bible_id"] = bible_id
        bible = LoreBible(**data)
        return bible, raw
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse bible: {e}")


def _save_bible(bible: LoreBible) -> None:
    """Save bible to YAML file."""
    path = BIBLES_DIR / f"{bible.bible_id}.yaml"
    # Convert to dict and dump as YAML
    data = bible.model_dump(exclude_none=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


@router.get("/bibles", response_model=BibleListResponse)
async def list_bibles():
    """List all available lore bibles."""
    bibles = []
    for path in BIBLES_DIR.glob("*.yaml"):
        if path.name.startswith("few_shot"):
            continue  # Skip example files
        try:
            bible, _ = _load_bible(path.stem)
            bibles.append(BibleInfo(
                bible_id=bible.bible_id,
                setting_name=bible.setting_name,
                tagline=bible.tagline,
                cluster_count=len(bible.proper_noun_clusters),
                faction_count=len(bible.faction_templates),
                tension_count=len(bible.narrative_tensions),
            ))
        except Exception:
            pass  # Skip invalid files

    return BibleListResponse(bibles=bibles)


@router.get("/bibles/{bible_id}")
async def get_bible(bible_id: str, format: str = Query("json", enum=["json", "yaml"])):
    """Get a specific bible."""
    bible, raw = _load_bible(bible_id)
    if format == "yaml":
        return {"bible_id": bible_id, "yaml": raw}
    return bible.model_dump()


@router.post("/bibles")
async def create_bible(bible: LoreBible):
    """Create a new lore bible."""
    path = BIBLES_DIR / f"{bible.bible_id}.yaml"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Bible already exists: {bible.bible_id}")

    _save_bible(bible)
    return {"created": bible.bible_id}


@router.put("/bibles/{bible_id}")
async def update_bible(bible_id: str, bible: LoreBible):
    """Update an existing bible."""
    path = BIBLES_DIR / f"{bible_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Bible not found: {bible_id}")

    bible.bible_id = bible_id  # Ensure ID matches
    _save_bible(bible)
    return {"updated": bible_id}


@router.post("/bibles/{bible_id}/propose_addition")
async def propose_bible_addition(bible_id: str, request: ProposeBibleAdditionRequest):
    """
    Propose an addition to a lore bible.

    Calls the LORE_CURATOR subagent (Opus) to validate the addition
    against existing bible content.
    """
    bible, raw_yaml = _load_bible(bible_id)
    caller = get_caller()

    try:
        log = caller.call_lore_curator(
            proposal_type=request.addition.addition_type,
            proposal=request.addition.model_dump(exclude_none=True),
            bible_content=raw_yaml,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Curator call failed: {e}")

    return {
        "success": log.parse_success,
        "decision": log.output_parsed,
        "log_id": log.log_id,
        "latency_ms": log.latency_ms,
        "errors": log.parse_errors,
    }


# =============================================================================
# Translation Endpoints
# =============================================================================

@router.post("/translate/triplet", response_model=TranslateTripletResponse)
async def translate_triplet(request: TranslateTripletRequest):
    """
    Translate a structural triplet to a new setting.

    Calls the TRANSLATION_ENGINE subagent (Sonnet) to generate
    new prose that preserves structure but matches target setting.
    """
    # Load target bible
    try:
        _, target_yaml = _load_bible(request.target_bible)
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail=f"Target bible not found: {request.target_bible}"
        )

    # Load few-shot examples if available
    few_shot_path = BIBLES_DIR / "few_shot_translations.yaml"
    few_shot_examples = None
    if few_shot_path.exists() and request.few_shot_count > 0:
        try:
            examples_data = yaml.safe_load(few_shot_path.read_text())
            # Extract a subset of examples
            all_examples = [v for k, v in examples_data.items() if k.startswith("example_")]
            few_shot_examples = all_examples[:request.few_shot_count]
        except Exception:
            pass

    caller = get_caller()

    try:
        log = caller.call_translation_engine(
            triplet=request.triplet.model_dump(),
            source_bible=request.source_bible,
            target_bible=request.target_bible,
            target_bible_content=target_yaml,
            few_shot_examples=few_shot_examples,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation call failed: {e}")

    # Check for new proper nouns that need curator review
    nouns_needing_review = []
    if log.output_parsed and "proper_nouns_introduced" in log.output_parsed:
        nouns_needing_review = log.output_parsed["proper_nouns_introduced"]

    return TranslateTripletResponse(
        success=log.parse_success,
        translation=log.output_parsed,
        log_id=log.log_id,
        latency_ms=log.latency_ms,
        model=log.model,
        errors=log.parse_errors,
        proper_nouns_needing_review=nouns_needing_review,
    )


# =============================================================================
# Synthetic Persistence Endpoints
# =============================================================================

def _get_synthetic_file(target_bible: str) -> Path:
    """Get path to synthetic corpus file."""
    target_dir = SYNTHETIC_DIR / target_bible
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / "corpus.jsonl"


@router.post("/synthetic/persist")
async def persist_synthetic(request: PersistSyntheticRequest):
    """Save a synthetic entry to the corpus."""
    synthetic = request.synthetic
    filepath = _get_synthetic_file(synthetic.target_bible)

    # Append to JSONL
    with open(filepath, "a") as f:
        f.write(json.dumps(synthetic.model_dump()) + "\n")

    return {
        "persisted": synthetic.synthetic_id,
        "target_bible": synthetic.target_bible,
        "filepath": str(filepath),
    }


@router.get("/synthetic/split/{target_bible}")
async def get_synthetic_split(
    target_bible: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Retrieve synthetics for a target setting."""
    filepath = _get_synthetic_file(target_bible)
    if not filepath.exists():
        return {"entries": [], "total": 0}

    entries = []
    total = 0
    with open(filepath) as f:
        for i, line in enumerate(f):
            total += 1
            if i >= offset and len(entries) < limit:
                entries.append(json.loads(line))

    return {"entries": entries, "total": total, "offset": offset}


@router.get("/synthetic/stats/{target_bible}", response_model=SyntheticStats)
async def get_synthetic_stats(target_bible: str):
    """Get statistics for synthetic corpus."""
    filepath = _get_synthetic_file(target_bible)
    if not filepath.exists():
        return SyntheticStats(
            target_bible=target_bible,
            total_entries=0,
            total_beats=0,
            by_arc_shape={},
            by_emotion={},
        )

    total_entries = 0
    total_beats = 0
    by_arc_shape: dict[str, int] = {}
    by_emotion: dict[str, int] = {}

    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            total_entries += 1
            texts = entry.get("translated_texts", [])
            total_beats += len(texts)

            # Count arc shapes
            triplet = entry.get("source_triplet", {})
            shape = triplet.get("arc_shape", "unknown")
            by_arc_shape[shape] = by_arc_shape.get(shape, 0) + 1

            # Count emotions in arc
            for beat in triplet.get("arc", []):
                emotion = beat.get("emotion", "unknown")
                by_emotion[emotion] = by_emotion.get(emotion, 0) + 1

    return SyntheticStats(
        target_bible=target_bible,
        total_entries=total_entries,
        total_beats=total_beats,
        by_arc_shape=by_arc_shape,
        by_emotion=by_emotion,
    )


@router.get("/synthetic/training/{target_bible}")
async def get_training_data(
    target_bible: str,
    limit: int = Query(1000, ge=1, le=10000),
):
    """
    Export synthetics in ML-ready training format.

    Returns one JSON object per beat (not per walk).
    """
    filepath = _get_synthetic_file(target_bible)
    if not filepath.exists():
        return {"entries": [], "total": 0}

    training_entries = []
    with open(filepath) as f:
        for line in f:
            entry = SyntheticEntry(**json.loads(line))
            training_entries.extend(entry.to_training_format())
            if len(training_entries) >= limit:
                break

    return {
        "entries": training_entries[:limit],
        "total": len(training_entries),
        "format": "per_beat",
    }


# =============================================================================
# Trace/Observability Endpoints
# =============================================================================

@router.get("/traces", response_model=TraceListResponse)
async def list_traces(
    limit: int = Query(50, ge=1, le=500),
    status: Optional[str] = Query(None, enum=["completed", "failed", "rejected", "in_progress"]),
):
    """List recent workflow traces."""
    store = get_trace_store()

    if status == "failed" or status == "rejected":
        traces = store.list_failures(limit)
    else:
        traces = store.list_recent(limit)

    # Filter by status if specified
    if status:
        traces = [t for t in traces if t.status.value == status]

    return TraceListResponse(
        traces=[t.to_dict() for t in traces[:limit]],
        total=len(traces),
    )


@router.get("/traces/stats", response_model=TraceStatsResponse)
async def get_trace_stats():
    """Get aggregate statistics across all traces."""
    store = get_trace_store()
    return TraceStatsResponse(stats=store.stats())


@router.get("/traces/{workflow_id}")
async def get_trace(workflow_id: str):
    """Get a specific workflow trace."""
    store = get_trace_store()
    trace = store.load(workflow_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {workflow_id}")
    return trace.to_dict()


# =============================================================================
# Workflow Endpoint - Full Pipeline
# =============================================================================

class WorkflowRequest(BaseModel):
    """Request to run full synthetic generation workflow."""
    walk: list[dict] = Field(description="Source dialogue walk")
    source_bible: str = Field(default="mojave")
    target_bible: str = Field(default="gallia")
    persist: bool = Field(default=True, description="Whether to persist result")
    validate_nouns: bool = Field(default=True, description="Whether to validate new nouns with curator")


class WorkflowResponse(BaseModel):
    """Response from workflow execution."""
    workflow_id: str
    status: str
    triplet: Optional[dict] = None
    translation: Optional[dict] = None
    synthetic_id: Optional[str] = None
    total_latency_ms: int
    total_cost_usd: float
    errors: list[str] = Field(default_factory=list)


@router.post("/workflow/generate", response_model=WorkflowResponse)
async def run_generation_workflow(request: WorkflowRequest):
    """
    Run the full synthetic generation workflow.

    1. Extract triplet from walk (Haiku)
    2. Translate to target setting (Sonnet)
    3. Optionally validate new nouns (Opus)
    4. Persist if successful
    """
    store = get_trace_store()
    caller = get_caller()

    # Initialize workflow trace
    trace = WorkflowTrace(
        source_game=request.walk[0].get("game", "unknown") if request.walk else "unknown",
        source_walk={"walk": request.walk},
        target_bible=request.target_bible,
    )

    errors = []

    # Step 1: Extract triplet
    try:
        extractor_log = caller.call_triplet_extractor(
            walk=request.walk,
            reference_bible=request.source_bible,
        )
        trace.add_extractor_log(extractor_log)

        if not extractor_log.parse_success:
            trace.fail("extraction", f"Parse failed: {extractor_log.parse_errors}")
            store.save(trace)
            return WorkflowResponse(
                workflow_id=trace.workflow_id,
                status=trace.status.value,
                total_latency_ms=trace.total_latency_ms,
                total_cost_usd=trace.total_cost_usd,
                errors=extractor_log.parse_errors,
            )
    except Exception as e:
        trace.fail("extraction", str(e))
        store.save(trace)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # Step 2: Translate
    try:
        _, target_yaml = _load_bible(request.target_bible)

        # Load few-shot examples
        few_shot_path = BIBLES_DIR / "few_shot_translations.yaml"
        few_shot_examples = None
        if few_shot_path.exists():
            try:
                examples_data = yaml.safe_load(few_shot_path.read_text())
                all_examples = [v for k, v in examples_data.items() if k.startswith("example_")]
                few_shot_examples = all_examples[:3]
            except Exception:
                pass

        translator_log = caller.call_translation_engine(
            triplet=trace.triplet,
            source_bible=request.source_bible,
            target_bible=request.target_bible,
            target_bible_content=target_yaml,
            few_shot_examples=few_shot_examples,
        )
        trace.add_translator_log(translator_log)

        if not translator_log.parse_success:
            trace.fail("translation", f"Parse failed: {translator_log.parse_errors}")
            store.save(trace)
            return WorkflowResponse(
                workflow_id=trace.workflow_id,
                status=trace.status.value,
                triplet=trace.triplet,
                total_latency_ms=trace.total_latency_ms,
                total_cost_usd=trace.total_cost_usd,
                errors=translator_log.parse_errors,
            )
    except HTTPException:
        raise
    except Exception as e:
        trace.fail("translation", str(e))
        store.save(trace)
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    # Step 3: Validate new nouns (optional)
    translation = translator_log.output_parsed
    new_nouns = translation.get("proper_nouns_introduced", [])

    if request.validate_nouns and new_nouns:
        try:
            for noun in new_nouns:
                curator_log = caller.call_lore_curator(
                    proposal_type="proper_noun",
                    proposal={
                        "proposed_noun": noun,
                        "context": f"Appeared in translated dialogue",
                    },
                    bible_content=target_yaml,
                )
                trace.add_curator_log(curator_log)

                if curator_log.parse_success:
                    decision = curator_log.output_parsed
                    if not decision.get("approved", False):
                        errors.append(f"Curator rejected noun '{noun}': {decision.get('reasoning', 'no reason')}")
        except Exception as e:
            errors.append(f"Curator validation failed: {e}")

    # Step 4: Persist if requested
    synthetic_id = None
    if request.persist and not errors:
        try:
            # Build triplet model
            triplet = StructuralTriplet(**trace.triplet)

            synthetic = SyntheticEntry(
                source_walk_id=trace.workflow_id,
                source_bible=request.source_bible,
                target_bible=request.target_bible,
                source_triplet=triplet,
                translated_texts=translation.get("translated_texts", []),
                proper_nouns_introduced=new_nouns,
                validation_score=translation.get("confidence", 0.5),
                workflow_id=trace.workflow_id,
            )

            filepath = _get_synthetic_file(request.target_bible)
            with open(filepath, "a") as f:
                f.write(json.dumps(synthetic.model_dump()) + "\n")

            synthetic_id = synthetic.synthetic_id
            trace.complete(synthetic.model_dump())
        except Exception as e:
            trace.fail("persistence", str(e))
            errors.append(f"Persistence failed: {e}")

    if not trace.completed_at:
        if errors:
            trace.fail("validation", "; ".join(errors))
        else:
            trace.complete()

    store.save(trace)

    return WorkflowResponse(
        workflow_id=trace.workflow_id,
        status=trace.status.value,
        triplet=trace.triplet,
        translation=translation,
        synthetic_id=synthetic_id,
        total_latency_ms=trace.total_latency_ms,
        total_cost_usd=trace.total_cost_usd,
        errors=errors,
    )
