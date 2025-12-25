"""
Subagent Orchestrator

Infrastructure for calling specialized Claude subagents with full observability.

Architecture:
    - observability.py: Logging, tracing, and persistence
    - subagent.py: Wrapper for Anthropic API calls with logging
    - models.py: Pydantic models for triplets, bibles, synthetics
    - validation.py: Schema validation against Pydantic models
"""

from .observability import (
    SubagentLog,
    WorkflowTrace,
    TraceStore,
)
from .subagent import SubagentCaller
from .models import (
    EmotionType,
    BeatFunction,
    ArchetypeRelation,
    StructuralBeat,
    StructuralTriplet,
    ProperNounCluster,
    FactionTemplate,
    LoreBible,
    SyntheticEntry,
    CuratorDecision,
    TranslationResult,
)
from .validation import (
    ValidationReport,
    ValidationStats,
    validate_subagent_output,
    validate_structural_triplet,
    validate_translation_result,
    validate_curator_decision,
)

__all__ = [
    # Observability
    "SubagentLog",
    "WorkflowTrace",
    "TraceStore",
    # Subagent caller
    "SubagentCaller",
    # Validation
    "ValidationReport",
    "ValidationStats",
    "validate_subagent_output",
    "validate_structural_triplet",
    "validate_translation_result",
    "validate_curator_decision",
    # Models
    "EmotionType",
    "BeatFunction",
    "ArchetypeRelation",
    "StructuralBeat",
    "StructuralTriplet",
    "ProperNounCluster",
    "FactionTemplate",
    "LoreBible",
    "SyntheticEntry",
    "CuratorDecision",
    "TranslationResult",
]
