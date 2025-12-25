"""
Pydantic Models for Synthetic Dialogue Pipeline

These models define the data structures that flow between:
- The dialogue graph API (source walks)
- The triplet extractor (structural parsing)
- The translation engine (prose generation)
- The lore curator (validation)
- The synthetic corpus (persistence)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# Enums - Controlled Vocabularies
# =============================================================================

class EmotionType(str, Enum):
    """Emotion types from TRDT subrecords (Bethesda facial animation tags)."""
    neutral = "neutral"
    happy = "happy"
    anger = "anger"
    sad = "sad"
    fear = "fear"
    surprise = "surprise"
    disgust = "disgust"
    pained = "pained"


class BeatFunction(str, Enum):
    """What a dialogue beat is DOING in the conversation."""
    establish_stakes = "establish_stakes"
    deliver_information = "deliver_information"
    negotiate = "negotiate"
    threaten = "threaten"
    plead = "plead"
    farewell = "farewell"
    react = "react"
    bark = "bark"
    query = "query"
    comply = "comply"
    refuse = "refuse"


class ArchetypeRelation(str, Enum):
    """The power/social dynamic between speaker and listener."""
    authority_to_subject = "authority_to_subject"
    subject_to_authority = "subject_to_authority"
    peer_to_peer = "peer_to_peer"
    merchant_to_customer = "merchant_to_customer"
    supplicant_to_power = "supplicant_to_power"
    power_to_supplicant = "power_to_supplicant"
    ally_to_ally = "ally_to_ally"
    enemy_to_enemy = "enemy_to_enemy"
    stranger_to_stranger = "stranger_to_stranger"


class BarrierType(str, Enum):
    """What stands between player and goal."""
    countdown = "countdown"
    confrontation = "confrontation"
    negotiation = "negotiation"
    investigation = "investigation"
    fetch = "fetch"
    escort = "escort"
    gatekeeping = "gatekeeping"
    revelation = "revelation"
    ambient = "ambient"
    unknown = "unknown"


class AttractorType(str, Enum):
    """What the player wants from this interaction."""
    survival = "survival"
    reward = "reward"
    information = "information"
    alliance = "alliance"
    revenge = "revenge"
    justice = "justice"
    power = "power"
    escape = "escape"


class ArcShape(str, Enum):
    """Overall pattern of the dialogue walk."""
    escalating_threat = "escalating_threat"
    de_escalation = "de_escalation"
    revelation_cascade = "revelation_cascade"
    negotiation_arc = "negotiation_arc"
    status_assertion = "status_assertion"
    plea_arc = "plea_arc"
    information_dump = "information_dump"
    ambient_chatter = "ambient_chatter"
    single_beat = "single_beat"


# =============================================================================
# Structural Models - Output of Triplet Extractor
# =============================================================================

class StructuralBeat(BaseModel):
    """A single beat in a dialogue arc."""
    beat: str = Field(description="Semantic label for this beat (e.g., 'ultimatum_initial')")
    text: str = Field(description="Original dialogue text")
    emotion: EmotionType = Field(description="Tagged emotion from source")
    function: BeatFunction = Field(description="What this beat DOES")
    archetype_relation: ArchetypeRelation = Field(description="Power dynamic")
    transition_from: Optional[EmotionType] = Field(
        default=None,
        description="Previous beat's emotion (None for first beat)"
    )
    source_game: Optional[str] = Field(default=None, description="For cross-game walks")


class StructuralTriplet(BaseModel):
    """
    The structural essence of a dialogue walk.

    This is what gets preserved during translation.
    The arc shape is sacred; the words serve the setting.
    """
    arc: list[StructuralBeat] = Field(description="Sequence of beats")
    proper_nouns_used: list[str] = Field(
        default_factory=list,
        description="Proper nouns found in source text"
    )
    barrier_type: BarrierType = Field(description="What blocks progress")
    attractor_type: AttractorType = Field(description="What player wants")
    arc_shape: ArcShape = Field(description="Overall pattern")
    source_walk_id: str = Field(default="", description="ID of source walk")
    source_game: str = Field(default="", description="Game the walk came from")


# =============================================================================
# Lore Bible Models - Managed by Curator
# =============================================================================

class ProperNounInstance(BaseModel):
    """A specific instance within a proper noun cluster."""
    instance: str = Field(description="The actual name/term")
    type: str = Field(description="Category: 'place', 'faction', 'person', etc.")
    usage: str = Field(description="When/how to use this instance")


class ProperNounCluster(BaseModel):
    """
    A cluster of related proper nouns.

    Proper nouns should cluster around meanings, not stand alone.
    E.g., 'Leclerc' -> [General, tank, grocery, prize]
    """
    model_config = {"extra": "ignore"}

    cluster_name: str = Field(description="Root name of cluster")
    instances: list[ProperNounInstance] = Field(
        default_factory=list,
        description="All instances in this cluster"
    )
    meaning_note: str = Field(description="What this cluster represents thematically")
    introduced_by_synthetic: Optional[str] = Field(
        default=None,
        description="ID of synthetic that introduced this cluster"
    )


class FactionTemplate(BaseModel):
    """A faction's structural identity for quest generation."""
    model_config = {"extra": "ignore", "populate_by_name": True}

    faction_id: str
    archetype: str = Field(description="Structural role: 'overextended_empire', 'desperate_outlaws', etc.")
    description: str = Field(default="")
    wants: str = Field(description="What this faction seeks")
    fears: str = Field(description="What this faction avoids/dreads")
    offers: str = Field(description="What this faction can provide")
    threatens: str = Field(default="", description="What this faction can harm")
    speech_register: str = Field(default="", alias="register", description="How this faction speaks")


class NarrativeTension(BaseModel):
    """A generative tension that drives stories."""
    model_config = {"extra": "ignore"}

    tension_id: str
    description: str
    manifests_as: list[str] = Field(default_factory=list)
    quest_shapes: list[str] = Field(default_factory=list)


class RevelationRule(BaseModel):
    """A rule for how information should be revealed to players."""
    rule: str
    example_good: str = Field(default="")
    example_bad: str = Field(default="")


class WorldLogic(BaseModel):
    """The deep structure of how a fictional world works."""
    terroir_equivalent: str = Field(default="", description="What makes places special")
    history_shape: str = Field(default="", description="How the past weighs on present")
    tone: str = Field(default="", description="Emotional/stylistic register")
    technology_level: str = Field(default="", description="What tools exist")
    mortality_rules: str = Field(default="", description="How death/consequence works")


class LoreBible(BaseModel):
    """
    Complete setting bible for synthetic generation.

    This defines everything a translator needs to write
    dialogue that sounds like it belongs in this world.
    """
    model_config = {"extra": "ignore"}  # Ignore extra YAML fields like game_mechanics

    bible_id: str
    setting_name: str
    tagline: str = Field(default="")
    world_logic: WorldLogic = Field(default_factory=WorldLogic)
    proper_noun_clusters: list[ProperNounCluster] = Field(default_factory=list)
    faction_templates: list[FactionTemplate] = Field(default_factory=list)
    narrative_tensions: list[NarrativeTension] = Field(default_factory=list)
    revelation_rules: list[RevelationRule] = Field(default_factory=list)

    def get_cluster(self, name: str) -> Optional[ProperNounCluster]:
        """Find a cluster by name."""
        for cluster in self.proper_noun_clusters:
            if cluster.cluster_name.lower() == name.lower():
                return cluster
        return None

    def get_faction(self, faction_id: str) -> Optional[FactionTemplate]:
        """Find a faction by ID."""
        for faction in self.faction_templates:
            if faction.faction_id == faction_id:
                return faction
        return None

    def all_proper_nouns(self) -> list[str]:
        """Get all proper noun instances across all clusters."""
        nouns = []
        for cluster in self.proper_noun_clusters:
            for inst in cluster.instances:
                nouns.append(inst.instance)
        return nouns


# =============================================================================
# Translation Models - Output of Translation Engine
# =============================================================================

class StructuralFidelity(BaseModel):
    """Self-reported structural match from translator."""
    emotion_arc_match: bool = True
    beat_count_match: bool = True
    archetype_preserved: bool = True


class TranslationResult(BaseModel):
    """Output of the translation engine."""
    translated_texts: list[str] = Field(description="One text per beat")
    proper_nouns_introduced: list[str] = Field(
        default_factory=list,
        description="New proper nouns used (need curator review)"
    )
    register_notes: str = Field(default="", description="Notes on tone/register")
    structural_fidelity: StructuralFidelity = Field(default_factory=StructuralFidelity)
    confidence: float = Field(ge=0.0, le=1.0, description="Self-rated confidence")


# =============================================================================
# Curator Models - Validation Decisions
# =============================================================================

class BibleAddition(BaseModel):
    """A proposed addition to a lore bible."""
    addition_type: str = Field(description="'proper_noun', 'faction', 'tension'")
    proposed_noun: Optional[str] = None
    proposed_cluster: Optional[str] = None
    proposed_meaning_update: Optional[str] = None
    context: str = Field(default="", description="Where this appeared")
    new_faction: Optional[FactionTemplate] = None
    new_tension: Optional[NarrativeTension] = None


class CuratorDecision(BaseModel):
    """
    The curator's decision on a proposed addition.

    The curator has VETO POWER - if approved is False,
    the synthetic must be revised or rejected.
    """
    approved: bool
    modified_addition: Optional[dict] = Field(
        default=None,
        description="The addition as approved (may differ from proposed)"
    )
    reasoning: str = Field(description="Why approved or rejected")
    suggested_alternatives: Optional[list[str]] = Field(
        default=None,
        description="Alternative proper nouns if rejected"
    )
    bible_update: Optional[dict] = Field(
        default=None,
        description="Changes to apply to bible if approved"
    )
    warnings: Optional[list[str]] = Field(
        default=None,
        description="Concerns even if approved"
    )


# =============================================================================
# Synthetic Entry - Final Output
# =============================================================================

class SyntheticEntry(BaseModel):
    """
    A completed synthetic dialogue entry.

    This is what gets persisted to the corpus.
    """
    synthetic_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    source_walk_id: str
    source_bible: str
    target_bible: str
    source_triplet: StructuralTriplet
    translated_texts: list[str]
    proper_nouns_introduced: list[str] = Field(default_factory=list)
    validation_score: float = Field(ge=0.0, le=1.0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="ID of workflow trace for debugging"
    )

    def to_training_format(self) -> list[dict]:
        """Convert to JSONL training format (one line per beat)."""
        entries = []
        for i, (beat, text) in enumerate(zip(
            self.source_triplet.arc, self.translated_texts
        )):
            entries.append({
                "text": text,
                "emotion": beat.emotion.value,
                "function": beat.function.value,
                "archetype": beat.archetype_relation.value,
                "beat_index": i,
                "arc_length": len(self.translated_texts),
                "arc_shape": self.source_triplet.arc_shape.value,
                "barrier": self.source_triplet.barrier_type.value,
                "attractor": self.source_triplet.attractor_type.value,
                "target_bible": self.target_bible,
                "synthetic_id": self.synthetic_id,
            })
        return entries


# =============================================================================
# Request/Response Models for API
# =============================================================================

class ExtractTripletRequest(BaseModel):
    """Request to extract triplet from a walk."""
    walk: list[dict] = Field(description="Dialogue nodes from graph API")
    reference_bible: str = Field(default="mojave", description="Source setting ID")


class TranslateTripletRequest(BaseModel):
    """Request to translate a triplet to a new setting."""
    triplet: StructuralTriplet
    source_bible: str
    target_bible: str
    few_shot_count: int = Field(default=3, ge=0, le=6)


class PersistSyntheticRequest(BaseModel):
    """Request to save a synthetic entry."""
    synthetic: SyntheticEntry


class ValidateSyntheticRequest(BaseModel):
    """Request to validate a synthetic entry."""
    synthetic_id: str
    checks: list[str] = Field(
        default=["emotion_arc", "proper_nouns", "register"],
        description="Which validations to run"
    )


class ProposeBibleAdditionRequest(BaseModel):
    """Request to propose an addition to a bible."""
    bible_id: str
    addition: BibleAddition
