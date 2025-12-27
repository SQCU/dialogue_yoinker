#!/usr/bin/env python3
"""
Hermeneutic Loop: Bidirectional Bible Enrichment During Translation

Implements the circular process of reading, interpreting, and enriching understanding.
Translation is literary criticism: finding analogies between source and target
necessarily produces annotations on BOTH.

Architecture:
- RunState: Global mutable state within an API run
- ProposedAddition: Queue of pending bible enrichments
- Systolic curator ticks: Periodic validation and merge to hot bible
- Sigmoid warmup: Explore (serial, low concurrency) → Exploit (parallel, high concurrency)

This should have been `from stanford_nlp import HermeneuticLoop` but here we are.
"""

import asyncio
import json
import hashlib
from math import exp
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import yaml


# =============================================================================
# Schemas
# =============================================================================

class AdditionType(str, Enum):
    """What kind of bible enrichment is this?"""
    # Target enrichments (new stuff for synthetic setting)
    PROPER_NOUN = "proper_noun"           # New name/place/thing
    FACTION = "faction"                    # New group with wants/fears
    TENSION = "tension"                    # New narrative conflict
    LOCATION = "location"                  # New place with implied properties
    IDIOM = "idiom"                        # New setting-specific phrase

    # Source annotations (inferred meaning from reference)
    STRUCTURAL_THEME = "structural_theme"  # "This quest is a gatekeeping pattern"
    CHARACTER_ROLE = "character_role"      # "This NPC functions as threshold guardian"
    ARC_PATTERN = "arc_pattern"            # "This dialogue exhibits negotiation_under_duress"


class AdditionDirection(str, Enum):
    """Is this enriching source understanding or target vocabulary?"""
    SOURCE = "source"  # Annotation on reference game
    TARGET = "target"  # Enrichment for synthetic setting


@dataclass
class ProposedAddition:
    """A pending bible enrichment awaiting curator validation."""
    id: str
    direction: AdditionDirection
    setting: str  # Which bible (source game or target setting)
    addition_type: AdditionType
    content: Dict[str, Any]  # The actual addition (varies by type)

    # Provenance
    source_walk_id: str
    source_text: str  # The dialogue that prompted this inference
    reasoning: str    # Why this addition is warranted

    # State
    proposed_at: str
    proposed_by: str  # Which translation batch
    status: str = "pending"  # pending, approved, rejected, merged
    curator_response: Optional[Dict] = None

    # Confidence/support
    support_count: int = 1  # How many translations suggested similar
    related_additions: List[str] = field(default_factory=list)


@dataclass
class EnrichedBible:
    """A bible with accumulated enrichments from a translation run."""
    setting: str
    base_bible_path: str

    # Accumulated enrichments (approved only)
    proper_nouns: Dict[str, List[str]] = field(default_factory=dict)  # cluster -> instances
    factions: List[Dict] = field(default_factory=list)
    tensions: List[str] = field(default_factory=list)
    locations: List[Dict] = field(default_factory=list)
    idioms: List[str] = field(default_factory=list)

    # Source annotations (if this is a reference bible)
    structural_themes: List[Dict] = field(default_factory=list)
    character_roles: Dict[str, str] = field(default_factory=dict)  # npc -> role
    arc_patterns: List[Dict] = field(default_factory=list)

    # Metadata
    enrichment_count: int = 0
    last_merge_tick: int = 0

    def to_prompt_context(self, max_chars: int = 3000) -> str:
        """Format enriched bible for inclusion in translation prompts."""
        sections = []

        # Base content
        if Path(self.base_bible_path).exists():
            base = Path(self.base_bible_path).read_text()[:1500]
            sections.append(f"## Base Setting\n{base}")

        # Enrichments
        if self.proper_nouns:
            nouns_str = "\n".join(
                f"- {cluster}: {', '.join(instances[:5])}"
                for cluster, instances in list(self.proper_nouns.items())[:10]
            )
            sections.append(f"## Discovered Names\n{nouns_str}")

        if self.factions:
            factions_str = "\n".join(
                f"- {f.get('name', '?')}: {f.get('archetype', '?')}"
                for f in self.factions[:5]
            )
            sections.append(f"## Discovered Factions\n{factions_str}")

        if self.tensions:
            sections.append(f"## Discovered Tensions\n" + "\n".join(f"- {t}" for t in self.tensions[:5]))

        if self.idioms:
            sections.append(f"## Setting Idioms\n" + "\n".join(f"- \"{i}\"" for i in self.idioms[:5]))

        result = "\n\n".join(sections)
        return result[:max_chars]

    def merge_addition(self, addition: ProposedAddition):
        """Merge an approved addition into this bible."""
        content = addition.content

        if addition.addition_type == AdditionType.PROPER_NOUN:
            cluster = content.get("cluster", "misc")
            instance = content.get("instance", "")
            if cluster not in self.proper_nouns:
                self.proper_nouns[cluster] = []
            if instance and instance not in self.proper_nouns[cluster]:
                self.proper_nouns[cluster].append(instance)

        elif addition.addition_type == AdditionType.FACTION:
            self.factions.append(content)

        elif addition.addition_type == AdditionType.TENSION:
            tension = content.get("tension", "")
            if tension and tension not in self.tensions:
                self.tensions.append(tension)

        elif addition.addition_type == AdditionType.LOCATION:
            self.locations.append(content)

        elif addition.addition_type == AdditionType.IDIOM:
            idiom = content.get("idiom", "")
            if idiom and idiom not in self.idioms:
                self.idioms.append(idiom)

        elif addition.addition_type == AdditionType.STRUCTURAL_THEME:
            self.structural_themes.append(content)

        elif addition.addition_type == AdditionType.CHARACTER_ROLE:
            npc = content.get("npc", "")
            role = content.get("role", "")
            if npc and role:
                self.character_roles[npc] = role

        elif addition.addition_type == AdditionType.ARC_PATTERN:
            self.arc_patterns.append(content)

        self.enrichment_count += 1


# =============================================================================
# Run State
# =============================================================================

@dataclass
class RunState:
    """
    Global mutable state within an API translation run.

    This is the hermeneutic context - the accumulated understanding
    that grows as we translate and discover metaphors.
    """
    run_id: str
    started_at: str

    # Hot bibles (enriched during run)
    hot_bibles: Dict[str, EnrichedBible] = field(default_factory=dict)

    # Addition queue (pending curator validation)
    addition_queue: List[ProposedAddition] = field(default_factory=list)

    # Curator state
    curator_clock: int = 0  # Systolic tick counter
    last_curator_run: Optional[str] = None

    # Warmup state
    warmup_progress: float = 0.0  # 0.0 → 1.0
    total_translations: int = 0
    target_translations: int = 100  # For warmup calculation

    # Metrics
    additions_proposed: int = 0
    additions_approved: int = 0
    additions_rejected: int = 0
    additions_merged: int = 0

    # Deduplication
    _seen_additions: Set[str] = field(default_factory=set)

    def get_or_create_bible(self, setting: str, base_path: str = "") -> EnrichedBible:
        """Get hot bible for setting, creating if needed."""
        if setting not in self.hot_bibles:
            # Try to find base bible
            if not base_path:
                base_path = f"bibles/{setting}.yaml"
            self.hot_bibles[setting] = EnrichedBible(
                setting=setting,
                base_bible_path=base_path,
            )
        return self.hot_bibles[setting]

    def propose_addition(self, addition: ProposedAddition) -> bool:
        """
        Add to queue if not duplicate.
        Returns True if added, False if duplicate.
        """
        # Content-based dedup key
        key = hashlib.sha256(
            json.dumps({
                "setting": addition.setting,
                "type": addition.addition_type,
                "content": addition.content,
            }, sort_keys=True).encode()
        ).hexdigest()[:16]

        if key in self._seen_additions:
            # Increment support count on existing
            for existing in self.addition_queue:
                if existing.id.endswith(key):
                    existing.support_count += 1
                    return False
            return False

        self._seen_additions.add(key)
        addition.id = f"add_{self.run_id}_{key}"
        self.addition_queue.append(addition)
        self.additions_proposed += 1
        return True

    def get_pending_additions(self, limit: int = 20) -> List[ProposedAddition]:
        """Get pending additions for curator batch."""
        return [a for a in self.addition_queue if a.status == "pending"][:limit]

    def mark_curated(
        self,
        addition_id: str,
        approved: bool,
        curator_response: Dict,
    ):
        """Mark addition as curated, merge if approved."""
        for addition in self.addition_queue:
            if addition.id == addition_id:
                addition.status = "approved" if approved else "rejected"
                addition.curator_response = curator_response

                if approved:
                    self.additions_approved += 1
                    # Merge to hot bible
                    bible = self.get_or_create_bible(addition.setting)
                    bible.merge_addition(addition)
                    bible.last_merge_tick = self.curator_clock
                    addition.status = "merged"
                    self.additions_merged += 1
                else:
                    self.additions_rejected += 1

                return

    def tick_curator(self):
        """Advance curator clock."""
        self.curator_clock += 1
        self.last_curator_run = datetime.now().isoformat()

    def update_warmup(self, completed: int = 1):
        """Update warmup progress based on completed translations."""
        self.total_translations += completed
        self.warmup_progress = min(1.0, self.total_translations / self.target_translations)

    def effective_concurrency(self, max_concurrency: int = 25) -> int:
        """
        Sigmoid warmup for concurrency.

        Explore phase (warmup < 0.3): Low concurrency, allow metaphor diffusion
        Exploit phase (warmup > 0.3): Scale up, bible is stable enough
        """
        # Sigmoid centered at 0.3, steep transition
        sigmoid = 1 / (1 + exp(-10 * (self.warmup_progress - 0.3)))
        return max(2, int(max_concurrency * sigmoid))

    def should_run_curator(self, queue_threshold: int = 10, tick_interval: int = 5) -> bool:
        """
        Decide if curator should run now.

        Run if:
        - Queue has enough items, OR
        - It's been too long since last run
        """
        pending = len([a for a in self.addition_queue if a.status == "pending"])

        if pending >= queue_threshold:
            return True

        # During explore phase, run curator more frequently
        if self.warmup_progress < 0.3:
            return pending > 0 and self.total_translations % 3 == 0

        return False

    def to_dict(self) -> Dict:
        """Serialize run state (for persistence/logging)."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "curator_clock": self.curator_clock,
            "warmup_progress": self.warmup_progress,
            "total_translations": self.total_translations,
            "additions_proposed": self.additions_proposed,
            "additions_approved": self.additions_approved,
            "additions_rejected": self.additions_rejected,
            "additions_merged": self.additions_merged,
            "queue_depth": len([a for a in self.addition_queue if a.status == "pending"]),
            "hot_bibles": {
                k: {"enrichment_count": v.enrichment_count}
                for k, v in self.hot_bibles.items()
            },
        }

    def save_snapshot(self, output_dir: Path):
        """Save run state snapshot for later analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # State summary
        with open(output_dir / "run_state.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Full addition queue
        with open(output_dir / "additions.jsonl", "w") as f:
            for a in self.addition_queue:
                f.write(json.dumps(asdict(a)) + "\n")

        # Hot bibles
        for setting, bible in self.hot_bibles.items():
            with open(output_dir / f"bible_{setting}.json", "w") as f:
                json.dump(asdict(bible), f, indent=2)


# =============================================================================
# Global Run State (singleton within API process)
# =============================================================================

_CURRENT_RUN: Optional[RunState] = None


def get_current_run() -> Optional[RunState]:
    """Get current run state, if any."""
    return _CURRENT_RUN


def start_run(
    run_id: Optional[str] = None,
    target_translations: int = 100,
    source_settings: List[str] = None,
    target_settings: List[str] = None,
) -> RunState:
    """Start a new translation run with fresh state."""
    global _CURRENT_RUN

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    _CURRENT_RUN = RunState(
        run_id=run_id,
        started_at=datetime.now().isoformat(),
        target_translations=target_translations,
    )

    # Initialize hot bibles for known settings
    for setting in (source_settings or []):
        _CURRENT_RUN.get_or_create_bible(setting)
    for setting in (target_settings or []):
        _CURRENT_RUN.get_or_create_bible(setting)

    return _CURRENT_RUN


def end_run(output_dir: Optional[Path] = None) -> Optional[RunState]:
    """End current run, optionally saving snapshot."""
    global _CURRENT_RUN

    if _CURRENT_RUN is None:
        return None

    run = _CURRENT_RUN

    if output_dir:
        run.save_snapshot(output_dir)

    _CURRENT_RUN = None
    return run


# =============================================================================
# Curator Batch Processing
# =============================================================================

CURATOR_BATCH_PROMPT = """You are validating proposed additions to a lore bible.

## Current Bible State
{bible_context}

## Proposed Additions (validate each)
{additions_json}

For EACH addition, respond with:
```json
{{
  "addition_id": "...",
  "approved": true|false,
  "reasoning": "Why this fits or doesn't fit",
  "modified_content": {{...}} | null,  // If approved but needs adjustment
  "warnings": ["..."] | null  // Concerns even if approved
}}
```

Validation criteria:
- PROPER_NOUN: Does it fit an existing cluster? Is naming convention consistent?
- FACTION: Does it have coherent wants/fears? Does it create contradictions?
- TENSION: Is it generative (produces quests)? Is it unresolvable?
- IDIOM: Does it match the setting's register?
- STRUCTURAL_THEME: Is the inference well-supported by the source text?

Return a JSON array of validation results, one per addition.
"""


async def run_curator_batch(
    run_state: RunState,
    llm_call: callable,
    batch_size: int = 10,
) -> List[Dict]:
    """
    Run curator validation on pending additions.

    Returns list of validation results.
    """
    pending = run_state.get_pending_additions(limit=batch_size)
    if not pending:
        return []

    # Group by setting for context
    by_setting: Dict[str, List[ProposedAddition]] = {}
    for a in pending:
        if a.setting not in by_setting:
            by_setting[a.setting] = []
        by_setting[a.setting].append(a)

    results = []

    for setting, additions in by_setting.items():
        bible = run_state.get_or_create_bible(setting)
        bible_context = bible.to_prompt_context(max_chars=2000)

        additions_json = json.dumps([
            {
                "addition_id": a.id,
                "type": a.addition_type,
                "content": a.content,
                "reasoning": a.reasoning,
                "source_text": a.source_text[:200],
                "support_count": a.support_count,
            }
            for a in additions
        ], indent=2)

        prompt = CURATOR_BATCH_PROMPT.format(
            bible_context=bible_context,
            additions_json=additions_json,
        )

        try:
            response = await llm_call(prompt)

            # Parse response
            import re
            # Try to find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                validations = json.loads(json_match.group())
            else:
                # Try parsing whole response as JSON
                validations = json.loads(response)

            # Apply validations
            for v in validations:
                addition_id = v.get("addition_id")
                approved = v.get("approved", False)

                run_state.mark_curated(
                    addition_id=addition_id,
                    approved=approved,
                    curator_response=v,
                )
                results.append(v)

        except Exception as e:
            # Log error but don't crash
            print(f"Curator batch error for {setting}: {e}")
            # Mark all as needing retry
            for a in additions:
                a.status = "pending"  # Will retry next tick

    run_state.tick_curator()
    return results


# =============================================================================
# Translation Integration Helpers
# =============================================================================

def extract_proposed_additions(
    source_walk_id: str,
    source_text: str,
    translated_text: str,
    source_setting: str,
    target_setting: str,
    structural_analysis: Optional[Dict] = None,
) -> List[ProposedAddition]:
    """
    Extract proposed additions from a translation result.

    This is called after translation to identify:
    - New proper nouns introduced in target
    - Inferred structural patterns from source
    """
    additions = []
    now = datetime.now().isoformat()

    # Look for capitalized words that might be proper nouns
    import re

    # Find capitalized multi-word phrases (potential proper nouns)
    proper_noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    potential_nouns = set(re.findall(proper_noun_pattern, translated_text))

    # Filter out common sentence starters and known words
    common_starters = {"The", "This", "That", "What", "When", "Where", "Why", "How", "It", "He", "She", "They", "We", "I"}
    potential_nouns = {n for n in potential_nouns if n not in common_starters and len(n) > 3}

    for noun in potential_nouns:
        # Heuristic: if it looks like a name or title
        if any(c in noun for c in ["Office", "Bureau", "Department", "District", "Agent", "Director"]):
            cluster = "bureaucratic_entities"
        elif any(c in noun for c in ["Street", "Plaza", "Tower", "Hall", "Quarter"]):
            cluster = "locations"
        else:
            cluster = "characters"

        additions.append(ProposedAddition(
            id="",  # Will be set by propose_addition
            direction=AdditionDirection.TARGET,
            setting=target_setting,
            addition_type=AdditionType.PROPER_NOUN,
            content={"cluster": cluster, "instance": noun},
            source_walk_id=source_walk_id,
            source_text=source_text[:200],
            reasoning=f"Introduced in translation as apparent {cluster}",
            proposed_at=now,
            proposed_by=f"translation_{source_walk_id}",
        ))

    # If structural analysis provided, extract source annotations
    if structural_analysis:
        if structural_analysis.get("barrier_type"):
            additions.append(ProposedAddition(
                id="",
                direction=AdditionDirection.SOURCE,
                setting=source_setting,
                addition_type=AdditionType.STRUCTURAL_THEME,
                content={
                    "pattern": "barrier",
                    "barrier_type": structural_analysis["barrier_type"],
                    "walk_id": source_walk_id,
                },
                source_walk_id=source_walk_id,
                source_text=source_text[:200],
                reasoning=f"Identified {structural_analysis['barrier_type']} barrier pattern",
                proposed_at=now,
                proposed_by=f"structural_parser_{source_walk_id}",
            ))

        if structural_analysis.get("arc_shape"):
            additions.append(ProposedAddition(
                id="",
                direction=AdditionDirection.SOURCE,
                setting=source_setting,
                addition_type=AdditionType.ARC_PATTERN,
                content={
                    "arc_shape": structural_analysis["arc_shape"],
                    "emotions": structural_analysis.get("emotions", []),
                    "walk_id": source_walk_id,
                },
                source_walk_id=source_walk_id,
                source_text=source_text[:200],
                reasoning=f"Identified {structural_analysis['arc_shape']} arc pattern",
                proposed_at=now,
                proposed_by=f"structural_parser_{source_walk_id}",
            ))

    return additions


async def translation_with_enrichment(
    source_walk: Dict,
    source_setting: str,
    target_setting: str,
    translate_fn: callable,
    run_state: Optional[RunState] = None,
) -> Tuple[str, List[ProposedAddition]]:
    """
    Wrapper for translation that handles bible enrichment.

    1. Get hot bible context for prompt enrichment
    2. Run translation
    3. Extract proposed additions
    4. Queue additions for curator
    5. Update warmup progress
    """
    if run_state is None:
        run_state = get_current_run()

    # Get enriched bible context
    target_bible = run_state.get_or_create_bible(target_setting) if run_state else None
    bible_context = target_bible.to_prompt_context() if target_bible else ""

    # Run translation with enriched context
    source_text = " ".join(b.get("text", "") for b in source_walk.get("beats", []))
    translated = await translate_fn(
        source_walk=source_walk,
        bible_context=bible_context,
    )

    # Extract additions
    additions = extract_proposed_additions(
        source_walk_id=source_walk.get("id", "unknown"),
        source_text=source_text,
        translated_text=translated,
        source_setting=source_setting,
        target_setting=target_setting,
        structural_analysis=source_walk.get("structural_analysis"),
    )

    # Queue additions
    if run_state:
        for a in additions:
            run_state.propose_addition(a)
        run_state.update_warmup(completed=1)

    return translated, additions


# =============================================================================
# CLI Test
# =============================================================================

async def main():
    """Test the hermeneutic loop."""
    print("=" * 60)
    print("HERMENEUTIC LOOP TEST")
    print("=" * 60)

    # Start a run
    run = start_run(
        run_id="test_001",
        target_translations=20,
        source_settings=["oblivion"],
        target_settings=["marmotte"],
    )

    print(f"Started run: {run.run_id}")
    print(f"Initial concurrency: {run.effective_concurrency()}")

    # Simulate some translations proposing additions
    for i in range(10):
        addition = ProposedAddition(
            id="",
            direction=AdditionDirection.TARGET,
            setting="marmotte",
            addition_type=AdditionType.PROPER_NOUN,
            content={"cluster": "bureaucratic_entities", "instance": f"Department {i}"},
            source_walk_id=f"walk_{i}",
            source_text="Some source dialogue...",
            reasoning="Introduced in translation",
            proposed_at=datetime.now().isoformat(),
            proposed_by=f"translation_{i}",
        )
        run.propose_addition(addition)
        run.update_warmup(completed=1)

    print(f"\nAfter 10 translations:")
    print(f"  Warmup: {run.warmup_progress:.2f}")
    print(f"  Concurrency: {run.effective_concurrency()}")
    print(f"  Queue depth: {len(run.get_pending_additions())}")
    print(f"  Should run curator: {run.should_run_curator()}")

    # Simulate curator approval
    for a in run.get_pending_additions()[:5]:
        run.mark_curated(a.id, approved=True, curator_response={"reasoning": "Fits well"})
    for a in run.get_pending_additions()[:3]:
        run.mark_curated(a.id, approved=False, curator_response={"reasoning": "Doesn't fit"})

    print(f"\nAfter curator batch:")
    print(f"  Approved: {run.additions_approved}")
    print(f"  Rejected: {run.additions_rejected}")
    print(f"  Merged: {run.additions_merged}")

    # Check hot bible
    bible = run.get_or_create_bible("marmotte")
    print(f"\nMarmotte hot bible:")
    print(f"  Enrichment count: {bible.enrichment_count}")
    print(f"  Proper nouns: {bible.proper_nouns}")

    # Simulate more translations (exploit phase)
    for i in range(50):
        run.update_warmup(completed=1)

    print(f"\nAfter 60 total translations:")
    print(f"  Warmup: {run.warmup_progress:.2f}")
    print(f"  Concurrency: {run.effective_concurrency()}")

    # Save snapshot
    output_dir = Path("output/hermeneutic_test")
    end_run(output_dir)
    print(f"\nSaved snapshot to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
