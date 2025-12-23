#!/usr/bin/env python3
"""
Lore Curator Validation Loop

Validates new proper nouns from translations against the target bible.
Handles:
- Batch validation of multiple synthetics
- Bible updates for approved additions
- Rejection tracking for revision
- Statistics on curator decisions

Usage:
    from workflow.curator_loop import CuratorLoop

    loop = CuratorLoop(bible_path="bibles/gallia.yaml")
    result = loop.validate_translation(translation_output)

    if result.all_approved:
        persist_synthetic(...)
    else:
        # Handle rejections
        for rejection in result.rejections:
            print(f"Rejected: {rejection.noun} - {rejection.reason}")
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from subagent_orchestrator import (
    SubagentCaller,
    LoreBible,
    CuratorDecision,
    ValidationReport,
    validate_curator_decision,
)


@dataclass
class NounValidation:
    """Result of validating a single proper noun."""
    noun: str
    approved: bool
    reasoning: str
    cluster_assignment: Optional[str] = None
    suggested_alternatives: Optional[list[str]] = None
    warnings: Optional[list[str]] = None
    latency_ms: int = 0
    cost_usd: float = 0.0


@dataclass
class TranslationValidationResult:
    """Result of validating all new nouns in a translation."""
    translation_id: str
    nouns_submitted: list[str]
    validations: list[NounValidation] = field(default_factory=list)
    bible_updates_proposed: list[dict] = field(default_factory=list)
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0

    @property
    def all_approved(self) -> bool:
        return all(v.approved for v in self.validations)

    @property
    def approvals(self) -> list[NounValidation]:
        return [v for v in self.validations if v.approved]

    @property
    def rejections(self) -> list[NounValidation]:
        return [v for v in self.validations if not v.approved]

    def summary(self) -> str:
        approved = len(self.approvals)
        rejected = len(self.rejections)
        total = len(self.validations)

        parts = [f"Curator Validation: {approved}/{total} approved"]

        if self.rejections:
            parts.append("\nRejections:")
            for r in self.rejections:
                alts = f" (try: {', '.join(r.suggested_alternatives[:3])})" if r.suggested_alternatives else ""
                parts.append(f"  ✗ {r.noun}: {r.reasoning[:80]}...{alts}")

        if self.approvals:
            parts.append("\nApprovals:")
            for a in self.approvals:
                cluster = f" → {a.cluster_assignment}" if a.cluster_assignment else ""
                parts.append(f"  ✓ {a.noun}{cluster}")

        parts.append(f"\nCost: ${self.total_cost_usd:.4f} | Latency: {self.total_latency_ms}ms")

        return "\n".join(parts)


@dataclass
class CuratorStats:
    """Aggregate statistics from curator validation."""
    total_nouns: int = 0
    approved: int = 0
    rejected: int = 0
    rejection_reasons: dict = field(default_factory=dict)  # reason_category -> count
    cluster_assignments: dict = field(default_factory=dict)  # cluster -> count
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0

    @property
    def approval_rate(self) -> float:
        return self.approved / self.total_nouns if self.total_nouns > 0 else 0.0

    def add_validation(self, v: NounValidation):
        self.total_nouns += 1
        self.total_cost_usd += v.cost_usd
        self.total_latency_ms += v.latency_ms

        if v.approved:
            self.approved += 1
            if v.cluster_assignment:
                self.cluster_assignments[v.cluster_assignment] = (
                    self.cluster_assignments.get(v.cluster_assignment, 0) + 1
                )
        else:
            self.rejected += 1
            # Categorize rejection reason (first word as category)
            category = v.reasoning.split()[0].lower() if v.reasoning else "unknown"
            self.rejection_reasons[category] = self.rejection_reasons.get(category, 0) + 1

    def summary(self) -> str:
        parts = [
            f"Curator Stats: {self.approved}/{self.total_nouns} approved ({self.approval_rate:.1%})",
            f"Total cost: ${self.total_cost_usd:.4f}",
        ]

        if self.rejection_reasons:
            parts.append("\nTop rejection reasons:")
            for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: -x[1])[:5]:
                parts.append(f"  {reason}: {count}")

        if self.cluster_assignments:
            parts.append("\nCluster assignments:")
            for cluster, count in sorted(self.cluster_assignments.items(), key=lambda x: -x[1])[:5]:
                parts.append(f"  {cluster}: {count}")

        return "\n".join(parts)


class CuratorLoop:
    """
    Orchestrates proper noun validation against lore bibles.

    The curator has VETO POWER. Synthetics with rejected nouns
    should not be persisted without revision.
    """

    def __init__(
        self,
        bible_path: Path | str,
        prompts_dir: Path | str = "claudefiles/subagents",
        auto_update_bible: bool = False,
    ):
        self.bible_path = Path(bible_path)
        self.prompts_dir = Path(prompts_dir)
        self.auto_update_bible = auto_update_bible
        self.stats = CuratorStats()

        # Load bible
        self._bible_content = self.bible_path.read_text()
        self._bible = self._parse_bible(self._bible_content)

        # Initialize caller
        self._caller = SubagentCaller(prompts_dir=prompts_dir)

    def _parse_bible(self, content: str) -> LoreBible:
        """Parse bible YAML into model."""
        data = yaml.safe_load(content)
        return LoreBible.model_validate(data)

    def validate_translation(
        self,
        translation_output: dict,
        translation_id: str = "",
        context: str = "",
    ) -> TranslationValidationResult:
        """
        Validate all new proper nouns in a translation.

        Args:
            translation_output: Output from translation-engine
            translation_id: ID for tracking
            context: Additional context about the translation

        Returns:
            TranslationValidationResult with all validations
        """
        nouns = translation_output.get("proper_nouns_introduced", [])

        result = TranslationValidationResult(
            translation_id=translation_id or f"t_{datetime.now(timezone.utc).strftime('%H%M%S')}",
            nouns_submitted=nouns,
        )

        if not nouns:
            # No new nouns to validate
            return result

        # Validate each noun
        for noun in nouns:
            validation = self._validate_single_noun(noun, context)
            result.validations.append(validation)
            result.total_latency_ms += validation.latency_ms
            result.total_cost_usd += validation.cost_usd
            self.stats.add_validation(validation)

            # Collect bible updates
            if validation.approved and hasattr(validation, '_bible_update'):
                result.bible_updates_proposed.append(validation._bible_update)

        # Optionally apply bible updates
        if self.auto_update_bible and result.bible_updates_proposed:
            self._apply_bible_updates(result.bible_updates_proposed)

        return result

    def _validate_single_noun(
        self,
        noun: str,
        context: str = "",
    ) -> NounValidation:
        """Validate a single proper noun against the bible."""

        # Check if noun already exists
        existing = self._check_existing_noun(noun)
        if existing:
            return NounValidation(
                noun=noun,
                approved=True,
                reasoning=f"Already exists in cluster '{existing}'",
                cluster_assignment=existing,
            )

        # Build context for curator
        translated_texts = context if context else "New proper noun from translation"

        # Call curator
        log = self._caller.call_lore_curator(
            proposal_type="proper_noun",
            proposal={
                "proposed_noun": noun,
                "context": translated_texts,
                "source": "translation_engine",
            },
            bible_content=self._bible_content,
        )

        validation = NounValidation(
            noun=noun,
            approved=False,
            reasoning="Parse error",
            latency_ms=log.latency_ms,
            cost_usd=log.cost_usd,
        )

        if log.parse_success and log.output_parsed:
            # Validate against schema
            decision, report = validate_curator_decision(log.output_parsed)

            if report.valid and decision:
                validation.approved = decision.approved
                validation.reasoning = decision.reasoning
                validation.suggested_alternatives = decision.suggested_alternatives
                validation.warnings = decision.warnings

                # Extract cluster assignment if approved
                if decision.approved and decision.bible_update:
                    if "cluster" in str(decision.bible_update).lower():
                        validation.cluster_assignment = self._extract_cluster(decision.bible_update)
                    validation._bible_update = decision.bible_update
            else:
                validation.reasoning = f"Schema validation failed: {report.summary()}"
        else:
            validation.reasoning = f"Parse failed: {log.parse_errors}"

        return validation

    def _check_existing_noun(self, noun: str) -> Optional[str]:
        """Check if noun already exists in bible, return cluster name if so."""
        noun_lower = noun.lower()
        for cluster in self._bible.proper_noun_clusters:
            for instance in cluster.instances:
                if instance.instance.lower() == noun_lower:
                    return cluster.cluster_name
        return None

    def _extract_cluster(self, bible_update: dict) -> Optional[str]:
        """Try to extract cluster name from bible update."""
        # This is heuristic - depends on curator output format
        update_str = str(bible_update).lower()
        for cluster in self._bible.proper_noun_clusters:
            if cluster.cluster_name.lower() in update_str:
                return cluster.cluster_name
        return None

    def _apply_bible_updates(self, updates: list[dict]):
        """Apply accumulated bible updates (placeholder for now)."""
        # TODO: Implement actual bible updates
        # This would merge approved additions into the YAML
        # For now, just log
        print(f"[CuratorLoop] Would apply {len(updates)} bible updates")

    def validate_batch(
        self,
        translations: list[dict],
        translation_ids: Optional[list[str]] = None,
    ) -> list[TranslationValidationResult]:
        """
        Validate a batch of translations.

        Args:
            translations: List of translation outputs
            translation_ids: Optional IDs for each translation

        Returns:
            List of validation results
        """
        results = []

        for i, translation in enumerate(translations):
            tid = translation_ids[i] if translation_ids else f"batch_{i}"
            result = self.validate_translation(translation, tid)
            results.append(result)

        return results

    def filter_valid_synthetics(
        self,
        synthetics: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Filter synthetics by curator approval.

        Returns:
            (approved_synthetics, rejected_synthetics)
        """
        approved = []
        rejected = []

        for synthetic in synthetics:
            translation = {
                "proper_nouns_introduced": synthetic.get("proper_nouns_introduced", []),
            }
            result = self.validate_translation(
                translation,
                synthetic.get("synthetic_id", ""),
            )

            if result.all_approved:
                approved.append(synthetic)
            else:
                rejected.append({
                    "synthetic": synthetic,
                    "rejections": [
                        {"noun": r.noun, "reason": r.reasoning, "alternatives": r.suggested_alternatives}
                        for r in result.rejections
                    ],
                })

        return approved, rejected


# =============================================================================
# CLI for standalone testing
# =============================================================================

def main():
    """Test curator loop with sample input."""
    import argparse

    parser = argparse.ArgumentParser(description="Test curator validation")
    parser.add_argument("--bible", default="bibles/gallia.yaml", help="Bible path")
    parser.add_argument("--noun", help="Single noun to validate")
    parser.add_argument("--context", default="", help="Context for validation")

    args = parser.parse_args()

    if not Path(args.bible).exists():
        print(f"Bible not found: {args.bible}")
        return 1

    loop = CuratorLoop(bible_path=args.bible)

    if args.noun:
        # Single noun validation
        result = loop.validate_translation(
            {"proper_nouns_introduced": [args.noun]},
            context=args.context,
        )
        print(result.summary())
    else:
        # Demo with sample nouns
        demo_nouns = ["Sous-Préfet Moreau", "La Bastide", "Section 47"]
        result = loop.validate_translation(
            {"proper_nouns_introduced": demo_nouns},
            context="Test validation",
        )
        print(result.summary())
        print("\n" + loop.stats.summary())

    return 0


if __name__ == "__main__":
    exit(main())
