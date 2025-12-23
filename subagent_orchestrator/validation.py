"""
Schema Validation for Subagent Outputs

Validates parsed JSON against Pydantic models with controlled vocabularies.
Reports specific field-level failures for debugging model behavior.
"""

from __future__ import annotations

from typing import Optional, TypeVar, Type, Any
from pydantic import BaseModel, ValidationError

from .models import (
    StructuralTriplet,
    StructuralBeat,
    TranslationResult,
    CuratorDecision,
    EmotionType,
    BeatFunction,
    ArchetypeRelation,
    BarrierType,
    AttractorType,
    ArcShape,
)


T = TypeVar("T", bound=BaseModel)


class ValidationReport:
    """Detailed report of schema validation results."""

    def __init__(self):
        self.valid: bool = True
        self.model_name: str = ""
        self.field_errors: list[dict] = []
        self.enum_violations: list[dict] = []
        self.missing_fields: list[str] = []
        self.extra_fields: list[str] = []
        self.coerced_values: list[dict] = []
        self.validated_instance: Optional[BaseModel] = None

    def add_field_error(self, field: str, value: Any, expected: str, error: str):
        self.valid = False
        self.field_errors.append({
            "field": field,
            "value": value,
            "expected": expected,
            "error": error,
        })

    def add_enum_violation(self, field: str, value: str, allowed: list[str]):
        self.valid = False
        self.enum_violations.append({
            "field": field,
            "value": value,
            "allowed": allowed,
        })

    def summary(self) -> str:
        """Human-readable summary."""
        if self.valid:
            return f"✓ Valid {self.model_name}"

        parts = [f"✗ Invalid {self.model_name}:"]

        if self.missing_fields:
            parts.append(f"  Missing: {', '.join(self.missing_fields)}")

        if self.extra_fields:
            parts.append(f"  Extra: {', '.join(self.extra_fields)}")

        for ev in self.enum_violations:
            parts.append(
                f"  Bad enum: {ev['field']}='{ev['value']}' "
                f"(allowed: {', '.join(ev['allowed'][:5])}{'...' if len(ev['allowed']) > 5 else ''})"
            )

        for fe in self.field_errors[:3]:  # Limit to first 3
            parts.append(f"  {fe['field']}: {fe['error']}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serializable report."""
        return {
            "valid": self.valid,
            "model": self.model_name,
            "field_errors": self.field_errors,
            "enum_violations": self.enum_violations,
            "missing_fields": self.missing_fields,
            "extra_fields": self.extra_fields,
        }


def validate_against_model(
    data: dict,
    model: Type[T],
    strict: bool = False,
) -> tuple[Optional[T], ValidationReport]:
    """
    Validate a dict against a Pydantic model.

    Args:
        data: Parsed JSON dict
        model: Pydantic model class to validate against
        strict: If True, fail on extra fields

    Returns:
        (validated_instance or None, ValidationReport)
    """
    report = ValidationReport()
    report.model_name = model.__name__

    try:
        instance = model.model_validate(data)
        report.validated_instance = instance
        return instance, report

    except ValidationError as e:
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_type = error["type"]

            # Check if it's an enum violation
            if "not a valid enumeration member" in msg or error_type == "enum":
                # Extract allowed values
                allowed = _get_enum_values_for_field(model, error["loc"])
                if isinstance(error.get("input"), str):
                    report.add_enum_violation(loc, error["input"], allowed)
                else:
                    report.add_field_error(loc, error.get("input"), "enum", msg)

            elif error_type == "missing":
                report.missing_fields.append(loc)
                report.valid = False

            else:
                report.add_field_error(
                    loc,
                    error.get("input"),
                    str(error.get("expected", "unknown")),
                    msg,
                )

        return None, report


def _get_enum_values_for_field(model: Type[BaseModel], loc: tuple) -> list[str]:
    """Try to extract allowed enum values for a field path."""
    # This is a simplified approach - handles common cases
    enum_maps = {
        "emotion": [e.value for e in EmotionType],
        "function": [e.value for e in BeatFunction],
        "archetype_relation": [e.value for e in ArchetypeRelation],
        "barrier_type": [e.value for e in BarrierType],
        "attractor_type": [e.value for e in AttractorType],
        "arc_shape": [e.value for e in ArcShape],
        "transition_from": [e.value for e in EmotionType] + [None],
    }

    # Check if any part of the loc matches known enum fields
    for part in loc:
        if isinstance(part, str) and part in enum_maps:
            return enum_maps[part]

    return ["<unknown enum>"]


def validate_structural_triplet(data: dict) -> tuple[Optional[StructuralTriplet], ValidationReport]:
    """Validate output from structural-parser/triplet-extractor."""
    return validate_against_model(data, StructuralTriplet)


def validate_translation_result(data: dict) -> tuple[Optional[TranslationResult], ValidationReport]:
    """Validate output from translation-engine."""
    return validate_against_model(data, TranslationResult)


def validate_curator_decision(data: dict) -> tuple[Optional[CuratorDecision], ValidationReport]:
    """Validate output from lore-curator."""
    return validate_against_model(data, CuratorDecision)


def validate_beat(data: dict) -> tuple[Optional[StructuralBeat], ValidationReport]:
    """Validate a single beat (for debugging individual nodes)."""
    return validate_against_model(data, StructuralBeat)


# =============================================================================
# Integration with SubagentLog
# =============================================================================

def validate_subagent_output(
    parsed_output: Optional[dict],
    subagent_type: str,
) -> ValidationReport:
    """
    Validate parsed output against expected schema for subagent type.

    Args:
        parsed_output: The output_parsed from SubagentLog
        subagent_type: "triplet_extractor", "translation_engine", or "lore_curator"

    Returns:
        ValidationReport with detailed results
    """
    if parsed_output is None:
        report = ValidationReport()
        report.valid = False
        report.add_field_error("root", None, "dict", "No parsed output")
        return report

    validators = {
        "triplet_extractor": validate_structural_triplet,
        "structural_parser": validate_structural_triplet,  # Alias
        "translation_engine": validate_translation_result,
        "lore_curator": validate_curator_decision,
    }

    validator = validators.get(subagent_type.lower().replace("-", "_"))
    if validator is None:
        report = ValidationReport()
        report.valid = False
        report.add_field_error("subagent_type", subagent_type, "known type", "Unknown subagent type")
        return report

    _, report = validator(parsed_output)
    return report


# =============================================================================
# Batch Validation Statistics
# =============================================================================

class ValidationStats:
    """Aggregate statistics from batch validation."""

    def __init__(self):
        self.total: int = 0
        self.valid: int = 0
        self.invalid: int = 0
        self.enum_violation_counts: dict[str, dict[str, int]] = {}  # field -> {bad_value: count}
        self.missing_field_counts: dict[str, int] = {}
        self.error_type_counts: dict[str, int] = {}

    def add_report(self, report: ValidationReport):
        self.total += 1
        if report.valid:
            self.valid += 1
        else:
            self.invalid += 1

        for ev in report.enum_violations:
            field = ev["field"]
            value = ev["value"]
            if field not in self.enum_violation_counts:
                self.enum_violation_counts[field] = {}
            self.enum_violation_counts[field][value] = (
                self.enum_violation_counts[field].get(value, 0) + 1
            )

        for mf in report.missing_fields:
            self.missing_field_counts[mf] = self.missing_field_counts.get(mf, 0) + 1

        for fe in report.field_errors:
            err_type = fe.get("error", "unknown")[:50]
            self.error_type_counts[err_type] = self.error_type_counts.get(err_type, 0) + 1

    @property
    def valid_rate(self) -> float:
        return self.valid / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        parts = [
            f"Validation Stats: {self.valid}/{self.total} valid ({self.valid_rate:.1%})",
        ]

        if self.enum_violation_counts:
            parts.append("\nEnum Violations:")
            for field, values in self.enum_violation_counts.items():
                top_bad = sorted(values.items(), key=lambda x: -x[1])[:3]
                parts.append(f"  {field}: {', '.join(f'{v}({c})' for v, c in top_bad)}")

        if self.missing_field_counts:
            parts.append("\nMissing Fields:")
            for field, count in sorted(self.missing_field_counts.items(), key=lambda x: -x[1]):
                parts.append(f"  {field}: {count}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "valid_rate": self.valid_rate,
            "enum_violations": self.enum_violation_counts,
            "missing_fields": self.missing_field_counts,
            "error_types": self.error_type_counts,
        }
