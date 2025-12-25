"""
Subagent Caller - Anthropic API Wrapper with Observability

Wraps Anthropic API calls with:
- Automatic system prompt loading from CLAUDE.md files
- Full request/response logging
- Timing and token counting
- JSON parsing with error capture
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Any

from .observability import (
    SubagentLog,
    SubagentType,
    Timer,
    hash_prompt,
)

# Lazy import to avoid requiring anthropic when not calling subagents
_anthropic_client = None


def get_anthropic_client():
    """Lazy-load Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic()
        except ImportError:
            raise ImportError(
                "anthropic package required for subagent calls. "
                "Install with: pip install anthropic"
            )
    return _anthropic_client


# Model mappings
MODEL_MAP = {
    "haiku": "claude-3-5-haiku-20241022",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}

# Default model assignments per subagent type
SUBAGENT_MODELS = {
    SubagentType.TRIPLET_EXTRACTOR: "haiku",
    SubagentType.TRANSLATION_ENGINE: "sonnet",
    SubagentType.LORE_CURATOR: "opus",
}


class SubagentCaller:
    """
    Orchestrator for calling specialized subagents.

    Handles:
    - Loading system prompts from CLAUDE.md files
    - Making API calls with appropriate models
    - Logging all calls for observability
    - Parsing JSON responses
    """

    def __init__(
        self,
        prompts_dir: Path | str = "claudefiles/subagents",
        traces_dir: Path | str = "traces",
    ):
        self.prompts_dir = Path(prompts_dir)
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        # Cache loaded prompts
        self._prompt_cache: dict[SubagentType, str] = {}

    def _load_prompt(self, subagent_type: SubagentType) -> str:
        """Load system prompt from CLAUDE.md file."""
        if subagent_type in self._prompt_cache:
            return self._prompt_cache[subagent_type]

        # Map subagent type to directory name
        dir_names = {
            SubagentType.TRIPLET_EXTRACTOR: "triplet_extractor",
            SubagentType.LORE_CURATOR: "lore_curator",
            SubagentType.TRANSLATION_ENGINE: "translation_engine",
        }

        prompt_path = self.prompts_dir / dir_names[subagent_type] / "CLAUDE.md"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt not found: {prompt_path}\n"
                f"Expected CLAUDE.md at: {prompt_path.absolute()}"
            )

        prompt = prompt_path.read_text()
        self._prompt_cache[subagent_type] = prompt
        return prompt

    def call(
        self,
        subagent_type: SubagentType,
        user_message: str,
        input_payload: Optional[dict] = None,
        model_override: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> SubagentLog:
        """
        Call a subagent and return a log entry.

        Args:
            subagent_type: Which subagent to call
            user_message: The prompt to send
            input_payload: Structured input data (for logging)
            model_override: Override default model (haiku/sonnet/opus)
            max_tokens: Maximum response tokens

        Returns:
            SubagentLog with full request/response data
        """
        # Resolve model
        model_key = model_override or SUBAGENT_MODELS[subagent_type]
        model = MODEL_MAP.get(model_key, model_key)

        # Load system prompt
        system_prompt = self._load_prompt(subagent_type)

        # Initialize log
        log = SubagentLog(
            subagent_type=subagent_type,
            model=model,
            system_prompt_hash=hash_prompt(system_prompt),
            input_payload=input_payload or {},
            user_message=user_message,
        )

        # Make API call with timing
        client = get_anthropic_client()

        with Timer() as timer:
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
            except Exception as e:
                log.latency_ms = timer.elapsed_ms
                log.parse_errors.append(f"API error: {type(e).__name__}: {str(e)}")
                return log

        log.latency_ms = timer.elapsed_ms

        # Extract response data
        log.output_raw = response.content[0].text
        log.input_tokens = response.usage.input_tokens
        log.output_tokens = response.usage.output_tokens

        # Try to parse as JSON
        log.output_parsed, log.parse_success, errors = self._parse_json_response(
            log.output_raw
        )
        log.parse_errors.extend(errors)

        return log

    def _parse_json_response(
        self, raw: str
    ) -> tuple[Optional[dict], bool, list[str]]:
        """
        Try to extract JSON from response.

        Handles common cases:
        - Pure JSON
        - JSON in markdown code blocks
        - JSON with surrounding text
        """
        errors = []

        # Try direct parse first
        try:
            return json.loads(raw), True, []
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        code_block = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw)
        if code_block:
            try:
                return json.loads(code_block.group(1)), True, []
            except json.JSONDecodeError as e:
                errors.append(f"JSON in code block invalid: {e}")

        # Try finding JSON object/array anywhere
        for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
            match = re.search(pattern, raw)
            if match:
                try:
                    return json.loads(match.group()), True, []
                except json.JSONDecodeError:
                    pass

        errors.append("Could not extract valid JSON from response")
        return None, False, errors

    def call_triplet_extractor(
        self,
        walk: list[dict],
        reference_bible: str = "mojave",
    ) -> SubagentLog:
        """
        Convenience method for triplet extraction.

        Args:
            walk: Dialogue nodes from graph API
            reference_bible: Source setting ID

        Returns:
            SubagentLog containing parsed StructuralTriplet (if successful)
        """
        payload = {
            "walk": walk,
            "reference_bible": reference_bible,
        }

        user_message = json.dumps(payload, indent=2)

        return self.call(
            subagent_type=SubagentType.TRIPLET_EXTRACTOR,
            user_message=user_message,
            input_payload=payload,
        )

    def call_translation_engine(
        self,
        triplet: dict,
        source_bible: str,
        target_bible: str,
        target_bible_content: Optional[str] = None,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> SubagentLog:
        """
        Convenience method for translation.

        Args:
            triplet: Structural triplet from extractor
            source_bible: Source setting ID
            target_bible: Target setting ID
            target_bible_content: Full bible YAML (required for good translation)
            few_shot_examples: Optional examples to include

        Returns:
            SubagentLog containing TranslationResult (if successful)
        """
        payload = {
            "triplet": triplet,
            "source_bible": source_bible,
            "target_bible": target_bible,
        }

        # Build user message
        parts = [
            "## Structural Triplet to Translate",
            "```json",
            json.dumps(triplet, indent=2),
            "```",
            "",
            f"Source setting: {source_bible}",
            f"Target setting: {target_bible}",
        ]

        if target_bible_content:
            parts.extend([
                "",
                "## Target Bible",
                "```yaml",
                target_bible_content,
                "```",
            ])

        if few_shot_examples:
            parts.extend([
                "",
                "## Few-Shot Examples",
                "```yaml",
                json.dumps(few_shot_examples, indent=2),
                "```",
            ])

        parts.extend([
            "",
            "Please translate this triplet to the target setting.",
            "Output valid JSON matching the TranslationResult schema.",
        ])

        user_message = "\n".join(parts)

        return self.call(
            subagent_type=SubagentType.TRANSLATION_ENGINE,
            user_message=user_message,
            input_payload=payload,
        )

    def call_lore_curator(
        self,
        proposal_type: str,
        proposal: dict,
        bible_content: str,
    ) -> SubagentLog:
        """
        Convenience method for curator validation.

        Args:
            proposal_type: 'proper_noun', 'faction', 'tension', etc.
            proposal: The proposed addition
            bible_content: Current bible YAML

        Returns:
            SubagentLog containing CuratorDecision (if successful)
        """
        payload = {
            "proposal_type": proposal_type,
            "proposal": proposal,
        }

        user_message = f"""## Proposal Type: {proposal_type}

## Proposed Addition
```json
{json.dumps(proposal, indent=2)}
```

## Current Bible
```yaml
{bible_content}
```

Please validate this proposal against the bible.
Output valid JSON matching the CuratorDecision schema.
"""

        return self.call(
            subagent_type=SubagentType.LORE_CURATOR,
            user_message=user_message,
            input_payload=payload,
        )


# Convenience functions for simple use cases

def extract_triplet(
    walk: list[dict],
    reference_bible: str = "mojave",
    prompts_dir: str = "claudefiles/subagents",
) -> tuple[Optional[dict], SubagentLog]:
    """
    One-shot triplet extraction.

    Returns:
        (triplet_dict, log) - triplet is None if extraction failed
    """
    caller = SubagentCaller(prompts_dir=prompts_dir)
    log = caller.call_triplet_extractor(walk, reference_bible)
    return log.output_parsed, log


def translate_triplet(
    triplet: dict,
    source_bible: str,
    target_bible: str,
    target_bible_content: str,
    prompts_dir: str = "claudefiles/subagents",
) -> tuple[Optional[dict], SubagentLog]:
    """
    One-shot triplet translation.

    Returns:
        (translation_dict, log) - translation is None if failed
    """
    caller = SubagentCaller(prompts_dir=prompts_dir)
    log = caller.call_translation_engine(
        triplet, source_bible, target_bible, target_bible_content
    )
    return log.output_parsed, log


def validate_addition(
    proposal_type: str,
    proposal: dict,
    bible_content: str,
    prompts_dir: str = "claudefiles/subagents",
) -> tuple[Optional[dict], SubagentLog]:
    """
    One-shot curator validation.

    Returns:
        (decision_dict, log) - decision is None if failed
    """
    caller = SubagentCaller(prompts_dir=prompts_dir)
    log = caller.call_lore_curator(proposal_type, proposal, bible_content)
    return log.output_parsed, log
