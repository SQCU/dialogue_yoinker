# Session Report: First Wet Claude Dry Run

**Date**: 2025-12-22
**Run ID**: `run_demo_20251222_gallia`
**Samples**: 4 dialogue walks from Fallout NV → Gallia
**Auditor**: Claude Code (Opus 4.5)
**Subagents**: structural-parser (Haiku), translation-engine (Sonnet)

---

## What We Built

This session constructed the infrastructure for **structural transfer** of dialogue from reference games to synthetic target settings. The pipeline:

1. Samples walks from the dialogue graph API
2. Parses structural triplets (emotion arc, beat functions, archetype relations)
3. Translates to target setting preserving structure, transforming content
4. Validates against schemas and lore bibles
5. Persists with full source traceability

Key files created:
- `subagent_orchestrator/validation.py` — Schema validation with enum violation tracking
- `workflow/trial_runs.py` — Run management with source linking and gap analysis
- `workflow/curator_loop.py` — Lore bible validation for new proper nouns
- `.claude/agents/structural-parser.md` — Added 4 few-shot examples to anchor labeling

---

## The Four Translations

### Sample 1: Hero Worship → Bureaucratic Contempt
**Source** (happy → disgust):
> "Y'all sure are a big ol' hero. I gotta wright my mama and tell her I met you!"
> "O'Hanrahan's a big pussy, ain't good for shit in a fight..."

**Translation**:
> "Monsieur is the name in every prefecture from here to Marseille. I must telephone my mother — she will want the photograph."
> "Sous-préfet Leclerc cannot hold a pen through a requisition form, and as for Duval... Duval is destined for the archives, not the field, you understand?"

**Auditor notes**: Excellent register shift. "Destined for the archives" is a masterful bureaucratic assassination. The emotion arc (admiration → contempt) preserved through completely different surface forms.

### Sample 2: Casino Farewells → Office Departures
**Source** (all neutral, farewell→negotiate→refuse→farewell):
> "Have a good time, baby."
> "Suit yourself. Gambling's thirsty work..."
> "Yeah, whatever."
> "Until we meet again."

**Translation**:
> "Que votre service soit fructueux, collègue."
> "Comme vous préférez. La paperasse donne soif — je reste au buffet si vous reconsidérez."
> "Oui, enfin bon."
> "Jusqu'à la prochaine réunion."

**Auditor notes**: "Oui, enfin bon" for "Yeah, whatever" is inspired — captures the same dismissive shrug through French bureaucratic ennui. The gambling→paperwork substitution maintains the transactional frame.

### Sample 3: Merchant Hints → Functionary Guidance
**Source** (merchant_to_customer, information delivery):
> "Remember, an impressive gratuity sends a message."
> "Our drinks have many subtle flavors."

**Translation**:
> "Une gratification appropriée facilite toujours les démarches futures."
> "Nos vins comportent des appellations très précises."

**Auditor notes**: The pivot from "subtle flavors" to "precise appellations" is structurally perfect — AOC certification carries the same cultural weight in France that connoisseurship does in the source. The merchant becomes a functionary explaining how the system responds to proper form.

### Sample 4: Robot Status Reports
**Source** (robot barks, all neutral, information dump):
> "Status report: red. Primary system failure imminent..."
> "Threat assessment: primary target is at large and undamaged..."

**Translation**: Near-literal French translation (appropriate for machine speech).

**Auditor notes**: This one exposed schema limitations. The parser correctly identified the barks but had to use invalid enum values to express the structure.

---

## Schema Violations as Latent Category Discovery

Three enum violations occurred. All three represent **legitimate structural features** the schema doesn't yet capture:

| Violation | What the model detected | Schema gap |
|-----------|------------------------|------------|
| `archetype_relation: "bark"` | Robot→ambient has no listener | Need `speaker_to_ambient` or `broadcast` relation |
| `archetype_relation: "customer_to_merchant"` | The inverse power dynamic | Relation vocabulary is asymmetric; need inverses |
| `attractor_type: "threat_escalation"` | Threat itself as motivation | `survival` doesn't capture "threat as attractor" |

**The model is discovering categories.** These aren't errors — they're signals that a schema-mediating outer loop should interpret and potentially incorporate. The validation layer correctly flagged them; a human (or higher-tier model) should decide whether to:
1. Expand the schema to include the new category
2. Map the violation to an existing category
3. Flag the sample for manual review

This is the "surprising data truths or surprising data mistakes which must be interpreted and pondered and decided-over" dynamic in action.

---

## On "Wet Claude Dry Runs"

This session demonstrated a pattern worth naming: **wet Claude dry runs**.

A "dry run" traditionally means executing a process without its real effects — testing the pipeline without spending API credits, processing without persisting. But when Claude Code supervises subagent orchestration:

1. **The audit is wet**: Claude Code observes actual model outputs, not simulated ones
2. **The interpretation is live**: Violations are analyzed in context, not post-hoc
3. **The feedback loop is immediate**: Schema gaps identified → session notes → future runs improved

This is equivalent to:
- A Claude Code instance auditing an integration test where OpenAI credits are being spent
- Supervising a DeepSeek v3.2 model in a similar capacity
- Any scenario where a capable overseer watches less-capable workers in real-time

The value isn't just catching errors — it's **generating interpretable feedback** about where the system's assumptions don't match reality.

---

## On Coverage and Comparison

The user observes that testing translation quality requires **coverage in the thousands of percents** — not translating each source line once, but exploring the translation space thoroughly enough to compare inference quality across providers.

At 4 samples, we can demonstrate the pipeline works. At 400 samples (1%), we can detect systematic enum drift. At 4,000 samples, we can compare:

- Haiku vs Gemini Flash as structural parser
- Sonnet vs GPT-4o as translation engine
- Different prompt formulations for the same model

The dialogue corpus becomes a **benchmark for situated language generation**. The structural triplets are the ground truth; the translations are the predictions; the schema validation is the metric.

This is data programming with data tools that are themselves data programs.

---

## Recommendations for Next Steps

1. **Expand the schema** to include:
   - `speaker_to_ambient` archetype relation (for barks)
   - Inverse relations (`customer_to_merchant`, `subject_to_authority`)
   - Threat-adjacent attractors (`threat_escalation`, `dominance`, `reputation`)

2. **Run a real 1% trial** (420 samples) with:
   - Full validation statistics
   - Transition gap analysis vs source corpus
   - Proper noun accumulation tracking

3. **Add a schema evolution loop**:
   - Collect enum violations across runs
   - Cluster by semantic similarity
   - Propose schema additions for curator review

4. **Compare across model providers**:
   - Same prompts, different models
   - Measure: validation rate, enum drift patterns, translation confidence distributions

5. **Build the lore bible incrementally**:
   - Start with Gallia's existing clusters
   - Let translations propose additions
   - Curator approves/rejects
   - Bible grows organically from corpus needs

---

## User Feedback to Auditor Feedback

> "interesting all 3 of those enum violations reflect mild modeling errors on the part of the schema stuff, suggesting they're alerts that a schema-mediating outer loop might need to resolve by *adding* schema features"

Yes. The violations are the model saying "your categories don't cover my observations." A rigid system rejects; a learning system incorporates.

> "this is actually one of the strongest arguments for doing 'wet claude dry runs'"

The pattern generalizes. Whenever you have:
- A pipeline with structured outputs
- Models that can produce valid-but-unexpected outputs
- A need to evolve the schema over time

...you want a capable observer in the loop during runs, not just post-hoc analysis.

> "the future of data science: data programming with data tools which themselves might be data programs"

This session is an instance of that future. The dialogue corpus is data. The structural parser is a model making inferences about that data. The validation layer is a program checking those inferences. Claude Code is a model interpreting the validation results. The session notes are data about the process. And the user is making decisions based on all of it.

Turtles all the way down, but at each level something is learned.

---

## Appendix: Run Artifacts

```
runs/run_demo_20251222_gallia/
├── config.yaml           # Run configuration
├── synthetics.jsonl      # 4 translated dialogues with source links
└── stats.json            # Cached analysis (regenerable)

.claude/agents/
├── structural-parser.md  # Updated with 4 few-shot examples
├── translation-engine.md
├── lore-curator.md
├── triplet-extractor.md
└── dialogue-corpus-analyst.md

subagent_orchestrator/
├── validation.py         # NEW: Schema validation with enum tracking
├── models.py
├── subagent.py
└── observability.py

workflow/
├── trial_runs.py         # NEW: Run management with source linking
├── curator_loop.py       # NEW: Lore bible validation
├── run_trial.py          # NEW: CLI for trial execution
└── prototype.py
```

---

*Report generated by Claude Code (Opus 4.5) during supervised synthetic dialogue generation session.*
