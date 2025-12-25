# Prose Expansion Specification
 
**Date**: 2025-12-25
**Status**: Ready for implementation
 
## Overview
 
Convert trajectory-structured dialogue into narrated prose at multiple reading levels. Leverage prefix caching for efficiency: same prompt prefix (examples + bible + framing) across all trajectories, with only the trajectory sample and target FK level varying.
 
## Architecture
 
```
                    ┌──────────────────────────────────────┐
                    │         CACHED PREFIX                │
                    │  • FK examples at grades 0,3,6,9     │
                    │  • Lore bible (setting-specific)     │
                    │  • Task framing                      │
                    └──────────────────────────────────────┘
                                     │
         ┌───────────────┬───────────┼───────────┬───────────────┐
         ▼               ▼           ▼           ▼               ▼
    trajectory_0    trajectory_1    ...    trajectory_N    (suffix only)
         │               │           │           │
         ▼               ▼           ▼           ▼
    ×4 FK levels    ×4 FK levels    ...    ×4 FK levels
         │               │           │           │
         ▼               ▼           ▼           ▼
    rejection      rejection        ...    rejection
    filter         filter                  filter
         │               │           │           │
         └───────────────┴───────────┴───────────┘
                         │
                         ▼
                  prose_corpus.jsonl
```
 
## Prompt Structure
 
### Prefix (Cached)
 
```
You are expanding dialogue trajectories into narrated prose at specific
reading levels. The Flesch-Kincaid grade level indicates target complexity:
 
GRADE 0 (age 5-6): Very short sentences. Common words only. "See Spot run."
GRADE 3 (age 8-9): Simple sentences. Basic vocabulary. Clear cause-effect.
GRADE 6 (age 11-12): Compound sentences. Some domain vocabulary. Subplots OK.
GRADE 9 (age 14-15): Complex sentences. Abstract concepts. Nuanced emotion.
 
## Examples by Grade Level
 
### Grade 0 Example
Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."
 
Output:
The man held up three fingers. "Three days," he said.
The boy came back. The man was mad. "Time is up," he said.
 
### Grade 3 Example
Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."
 
Output:
The soldier gave a warning. "You have three days to leave," he said.
Two days later, the soldier returned. He frowned when he saw the family
still there. "I told you three days. Time's up now."
 
### Grade 6 Example
Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."
 
Output:
The officer's voice was flat, businesslike. "Three days. That's what
you have to settle your affairs and vacate the premises."
 
When he returned forty-eight hours later, his expression shifted from
neutral to something harder. "Still here? I was hoping you'd prove
smarter than the last family." He pulled out a form. "Time's up.
We do this the official way now."
 
### Grade 9 Example
Input trajectory:
- [neutral] "Three days."
- [neutral] "Still here?"
- [anger] "Time's up."
 
Output:
"Three days." The bureaucrat didn't look up from his ledger as he
spoke, the words carrying the weight of procedure rather than
personal animosity. It was simply how these things were done.
 
The second visit carried a different tenor. He paused at the
threshold, surveying the unchanged interior with something between
disappointment and resignation. "Still here?" The question was
almost rhetorical. They both knew the answer, and they both knew
what came next.
 
"Time's up." This time he met her eyes, and whatever professional
detachment he'd maintained had calcified into something colder.
The machinery of displacement would proceed regardless of
individual sentiment.
 
---
 
## Target Setting
 
{lore_bible_content}
 
---
 
## Task
 
Expand the following trajectory to narrated prose at Flesch-Kincaid
grade level {target_fk_level}. Preserve the emotion arc and beat
structure. Use proper nouns and idiom from the setting bible.
Speaker attribution style should match the grade level (simpler
at lower levels, varied at higher levels).
 
Trajectory:
```
 
### Suffix (Per-Request)
 
```
{trajectory_json}
 
Flesch-Kincaid target: Grade {0|3|6|9}
 
Prose:
```
 
## Trajectory Format
 
Input trajectories from `trajectories.json`:
 
```json
{
  "arc": {
    "shape": "escalating_threat",
    "emotions": ["neutral", "neutral", "anger"]
  },
  "beats": [
    {"emotion": "neutral", "function": "establish_stakes", "target_text": "Seventy-two hours."},
    {"emotion": "neutral", "function": "threaten", "target_text": "The Hexagon expects compliance."},
    {"emotion": "anger", "function": "react", "target_text": "The Leclerc is outside."}
  ],
  "archetype_relation": "authority_to_subject",
  "source_game": "falloutnv"
}
```
 
## Output Format
 
```json
{
  "prose": "In the prefecture office, a clerk shuffled papers...",
  "trajectory_id": "traj_a8c3d2e1",
  "target_fk_level": 3,
  "measured_fk_level": 3.2,
  "word_count": 89,
  "setting": "gallia",
  "source_arc": "escalating_threat",
  "beat_count": 3,
  "passed_filters": true
}
```
 
## Rejection Filters
 
1. **FK Score**: `|measured - target| <= 1.5 grade levels`
2. **Word Count**: `20 <= words <= 500` (scale with beat count)
3. **Proper Noun Check**: All names/places exist in bible OR are plausible compounds
4. **Beat Coverage**: All trajectory beats appear in output (fuzzy match)
5. **No Meta-Commentary**: Reject if contains "As an AI", "I'll write", etc.
 
Optional (if quality issues emerge):
- Small classifier for "natural prose" vs "AI slop"
- Semantic similarity to trajectory (embedding distance)
 
## Efficiency Model
 
Assumptions:
- Prefix: ~3000 tokens (examples + bible + framing)
- Suffix: ~200 tokens (trajectory + FK target)
- Output: ~150 tokens average
 
With prefix caching:
- First request per prefix: 3200 input tokens
- Subsequent requests: ~200 input tokens (cache hit on prefix)
- 4 FK levels × N trajectories = 4N requests
- Effective input cost: `3000 + (4N × 200)` vs `4N × 3200` without caching
- At N=100 trajectories: 83K tokens vs 1.28M tokens (15× savings)
 
## Concurrency
 
Prefix caching enables high parallelism:
- Same prefix across all concurrent requests
- No dependency between trajectories
- Batch dispatch: 50-100 requests per wave
- Rate limit is the constraint, not architecture
 
Suggested dispatch pattern:
```python
async def expand_trajectories(trajectories, bible, fk_levels=[0,3,6,9]):
    prefix = build_prefix(bible)
 
    tasks = []
    for traj in trajectories:
        for fk in fk_levels:
            suffix = build_suffix(traj, fk)
            tasks.append(call_llm(prefix, suffix))
 
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if passes_filters(r)]
```
 
## Implementation Path
 
1. **Build prefix template** with FK examples at 0/3/6/9
2. **Add FK measurement** (textstat library or similar)
3. **Implement rejection filters**
4. **Batch dispatch** against existing trajectories
5. **Collect** → filter → compile to `prose_training.jsonl`
 
## Open Questions
 
- Should we include the source_text (hashed ref) in trajectory input, or is target_text sufficient?
- Do we want FK modes at 0/3/6/9 or different distribution (0/2/4/6/8)?
- Should bible be truncated per-trajectory (relevant section only) or full?
- Retry budget: how many attempts before giving up on a trajectory×level combo?