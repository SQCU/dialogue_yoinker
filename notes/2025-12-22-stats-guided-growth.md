# Stats-Guided Graph Growth

**Date**: 2025-12-22
**Status**: Implementation approach

## The Simplification

We don't need:
- State machine parsing of games
- Complex state change annotation
- Search/prune Suttonian approaches

We already have:
- Graph statistics from reference corpus
- Emotion transition matrices
- Topic degree distributions
- Arc shape frequencies
- Sliding window local probes

**The reference corpus IS the model. Sample from its statistics.**

## The Algorithm

```
while target_graph needs growth:
    1. MEASURE: Compute stats of current target graph
    2. COMPARE: Find gaps between target stats and reference stats
       - Underrepresented emotion transitions?
       - Missing arc shapes?
       - Degree distribution skew?

    3. SAMPLE: Query reference corpus for walks that would close the gap
       - "Give me walks with anger→neutral transition"
       - "Give me negotiation_arc with merchant_to_customer"
       - "Give me high-degree hub connections"

    4. GENERATE: Use sampled walk as template, generate target-setting analogue
       - Local similarity to template is FINE
       - The divergence comes from WHERE we attach it

    5. ATTACH: Connect new content to target graph
       - Choose attachment point to improve global stats
       - The topology differs even if local texture is similar
```

## Why This Works

The reference graph has certain statistical properties:
- 74.6% emotion self-loop rate
- Power-law-ish topic degree distribution
- Specific arc shape frequencies

If we sample new edges/nodes to match these statistics, the target graph will be
**statistically similar** to reference but **topologically different** because:
- We're building a new graph, not copying the old one
- Attachment points are chosen to balance stats, not to mirror source
- Same local texture, different global structure

## What We Already Have

From `cross_game.py` and `emotion_bridge.py`:
- Emotion transition matrices per game
- Cross-game emotion bridges
- Topic connectivity stats

From `topic_graph.py`:
- Topic→topic transition frequencies
- Hub detection (high-degree topics)
- Chain/path analysis

From `query_graph.py`:
- Sliding window emotion sequences
- Local neighborhood probes
- Path sampling

## Implementation Sketch

```python
class StatsGuidedGrowth:
    def __init__(self, reference_stats, target_graph):
        self.ref = reference_stats  # From reference corpus
        self.target = target_graph

    def identify_gap(self):
        """Find biggest statistical gap between target and reference."""
        target_stats = compute_stats(self.target)

        gaps = []
        for stat_name, ref_value in self.ref.items():
            target_value = target_stats.get(stat_name, 0)
            gap = abs(ref_value - target_value)
            gaps.append((stat_name, gap, ref_value, target_value))

        return max(gaps, key=lambda x: x[1])

    def sample_reference_for_gap(self, gap):
        """Find reference walks that would help close this gap."""
        stat_name, _, ref_value, target_value = gap

        if stat_name.startswith("emotion_transition_"):
            # Find walks with this emotion transition
            return query_walks_by_emotion_transition(stat_name)

        elif stat_name.startswith("arc_shape_"):
            # Find walks with this arc shape
            return query_walks_by_arc_shape(stat_name)

        # etc.

    def generate_and_attach(self, reference_walk, attachment_point):
        """Generate target-setting analogue and attach to graph."""
        # Generate (accepting local similarity)
        new_content = translate_walk(reference_walk, target_setting)

        # Attach at point that improves stats
        self.target.add_edge(attachment_point, new_content)
```

## The Key Insight

> We don't have to do hardly any complicated tricks at all.

The statistics ARE the model. We're not learning anything new - we're just
ensuring the synthetic graph has the same statistical signature as the reference.

Local similarity is fine. Global topology differs because we're constructing
a new graph guided by stats, not copying structure.

## Existing Tools to Use

- `GET /api/transitions/{game}` - emotion transition matrix
- `GET /api/stats/{game}` - graph statistics
- `POST /api/sample` with emotion/arc filters - get walks matching criteria
- `query_graph.py` sliding window - local neighborhood stats

Just wire these together with a growth loop.
