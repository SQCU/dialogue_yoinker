# Stats-Guided Growth Orchestration Spec

## Goal

Synthetic graphs become locally similar to reference after 1% warmup (~500 nodes) and stay similar through 300% of reference size (~150k nodes).

## Current Infrastructure

**What exists:**
- `stats_guided_growth.py` - ReferenceCorpus, gap identification, emotion/arc sampling
- `workflow/guided_sampling.py` - bridges stats to ticket queue
- `run_batch.py` - translate/link/extend operations
- `--guided` flag for stats-aware sampling

**What's tracked:**
- Emotion transition matrix
- Arc shape distribution
- Walk length distribution
- Topic degree distribution

**What's missing:**
- Node degree distribution (not topic degree)
- Hub density metric
- Chain ratio metric
- Topic hub aggregation operation
- Closed-loop controller

## Implementation Plan

### Phase 1: Add Missing Metrics

```python
# In stats_guided_growth.py, add to ReferenceStats:

@dataclass
class ReferenceStats:
    # ... existing fields ...

    # NEW: Node-level degree distributions
    node_out_degrees: Dict[int, int] = field(default_factory=dict)
    node_in_degrees: Dict[int, int] = field(default_factory=dict)

    # NEW: Structural metrics
    hub_density: float = 0.0      # nodes with degree > 5 per 100 nodes
    chain_ratio: float = 0.0      # fraction of nodes with in=out=1
    leaf_ratio: float = 0.0       # fraction with out=0
    entry_ratio: float = 0.0      # fraction with in=0
```

### Phase 2: Topic Hub Aggregation Operation

```python
# New file: scripts/aggregate_topic_hubs.py

def aggregate_topic_hubs(graph_path: Path) -> dict:
    """
    Create hub nodes that aggregate arcs by synthetic_topic.

    This mimics reference DIAL→INFO structure where topics are hubs.
    """
    graph = json.loads(graph_path.read_text())
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Group arc entries by topic
    topics = defaultdict(list)
    for node in nodes:
        if node.get("beat_index") == 0:  # arc entry
            topic = node.get("topic") or node.get("arc_shape") or "unknown"
            topics[topic].append(node["id"])

    # Create hub for topics with multiple entries
    new_nodes = []
    new_edges = []

    for topic, entry_ids in topics.items():
        if len(entry_ids) < 2:
            continue

        hub_id = f"hub_{hashlib.sha256(topic.encode()).hexdigest()[:8]}"
        hub_node = {
            "id": hub_id,
            "text": f"[TOPIC: {topic}]",
            "emotion": "neutral",
            "node_type": "topic_hub",
            "topic": topic,
        }
        new_nodes.append(hub_node)

        for entry_id in entry_ids:
            new_edges.append({
                "source": hub_id,
                "target": entry_id,
                "type": "topic_branch",
            })

    graph["nodes"].extend(new_nodes)
    graph["edges"].extend(new_edges)
    graph_path.write_text(json.dumps(graph, indent=2))

    return {"hubs_created": len(new_nodes), "edges_added": len(new_edges)}
```

### Phase 3: Closed-Loop Controller

```python
# New file: scripts/growth_controller.py

@dataclass
class GrowthState:
    """Current state of growth process."""
    nodes: int
    edges: int
    phase: str  # "warmup", "growth", "extension"
    stats: ReferenceStats
    gaps: List[StatisticalGap]

class GrowthController:
    """
    Closed-loop controller for stats-guided growth.
    """

    def __init__(self, setting: str, version: int, target_size: int):
        self.setting = setting
        self.version = version
        self.target_size = target_size
        self.reference = ReferenceCorpus()
        self.reference.load(["oblivion", "falloutnv"])
        self.ref_stats = self.reference.compute_stats().normalize()

        # Thresholds
        self.warmup_size = 500
        self.hub_aggregation_interval = 1000
        self.stats_check_interval = 500

        # Gap thresholds for action selection
        self.chain_excess_threshold = 0.1
        self.hub_deficit_threshold = 0.02
        self.degree_divergence_threshold = 0.1

    def get_state(self) -> GrowthState:
        """Measure current graph state."""
        graph_path = Path(f"synthetic/{self.setting}_v{self.version}/graph.json")
        graph = json.loads(graph_path.read_text())

        stats = compute_synthetic_stats(graph)
        gaps = compute_gaps(stats, self.ref_stats)

        nodes = len(graph["nodes"])
        phase = (
            "warmup" if nodes < self.warmup_size else
            "growth" if nodes < self.target_size else
            "extension"
        )

        return GrowthState(
            nodes=nodes,
            edges=len(graph["edges"]),
            phase=phase,
            stats=stats,
            gaps=gaps,
        )

    def select_action(self, state: GrowthState) -> str:
        """
        Select next action based on statistical gaps.

        Returns: "translate", "link", "aggregate", "extend"
        """
        # Extract key gaps
        chain_excess = state.stats.chain_ratio - self.ref_stats.chain_ratio
        hub_deficit = self.ref_stats.hub_density - state.stats.hub_density

        # Priority order
        if chain_excess > self.chain_excess_threshold:
            return "link"
        elif hub_deficit > self.hub_deficit_threshold:
            return "aggregate"
        elif state.phase == "extension":
            return "extend"
        else:
            return "translate"

    def step(self, batch_size: int = 50) -> dict:
        """
        Execute one step of growth.
        """
        state = self.get_state()
        action = self.select_action(state)

        result = {"action": action, "state_before": asdict(state)}

        if action == "translate":
            # Sample walks guided by emotion/arc gaps
            run_id = create_translation_run(
                self.setting, batch_size, self.version, guided=True
            )
            process_translation(run_id)
            compile_results(run_id, self.setting, self.version)
            result["run_id"] = run_id

        elif action == "link":
            # Add cross-arc connections
            run_id = create_linking_run(
                self.setting, self.version, batch_size, guided=True
            )
            process_linking(run_id)
            apply_results(run_id, self.setting, self.version)
            result["run_id"] = run_id

        elif action == "aggregate":
            # Create topic hubs
            result.update(aggregate_topic_hubs(
                Path(f"synthetic/{self.setting}_v{self.version}/graph.json")
            ))

        elif action == "extend":
            # Resolve extension candidates
            run_id = create_extension_run(...)
            process_extension(run_id)
            apply_results(run_id, self.setting, self.version)
            result["run_id"] = run_id

        # Measure after
        result["state_after"] = asdict(self.get_state())

        return result

    def run_to_target(self) -> List[dict]:
        """
        Run growth loop until target size reached.
        """
        history = []

        while True:
            state = self.get_state()

            if state.nodes >= self.target_size:
                break

            # Determine batch size by phase
            batch_size = {
                "warmup": 100,
                "growth": 200,
                "extension": 500,
            }[state.phase]

            result = self.step(batch_size)
            history.append(result)

            # Periodic hub aggregation
            if state.nodes % self.hub_aggregation_interval == 0:
                agg_result = aggregate_topic_hubs(...)
                history.append({"action": "aggregate", **agg_result})

            # Log progress
            print(f"[{state.phase}] {state.nodes} nodes, action={result['action']}")

        return history
```

### Phase 4: CLI Integration

```bash
# Add to run_batch.py or new script

# Run warmup (0 → 500 nodes)
python scripts/growth_controller.py warmup gallia:7 --target 500

# Run growth (500 → 49k nodes)
python scripts/growth_controller.py grow gallia:7 --target 49000

# Run extension (49k → 150k nodes)
python scripts/growth_controller.py extend gallia:7 --target 150000

# Full auto-run
python scripts/growth_controller.py auto gallia:7 --target 150000
```

## Key Insights

1. **Topic hub aggregation** is the main structural operation needed - it creates the heavy-tailed degree distribution that reference graphs have.

2. **Action selection** based on gap analysis ensures we're always closing the largest statistical gap.

3. **Phase-aware batch sizing** matches the natural rhythm of each phase.

4. **The chain_ratio metric** is the key indicator of when to link vs translate - high chain ratio means too linear, need more cross-connections.

5. **Hub density metric** indicates when to run topic aggregation.

## Measured Reference Baselines (Dec 2024)

| Metric | Oblivion | FalloutNV | gallia_v6 (351 nodes) |
|--------|----------|-----------|------------------------|
| chain_ratio | 53.6% | 27.5% | 32.2% ✓ |
| leaf_ratio | 15.0% | 39.8% | 10.8% |
| hub_density | 34.6% | 51.4% | 16.2% (needs +18-35%) |
| entry_ratio | 1.4% | 4.0% | 2.0% |
| edges_per_node | 5.5 | 20.6 | 2.2 (needs 2.5-9x more) |

Key finding: chain_ratio is already within reference range after linking/extension.
The main gaps are hub_density and edges_per_node.

## Success Criteria

After warmup (500 nodes):
- Chain ratio: 27-54% (within reference range)
- Hub density: >25% (halfway to reference minimum)
- Edges/node: >4.0 (approaching Oblivion baseline)
- Emotion transition JS-divergence < 0.1
- Arc shape chi-squared p-value > 0.05

Maintained through growth/extension:
- Same metrics stay within bounds
- No regression as graph scales
