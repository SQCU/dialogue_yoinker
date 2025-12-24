#!/usr/bin/env python3
"""
Dialogue Graph API Server

A simple async REST API for exploring dialogue graphs.
Self-documenting via OpenAPI at /docs.

Run with: uvicorn api_server:app --reload
Or: python api_server.py

Endpoints:
    GET  /                      - Landing page with simple visualization
    GET  /api                   - API overview (for Claude/LLM discovery)
    GET  /api/games             - List available parsed games
    GET  /api/stats/{game}      - Graph statistics for a game
    GET  /api/transitions/{game} - Emotion transition matrix
    POST /api/sample            - Sample chains/walks from graph
    POST /api/subgraph          - Extract subgraph around a node
    GET  /docs                  - OpenAPI interactive documentation
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Local imports
from dialogue_graph import DialogueGraph, load_dialogue
from chain_linker import ChainLinker
from topic_graph import TopicGraph
from cross_game import CrossGameLinker
from emotion_bridge import EmotionBridgeGraph, load_emotion_bridge
from query_graph import QueryGraph, load_query_graph

# Synthetic dialogue generation (optional - requires anthropic package)
try:
    from subagent_orchestrator.api_routes import router as synthetic_router
    SYNTHETIC_AVAILABLE = True
except ImportError:
    SYNTHETIC_AVAILABLE = False
    synthetic_router = None

# Ticket queue for decoupled orchestration
try:
    from workflow.ticket_routes import router as ticket_router
    from workflow.dispatch_routes import router as dispatch_router
    TICKET_QUEUE_AVAILABLE = True
except ImportError:
    TICKET_QUEUE_AVAILABLE = False
    ticket_router = None
    dispatch_router = None

# Synthetic data routes (compiled translations)
try:
    from workflow.synthetic_routes import router as synthetic_data_router
    SYNTHETIC_DATA_AVAILABLE = True
except ImportError:
    SYNTHETIC_DATA_AVAILABLE = False
    synthetic_data_router = None

# Graph diagnostics for topology comparison
try:
    from graph_diagnostics import analyze_graph
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

# Landing page HTML (extracted to reduce file size)
from landing_html import LANDING_HTML


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("dialogue_data")
SYNTHETIC_DIR = Path("synthetic")
CACHE: Dict[str, DialogueGraph] = {}
TOPIC_CACHE: Dict[str, TopicGraph] = {}
CROSS_GAME_CACHE: Optional[CrossGameLinker] = None
BRIDGE_CACHE: Optional[EmotionBridgeGraph] = None
QUERY_CACHE: Optional[QueryGraph] = None
SYNTHETIC_GRAPH_CACHE: Dict[str, Any] = {}


# =============================================================================
# Pydantic Models (self-documenting request/response schemas)
# =============================================================================

class GameInfo(BaseModel):
    """Information about an available game dataset."""
    name: str = Field(description="Game identifier (e.g., 'oblivion', 'falloutnv')")
    dialogue_count: int = Field(description="Number of dialogue lines")
    file_path: str = Field(description="Path to dialogue JSON file")


class GraphStats(BaseModel):
    """Statistics about a dialogue graph."""
    nodes: int = Field(description="Number of dialogue nodes")
    edges: int = Field(description="Number of edges (transitions)")
    topics: int = Field(description="Unique dialogue topics")
    quests: int = Field(description="Unique quest contexts")
    speakers: int = Field(description="Unique named speakers")
    edge_types: Dict[str, int] = Field(description="Count by edge type")
    avg_in_degree: float = Field(description="Average incoming connections")
    avg_out_degree: float = Field(description="Average outgoing connections")
    max_in_degree: int = Field(description="Most connected (incoming)")
    max_out_degree: int = Field(description="Most connected (outgoing)")
    isolated_nodes: int = Field(description="Nodes with no connections")


class TransitionMatrix(BaseModel):
    """Emotion transition counts between dialogue lines."""
    game: str
    transitions: Dict[str, Dict[str, int]] = Field(
        description="Nested dict: source_emotion -> target_emotion -> count"
    )
    total_transitions: int


class SampleRequest(BaseModel):
    """Request to sample from dialogue graph."""
    game: str = Field(description="Game to sample from")
    method: str = Field(
        default="walk",
        description="Sampling method: 'walk' (random walk), 'chain' (quest chain), 'hub' (from high-degree node)"
    )
    count: int = Field(default=3, ge=1, le=20, description="Number of samples")
    max_length: int = Field(default=8, ge=2, le=50, description="Max nodes per sample")
    allow_cycles: bool = Field(default=False, description="Allow revisiting nodes in walks")
    quest_filter: Optional[str] = Field(default=None, description="Filter to specific quest")
    emotion_filter: Optional[str] = Field(default=None, description="Filter to specific emotion")


class DialogueSample(BaseModel):
    """A sampled dialogue sequence."""
    method: str
    length: int
    nodes: List[Dict[str, Any]] = Field(description="Sequence of dialogue nodes")


class SampleResponse(BaseModel):
    """Response containing sampled dialogues."""
    game: str
    samples: List[DialogueSample]


class SubgraphRequest(BaseModel):
    """Request to extract a subgraph."""
    game: str
    center_id: str = Field(description="Form ID of center node")
    radius: int = Field(default=2, ge=1, le=5, description="BFS radius from center")


class SubgraphResponse(BaseModel):
    """A subgraph around a center node."""
    center_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, int]


class PageRankResult(BaseModel):
    """PageRank analysis result."""
    game: str
    top_nodes: List[Dict[str, Any]] = Field(description="Top nodes by PageRank score")


class CommunityResult(BaseModel):
    """Community detection result."""
    game: str
    algorithm: str
    community_count: int
    communities: List[Dict[str, Any]] = Field(description="Detected communities with stats")


class PathRequest(BaseModel):
    """Request to find path between nodes."""
    game: str
    source: str = Field(description="Source node form_id")
    target: str = Field(description="Target node form_id")
    max_length: int = Field(default=10, ge=2, le=30)


class PathResponse(BaseModel):
    """Path between two nodes."""
    source: str
    target: str
    path_length: int
    path: Optional[List[Dict[str, Any]]] = Field(description="Nodes along path, or null if no path")


class CentralityResult(BaseModel):
    """Centrality analysis result."""
    game: str
    metrics: Dict[str, List[Dict[str, Any]]] = Field(
        description="Top nodes by each centrality metric (degree, betweenness, closeness)"
    )


class SCCResult(BaseModel):
    """Strongly connected components result."""
    game: str
    scc_count: int
    components: List[Dict[str, Any]] = Field(description="Non-trivial SCCs (size > 1)")


class TopicGraphStats(BaseModel):
    """Topic graph statistics."""
    game: str
    topics: int
    edges: int
    total_lines: int
    avg_lines_per_topic: float
    avg_degree: float
    max_degree: int
    hubs: List[Dict[str, Any]] = Field(description="Top hub topics by degree")


class TopicChain(BaseModel):
    """A chain of related topics."""
    length: int
    topics: List[Dict[str, Any]]


class TopicPathResult(BaseModel):
    """Topic→text→topic paths from a source topic."""
    source_topic: str
    paths: List[Dict[str, Any]]


class BridgeWalkRequest(BaseModel):
    """Request for cross-game emotion bridge walk."""
    max_steps: int = Field(default=10, ge=2, le=30, description="Maximum walk length")
    start_game: Optional[str] = Field(default=None, description="Starting game (random if not specified)")
    cross_probability: float = Field(default=0.3, ge=0.0, le=1.0, description="Probability of crossing to another game at bridge points")
    prefer_off_diagonal: bool = Field(default=True, description="Prefer interesting (non-neutral) emotion transitions for bridges")


class CoverageRequest(BaseModel):
    """Request for coverage trajectory across games."""
    min_length: int = Field(default=5, ge=2, le=20)
    max_length: int = Field(default=20, ge=5, le=50)
    target_games: Optional[List[str]] = Field(default=None, description="Games to cover (all if not specified)")


class QuerySampleRequest(BaseModel):
    """Request to sample responses for a topic/query."""
    topic: Optional[str] = Field(default=None, description="Specific topic to sample from")
    category: Optional[str] = Field(default=None, description="Semantic category (GREETING, COMBAT, QUEST, etc.)")
    n: int = Field(default=5, ge=1, le=20, description="Number of responses to sample")
    cross_game: bool = Field(default=False, description="Try to sample from different games")


class QueryWalkRequest(BaseModel):
    """Request for topic chain walk."""
    start_topic: Optional[str] = Field(default=None, description="Starting topic (random if not specified)")
    max_steps: int = Field(default=5, ge=2, le=15, description="Maximum chain length")
    cross_game: bool = Field(default=True, description="Prefer cross-game transitions")


# =============================================================================
# Graph Loading & Caching
# =============================================================================

def get_available_games() -> List[GameInfo]:
    """Find all parsed dialogue files."""
    games = []
    for path in DATA_DIR.glob("*_dialogue.json"):
        name = path.stem.replace("_dialogue", "")
        try:
            with open(path) as f:
                data = json.load(f)
            count = len(data.get("dialogue", []))
            games.append(GameInfo(name=name, dialogue_count=count, file_path=str(path)))
        except Exception:
            pass
    return games


async def get_graph(game: str) -> DialogueGraph:
    """Get or load a dialogue graph (with caching)."""
    if game in CACHE:
        return CACHE[game]

    # Try full dialogue first (includes DLCs), then regular
    path = DATA_DIR / f"{game}_full_dialogue.json"
    if not path.exists():
        path = DATA_DIR / f"{game}_dialogue.json"
    if not path.exists():
        raise HTTPException(404, f"Game '{game}' not found. Check /api/games for available games.")

    # Load in executor to not block
    loop = asyncio.get_event_loop()
    dialogue = await loop.run_in_executor(None, load_dialogue, path)
    graph = await loop.run_in_executor(None, DialogueGraph.from_dialogue_data, dialogue)

    CACHE[game] = graph
    return graph


async def get_topic_graph(game: str, filter_hubs: bool = True) -> TopicGraph:
    """Get or load a topic graph (with caching)."""
    cache_key = f"{game}_{'filtered' if filter_hubs else 'raw'}"
    if cache_key in TOPIC_CACHE:
        return TOPIC_CACHE[cache_key]

    # Try full dialogue first
    path = DATA_DIR / f"{game}_full_dialogue.json"
    if not path.exists():
        path = DATA_DIR / f"{game}_dialogue.json"
    if not path.exists():
        raise HTTPException(404, f"Game '{game}' not found.")

    loop = asyncio.get_event_loop()
    dialogue = await loop.run_in_executor(None, load_dialogue, path)
    graph = await loop.run_in_executor(None, TopicGraph.from_dialogue, dialogue)

    if filter_hubs:
        graph = graph.filter_common_hubs()

    TOPIC_CACHE[cache_key] = graph
    return graph


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    print(f"Dialogue Graph API starting...")
    print(f"Data directory: {DATA_DIR.absolute()}")
    games = get_available_games()
    print(f"Available games: {[g.name for g in games]}")
    yield
    CACHE.clear()


app = FastAPI(
    title="Dialogue Graph API",
    description="""
REST API for exploring dialogue graphs extracted from Bethesda games.

## Quick Start (for Claude/LLMs)

1. `GET /api/games` - discover available datasets
2. `GET /api/stats/{game}` - get graph statistics
3. `POST /api/sample` - sample dialogue sequences
4. `GET /api/transitions/{game}` - get emotion transition matrix

## Example Sample Request

```json
POST /api/sample
{
    "game": "oblivion",
    "method": "walk",
    "count": 3,
    "max_length": 6
}
```
    """,
    version="0.1.0",
    lifespan=lifespan
)

# Include synthetic dialogue generation routes if available
if SYNTHETIC_AVAILABLE and synthetic_router:
    app.include_router(synthetic_router)

# Include ticket queue routes if available
if TICKET_QUEUE_AVAILABLE and ticket_router:
    app.include_router(ticket_router)

# Include multi-backend dispatch routes if available
if TICKET_QUEUE_AVAILABLE and dispatch_router:
    app.include_router(dispatch_router)

# Include synthetic data routes if available
if SYNTHETIC_DATA_AVAILABLE and synthetic_data_router:
    app.include_router(synthetic_data_router)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api", response_class=JSONResponse)
async def api_overview():
    """
    API overview for discovery by Claude/LLMs.

    Returns a summary of available endpoints and their purposes.
    """
    return {
        "name": "Dialogue Graph API",
        "description": "REST API for exploring situated dialogue from Bethesda games",
        "endpoints": {
            "GET /api/games": "List available game datasets",
            "GET /api/stats/{game}": "Get graph statistics for a game",
            "GET /api/transitions/{game}": "Get emotion transition matrix",
            "POST /api/sample": "Sample dialogue sequences (walks, chains, hubs)",
            "POST /api/subgraph": "Extract subgraph around a node",
            "GET /api/pagerank/{game}": "PageRank analysis - find important nodes",
            "GET /api/communities/{game}": "Community detection - find dialogue clusters",
            "POST /api/path": "Find shortest path between two nodes",
            "GET /api/centrality/{game}": "Centrality analysis (degree, betweenness, closeness)",
            "GET /api/components/{game}": "Find strongly connected components (dialogue loops)",
            "GET /api/topics/{game}": "Topic graph stats and hubs",
            "GET /api/topics/{game}/chains": "Find topic conversation chains",
            "GET /api/topics/{game}/paths/{topic}": "Topic→text→topic paths",
            "GET /api/topics/{game}/pagerank": "PageRank on topic graph",
            "GET /api/crossgame/stats": "Cross-game linking statistics",
            "GET /api/crossgame/pairs": "Sample cross-game pairs by emotion",
            "GET /api/crossgame/unified": "Unified cross-game graph",
            "GET /api/bridge/stats": "Emotion bridge graph statistics",
            "GET /api/bridge/cells": "Bridge cells (cross-game emotion transitions)",
            "GET /api/bridge/matrix": "Full transition matrix with bridge info",
            "POST /api/bridge/walk": "Cross-game walk via emotion bridges",
            "POST /api/bridge/coverage": "Coverage trajectory across all games",
            "GET /api/bridge/visualization": "D3.js visualization data for bridge graph",
            "GET /api/query/stats": "Query graph stats (topics as gaps)",
            "GET /api/query/topics": "List topics by category",
            "GET /api/query/categories": "Semantic category breakdown",
            "POST /api/query/sample": "Sample responses for a topic/category",
            "POST /api/query/walk": "Topic chain walk (interleaved query-text)",
            "GET /api/query/cross-game": "Topics shared across games",
            "GET /api/query/visualization": "Bipartite topic-text graph data",
            "GET /api/query/transitions": "Topic-to-topic transition stats",
            "GET /api/query/cycles": "Find conversational cycles in topic graph",
            "GET /docs": "Interactive OpenAPI documentation",
            # Synthetic dialogue generation (if anthropic package installed)
            "POST /api/extract/triplet": "Extract structural triplet from dialogue walk",
            "GET /api/bibles": "List available lore bibles",
            "GET /api/bibles/{bible_id}": "Get a specific lore bible",
            "POST /api/bibles/{bible_id}/propose_addition": "Propose addition to bible (curator review)",
            "POST /api/translate/triplet": "Translate triplet to new setting",
            "POST /api/synthetic/persist": "Save synthetic entry to corpus",
            "GET /api/synthetic/split/{target_bible}": "Get synthetics for a setting",
            "GET /api/synthetic/stats/{target_bible}": "Synthetic corpus statistics",
            "GET /api/synthetic/training/{target_bible}": "Export training data (per-beat)",
            "POST /api/workflow/generate": "Full pipeline: extract -> translate -> persist",
            "GET /api/traces": "List workflow traces (observability)",
            "GET /api/traces/stats": "Aggregate trace statistics",
            "GET /api/traces/{workflow_id}": "Get specific workflow trace",
            # Compiled synthetic data (from translation runs)
            "GET /api/synthetic/settings": "List compiled synthetic settings",
            "GET /api/synthetic/{setting}/stats": "Stats for synthetic setting",
            "GET /api/synthetic/{setting}/dialogue": "Raw dialogue entries",
            "GET /api/synthetic/{setting}/trajectories": "Full trajectory view with arcs",
            "POST /api/synthetic/{setting}/sample": "Sample from synthetic data",
            "GET /api/synthetic/{setting}/concept-mappings": "View concept transformations",
            "GET /api/synthetic/{setting}/compare/{source}": "Side-by-side source/target",
            # Synthetic graph topology analysis
            "GET /api/synthetic-graph/settings": "List synthetic settings with graphs",
            "GET /api/synthetic-graph/{setting}/stats": "Topology stats (degree distribution, hubs)",
            "GET /api/synthetic-graph/{setting}/compare/{reference}": "Compare to reference game",
            "GET /api/synthetic-graph/{setting}/pagerank": "PageRank on synthetic graph",
            "GET /api/synthetic-graph/{setting}/centrality": "Hub detection (in/out degree)",
            "GET /api/synthetic-graph/{setting}/communities": "Community detection",
            "GET /api/synthetic-graph/{setting}/components": "Connected components analysis",
        },
        "example_request": {
            "endpoint": "POST /api/sample",
            "body": {
                "game": "oblivion",
                "method": "walk",
                "count": 3,
                "max_length": 6
            }
        },
        "available_games": [g.name for g in get_available_games()]
    }


@app.get("/api/games", response_model=List[GameInfo])
async def list_games():
    """
    List all available parsed game datasets.

    Returns basic info about each game including dialogue count.
    """
    return get_available_games()


@app.get("/api/stats/{game}", response_model=GraphStats)
async def get_stats(game: str):
    """
    Get graph statistics for a specific game.

    Includes node/edge counts, degree distribution, edge type breakdown.
    """
    graph = await get_graph(game)
    stats = graph.get_statistics()
    return GraphStats(**stats)


@app.get("/api/transitions/{game}", response_model=TransitionMatrix)
async def get_transitions(game: str):
    """
    Get the emotion transition matrix for a game.

    Shows how often each emotion leads to each other emotion in the dialogue graph.
    Useful for understanding emotional dynamics of the narrative.
    """
    graph = await get_graph(game)

    transitions: Dict[str, Dict[str, int]] = {}
    total = 0

    for edge in graph.edges:
        if edge.source in graph.nodes and edge.target in graph.nodes:
            src_emo = graph.nodes[edge.source].emotion
            tgt_emo = graph.nodes[edge.target].emotion

            if src_emo not in transitions:
                transitions[src_emo] = {}
            transitions[src_emo][tgt_emo] = transitions[src_emo].get(tgt_emo, 0) + 1
            total += 1

    return TransitionMatrix(game=game, transitions=transitions, total_transitions=total)


@app.post("/api/sample", response_model=SampleResponse)
async def sample_dialogue(request: SampleRequest):
    """
    Sample dialogue sequences from the graph.

    Methods:
    - 'walk': Random walk through the graph
    - 'chain': Quest-based chains (using ChainLinker)
    - 'hub': Start from high-degree hub nodes

    Filters can restrict to specific quests or emotions.
    """
    graph = await get_graph(request.game)
    samples = []

    if request.method == "walk":
        for _ in range(request.count):
            # Optional: start from filtered nodes
            start = None
            if request.quest_filter or request.emotion_filter:
                candidates = [
                    nid for nid, node in graph.nodes.items()
                    if (not request.quest_filter or node.quest == request.quest_filter)
                    and (not request.emotion_filter or node.emotion == request.emotion_filter)
                    and node.outgoing
                ]
                if candidates:
                    import random
                    start = random.choice(candidates)

            path = graph.random_walk(
                start=start,
                max_steps=request.max_length,
                allow_cycles=request.allow_cycles
            )
            nodes = [
                {
                    "id": n.id,
                    "text": n.text,
                    "speaker": n.speaker,
                    "emotion": n.emotion,
                    "topic": n.topic,
                    "quest": n.quest
                }
                for n in path
            ]
            samples.append(DialogueSample(method="walk", length=len(nodes), nodes=nodes))

    elif request.method == "hub":
        hubs = graph.find_hubs(min_degree=5)
        import random
        for _ in range(min(request.count, len(hubs))):
            hub_id = random.choice(hubs[:20])[0]
            path = graph.random_walk(
                start=hub_id,
                max_steps=request.max_length,
                allow_cycles=request.allow_cycles
            )
            nodes = [
                {
                    "id": n.id,
                    "text": n.text,
                    "speaker": n.speaker,
                    "emotion": n.emotion,
                    "topic": n.topic,
                    "quest": n.quest
                }
                for n in path
            ]
            samples.append(DialogueSample(method="hub", length=len(nodes), nodes=nodes))

    elif request.method == "chain":
        # Use ChainLinker for quest-based chains
        path = DATA_DIR / f"{request.game}_dialogue.json"
        dialogue = load_dialogue(path)
        linker = ChainLinker(dialogue)
        chains = linker.sample_chains(n=request.count, min_length=request.max_length)

        for chain in chains:
            nodes = [
                {
                    "id": line.form_id,
                    "text": line.text,
                    "speaker": line.speaker,
                    "emotion": line.emotion,
                    "topic": chain.topic,
                    "quest": chain.quest
                }
                for line in chain.lines[:request.max_length]
            ]
            samples.append(DialogueSample(method="chain", length=len(nodes), nodes=nodes))

    return SampleResponse(game=request.game, samples=samples)


@app.post("/api/subgraph", response_model=SubgraphResponse)
async def extract_subgraph(request: SubgraphRequest):
    """
    Extract a subgraph around a center node.

    Returns all nodes within 'radius' hops of the center, plus connecting edges.
    Useful for examining local dialogue structure.
    """
    graph = await get_graph(request.game)

    if request.center_id not in graph.nodes:
        raise HTTPException(404, f"Node '{request.center_id}' not found in {request.game}")

    sub = graph.sample_subgraph(request.center_id, radius=request.radius)

    nodes = [
        {
            "id": n.id,
            "text": n.text,
            "speaker": n.speaker,
            "emotion": n.emotion,
            "topic": n.topic,
            "quest": n.quest,
            "in_degree": len(n.incoming),
            "out_degree": len(n.outgoing)
        }
        for n in sub.nodes.values()
    ]

    edges = [
        {
            "source": e.source,
            "target": e.target,
            "type": e.edge_type,
            "weight": e.weight
        }
        for e in sub.edges
    ]

    return SubgraphResponse(
        center_id=request.center_id,
        nodes=nodes,
        edges=edges,
        stats={"nodes": len(nodes), "edges": len(edges)}
    )


# =============================================================================
# NetworkX Analysis Endpoints
# =============================================================================

@app.get("/api/pagerank/{game}", response_model=PageRankResult)
async def get_pagerank(
    game: str,
    top_n: int = Query(default=20, ge=5, le=100, description="Number of top nodes to return")
):
    """
    Compute PageRank to find important dialogue nodes.

    High PageRank = many paths lead here, or important paths lead here.
    These are often narrative bottlenecks or key conversation hubs.
    """
    graph = await get_graph(game)

    try:
        results = graph.pagerank(top_n=top_n)
    except ImportError:
        raise HTTPException(500, "networkx not installed. Run: uv sync --extra server")

    top_nodes = [
        {
            "id": node_id,
            "score": round(score, 6),
            **info
        }
        for node_id, score, info in results
    ]

    return PageRankResult(game=game, top_nodes=top_nodes)


@app.get("/api/communities/{game}", response_model=CommunityResult)
async def get_communities(
    game: str,
    algorithm: str = Query(
        default="louvain",
        description="Algorithm: louvain (default), label_propagation, greedy_modularity"
    )
):
    """
    Detect communities (clusters) of related dialogue.

    Communities often correspond to:
    - Quest-related dialogue groupings
    - Character relationship clusters
    - Thematic/emotional groupings
    """
    if algorithm not in ("louvain", "label_propagation", "greedy_modularity"):
        raise HTTPException(400, f"Unknown algorithm: {algorithm}. Use louvain, label_propagation, or greedy_modularity")

    graph = await get_graph(game)

    try:
        communities = graph.communities(algorithm=algorithm)
    except ImportError:
        raise HTTPException(500, "networkx not installed. Run: uv sync --extra server")

    return CommunityResult(
        game=game,
        algorithm=algorithm,
        community_count=len(communities),
        communities=communities
    )


@app.post("/api/path", response_model=PathResponse)
async def find_path(request: PathRequest):
    """
    Find shortest path between two dialogue nodes.

    Useful for understanding how dialogue flows connect distant nodes.
    Returns null path if no connection exists.
    """
    graph = await get_graph(request.game)

    if request.source not in graph.nodes:
        raise HTTPException(404, f"Source node '{request.source}' not found")
    if request.target not in graph.nodes:
        raise HTTPException(404, f"Target node '{request.target}' not found")

    try:
        path = graph.find_path(request.source, request.target, max_length=request.max_length)
    except ImportError:
        raise HTTPException(500, "networkx not installed. Run: uv sync --extra server")

    return PathResponse(
        source=request.source,
        target=request.target,
        path_length=len(path) if path else 0,
        path=path
    )


@app.get("/api/centrality/{game}", response_model=CentralityResult)
async def get_centrality(
    game: str,
    top_n: int = Query(default=10, ge=5, le=50, description="Number of top nodes per metric")
):
    """
    Compute multiple centrality measures to find important nodes.

    Metrics:
    - degree: Most connected nodes (chatty characters)
    - betweenness: Nodes on many shortest paths (narrative bottlenecks)
    - closeness: Nodes close to all others (central to the graph)
    """
    graph = await get_graph(game)

    try:
        results = graph.centrality_analysis(top_n=top_n)
    except ImportError:
        raise HTTPException(500, "networkx not installed. Run: uv sync --extra server")

    # Format results
    metrics = {}
    for metric_name, entries in results.items():
        metrics[metric_name] = [
            {"id": nid, "score": round(score, 6), "text": text}
            for nid, score, text in entries
        ]

    return CentralityResult(game=game, metrics=metrics)


@app.get("/api/components/{game}", response_model=SCCResult)
async def get_sccs(game: str):
    """
    Find strongly connected components (dialogue cycles/loops).

    An SCC is a set of nodes where every node can reach every other node.
    These represent repeatable conversation structures - places where
    dialogue can loop back on itself.
    """
    graph = await get_graph(game)

    try:
        sccs = graph.strongly_connected_components()
    except ImportError:
        raise HTTPException(500, "networkx not installed. Run: uv sync --extra server")

    return SCCResult(
        game=game,
        scc_count=len(sccs),
        components=sccs
    )


# =============================================================================
# Topic Graph Endpoints
# =============================================================================

@app.get("/api/topics/{game}", response_model=TopicGraphStats)
async def get_topic_stats(
    game: str,
    filter_hubs: bool = Query(default=True, description="Filter common hub topics (GREETING, GOODBYE, etc.)")
):
    """
    Get topic graph statistics for a game.

    Topic graphs model dialogue at the topic level rather than individual lines.
    Topics are containers of related dialogue, and edges represent conversational flow.
    """
    graph = await get_topic_graph(game, filter_hubs=filter_hubs)
    stats = graph.get_statistics()

    hubs = [
        {
            'topic': topic_id,
            'degree': degree,
            'line_count': node.line_count,
            'dominant_emotion': node.dominant_emotion(),
            'sample': node.sample_lines[0][:80] if node.sample_lines else '',
        }
        for topic_id, degree, node in graph.find_hubs(min_degree=5)[:15]
    ]

    return TopicGraphStats(
        game=game,
        topics=stats['topics'],
        edges=stats['edges'],
        total_lines=stats['total_lines'],
        avg_lines_per_topic=stats['avg_lines_per_topic'],
        avg_degree=stats['avg_degree'],
        max_degree=stats['max_degree'],
        hubs=hubs
    )


@app.get("/api/topics/{game}/chains")
async def get_topic_chains(
    game: str,
    min_length: int = Query(default=3, ge=2, le=20),
    filter_hubs: bool = Query(default=True),
    top_k_exclude: int = Query(default=10, ge=0, le=50, description="Exclude top K hub topics")
):
    """
    Find linear chains of topics.

    Chains are sequences of topics that form conversation arcs without branching.
    Useful for understanding narrative flow.
    """
    graph = await get_topic_graph(game, filter_hubs=filter_hubs)

    if top_k_exclude > 0:
        graph = graph.filter_topics(top_k_exclude=top_k_exclude)

    chains = graph.find_chains(min_length=min_length)

    result = []
    for chain in chains[:20]:
        topics = []
        for topic_id in chain:
            node = graph.nodes.get(topic_id)
            if node:
                topics.append({
                    'topic': topic_id,
                    'emotion': node.dominant_emotion(),
                    'sample': node.sample_lines[0][:100] if node.sample_lines else '',
                })
        result.append(TopicChain(length=len(chain), topics=topics))

    return {'game': game, 'chain_count': len(result), 'chains': result}


@app.get("/api/topics/{game}/paths/{topic}")
async def get_topic_paths(
    game: str,
    topic: str,
    filter_hubs: bool = Query(default=True)
):
    """
    Get topic→text→topic paths from a source topic.

    Shows how dialogue flows from one topic to connected topics,
    including sample text from each.
    """
    graph = await get_topic_graph(game, filter_hubs=filter_hubs)

    if topic not in graph.nodes:
        raise HTTPException(404, f"Topic '{topic}' not found")

    paths = graph.topic_text_topic_paths(topic)

    return TopicPathResult(source_topic=topic, paths=paths)


@app.get("/api/topics/{game}/pagerank")
async def get_topic_pagerank(
    game: str,
    top_n: int = Query(default=20, ge=5, le=100),
    filter_hubs: bool = Query(default=True)
):
    """
    Compute PageRank on topic graph.

    High PageRank = important topics that many conversation paths flow through.
    """
    graph = await get_topic_graph(game, filter_hubs=filter_hubs)

    try:
        results = graph.pagerank(top_n=top_n)
    except ImportError:
        raise HTTPException(500, "networkx not installed")

    return {
        'game': game,
        'top_topics': [
            {
                'topic': topic_id,
                'score': round(score, 6),
                'line_count': node.line_count,
                'emotion': node.dominant_emotion(),
                'sample': node.sample_lines[0][:80] if node.sample_lines else '',
            }
            for topic_id, score, node in results
        ]
    }


# =============================================================================
# Cross-Game Endpoints
# =============================================================================

async def get_cross_game_linker() -> CrossGameLinker:
    """Get or create cross-game linker (loads all available games)."""
    global CROSS_GAME_CACHE
    if CROSS_GAME_CACHE is not None:
        return CROSS_GAME_CACHE

    linker = CrossGameLinker()

    # Load all available dialogue files
    for path in DATA_DIR.glob("*_dialogue.json"):
        if path.exists():
            linker.load_game(path)
    for path in DATA_DIR.glob("*_full_dialogue.json"):
        if path.exists():
            game_name = path.stem.replace('_full_dialogue', '')
            linker.load_game(path, game_name)

    linker.build_emotion_clusters()
    CROSS_GAME_CACHE = linker
    return linker


@app.get("/api/crossgame/stats")
async def get_crossgame_stats():
    """
    Get cross-game linking statistics.

    Shows which emotions are linkable across games and their distributions.
    """
    linker = await get_cross_game_linker()
    stats = linker.get_cross_game_stats()
    emotion_dist = linker.get_emotion_distribution()

    return {
        'games': stats['games'],
        'total_nodes': stats['total_nodes'],
        'nodes_per_game': stats['nodes_per_game'],
        'linkable_emotions': stats['linkable_emotions'],
        'emotion_distribution': emotion_dist,
    }


@app.get("/api/crossgame/pairs")
async def get_crossgame_pairs(
    emotion: str = Query(default=None, description="Filter to specific emotion"),
    n: int = Query(default=10, ge=1, le=50, description="Number of pairs to sample")
):
    """
    Sample cross-game dialogue pairs linked by emotion.

    Returns pairs of dialogue from different games sharing the same emotion.
    Useful for cross-game training data or finding parallel narrative structures.
    """
    linker = await get_cross_game_linker()
    pairs = linker.sample_cross_game_pairs(emotion=emotion, n=n)
    return {'pairs': pairs, 'count': len(pairs)}


@app.get("/api/crossgame/unified")
async def get_unified_graph():
    """
    Get unified cross-game graph structure.

    Shows how dialogue from different games links via shared emotions.
    """
    linker = await get_cross_game_linker()
    graph = linker.build_unified_graph()
    return graph


# =============================================================================
# Emotion Bridge Endpoints (Cross-Game via Emotion Transitions)
# =============================================================================

async def get_bridge_graph() -> EmotionBridgeGraph:
    """Get or load the emotion bridge graph."""
    global BRIDGE_CACHE
    if BRIDGE_CACHE is not None:
        return BRIDGE_CACHE

    loop = asyncio.get_event_loop()
    BRIDGE_CACHE = await loop.run_in_executor(None, load_emotion_bridge, DATA_DIR)
    return BRIDGE_CACHE


@app.get("/api/bridge/stats")
async def get_bridge_stats():
    """
    Get statistics about the emotion bridge graph.

    The bridge graph connects dialogue from different games via matching
    emotion transitions (e.g., happy->sad in Skyrim links to happy->sad in FNV).
    """
    graph = await get_bridge_graph()
    return graph.get_statistics()


@app.get("/api/bridge/cells")
async def get_bridge_cells(
    top_n: int = Query(default=20, ge=5, le=100, description="Number of top bridge cells to return")
):
    """
    Get emotion transition cells that bridge multiple games.

    Each cell represents a (source_emotion -> target_emotion) transition.
    Cells that appear in multiple games are "bridges" that enable cross-game walks.
    """
    graph = await get_bridge_graph()
    cells = graph.get_bridge_cells()
    return {
        'total_bridge_cells': len(cells),
        'cells': cells[:top_n],
    }


@app.get("/api/bridge/matrix")
async def get_bridge_matrix():
    """
    Get the full emotion transition matrix with cross-game bridge info.

    Returns a matrix where each cell shows:
    - Total edges with that transition
    - Edges per game
    - Whether it's a cross-game bridge point
    """
    graph = await get_bridge_graph()
    return graph.get_transition_matrix_data()


@app.post("/api/bridge/walk")
async def bridge_walk(request: BridgeWalkRequest):
    """
    Random walk that can jump between games at emotion transitions.

    When the walk encounters an emotion transition that exists in multiple games,
    it may "bridge" to another game with probability `cross_probability`.

    This enables sampling trajectories that cross game boundaries while
    maintaining emotional coherence.
    """
    graph = await get_bridge_graph()

    if request.start_game and request.start_game not in graph.games:
        raise HTTPException(404, f"Game '{request.start_game}' not found. Available: {list(graph.games)}")

    path = graph.cross_game_walk(
        max_steps=request.max_steps,
        start_game=request.start_game,
        cross_probability=request.cross_probability,
        prefer_off_diagonal=request.prefer_off_diagonal,
    )

    # Annotate with game transitions
    transitions = []
    current_game = None
    for node in path:
        if current_game and node['game'] != current_game:
            transitions.append({
                'from_game': current_game,
                'to_game': node['game'],
                'at_step': node['step'],
                'bridge_emotion': node.get('bridge_emotion', 'unknown'),
            })
        current_game = node['game']

    return {
        'path': path,
        'length': len(path),
        'games_visited': list(set(n['game'] for n in path)),
        'transitions': transitions,
    }


@app.post("/api/bridge/coverage")
async def bridge_coverage(request: CoverageRequest):
    """
    Sample a trajectory that tries to cover multiple games.

    Useful for generating training data that spans the full corpus
    while maintaining emotional coherence through bridge transitions.
    """
    graph = await get_bridge_graph()

    target_games = request.target_games
    if target_games:
        invalid = [g for g in target_games if g not in graph.games]
        if invalid:
            raise HTTPException(404, f"Games not found: {invalid}. Available: {list(graph.games)}")
    else:
        target_games = list(graph.games)

    path = graph.sample_coverage_trajectory(
        target_games=target_games,
        min_length=request.min_length,
        max_length=request.max_length,
    )

    games_hit = set(n['game'] for n in path)

    return {
        'path': path,
        'length': len(path),
        'target_games': target_games,
        'games_covered': list(games_hit),
        'coverage_ratio': len(games_hit) / len(target_games),
    }


@app.get("/api/bridge/visualization")
async def get_bridge_visualization():
    """
    Get D3.js-compatible visualization data for the emotion bridge graph.

    Returns:
    - nodes: Emotion transition cells (matrix entries)
    - edges: Connections between cells (target emotion matches source)
    - Bridge cells (cross-game) are marked for highlighting
    """
    graph = await get_bridge_graph()
    return graph.get_visualization_data()


# =============================================================================
# Query Graph Endpoints (Topics as Gaps Between Text)
# =============================================================================

async def get_query_graph() -> QueryGraph:
    """Get or load the query graph."""
    global QUERY_CACHE
    if QUERY_CACHE is not None:
        return QUERY_CACHE

    loop = asyncio.get_event_loop()
    QUERY_CACHE = await loop.run_in_executor(None, load_query_graph, DATA_DIR)
    return QUERY_CACHE


@app.get("/api/query/stats")
async def get_query_stats():
    """
    Get statistics about the query graph.

    The query graph models topics as "gaps" or "queries" that text nodes respond to.
    This is a bipartite graph: Topic nodes <-> Text nodes.
    """
    graph = await get_query_graph()
    return graph.get_statistics()


@app.get("/api/query/topics")
async def get_query_topics(
    category: str = Query(default=None, description="Filter by category (GREETING, COMBAT, QUEST, etc.)"),
    cross_game: bool = Query(default=False, description="Only topics appearing in multiple games"),
    top_n: int = Query(default=50, ge=10, le=200, description="Number of topics to return"),
):
    """
    List topics (query nodes) in the graph.

    Topics are the semantic "gaps" that dialogue responds to.
    Categories: GREETING, FAREWELL, COMBAT, TRADE, QUEST, RUMORS, AMBIENT, SERVICE, COMPANION, CRIME
    """
    graph = await get_query_graph()

    if cross_game:
        topics = graph.get_cross_game_topics()
    elif category:
        topics = graph.get_topics_by_category(category.upper())
    else:
        topics = sorted(graph.topic_nodes.values(), key=lambda t: t.response_count, reverse=True)

    return {
        'count': len(topics[:top_n]),
        'topics': [
            {
                'id': t.id,
                'category': t.category,
                'games': list(t.games),
                'response_count': t.response_count,
                'dominant_emotion': t.dominant_emotion(),
                'sample_texts': t.sample_texts[:2],
            }
            for t in topics[:top_n]
        ],
    }


@app.get("/api/query/categories")
async def get_query_categories():
    """
    List all semantic categories and their topic counts.

    Categories represent types of conversational moves/gaps.
    """
    graph = await get_query_graph()
    stats = graph.get_statistics()
    return {
        'categories': stats['categories'],
        'total_topics': stats['topic_nodes'],
        'categorized_topics': sum(stats['categories'].values()),
    }


@app.post("/api/query/sample")
async def sample_query_responses(request: QuerySampleRequest):
    """
    Sample text responses for a topic/category.

    Topics are the "queries" or "gaps" - this returns the text nodes that fill them.
    Use cross_game=true to get responses from different games for the same topic.
    """
    graph = await get_query_graph()

    result = graph.sample_topic_responses(
        topic=request.topic,
        category=request.category.upper() if request.category else None,
        n=request.n,
        cross_game=request.cross_game,
    )

    if 'error' in result:
        raise HTTPException(404, result['error'])

    return result


@app.post("/api/query/walk")
async def query_walk(request: QueryWalkRequest):
    """
    Walk through topics following semantic connections.

    Creates a topic→text→topic→text chain, showing how different
    semantic gaps connect through their responses.

    This is the "interleaved" view: alternating between query nodes
    and response nodes.
    """
    graph = await get_query_graph()

    path = graph.walk_topic_chain(
        start_topic=request.start_topic,
        max_steps=request.max_steps,
        cross_game=request.cross_game,
    )

    # Analyze the walk
    games_visited = set()
    categories_visited = set()
    for step in path:
        games_visited.add(step['response']['game'])
        if step['category']:
            categories_visited.add(step['category'])

    return {
        'path': path,
        'length': len(path),
        'games_visited': list(games_visited),
        'categories_visited': list(categories_visited),
    }


@app.get("/api/query/cross-game")
async def get_cross_game_topics():
    """
    Get topics that appear across multiple games.

    These represent universal dialogue patterns - the same semantic "gap"
    appearing in different game worlds (e.g., GREETING, Attack, Flee).
    """
    graph = await get_query_graph()
    topics = graph.get_cross_game_topics()

    return {
        'count': len(topics),
        'topics': [
            {
                'id': t.id,
                'category': t.category,
                'games': list(t.games),
                'response_count': t.response_count,
                'dominant_emotion': t.dominant_emotion(),
                'responses_by_game': {
                    game: len([
                        tid for tid in graph.topic_to_texts[t.id]
                        if graph.text_nodes[tid].game == game
                    ])
                    for game in t.games
                },
            }
            for t in sorted(topics, key=lambda t: t.response_count, reverse=True)
        ],
    }


@app.get("/api/query/visualization")
async def get_query_visualization(
    max_topics: int = Query(default=30, ge=10, le=100, description="Max topic nodes to include"),
):
    """
    Get D3.js-compatible visualization data for the query graph.

    Returns a bipartite graph with:
    - Topic nodes (queries/gaps)
    - Text nodes (responses)
    - Edges connecting topics to their responses
    """
    graph = await get_query_graph()
    return graph.get_visualization_data(max_topics=max_topics)


@app.get("/api/query/transitions")
async def get_topic_transitions():
    """
    Get statistics about topic-to-topic transitions.

    Shows how topics connect to each other through their text responses.
    Identifies hub topics (high in/out degree) in the conversation flow.
    """
    graph = await get_query_graph()
    return graph.get_topic_transition_stats()


@app.get("/api/query/cycles")
async def get_topic_cycles(
    max_cycles: int = Query(default=10, ge=1, le=30, description="Maximum cycles to return"),
    max_cycle_length: int = Query(default=5, ge=2, le=8, description="Maximum cycle length"),
):
    """
    Find cycles in the topic transition graph.

    Cycles show conversational loops: topic → text → topic → text → ... → (back to start)

    Example cycle: GOODBYE → ThievesGuild → GREETING → QuestTopic → GOODBYE
    This reveals how dialogue can loop through related topics.
    """
    graph = await get_query_graph()
    cycles = graph.get_cycle_examples(
        max_cycles=max_cycles,
        max_cycle_length=max_cycle_length,
    )

    return {
        'count': len(cycles),
        'cycles': cycles,
    }



# =============================================================================
# Synthetic Graph Analysis Endpoints
# =============================================================================

def get_available_synthetic_settings() -> List[str]:
    """Find all synthetic settings with graph.json."""
    settings = []
    for path in SYNTHETIC_DIR.glob("*/graph.json"):
        settings.append(path.parent.name)
    return sorted(settings)


async def get_synthetic_graph(setting: str) -> dict:
    """Load a synthetic graph."""
    if setting in SYNTHETIC_GRAPH_CACHE:
        return SYNTHETIC_GRAPH_CACHE[setting]

    path = SYNTHETIC_DIR / setting / "graph.json"
    if not path.exists():
        raise HTTPException(404, f"Synthetic setting '{setting}' not found. Available: {get_available_synthetic_settings()}")

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, lambda: json.loads(path.read_text()))
    SYNTHETIC_GRAPH_CACHE[setting] = data
    return data


@app.get("/api/synthetic-graph/settings")
async def list_synthetic_graph_settings():
    """
    List available synthetic graph settings.

    These are generated dialogue corpora that can be analyzed for topology.
    """
    settings = get_available_synthetic_settings()
    result = []
    for s in settings:
        path = SYNTHETIC_DIR / s / "graph.json"
        if path.exists():
            data = json.loads(path.read_text())
            nodes = data.get('nodes', {})
            edges = data.get('edges', [])
            result.append({
                'setting': s,
                'nodes': len(nodes),
                'edges': len(edges),
            })
    return {'settings': result}


@app.get("/api/synthetic-graph/{setting}/stats")
async def get_synthetic_graph_stats(setting: str):
    """
    Get topology statistics for a synthetic graph.

    Includes node/edge counts, degree distributions, hub detection.
    """
    if not DIAGNOSTICS_AVAILABLE:
        raise HTTPException(500, "graph_diagnostics module not available")

    path = SYNTHETIC_DIR / setting / "graph.json"
    if not path.exists():
        raise HTTPException(404, f"Setting '{setting}' not found")

    loop = asyncio.get_event_loop()
    analysis = await loop.run_in_executor(None, analyze_graph, path, setting)
    return analysis


@app.get("/api/synthetic-graph/{setting}/compare/{reference}")
async def compare_synthetic_to_reference(setting: str, reference: str):
    """
    Compare synthetic graph topology to a reference game.

    Shows side-by-side metrics: branching factor, hub structure, component count.
    """
    if not DIAGNOSTICS_AVAILABLE:
        raise HTTPException(500, "graph_diagnostics module not available")

    syn_path = SYNTHETIC_DIR / setting / "graph.json"
    ref_path = DATA_DIR / f"{reference}_dialogue_graph.json"

    if not syn_path.exists():
        raise HTTPException(404, f"Synthetic setting '{setting}' not found")
    if not ref_path.exists():
        raise HTTPException(404, f"Reference game '{reference}' not found")

    loop = asyncio.get_event_loop()
    syn_analysis = await loop.run_in_executor(None, analyze_graph, syn_path, f"synthetic:{setting}")
    ref_analysis = await loop.run_in_executor(None, analyze_graph, ref_path, f"reference:{reference}")

    # Compute comparison metrics
    comparison = {
        'synthetic': syn_analysis,
        'reference': ref_analysis,
        'comparison': {
            'branching_factor_ratio': syn_analysis['branching_factor'] / ref_analysis['branching_factor'] if ref_analysis['branching_factor'] > 0 else 0,
            'node_ratio': syn_analysis['node_count'] / ref_analysis['node_count'] if ref_analysis['node_count'] > 0 else 0,
            'component_difference': syn_analysis['components']['component_count'] - ref_analysis['components']['component_count'],
            'leaf_ratio_difference': syn_analysis['leaves']['leaf_ratio'] - ref_analysis['leaves']['leaf_ratio'],
        }
    }
    return comparison


@app.get("/api/synthetic-graph/{setting}/pagerank")
async def get_synthetic_pagerank(
    setting: str,
    top_n: int = Query(default=20, ge=5, le=100)
):
    """
    PageRank analysis on synthetic graph.

    Find important nodes in the generated dialogue graph.
    """
    data = await get_synthetic_graph(setting)
    nodes = data.get('nodes', {})
    edges = data.get('edges', [])

    try:
        import networkx as nx
    except ImportError:
        raise HTTPException(500, "networkx not installed")

    G = nx.DiGraph()
    for nid, node in nodes.items():
        G.add_node(nid, **node)
    for edge in edges:
        src = edge.get('source') or edge.get('from')
        tgt = edge.get('target') or edge.get('to')
        if src and tgt:
            G.add_edge(src, tgt)

    pr = nx.pagerank(G, alpha=0.85)
    top = sorted(pr.items(), key=lambda x: -x[1])[:top_n]

    result = []
    for nid, score in top:
        node = nodes.get(nid, {})
        result.append({
            'id': nid,
            'score': round(score, 6),
            'text': node.get('text', '')[:100],
            'emotion': node.get('emotion', 'unknown'),
        })

    return {'setting': setting, 'top_nodes': result}


@app.get("/api/synthetic-graph/{setting}/centrality")
async def get_synthetic_centrality(
    setting: str,
    top_n: int = Query(default=10, ge=5, le=50)
):
    """
    Centrality analysis on synthetic graph.

    Find hubs and bottlenecks in generated dialogue.
    """
    data = await get_synthetic_graph(setting)
    nodes = data.get('nodes', {})
    edges = data.get('edges', [])

    try:
        import networkx as nx
    except ImportError:
        raise HTTPException(500, "networkx not installed")

    G = nx.DiGraph()
    for nid in nodes:
        G.add_node(nid)
    for edge in edges:
        src = edge.get('source') or edge.get('from')
        tgt = edge.get('target') or edge.get('to')
        if src and tgt:
            G.add_edge(src, tgt)

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # Top by in-degree (hubs - many incoming)
    top_in = sorted(in_deg.items(), key=lambda x: -x[1])[:top_n]
    # Top by out-degree (branches - many outgoing)
    top_out = sorted(out_deg.items(), key=lambda x: -x[1])[:top_n]

    return {
        'setting': setting,
        'top_in_degree': [
            {'id': nid, 'in_degree': deg, 'text': nodes.get(nid, {}).get('text', '')[:80]}
            for nid, deg in top_in
        ],
        'top_out_degree': [
            {'id': nid, 'out_degree': deg, 'text': nodes.get(nid, {}).get('text', '')[:80]}
            for nid, deg in top_out
        ],
    }


@app.get("/api/synthetic-graph/{setting}/communities")
async def get_synthetic_communities(
    setting: str,
    algorithm: str = Query(default="louvain", description="louvain, label_propagation")
):
    """
    Community detection on synthetic graph.

    Find clusters of related dialogue in the generated corpus.
    """
    data = await get_synthetic_graph(setting)
    nodes = data.get('nodes', {})
    edges = data.get('edges', [])

    try:
        import networkx as nx
        from networkx.algorithms import community
    except ImportError:
        raise HTTPException(500, "networkx not installed")

    # Use undirected for community detection
    G = nx.Graph()
    for nid in nodes:
        G.add_node(nid)
    for edge in edges:
        src = edge.get('source') or edge.get('from')
        tgt = edge.get('target') or edge.get('to')
        if src and tgt:
            G.add_edge(src, tgt)

    if algorithm == "louvain":
        communities_gen = community.louvain_communities(G)
    elif algorithm == "label_propagation":
        communities_gen = community.label_propagation_communities(G)
    else:
        raise HTTPException(400, f"Unknown algorithm: {algorithm}")

    communities_list = list(communities_gen)

    result = []
    for i, comm in enumerate(sorted(communities_list, key=len, reverse=True)[:20]):
        # Sample some nodes from the community
        sample_ids = list(comm)[:5]
        sample_texts = [nodes.get(nid, {}).get('text', '')[:60] for nid in sample_ids]

        # Get dominant emotion
        emotions = [nodes.get(nid, {}).get('emotion', 'neutral') for nid in comm]
        from collections import Counter
        emotion_counts = Counter(emotions)
        dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral'

        result.append({
            'community_id': i,
            'size': len(comm),
            'dominant_emotion': dominant,
            'sample_texts': sample_texts,
        })

    return {
        'setting': setting,
        'algorithm': algorithm,
        'community_count': len(communities_list),
        'communities': result,
    }


@app.get("/api/synthetic-graph/{setting}/components")
async def get_synthetic_components(setting: str):
    """
    Find connected components in synthetic graph.

    Shows graph fragmentation - ideally one large component.
    """
    data = await get_synthetic_graph(setting)
    nodes = data.get('nodes', {})
    edges = data.get('edges', [])

    try:
        import networkx as nx
    except ImportError:
        raise HTTPException(500, "networkx not installed")

    G = nx.DiGraph()
    for nid in nodes:
        G.add_node(nid)
    for edge in edges:
        src = edge.get('source') or edge.get('from')
        tgt = edge.get('target') or edge.get('to')
        if src and tgt:
            G.add_edge(src, tgt)

    # Weakly connected components
    wccs = list(nx.weakly_connected_components(G))
    wcc_sizes = sorted([len(c) for c in wccs], reverse=True)

    # Strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    non_trivial_sccs = [c for c in sccs if len(c) > 1]

    return {
        'setting': setting,
        'weakly_connected': {
            'count': len(wccs),
            'largest': wcc_sizes[0] if wcc_sizes else 0,
            'size_distribution': wcc_sizes[:10],
        },
        'strongly_connected': {
            'count': len(sccs),
            'non_trivial_count': len(non_trivial_sccs),
            'largest': max(len(c) for c in sccs) if sccs else 0,
        },
    }


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Landing page with interactive visualization."""
    return LANDING_HTML


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
