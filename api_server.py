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


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("dialogue_data")
CACHE: Dict[str, DialogueGraph] = {}
TOPIC_CACHE: Dict[str, TopicGraph] = {}
CROSS_GAME_CACHE: Optional[CrossGameLinker] = None
BRIDGE_CACHE: Optional[EmotionBridgeGraph] = None


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
            "GET /docs": "Interactive OpenAPI documentation",
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
# Landing Page
# =============================================================================

LANDING_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dialogue Graph Explorer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; margin-bottom: 5px; }
        h2 { color: #ff6b6b; margin-top: 1.5em; }
        h3 { color: #fbbf24; margin-bottom: 10px; }
        a { color: #00d9ff; }
        .subtitle { color: #888; margin-bottom: 20px; }
        .layout { display: grid; grid-template-columns: 1fr 350px; gap: 20px; }
        @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .sidebar .card { margin: 0 0 15px 0; }
        button {
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            margin: 3px;
            font-size: 13px;
        }
        button:hover { background: #00b8d9; }
        button.secondary { background: #4a5568; color: #eee; }
        button.secondary:hover { background: #5a6578; }
        select, input {
            background: #0f3460;
            color: #eee;
            border: 1px solid #00d9ff;
            padding: 6px 10px;
            border-radius: 4px;
            margin: 3px;
            font-size: 13px;
        }
        pre, code {
            background: #0f3460;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        pre { padding: 12px; overflow-x: auto; }
        .sample {
            background: #1f4068;
            padding: 10px;
            margin: 8px 0;
            border-radius: 4px;
            border-left: 3px solid #00d9ff;
            font-size: 14px;
        }
        .emotion-neutral { border-left-color: #888; }
        .emotion-happy { border-left-color: #4ade80; }
        .emotion-anger { border-left-color: #f87171; }
        .emotion-sad { border-left-color: #60a5fa; }
        .emotion-fear { border-left-color: #a78bfa; }
        .emotion-surprise { border-left-color: #fbbf24; }
        .emotion-disgust { border-left-color: #84cc16; }
        .speaker { color: #00d9ff; font-weight: bold; }
        .emotion-tag {
            font-size: 10px;
            background: #0f3460;
            padding: 2px 5px;
            border-radius: 3px;
            margin-left: 6px;
        }
        .node-id {
            font-size: 10px;
            color: #888;
            cursor: pointer;
            float: right;
        }
        .node-id:hover { color: #00d9ff; }
        .quest-tag, .topic-tag {
            font-size: 10px;
            background: #2d3748;
            padding: 2px 5px;
            border-radius: 3px;
            margin-left: 4px;
            color: #a0aec0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .stat-item {
            background: #0f3460;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
        }
        .stat-value { font-size: 20px; color: #00d9ff; }
        .stat-label { font-size: 10px; color: #888; }
        .matrix {
            display: grid;
            gap: 2px;
            font-size: 10px;
        }
        .matrix-cell {
            padding: 3px;
            text-align: center;
            background: #0f3460;
            cursor: pointer;
        }
        .matrix-cell:hover { outline: 1px solid #00d9ff; }
        .matrix-header { background: #1f4068; font-weight: bold; cursor: default; }
        .resource-list { font-size: 12px; line-height: 1.8; }
        .resource-list a { display: block; padding: 4px 0; }
        .api-example { font-size: 11px; line-height: 1.4; }
        .tabs { display: flex; gap: 5px; margin-bottom: 10px; }
        .tab { padding: 6px 12px; background: #0f3460; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .tab.active { background: #00d9ff; color: #1a1a2e; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        #loading { color: #fbbf24; font-size: 12px; }
        .subgraph-view { margin-top: 10px; padding: 10px; background: #0f3460; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🎭 Dialogue Graph Explorer</h1>
    <p class="subtitle">Situated dialogue from Bethesda games — with emotion, speaker, and quest annotations</p>

    <div class="card" style="margin-bottom:20px">
        <select id="gameSelect" onchange="loadGame()" style="font-size:15px;padding:10px">
            <option value="">Loading...</option>
        </select>
        <span id="loading"></span>
        <a href="/docs" style="float:right;margin-top:8px">📖 OpenAPI Docs</a>
    </div>

    <div class="layout">
        <div class="main">
            <div id="statsSection" style="display:none">
                <h2>📊 Graph Overview</h2>
                <div class="card">
                    <div id="statsGrid" class="stats-grid"></div>
                    <div style="margin-top:15px">
                        <h3>Emotion Transitions</h3>
                        <p style="font-size:11px;color:#888;margin-bottom:8px">Click a cell to sample that emotion pair</p>
                        <div id="transitionMatrix"></div>
                    </div>
                </div>

                <h2>🎲 Sample Dialogue</h2>
                <div class="card">
                    <div style="margin-bottom:10px">
                        <label>Method:</label>
                        <select id="sampleMethod">
                            <option value="walk">Random Walk</option>
                            <option value="chain">Quest Chain</option>
                            <option value="hub">From Hub</option>
                        </select>
                        <label>×</label>
                        <input type="number" id="sampleCount" value="2" min="1" max="10" style="width:50px">
                        <label>len:</label>
                        <input type="number" id="sampleLength" value="5" min="2" max="20" style="width:50px">
                        <button onclick="sampleDialogue()">Sample</button>
                        <button class="secondary" onclick="sampleDialogue('happy')">😊 Happy</button>
                        <button class="secondary" onclick="sampleDialogue('anger')">😠 Anger</button>
                    </div>
                    <div id="samples"><p style="color:#888">Click Sample to explore dialogue paths</p></div>
                </div>

                <div id="subgraphSection" style="display:none">
                    <h2>🔍 Subgraph View</h2>
                    <div class="card">
                        <p style="font-size:12px">Exploring neighborhood of <code id="subgraphCenter"></code></p>
                        <div id="subgraphView"></div>
                    </div>
                </div>

                <h2>🧠 Graph Analysis</h2>
                <div class="card">
                    <div style="margin-bottom:15px">
                        <button onclick="loadPageRank()">📊 PageRank</button>
                        <button onclick="loadCommunities()">🏘️ Communities</button>
                        <button onclick="loadCentrality()">🎯 Centrality</button>
                        <button onclick="loadSCCs()">🔄 Loops (SCCs)</button>
                    </div>
                    <div id="analysisResults"><p style="color:#888">Click an analysis button to explore graph structure</p></div>
                </div>

                <div id="pathSection" style="display:none">
                    <h2>🛤️ Path Finder</h2>
                    <div class="card">
                        <div style="margin-bottom:10px">
                            <label>From:</label>
                            <input type="text" id="pathSource" placeholder="0x..." style="width:120px">
                            <label>To:</label>
                            <input type="text" id="pathTarget" placeholder="0x..." style="width:120px">
                            <button onclick="findPath()">Find Path</button>
                        </div>
                        <div id="pathResult"></div>
                    </div>
                </div>

                <h2>🌉 Cross-Game Emotion Bridge</h2>
                <div class="card">
                    <p style="font-size:12px;color:#888;margin-bottom:10px">
                        Emotion transitions link dialogue across games. Click a cell to sample cross-game walks through that emotion bridge.
                    </p>
                    <div style="margin-bottom:10px">
                        <button onclick="loadBridgeGraph()">Load Bridge Graph</button>
                        <button onclick="sampleBridgeWalk()" class="secondary">🚶 Cross-Game Walk</button>
                        <button onclick="sampleCoverage()" class="secondary">🎯 Coverage Walk</button>
                        <label style="margin-left:10px">Cross prob:</label>
                        <input type="range" id="crossProb" min="0" max="100" value="40" style="width:80px" title="Probability of crossing to another game">
                        <span id="crossProbVal">40%</span>
                    </div>
                    <div id="bridgeStats" style="display:none;margin-bottom:15px">
                        <div class="stats-grid" id="bridgeStatsGrid"></div>
                    </div>
                    <div id="bridgeMatrix" style="margin-bottom:15px"></div>
                    <div id="bridgeWalkResults"></div>
                </div>
            </div>
        </div>

        <div class="sidebar">
            <div class="card">
                <h3>📁 Resources</h3>
                <div class="resource-list" id="resourceList">
                    <p style="color:#888">Select a game...</p>
                </div>
            </div>

            <div class="card">
                <h3>🔌 API Reference</h3>
                <div class="tabs">
                    <div class="tab active" onclick="showTab('curl')">curl</div>
                    <div class="tab" onclick="showTab('fetch')">fetch</div>
                    <div class="tab" onclick="showTab('python')">python</div>
                </div>
                <div id="tab-curl" class="tab-content active">
                    <pre class="api-example" id="curlExample">curl localhost:8000/api/games</pre>
                </div>
                <div id="tab-fetch" class="tab-content">
                    <pre class="api-example" id="fetchExample">fetch('/api/games').then(r=>r.json())</pre>
                </div>
                <div id="tab-python" class="tab-content">
                    <pre class="api-example" id="pythonExample">import requests
requests.get('http://localhost:8000/api/games').json()</pre>
                </div>
            </div>

            <div class="card">
                <h3>🤖 For Claude/LLMs</h3>
                <p style="font-size:11px;color:#888;line-height:1.5">
                    Hit <code>GET /api</code> for discovery.<br>
                    Use <code>POST /api/sample</code> with:<br>
                </p>
                <pre class="api-example">{"game":"oblivion",
 "method":"walk",
 "count":3}</pre>
            </div>
        </div>
    </div>

    <script>
        const API = '/api';
        let currentGame = '';

        async function init() {
            const resp = await fetch(`${API}/games`);
            const games = await resp.json();
            const select = document.getElementById('gameSelect');
            select.innerHTML = games.map(g =>
                `<option value="${g.name}">${g.name} (${g.dialogue_count.toLocaleString()} lines)</option>`
            ).join('');
            if (games.length > 0) loadGame();
        }

        async function loadGame() {
            currentGame = document.getElementById('gameSelect').value;
            if (!currentGame) return;

            document.getElementById('loading').textContent = 'Loading graph...';
            document.getElementById('statsSection').style.display = 'block';

            // Update resource links
            document.getElementById('resourceList').innerHTML = `
                <a href="/api/stats/${currentGame}">📊 GET /api/stats/${currentGame}</a>
                <a href="/api/transitions/${currentGame}">😊 GET /api/transitions/${currentGame}</a>
                <a href="#" onclick="showSampleRequest();return false">🎲 POST /api/sample</a>
                <a href="#" onclick="showSubgraphRequest();return false">🔍 POST /api/subgraph</a>
                <hr style="border-color:#2d3748;margin:8px 0">
                <span style="color:#fbbf24;font-size:11px">Graph Analysis:</span>
                <a href="/api/pagerank/${currentGame}">📈 GET /api/pagerank/${currentGame}</a>
                <a href="/api/communities/${currentGame}">🏘️ GET /api/communities/${currentGame}</a>
                <a href="/api/centrality/${currentGame}">🎯 GET /api/centrality/${currentGame}</a>
                <a href="/api/components/${currentGame}">🔄 GET /api/components/${currentGame}</a>
                <a href="#" onclick="showPathRequest();return false">🛤️ POST /api/path</a>
                <hr style="border-color:#2d3748;margin:8px 0">
                <a href="/docs#/default/get_stats_api_stats__game__get">📖 Full API Docs</a>
            `;

            // Load stats
            const statsResp = await fetch(`${API}/stats/${currentGame}`);
            const stats = await statsResp.json();

            document.getElementById('statsGrid').innerHTML = `
                <div class="stat-item"><div class="stat-value">${stats.nodes.toLocaleString()}</div><div class="stat-label">Nodes</div></div>
                <div class="stat-item"><div class="stat-value">${stats.edges.toLocaleString()}</div><div class="stat-label">Edges</div></div>
                <div class="stat-item"><div class="stat-value">${stats.topics.toLocaleString()}</div><div class="stat-label">Topics</div></div>
                <div class="stat-item"><div class="stat-value">${stats.quests}</div><div class="stat-label">Quests</div></div>
                <div class="stat-item"><div class="stat-value">${stats.speakers}</div><div class="stat-label">Speakers</div></div>
                <div class="stat-item"><div class="stat-value">${stats.avg_out_degree.toFixed(1)}</div><div class="stat-label">Avg Degree</div></div>
            `;

            // Load transitions
            const transResp = await fetch(`${API}/transitions/${currentGame}`);
            const trans = await transResp.json();
            renderTransitionMatrix(trans.transitions);

            document.getElementById('loading').textContent = '';
            updateApiExamples('stats');
        }

        function renderTransitionMatrix(transitions) {
            const emotions = ['neutral', 'happy', 'anger', 'sad', 'fear', 'surprise', 'disgust'];
            const present = emotions.filter(e => transitions[e] ||
                emotions.some(e2 => transitions[e2] && transitions[e2][e]));

            let html = '<div class="matrix" style="grid-template-columns: repeat(' + (present.length + 1) + ', 1fr)">';
            html += '<div class="matrix-cell matrix-header">→</div>';
            present.forEach(e => html += `<div class="matrix-cell matrix-header">${e.slice(0,3)}</div>`);

            present.forEach(src => {
                html += `<div class="matrix-cell matrix-header">${src.slice(0,3)}</div>`;
                present.forEach(tgt => {
                    const count = (transitions[src] && transitions[src][tgt]) || 0;
                    const intensity = Math.min(count / 200, 1);
                    const bg = count > 0 ? `rgba(0, 217, 255, ${intensity * 0.6})` : '';
                    html += `<div class="matrix-cell" style="background:${bg}"
                        onclick="sampleEmotionPair('${src}','${tgt}')"
                        title="${src}→${tgt}: ${count}">${count || '-'}</div>`;
                });
            });
            html += '</div>';
            document.getElementById('transitionMatrix').innerHTML = html;
        }

        async function sampleDialogue(emotionFilter = null) {
            const method = document.getElementById('sampleMethod').value;
            const count = parseInt(document.getElementById('sampleCount').value);
            const maxLength = parseInt(document.getElementById('sampleLength').value);

            document.getElementById('samples').innerHTML = '<p>Sampling...</p>';

            const body = {game: currentGame, method, count, max_length: maxLength};
            if (emotionFilter) body.emotion_filter = emotionFilter;

            const resp = await fetch(`${API}/sample`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            const data = await resp.json();
            renderSamples(data.samples);
            updateApiExamples('sample', body);
        }

        async function sampleEmotionPair(src, tgt) {
            document.getElementById('samples').innerHTML = '<p>Sampling ' + src + '→' + tgt + '...</p>';
            const body = {game: currentGame, method: 'walk', count: 3, max_length: 6, emotion_filter: src};
            const resp = await fetch(`${API}/sample`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            const data = await resp.json();
            renderSamples(data.samples);
            updateApiExamples('sample', body);
        }

        function renderSamples(samples) {
            let html = '';
            samples.forEach((sample, i) => {
                const quest = sample.nodes[0]?.quest;
                html += `<div style="margin-bottom:15px">`;
                html += `<strong>Sample ${i + 1}</strong> <span style="color:#888">(${sample.method}, ${sample.length} nodes)</span>`;
                if (quest) html += `<span class="quest-tag">📜 ${quest}</span>`;
                sample.nodes.forEach(node => {
                    const emo = node.emotion || 'neutral';
                    html += `<div class="sample emotion-${emo}">`;
                    html += `<span class="node-id" onclick="loadSubgraph('${node.id}')" title="Click to explore subgraph">${node.id}</span>`;
                    html += `<span class="speaker">${node.speaker || 'NPC'}</span>`;
                    if (emo !== 'neutral') html += `<span class="emotion-tag">${emo}</span>`;
                    if (node.topic) html += `<span class="topic-tag">${node.topic}</span>`;
                    html += `<br>${node.text}`;
                    html += '</div>';
                });
                html += '</div>';
            });
            document.getElementById('samples').innerHTML = html || '<p style="color:#888">No samples returned</p>';
        }

        async function loadSubgraph(nodeId) {
            document.getElementById('subgraphSection').style.display = 'block';
            document.getElementById('subgraphCenter').textContent = nodeId;
            document.getElementById('subgraphView').innerHTML = '<p>Loading...</p>';

            const resp = await fetch(`${API}/subgraph`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({game: currentGame, center_id: nodeId, radius: 2})
            });
            const data = await resp.json();

            let html = `<p style="font-size:11px;color:#888">${data.stats.nodes} nodes, ${data.stats.edges} edges in radius-2 neighborhood</p>`;
            data.nodes.slice(0, 10).forEach(node => {
                const emo = node.emotion || 'neutral';
                const isCenter = node.id === nodeId;
                html += `<div class="sample emotion-${emo}" style="${isCenter ? 'border-width:3px' : ''}">`;
                html += `<span class="node-id">${node.id}</span>`;
                html += `<span class="speaker">${node.speaker || 'NPC'}</span>`;
                if (emo !== 'neutral') html += `<span class="emotion-tag">${emo}</span>`;
                html += ` <span style="font-size:10px;color:#666">in:${node.in_degree} out:${node.out_degree}</span>`;
                html += `<br>${node.text}`;
                html += '</div>';
            });
            if (data.nodes.length > 10) html += `<p style="color:#888">...and ${data.nodes.length - 10} more</p>`;
            document.getElementById('subgraphView').innerHTML = html;

            updateApiExamples('subgraph', {game: currentGame, center_id: nodeId, radius: 2});
            document.getElementById('subgraphSection').scrollIntoView({behavior: 'smooth'});
        }

        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tab==='curl'?1:tab==='fetch'?2:3})`).classList.add('active');
            document.getElementById('tab-' + tab).classList.add('active');
        }

        function updateApiExamples(type, body = null) {
            let curl, fetchEx, python;
            if (type === 'stats') {
                curl = `curl localhost:8000/api/stats/${currentGame}`;
                fetchEx = `fetch('/api/stats/${currentGame}').then(r=>r.json())`;
                python = `requests.get('http://localhost:8000/api/stats/${currentGame}').json()`;
            } else if (type === 'sample' && body) {
                const json = JSON.stringify(body);
                curl = `curl -X POST localhost:8000/api/sample \\
  -H "Content-Type: application/json" \\
  -d '${json}'`;
                fetchEx = `fetch('/api/sample', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: '${json}'
}).then(r=>r.json())`;
                python = `requests.post('http://localhost:8000/api/sample',
  json=${json}).json()`;
            } else if (type === 'subgraph' && body) {
                const json = JSON.stringify(body);
                curl = `curl -X POST localhost:8000/api/subgraph \\
  -H "Content-Type: application/json" \\
  -d '${json}'`;
                fetchEx = `fetch('/api/subgraph', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: '${json}'
}).then(r=>r.json())`;
                python = `requests.post('http://localhost:8000/api/subgraph',
  json=${json}).json()`;
            }
            if (curl) {
                document.getElementById('curlExample').textContent = curl;
                document.getElementById('fetchExample').textContent = fetchEx;
                document.getElementById('pythonExample').textContent = python;
            }
        }

        function showSampleRequest() {
            updateApiExamples('sample', {game: currentGame, method: 'walk', count: 3, max_length: 6});
        }
        function showSubgraphRequest() {
            updateApiExamples('subgraph', {game: currentGame, center_id: '0x...', radius: 2});
        }
        function showPathRequest() {
            document.getElementById('pathSection').style.display = 'block';
            document.getElementById('pathSection').scrollIntoView({behavior: 'smooth'});
        }

        // Graph Analysis Functions
        async function loadPageRank() {
            document.getElementById('analysisResults').innerHTML = '<p>Computing PageRank...</p>';
            const resp = await fetch(`${API}/pagerank/${currentGame}?top_n=15`);
            const data = await resp.json();

            let html = '<h4 style="margin-top:0">Top Nodes by PageRank</h4>';
            html += '<p style="font-size:11px;color:#888">High PageRank = important narrative hub</p>';
            data.top_nodes.forEach((node, i) => {
                const emo = node.emotion || 'neutral';
                html += `<div class="sample emotion-${emo}">`;
                html += `<span class="node-id" onclick="loadSubgraph('${node.id}')">${node.id}</span>`;
                html += `<strong>#${i+1}</strong> <span style="color:#fbbf24">${node.score.toFixed(5)}</span>`;
                html += ` <span class="speaker">${node.speaker || 'NPC'}</span>`;
                if (node.quest) html += `<span class="quest-tag">📜 ${node.quest}</span>`;
                html += `<br>${node.text}`;
                html += '</div>';
            });
            document.getElementById('analysisResults').innerHTML = html;
            document.getElementById('pathSection').style.display = 'block';
        }

        async function loadCommunities() {
            document.getElementById('analysisResults').innerHTML = '<p>Detecting communities...</p>';
            const resp = await fetch(`${API}/communities/${currentGame}?algorithm=louvain`);
            const data = await resp.json();

            let html = `<h4 style="margin-top:0">Detected ${data.community_count} Communities</h4>`;
            html += `<p style="font-size:11px;color:#888">Algorithm: ${data.algorithm}</p>`;
            data.communities.slice(0, 10).forEach((comm, i) => {
                html += `<div class="card" style="margin:8px 0;padding:10px">`;
                html += `<strong>Community ${comm.id + 1}</strong> (${comm.size} nodes)`;
                html += ` <span class="emotion-tag">${comm.dominant_emotion}</span>`;
                if (comm.dominant_quest) html += `<span class="quest-tag">📜 ${comm.dominant_quest}</span>`;
                html += '<div style="margin-top:8px">';
                comm.sample_members.slice(0, 3).forEach(m => {
                    html += `<div style="font-size:12px;padding:3px 0;border-left:2px solid #444;padding-left:8px;margin:3px 0">`;
                    html += `<span class="node-id" onclick="loadSubgraph('${m.id}')" style="font-size:9px">${m.id}</span>`;
                    html += m.text;
                    html += '</div>';
                });
                html += '</div></div>';
            });
            document.getElementById('analysisResults').innerHTML = html;
        }

        async function loadCentrality() {
            document.getElementById('analysisResults').innerHTML = '<p>Computing centrality metrics...</p>';
            const resp = await fetch(`${API}/centrality/${currentGame}?top_n=8`);
            const data = await resp.json();

            let html = '<h4 style="margin-top:0">Centrality Analysis</h4>';
            for (const [metric, nodes] of Object.entries(data.metrics)) {
                const desc = {
                    degree: '🔗 Most connected',
                    betweenness: '🚧 Narrative bottlenecks',
                    closeness: '🎯 Central to graph'
                }[metric] || metric;
                html += `<div style="margin-bottom:15px">`;
                html += `<h5 style="margin:5px 0;color:#fbbf24">${desc}</h5>`;
                nodes.slice(0, 5).forEach((node, i) => {
                    html += `<div style="font-size:12px;padding:4px 0">`;
                    html += `<span class="node-id" onclick="loadSubgraph('${node.id}')" style="font-size:9px">${node.id}</span>`;
                    html += `<strong>${i+1}.</strong> <span style="color:#00d9ff">${node.score.toFixed(4)}</span> `;
                    html += node.text.slice(0, 50) + (node.text.length > 50 ? '...' : '');
                    html += '</div>';
                });
                html += '</div>';
            }
            document.getElementById('analysisResults').innerHTML = html;
            document.getElementById('pathSection').style.display = 'block';
        }

        async function loadSCCs() {
            document.getElementById('analysisResults').innerHTML = '<p>Finding dialogue loops...</p>';
            const resp = await fetch(`${API}/components/${currentGame}`);
            const data = await resp.json();

            let html = `<h4 style="margin-top:0">Strongly Connected Components (${data.scc_count} loops)</h4>`;
            html += '<p style="font-size:11px;color:#888">Dialogue that can loop back on itself</p>';
            if (data.components.length === 0) {
                html += '<p style="color:#888">No non-trivial SCCs found (all dialogue is linear)</p>';
            }
            data.components.slice(0, 10).forEach(scc => {
                html += `<div class="card" style="margin:8px 0;padding:10px">`;
                html += `<strong>Loop ${scc.id + 1}</strong> (${scc.size} nodes)`;
                if (scc.dominant_quest) html += `<span class="quest-tag">📜 ${scc.dominant_quest}</span>`;
                html += '<div style="margin-top:8px">';
                scc.sample_nodes.forEach(n => {
                    html += `<div style="font-size:12px;padding:3px 0;border-left:2px solid #ff6b6b;padding-left:8px;margin:3px 0">`;
                    html += `<span class="node-id" onclick="loadSubgraph('${n.id}')" style="font-size:9px">${n.id}</span>`;
                    html += `<span class="topic-tag">${n.topic}</span> `;
                    html += n.text.slice(0, 60) + (n.text.length > 60 ? '...' : '');
                    html += '</div>';
                });
                html += '</div></div>';
            });
            document.getElementById('analysisResults').innerHTML = html;
        }

        async function findPath() {
            const source = document.getElementById('pathSource').value.trim();
            const target = document.getElementById('pathTarget').value.trim();
            if (!source || !target) {
                document.getElementById('pathResult').innerHTML = '<p style="color:#ff6b6b">Enter both source and target node IDs</p>';
                return;
            }

            document.getElementById('pathResult').innerHTML = '<p>Finding path...</p>';
            const resp = await fetch(`${API}/path`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({game: currentGame, source, target})
            });

            if (!resp.ok) {
                const err = await resp.json();
                document.getElementById('pathResult').innerHTML = `<p style="color:#ff6b6b">${err.detail}</p>`;
                return;
            }

            const data = await resp.json();
            if (!data.path || data.path.length === 0) {
                document.getElementById('pathResult').innerHTML = '<p style="color:#888">No path found between these nodes</p>';
                return;
            }

            let html = `<p style="font-size:12px;color:#888">Path length: ${data.path_length} nodes</p>`;
            data.path.forEach((node, i) => {
                const emo = node.emotion || 'neutral';
                html += `<div class="sample emotion-${emo}" style="margin:4px 0">`;
                html += `<span class="node-id" onclick="loadSubgraph('${node.id}')">${node.id}</span>`;
                html += `<strong>${i + 1}.</strong> <span class="speaker">${node.speaker || 'NPC'}</span>`;
                if (emo !== 'neutral') html += `<span class="emotion-tag">${emo}</span>`;
                html += `<br>${node.text}`;
                html += '</div>';
                if (i < data.path.length - 1) {
                    html += '<div style="text-align:center;color:#444">↓</div>';
                }
            });
            document.getElementById('pathResult').innerHTML = html;
        }

        // ===== Bridge Graph Functions =====
        let bridgeData = null;

        document.getElementById('crossProb').addEventListener('input', (e) => {
            document.getElementById('crossProbVal').textContent = e.target.value + '%';
        });

        async function loadBridgeGraph() {
            document.getElementById('bridgeMatrix').innerHTML = '<p>Loading bridge graph...</p>';

            const [matrixResp, statsResp] = await Promise.all([
                fetch(`${API}/bridge/matrix`),
                fetch(`${API}/bridge/stats`)
            ]);

            const matrixData = await matrixResp.json();
            const statsData = await statsResp.json();
            bridgeData = matrixData;

            // Show stats
            document.getElementById('bridgeStats').style.display = 'block';
            document.getElementById('bridgeStatsGrid').innerHTML = `
                <div class="stat-item"><div class="stat-value">${statsData.total_nodes.toLocaleString()}</div><div class="stat-label">Total Nodes</div></div>
                <div class="stat-item"><div class="stat-value">${statsData.bridge_cells}</div><div class="stat-label">Bridge Cells</div></div>
                <div class="stat-item"><div class="stat-value">${statsData.games.length}</div><div class="stat-label">Games</div></div>
            `;

            // Render matrix with bridge highlighting
            renderBridgeMatrix(matrixData);
        }

        function renderBridgeMatrix(data) {
            const emotions = data.emotions.filter(e =>
                data.emotions.some(e2 => data.matrix[e] && data.matrix[e][e2] && data.matrix[e][e2].total > 0)
            );

            let html = '<div class="matrix" style="grid-template-columns: repeat(' + (emotions.length + 1) + ', 1fr)">';
            html += '<div class="matrix-cell matrix-header">→</div>';
            emotions.forEach(e => html += `<div class="matrix-cell matrix-header">${e.slice(0,3)}</div>`);

            emotions.forEach(src => {
                html += `<div class="matrix-cell matrix-header">${src.slice(0,3)}</div>`;
                emotions.forEach(tgt => {
                    const cell = data.matrix[src] && data.matrix[src][tgt];
                    const count = cell ? cell.total : 0;
                    const isBridge = cell && cell.is_bridge;
                    const games = cell && cell.by_game ? Object.keys(cell.by_game).length : 0;

                    // Color based on bridge status and count
                    let bg = '';
                    if (count > 0) {
                        const intensity = Math.min(count / 500, 1);
                        if (isBridge) {
                            bg = `rgba(251, 191, 36, ${0.3 + intensity * 0.5})`; // Gold for bridges
                        } else {
                            bg = `rgba(0, 217, 255, ${intensity * 0.4})`;
                        }
                    }

                    const title = isBridge
                        ? `${src}→${tgt}: ${count} edges across ${games} games (BRIDGE)`
                        : `${src}→${tgt}: ${count} edges`;

                    html += `<div class="matrix-cell" style="background:${bg};${isBridge ? 'font-weight:bold' : ''}"
                        onclick="sampleBridgeWalk('${src}', '${tgt}')"
                        title="${title}">${count || '-'}</div>`;
                });
            });
            html += '</div>';
            html += '<p style="font-size:10px;color:#888;margin-top:5px">🟡 Gold = cross-game bridge cells</p>';
            document.getElementById('bridgeMatrix').innerHTML = html;
        }

        async function sampleBridgeWalk(srcEmo, tgtEmo) {
            const crossProb = parseInt(document.getElementById('crossProb').value) / 100;
            document.getElementById('bridgeWalkResults').innerHTML = '<p>Sampling cross-game walk...</p>';

            const resp = await fetch(`${API}/bridge/walk`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    max_steps: 12,
                    cross_probability: crossProb,
                    prefer_off_diagonal: true
                })
            });

            const data = await resp.json();
            renderBridgeWalk(data);
        }

        async function sampleCoverage() {
            document.getElementById('bridgeWalkResults').innerHTML = '<p>Sampling coverage trajectory...</p>';

            const resp = await fetch(`${API}/bridge/coverage`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({min_length: 8, max_length: 20})
            });

            const data = await resp.json();
            renderBridgeWalk(data, true);
        }

        function renderBridgeWalk(data, isCoverage = false) {
            let html = `<div style="margin-bottom:10px">`;
            html += `<strong>${isCoverage ? 'Coverage' : 'Bridge'} Walk</strong> `;
            html += `<span style="color:#888">(${data.length} steps, games: ${data.games_visited?.join(', ') || data.games_covered?.join(', ')})</span>`;
            if (isCoverage) {
                html += ` <span style="color:#fbbf24">Coverage: ${Math.round(data.coverage_ratio * 100)}%</span>`;
            }
            if (data.transitions && data.transitions.length > 0) {
                html += `<br><span style="font-size:11px;color:#00d9ff">Bridges: ${data.transitions.map(t =>
                    `${t.from_game}→${t.to_game} @${t.at_step}`).join(', ')}</span>`;
            }
            html += '</div>';

            let currentGame = null;
            data.path.forEach((node, i) => {
                const gameChanged = currentGame && node.game !== currentGame;
                currentGame = node.game;

                if (gameChanged) {
                    html += `<div style="text-align:center;padding:5px;background:#2d3748;margin:3px 0;border-radius:4px;font-size:11px;color:#fbbf24">
                        🌉 BRIDGE TO ${node.game.toUpperCase()}</div>`;
                }

                const emo = node.emotion || 'neutral';
                const gameColor = {'skyrim': '#60a5fa', 'falloutnv': '#4ade80', 'oblivion': '#f472b6'}[node.game] || '#888';

                html += `<div class="sample emotion-${emo}" style="border-left-color:${gameColor}">`;
                html += `<span style="font-size:10px;color:${gameColor};float:right">${node.game}</span>`;
                html += `<span class="speaker">${node.speaker || 'NPC'}</span>`;
                if (emo !== 'neutral') html += `<span class="emotion-tag">${emo}</span>`;
                html += `<br>${node.text || '(no text)'}`;
                html += '</div>';
            });

            document.getElementById('bridgeWalkResults').innerHTML = html;
        }

        init();
    </script>
</body>
</html>
"""

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
