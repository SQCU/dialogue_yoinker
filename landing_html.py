"""
Landing page HTML for the Dialogue Graph API.

Extracted from api_server.py to reduce token count for LLM editing.
"""

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

                <h2>🔗 Query Graph (Topics as Gaps)</h2>
                <div class="card">
                    <p style="font-size:12px;color:#888;margin-bottom:10px">
                        Topics are semantic "gaps" that text responses fill. Explore the bipartite topic↔text structure and find conversational cycles.
                    </p>
                    <div style="margin-bottom:10px">
                        <button onclick="loadQueryStats()">📊 Load Stats</button>
                        <button onclick="loadQueryCycles()" class="secondary">🔄 Find Cycles</button>
                        <button onclick="loadQueryTransitions()" class="secondary">🔀 Transitions</button>
                        <button onclick="sampleQueryWalk()" class="secondary">🚶 Topic Walk</button>
                    </div>
                    <div style="margin-bottom:10px">
                        <select id="queryCategory" style="width:140px">
                            <option value="">Any Category</option>
                            <option value="GREETING">GREETING</option>
                            <option value="FAREWELL">FAREWELL</option>
                            <option value="COMBAT">COMBAT</option>
                            <option value="QUEST">QUEST</option>
                            <option value="TRADE">TRADE</option>
                            <option value="CRIME">CRIME</option>
                            <option value="COMPANION">COMPANION</option>
                            <option value="RUMORS">RUMORS</option>
                        </select>
                        <button onclick="sampleQueryCategory()" class="secondary">Sample Category</button>
                        <label><input type="checkbox" id="queryCrossGame"> Cross-game</label>
                    </div>
                    <div id="queryStats" style="display:none;margin-bottom:15px">
                        <div class="stats-grid" id="queryStatsGrid"></div>
                    </div>
                    <div id="queryResults"></div>
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

    <!-- Synthetic Dialogue Section (separate from reference games) -->
    <div id="syntheticSection" style="margin-top:40px;display:none">
        <h2 style="color:#84cc16">🧪 Synthetic Dialogue (Transposed Settings)</h2>
        <p class="subtitle">Dialogue generated by setting transposition from source games to new fictional settings</p>

        <div class="card" style="margin-bottom:15px">
            <select id="syntheticSelect" onchange="loadSynthetic()" style="font-size:15px;padding:10px">
                <option value="">Select a synthetic setting...</option>
            </select>
            <span id="syntheticLoading"></span>
        </div>

        <div id="syntheticContent" style="display:none">
            <div class="layout">
                <div class="main">
                    <div class="card">
                        <h3>📊 Statistics</h3>
                        <div class="stats-grid" id="syntheticStatsGrid"></div>
                        <div style="margin-top:15px">
                            <h4>Arc Shapes</h4>
                            <div id="syntheticArcShapes"></div>
                        </div>
                    </div>

                    <h3 style="margin-top:20px">🔀 Sample Trajectories</h3>
                    <div class="card">
                        <div style="margin-bottom:10px">
                            <button onclick="sampleSyntheticTrajectory()">Random Trajectory</button>
                            <select id="syntheticArcFilter" style="margin-left:10px">
                                <option value="">Any arc shape</option>
                            </select>
                            <button onclick="sampleSyntheticByArc()" class="secondary">Sample by Arc</button>
                        </div>
                        <div id="syntheticSamples"><p style="color:#888">Click to sample transposed dialogue</p></div>
                    </div>

                    <h3 style="margin-top:20px">🔄 Concept Mappings</h3>
                    <div class="card">
                        <p style="font-size:12px;color:#888;margin-bottom:10px">How source setting concepts were transposed to target setting</p>
                        <div id="syntheticMappings"><p style="color:#888">Loading...</p></div>
                    </div>

                    <h3 style="margin-top:20px">📝 Source → Target Comparison</h3>
                    <div class="card">
                        <select id="syntheticSourceGame">
                            <option value="">Select source game...</option>
                        </select>
                        <button onclick="loadSyntheticComparison()" class="secondary" style="margin-left:10px">Compare</button>
                        <div id="syntheticComparison" style="margin-top:15px"></div>
                    </div>
                </div>

                <div class="sidebar">
                    <div class="card">
                        <h3>📁 Synthetic Resources</h3>
                        <div class="resource-list" id="syntheticResourceList">
                            <p style="color:#888">Select a setting...</p>
                        </div>
                    </div>
                    <div class="card">
                        <h3>🤖 API Example</h3>
                        <pre class="api-example" id="syntheticApiExample">GET /api/synthetic/settings</pre>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Synthetic Graph Analysis Section -->
    <div id="syntheticGraphSection" style="margin-top:40px;display:none">
        <h2 style="color:#fbbf24">📊 Synthetic Graph Topology</h2>
        <p class="subtitle">Analyze the structure of generated dialogue graphs - compare to reference games</p>

        <div class="card" style="margin-bottom:15px">
            <select id="syntheticGraphSelect" onchange="loadSyntheticGraph()" style="font-size:15px;padding:10px">
                <option value="">Select a synthetic graph...</option>
            </select>
            <span id="syntheticGraphLoading"></span>
        </div>

        <div id="syntheticGraphContent" style="display:none">
            <div class="layout">
                <div class="main">
                    <div class="card">
                        <h3>📈 Topology Statistics</h3>
                        <div class="stats-grid" id="synGraphStatsGrid"></div>
                        <div style="margin-top:15px">
                            <h4>Degree Distribution</h4>
                            <div id="synGraphDegrees"></div>
                        </div>
                    </div>

                    <h3 style="margin-top:20px">🔬 Graph Analysis</h3>
                    <div class="card">
                        <div style="margin-bottom:10px">
                            <button onclick="loadSynGraphPageRank()">📊 PageRank</button>
                            <button onclick="loadSynGraphCentrality()">🎯 Hubs</button>
                            <button onclick="loadSynGraphCommunities()">🏘️ Communities</button>
                            <button onclick="loadSynGraphComponents()">🔗 Components</button>
                        </div>
                        <div id="synGraphAnalysis"><p style="color:#888">Click to analyze synthetic graph structure</p></div>
                    </div>

                    <h3 style="margin-top:20px">🔄 Compare to Reference</h3>
                    <div class="card">
                        <select id="synGraphRefGame">
                            <option value="">Select reference game...</option>
                            <option value="oblivion">Oblivion</option>
                            <option value="falloutnv">Fallout NV</option>
                        </select>
                        <button onclick="loadSynGraphComparison()" class="secondary" style="margin-left:10px">Compare</button>
                        <div id="synGraphComparison" style="margin-top:15px"></div>
                    </div>
                </div>

                <div class="sidebar">
                    <div class="card">
                        <h3>📁 Graph API</h3>
                        <div class="resource-list" id="synGraphResourceList">
                            <p style="color:#888">Select a graph...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API = '/api';
        let currentGame = '';
        let currentSynthetic = '';

        async function init() {
            const resp = await fetch(`${API}/games`);
            const games = await resp.json();
            const select = document.getElementById('gameSelect');
            select.innerHTML = games.map(g =>
                `<option value="${g.name}">${g.name} (${g.dialogue_count.toLocaleString()} lines)</option>`
            ).join('');
            if (games.length > 0) loadGame();

            // Also load synthetic settings if available
            initSynthetic();
        }

        // =========== SYNTHETIC DIALOGUE FUNCTIONS ===========

        async function initSynthetic() {
            try {
                const resp = await fetch(`${API}/synthetic/settings`);
                const settings = await resp.json();
                if (settings.length > 0) {
                    document.getElementById('syntheticSection').style.display = 'block';
                    const select = document.getElementById('syntheticSelect');
                    select.innerHTML = '<option value="">Select a synthetic setting...</option>' +
                        settings.map(s =>
                            `<option value="${s.setting}">${s.setting} (${s.total_beats} beats, ${s.total_trajectories} trajectories)</option>`
                        ).join('');
                }
            } catch (e) {
                // Synthetic routes not available, that's fine
                console.log('Synthetic routes not available:', e);
            }
        }

        async function loadSynthetic() {
            currentSynthetic = document.getElementById('syntheticSelect').value;
            if (!currentSynthetic) {
                document.getElementById('syntheticContent').style.display = 'none';
                return;
            }

            document.getElementById('syntheticLoading').textContent = 'Loading...';
            document.getElementById('syntheticContent').style.display = 'block';

            // Load stats
            const statsResp = await fetch(`${API}/synthetic/${currentSynthetic}/stats`);
            const stats = await statsResp.json();

            document.getElementById('syntheticStatsGrid').innerHTML = `
                <div class="stat-item"><div class="stat-value">${stats.total_beats.toLocaleString()}</div><div class="stat-label">Beats</div></div>
                <div class="stat-item"><div class="stat-value">${stats.total_trajectories}</div><div class="stat-label">Trajectories</div></div>
                <div class="stat-item"><div class="stat-value">${stats.avg_confidence.toFixed(2)}</div><div class="stat-label">Avg Confidence</div></div>
                <div class="stat-item"><div class="stat-value">${stats.total_concept_mappings}</div><div class="stat-label">Concept Maps</div></div>
                <div class="stat-item"><div class="stat-value">${stats.unique_source_concepts}</div><div class="stat-label">Unique Concepts</div></div>
                <div class="stat-item"><div class="stat-value">${stats.source_games.length}</div><div class="stat-label">Source Games</div></div>
            `;

            // Arc shapes
            let arcHtml = '<div style="display:flex;flex-wrap:wrap;gap:5px">';
            for (const [shape, count] of Object.entries(stats.arc_shapes)) {
                arcHtml += `<span style="background:#0f3460;padding:4px 8px;border-radius:4px;font-size:12px">${shape}: ${count}</span>`;
            }
            arcHtml += '</div>';
            document.getElementById('syntheticArcShapes').innerHTML = arcHtml;

            // Update arc filter dropdown
            const arcSelect = document.getElementById('syntheticArcFilter');
            arcSelect.innerHTML = '<option value="">Any arc shape</option>' +
                Object.keys(stats.arc_shapes).map(s => `<option value="${s}">${s}</option>`).join('');

            // Update source game dropdown
            const sourceSelect = document.getElementById('syntheticSourceGame');
            sourceSelect.innerHTML = '<option value="">Select source game...</option>' +
                stats.source_games.map(g => `<option value="${g}">${g}</option>`).join('');

            // Update resource links
            document.getElementById('syntheticResourceList').innerHTML = `
                <a href="/api/synthetic/${currentSynthetic}/stats">📊 GET /api/synthetic/${currentSynthetic}/stats</a>
                <a href="/api/synthetic/${currentSynthetic}/trajectories">🔀 GET /api/synthetic/${currentSynthetic}/trajectories</a>
                <a href="/api/synthetic/${currentSynthetic}/dialogue">📝 GET /api/synthetic/${currentSynthetic}/dialogue</a>
                <a href="/api/synthetic/${currentSynthetic}/concept-mappings">🔄 GET /api/synthetic/${currentSynthetic}/concept-mappings</a>
            `;

            // Load concept mappings
            loadConceptMappings();

            document.getElementById('syntheticLoading').textContent = '';
        }

        async function loadConceptMappings() {
            const resp = await fetch(`${API}/synthetic/${currentSynthetic}/concept-mappings?min_occurrences=1`);
            const data = await resp.json();

            let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px">';
            data.mappings.slice(0, 20).forEach(m => {
                html += `<div style="background:#0f3460;padding:6px;border-radius:4px">`;
                html += `<span style="color:#ff6b6b">${m.source}</span>`;
                html += ` → <span style="color:#84cc16">${m.target}</span>`;
                html += ` <span style="color:#888">(${m.count})</span>`;
                html += `</div>`;
            });
            html += '</div>';
            if (data.mappings.length > 20) {
                html += `<p style="color:#888;font-size:11px;margin-top:8px">...and ${data.mappings.length - 20} more</p>`;
            }
            document.getElementById('syntheticMappings').innerHTML = html;
        }

        async function sampleSyntheticTrajectory() {
            document.getElementById('syntheticSamples').innerHTML = '<p>Sampling...</p>';

            const resp = await fetch(`${API}/synthetic/${currentSynthetic}/sample`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: currentSynthetic, method: 'by_arc', count: 2})
            });
            const data = await resp.json();
            renderSyntheticSamples(data.samples);
        }

        async function sampleSyntheticByArc() {
            const arcFilter = document.getElementById('syntheticArcFilter').value;
            document.getElementById('syntheticSamples').innerHTML = '<p>Sampling...</p>';

            const resp = await fetch(`${API}/synthetic/${currentSynthetic}/sample`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: currentSynthetic, method: 'by_arc', count: 2, arc_filter: arcFilter || null})
            });
            const data = await resp.json();
            renderSyntheticSamples(data.samples);
        }

        function renderSyntheticSamples(samples) {
            if (!samples || samples.length === 0) {
                document.getElementById('syntheticSamples').innerHTML = '<p style="color:#888">No samples found</p>';
                return;
            }

            let html = '';
            samples.forEach((traj, i) => {
                const arc = traj.arc || {};
                html += `<div style="margin-bottom:20px">`;
                html += `<strong>Trajectory ${i + 1}</strong>`;
                html += ` <span style="color:#84cc16">${arc.shape || 'unknown'}</span>`;
                html += ` <span style="color:#888">(conf: ${(traj.confidence || 0).toFixed(2)})</span>`;
                html += ` <span style="color:#fbbf24;font-size:11px">from ${traj.source_game}</span>`;

                (traj.beats || []).forEach((beat, j) => {
                    const emo = beat.emotion || 'neutral';
                    html += `<div class="sample emotion-${emo}" style="margin-top:8px">`;
                    html += `<span style="color:#888;font-size:10px;float:right">${beat.function || ''} | ${beat.archetype_relation || ''}</span>`;
                    if (emo !== 'neutral') html += `<span class="emotion-tag">${emo}</span>`;
                    html += `<br><span style="color:#84cc16">${beat.target_text || ''}</span>`;
                    html += `<br><span style="color:#666;font-size:11px">← ${beat.source_text || ''}</span>`;
                    html += '</div>';
                });

                // Show concept mappings for this trajectory
                if (traj.concept_mappings && traj.concept_mappings.length > 0) {
                    html += '<div style="margin-top:8px;font-size:11px;color:#888">';
                    html += '<strong>Mappings:</strong> ';
                    html += traj.concept_mappings.map(m => `${m.source}→${m.target}`).join(', ');
                    html += '</div>';
                }

                html += '</div>';
            });
            document.getElementById('syntheticSamples').innerHTML = html;
        }

        async function loadSyntheticComparison() {
            const sourceGame = document.getElementById('syntheticSourceGame').value;
            if (!sourceGame) {
                document.getElementById('syntheticComparison').innerHTML = '<p style="color:#888">Select a source game</p>';
                return;
            }

            document.getElementById('syntheticComparison').innerHTML = '<p>Loading comparison...</p>';

            const resp = await fetch(`${API}/synthetic/${currentSynthetic}/compare/${sourceGame}?limit=5`);
            const data = await resp.json();

            let html = `<p style="font-size:12px;color:#888">${data.total_comparisons} beat comparisons from ${sourceGame}</p>`;
            (data.comparisons || []).forEach(c => {
                const emo = c.emotion || 'neutral';
                html += `<div class="sample emotion-${emo}">`;
                html += `<span style="color:#888;font-size:10px;float:right">${c.arc_shape} | ${c.function}</span>`;
                html += `<span class="emotion-tag">${emo}</span>`;
                html += `<br><strong style="color:#84cc16">Target:</strong> ${c.target_text}`;
                html += `<br><strong style="color:#ff6b6b">Source:</strong> ${c.source_text}`;
                html += '</div>';
            });

            document.getElementById('syntheticComparison').innerHTML = html;
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

        // ===== Query Graph Functions =====

        async function loadQueryStats() {
            document.getElementById('queryResults').innerHTML = '<p>Loading query graph stats...</p>';

            const resp = await fetch(`${API}/query/stats`);
            const data = await resp.json();

            document.getElementById('queryStats').style.display = 'block';
            document.getElementById('queryStatsGrid').innerHTML = `
                <div class="stat-item"><div class="stat-value">${data.topic_nodes.toLocaleString()}</div><div class="stat-label">Topics (Gaps)</div></div>
                <div class="stat-item"><div class="stat-value">${data.text_nodes.toLocaleString()}</div><div class="stat-label">Text Responses</div></div>
                <div class="stat-item"><div class="stat-value">${data.cross_game_topics}</div><div class="stat-label">Cross-Game</div></div>
            `;

            let html = '<div style="font-size:12px"><strong>Categories:</strong> ';
            html += Object.entries(data.categories).map(([k, v]) => `${k}(${v})`).join(', ');
            html += '</div>';
            document.getElementById('queryResults').innerHTML = html;
        }

        async function loadQueryCycles() {
            document.getElementById('queryResults').innerHTML = '<p>Finding topic cycles...</p>';

            const resp = await fetch(`${API}/query/cycles?max_cycles=8&max_cycle_length=5`);
            const data = await resp.json();

            let html = `<p style="font-size:12px;color:#888">Found ${data.count} conversational cycles</p>`;

            data.cycles.forEach((cycle, idx) => {
                html += `<div class="sample" style="margin:8px 0">`;
                html += `<strong>Cycle ${idx + 1}</strong> <span style="color:#888">(${cycle.cycle_length} topics, games: ${cycle.games.join(', ')})</span>`;
                html += `<div style="font-size:11px;color:#fbbf24;margin:5px 0">${cycle.topics.join(' → ')} → ↩</div>`;

                cycle.steps.slice(0, 3).forEach(step => {
                    if (step.sample_response) {
                        const gameColor = {'skyrim': '#60a5fa', 'falloutnv': '#4ade80', 'oblivion': '#f472b6'}[step.sample_response.game] || '#888';
                        html += `<div style="margin:3px 0;padding:5px;background:#0f3460;border-radius:3px;font-size:12px">`;
                        html += `<span style="color:#00d9ff">[${step.topic}]</span>`;
                        html += `<span style="color:${gameColor};font-size:10px;float:right">${step.sample_response.game}</span>`;
                        html += `<br>"${step.sample_response.text.slice(0, 60)}..."`;
                        html += `</div>`;
                    }
                });
                if (cycle.steps.length > 3) {
                    html += `<div style="color:#888;font-size:11px">... +${cycle.steps.length - 3} more steps</div>`;
                }
                html += '</div>';
            });

            document.getElementById('queryResults').innerHTML = html;
        }

        async function loadQueryTransitions() {
            document.getElementById('queryResults').innerHTML = '<p>Loading topic transitions...</p>';

            const resp = await fetch(`${API}/query/transitions`);
            const data = await resp.json();

            let html = `<p style="font-size:12px;color:#888">${data.total_transitions.toLocaleString()} topic→topic transitions</p>`;

            html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px">';

            html += '<div><strong style="color:#4ade80">Top Sources (outgoing):</strong>';
            data.top_sources.slice(0, 6).forEach(t => {
                html += `<div style="margin:3px 0;padding:3px;background:#0f3460;border-radius:3px">`;
                html += `${t.topic} <span style="color:#888">(${t.category || 'misc'})</span>`;
                html += `<span style="float:right;color:#4ade80">${t.out_degree}→</span></div>`;
            });
            html += '</div>';

            html += '<div><strong style="color:#f472b6">Top Sinks (incoming):</strong>';
            data.top_sinks.slice(0, 6).forEach(t => {
                html += `<div style="margin:3px 0;padding:3px;background:#0f3460;border-radius:3px">`;
                html += `${t.topic} <span style="color:#888">(${t.category || 'misc'})</span>`;
                html += `<span style="float:right;color:#f472b6">→${t.in_degree}</span></div>`;
            });
            html += '</div>';

            html += '</div>';
            document.getElementById('queryResults').innerHTML = html;
        }

        async function sampleQueryWalk() {
            document.getElementById('queryResults').innerHTML = '<p>Walking topic chain...</p>';

            const crossGame = document.getElementById('queryCrossGame').checked;
            const resp = await fetch(`${API}/query/walk`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({max_steps: 6, cross_game: crossGame})
            });
            const data = await resp.json();

            let html = `<p style="font-size:12px;color:#888">Topic walk: ${data.length} steps, games: ${data.games_visited.join(', ')}</p>`;

            data.path.forEach((step, i) => {
                const gameColor = {'skyrim': '#60a5fa', 'falloutnv': '#4ade80', 'oblivion': '#f472b6'}[step.response.game] || '#888';
                const emo = step.response.emotion || 'neutral';

                html += `<div class="sample emotion-${emo}">`;
                html += `<span style="color:#fbbf24;font-size:11px">[${step.topic}]</span>`;
                html += `<span style="color:#888;font-size:10px"> (${step.category || 'misc'})</span>`;
                html += `<span style="font-size:10px;color:${gameColor};float:right">${step.response.game}</span>`;
                html += `<br><span class="speaker">${step.response.speaker || 'NPC'}</span>`;
                if (emo !== 'neutral') html += `<span class="emotion-tag">${emo}</span>`;
                html += `<br>${step.response.text || '(no text)'}`;
                html += '</div>';

                if (i < data.path.length - 1) {
                    html += '<div style="text-align:center;color:#444;font-size:10px">↓ topic transition ↓</div>';
                }
            });

            document.getElementById('queryResults').innerHTML = html;
        }

        async function sampleQueryCategory() {
            const category = document.getElementById('queryCategory').value;
            const crossGame = document.getElementById('queryCrossGame').checked;

            document.getElementById('queryResults').innerHTML = '<p>Sampling category...</p>';

            const body = {n: 5, cross_game: crossGame};
            if (category) body.category = category;

            const resp = await fetch(`${API}/query/sample`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            const data = await resp.json();

            let html = `<p style="font-size:12px;color:#888">Topic: <strong>${data.topic}</strong>`;
            html += ` (${data.category || 'misc'}) - ${data.total_responses} total responses</p>`;

            data.sampled_responses.forEach(r => {
                const gameColor = {'skyrim': '#60a5fa', 'falloutnv': '#4ade80', 'oblivion': '#f472b6'}[r.game] || '#888';
                const emo = r.emotion || 'neutral';

                html += `<div class="sample emotion-${emo}">`;
                html += `<span style="font-size:10px;color:${gameColor};float:right">${r.game}</span>`;
                html += `<span class="speaker">${r.speaker || 'NPC'}</span>`;
                if (emo !== 'neutral') html += `<span class="emotion-tag">${emo}</span>`;
                html += `<br>${r.text || '(no text)'}`;
                html += '</div>';
            });

            document.getElementById('queryResults').innerHTML = html;
        }

        // =========== SYNTHETIC GRAPH FUNCTIONS ===========

        let currentSynGraph = '';

        async function initSyntheticGraph() {
            try {
                const resp = await fetch(`${API}/synthetic-graph/settings`);
                const data = await resp.json();
                if (data.settings && data.settings.length > 0) {
                    document.getElementById('syntheticGraphSection').style.display = 'block';
                    const select = document.getElementById('syntheticGraphSelect');
                    select.innerHTML = '<option value="">Select a synthetic graph...</option>' +
                        data.settings.map(s =>
                            `<option value="${s.setting}">${s.setting} (${s.nodes} nodes, ${s.edges} edges)</option>`
                        ).join('');
                }
            } catch (e) {
                console.log('Synthetic graph routes not available:', e);
            }
        }

        async function loadSyntheticGraph() {
            currentSynGraph = document.getElementById('syntheticGraphSelect').value;
            if (!currentSynGraph) {
                document.getElementById('syntheticGraphContent').style.display = 'none';
                return;
            }

            document.getElementById('syntheticGraphLoading').textContent = 'Loading...';
            document.getElementById('syntheticGraphContent').style.display = 'block';

            const resp = await fetch(`${API}/synthetic-graph/${currentSynGraph}/stats`);
            const stats = await resp.json();

            document.getElementById('synGraphStatsGrid').innerHTML = `
                <div class="stat-item"><div class="stat-value">${stats.node_count.toLocaleString()}</div><div class="stat-label">Nodes</div></div>
                <div class="stat-item"><div class="stat-value">${stats.edge_count.toLocaleString()}</div><div class="stat-label">Edges</div></div>
                <div class="stat-item"><div class="stat-value">${stats.branching_factor.toFixed(3)}</div><div class="stat-label">Branching Factor</div></div>
                <div class="stat-item"><div class="stat-value">${stats.components?.component_count || 1}</div><div class="stat-label">Components</div></div>
                <div class="stat-item"><div class="stat-value">${stats.leaves?.leaf_count || 0}</div><div class="stat-label">Leaves</div></div>
                <div class="stat-item"><div class="stat-value">${stats.hub_count || 0}</div><div class="stat-label">Hubs</div></div>
            `;

            // Degree distribution
            let degHtml = '<div style="font-size:12px">';
            degHtml += '<strong>Out-degree:</strong> ';
            if (stats.out_degree?.distribution) {
                degHtml += Object.entries(stats.out_degree.distribution).map(([k,v]) => `${k}:${v}`).join(', ');
            }
            degHtml += '<br><strong>In-degree:</strong> ';
            if (stats.in_degree?.distribution) {
                degHtml += Object.entries(stats.in_degree.distribution).map(([k,v]) => `${k}:${v}`).join(', ');
            }
            degHtml += '</div>';
            document.getElementById('synGraphDegrees').innerHTML = degHtml;

            // Update resource links
            document.getElementById('synGraphResourceList').innerHTML = `
                <a href="/api/synthetic-graph/${currentSynGraph}/stats">📈 Stats</a>
                <a href="/api/synthetic-graph/${currentSynGraph}/pagerank">📊 PageRank</a>
                <a href="/api/synthetic-graph/${currentSynGraph}/centrality">🎯 Centrality</a>
                <a href="/api/synthetic-graph/${currentSynGraph}/communities">🏘️ Communities</a>
                <a href="/api/synthetic-graph/${currentSynGraph}/components">🔗 Components</a>
            `;

            document.getElementById('syntheticGraphLoading').textContent = '';
        }

        async function loadSynGraphPageRank() {
            if (!currentSynGraph) return;
            document.getElementById('synGraphAnalysis').innerHTML = '<p>Computing PageRank...</p>';

            const resp = await fetch(`${API}/synthetic-graph/${currentSynGraph}/pagerank?top_n=10`);
            const data = await resp.json();

            let html = '<h4 style="margin-top:0">Top Nodes by PageRank</h4>';
            data.top_nodes.forEach((node, i) => {
                const emo = node.emotion || 'neutral';
                html += `<div class="sample emotion-${emo}" style="margin:5px 0">`;
                html += `<strong>#${i+1}</strong> <span style="color:#fbbf24">${node.score.toFixed(5)}</span>`;
                html += `<span class="emotion-tag">${emo}</span>`;
                html += `<br>${node.text}`;
                html += '</div>';
            });
            document.getElementById('synGraphAnalysis').innerHTML = html;
        }

        async function loadSynGraphCentrality() {
            if (!currentSynGraph) return;
            document.getElementById('synGraphAnalysis').innerHTML = '<p>Finding hubs...</p>';

            const resp = await fetch(`${API}/synthetic-graph/${currentSynGraph}/centrality?top_n=8`);
            const data = await resp.json();

            let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px">';

            html += '<div><h4 style="margin-top:0;color:#4ade80">Top In-Degree (Hubs)</h4>';
            data.top_in_degree.forEach((n, i) => {
                html += `<div style="font-size:12px;padding:4px;background:#0f3460;border-radius:3px;margin:3px 0">`;
                html += `<strong>${i+1}.</strong> in=${n.in_degree} ${n.text.slice(0,40)}...`;
                html += '</div>';
            });
            html += '</div>';

            html += '<div><h4 style="margin-top:0;color:#f472b6">Top Out-Degree (Branches)</h4>';
            data.top_out_degree.forEach((n, i) => {
                html += `<div style="font-size:12px;padding:4px;background:#0f3460;border-radius:3px;margin:3px 0">`;
                html += `<strong>${i+1}.</strong> out=${n.out_degree} ${n.text.slice(0,40)}...`;
                html += '</div>';
            });
            html += '</div>';

            html += '</div>';
            document.getElementById('synGraphAnalysis').innerHTML = html;
        }

        async function loadSynGraphCommunities() {
            if (!currentSynGraph) return;
            document.getElementById('synGraphAnalysis').innerHTML = '<p>Detecting communities...</p>';

            const resp = await fetch(`${API}/synthetic-graph/${currentSynGraph}/communities?algorithm=louvain`);
            const data = await resp.json();

            let html = `<h4 style="margin-top:0">Detected ${data.community_count} Communities</h4>`;
            data.communities.slice(0, 8).forEach((comm, i) => {
                html += `<div class="sample" style="margin:5px 0">`;
                html += `<strong>Community ${i+1}</strong> (${comm.size} nodes)`;
                html += `<span class="emotion-tag">${comm.dominant_emotion}</span>`;
                html += '<div style="font-size:11px;margin-top:5px">';
                comm.sample_texts.forEach(t => html += `<div style="color:#888">• ${t}</div>`);
                html += '</div></div>';
            });
            document.getElementById('synGraphAnalysis').innerHTML = html;
        }

        async function loadSynGraphComponents() {
            if (!currentSynGraph) return;
            document.getElementById('synGraphAnalysis').innerHTML = '<p>Finding components...</p>';

            const resp = await fetch(`${API}/synthetic-graph/${currentSynGraph}/components`);
            const data = await resp.json();

            let html = '<h4 style="margin-top:0">Connected Components</h4>';
            html += `<div class="stats-grid">`;
            html += `<div class="stat-item"><div class="stat-value">${data.weakly_connected.count}</div><div class="stat-label">Weak Components</div></div>`;
            html += `<div class="stat-item"><div class="stat-value">${data.weakly_connected.largest}</div><div class="stat-label">Largest WCC</div></div>`;
            html += `<div class="stat-item"><div class="stat-value">${data.strongly_connected.non_trivial_count}</div><div class="stat-label">SCCs (loops)</div></div>`;
            html += '</div>';

            html += '<div style="margin-top:10px;font-size:12px">';
            html += `<strong>WCC sizes:</strong> ${data.weakly_connected.size_distribution.join(', ')}`;
            html += '</div>';

            document.getElementById('synGraphAnalysis').innerHTML = html;
        }

        async function loadSynGraphComparison() {
            const refGame = document.getElementById('synGraphRefGame').value;
            if (!currentSynGraph || !refGame) {
                document.getElementById('synGraphComparison').innerHTML = '<p style="color:#888">Select both synthetic graph and reference game</p>';
                return;
            }

            document.getElementById('synGraphComparison').innerHTML = '<p>Comparing...</p>';

            try {
                const resp = await fetch(`${API}/synthetic-graph/${currentSynGraph}/compare/${refGame}`);
                const data = await resp.json();

                let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;font-size:12px">';

                html += `<div style="background:#0f3460;padding:10px;border-radius:4px">`;
                html += `<h4 style="color:#84cc16;margin-top:0">Synthetic: ${currentSynGraph}</h4>`;
                html += `<div>Nodes: ${data.synthetic.node_count}</div>`;
                html += `<div>Edges: ${data.synthetic.edge_count}</div>`;
                html += `<div>BF: ${data.synthetic.branching_factor.toFixed(3)}</div>`;
                html += `<div>Components: ${data.synthetic.components?.component_count || 1}</div>`;
                html += '</div>';

                html += `<div style="background:#0f3460;padding:10px;border-radius:4px">`;
                html += `<h4 style="color:#ff6b6b;margin-top:0">Reference: ${refGame}</h4>`;
                html += `<div>Nodes: ${data.reference.node_count}</div>`;
                html += `<div>Edges: ${data.reference.edge_count}</div>`;
                html += `<div>BF: ${data.reference.branching_factor.toFixed(3)}</div>`;
                html += `<div>Components: ${data.reference.components?.component_count || 1}</div>`;
                html += '</div>';

                html += '</div>';

                html += '<div style="margin-top:10px;padding:10px;background:#1f4068;border-radius:4px;font-size:12px">';
                html += '<strong>Comparison:</strong><br>';
                html += `Size ratio: ${(data.comparison.node_ratio * 100).toFixed(1)}% of reference<br>`;
                html += `BF ratio: ${(data.comparison.branching_factor_ratio * 100).toFixed(1)}% of reference<br>`;
                html += '</div>';

                document.getElementById('synGraphComparison').innerHTML = html;
            } catch (e) {
                document.getElementById('synGraphComparison').innerHTML = `<p style="color:#ff6b6b">Error: ${e.message}</p>`;
            }
        }

        // Initialize both synthetic sections
        async function initAll() {
            await init();
            await initSyntheticGraph();
        }

        initAll();
    </script>
</body>
</html>
"""
