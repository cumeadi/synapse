/**
 * Synapse Studio Vanilla JS App
 */

// State
let network = null;
let currentNamespaceId = null;
let currentGraphData = { nodes: [], edges: [] };
let selectedElement = null; // { type: 'node'|'edge', id: str }
let apiKey = localStorage.getItem('synapse_api_key') || '';
let autoRefreshInterval = null;

// DOM Elements
const nsSelect = document.getElementById('namespace-select');
const refreshBtn = document.getElementById('refresh-btn');
const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
const graphContainer = document.getElementById('graph-container');

// Inspector Elements
const inspectorPanel = document.getElementById('inspector-panel');
const closeInspectorBtn = document.getElementById('close-inspector-btn');
const nodeDetails = document.getElementById('node-details');
const edgeDetails = document.getElementById('edge-details');
const deleteBtn = document.getElementById('delete-btn');
const deleteLabel = document.getElementById('delete-type-label');

// Status Overlay Elements
const statusOverlay = document.getElementById('status-overlay');
const statusTitle = document.getElementById('status-title');
const statusMessage = document.getElementById('status-message');
const statusSpinner = document.getElementById('status-spinner');

// API Key Modal
const apiKeyModal = document.getElementById('api-key-modal');
const apiKeyInput = document.getElementById('api-key-input');
const apiKeySubmit = document.getElementById('api-key-submit');

// VisNetwork Options
const networkOptions = {
    nodes: {
        shape: 'dot',
        size: 20,
        font: {
            size: 14,
            color: '#f3f4f6', // text-gray-100
            face: 'sans-serif'
        },
        borderWidth: 2,
        shadow: true
    },
    edges: {
        width: 2,
        color: {
            color: '#6b7280', // text-gray-500
            highlight: '#60a5fa', // text-blue-400
            hover: '#9ca3af'
        },
        arrows: {
            to: { enabled: true, scaleFactor: 0.5 }
        },
        font: {
            size: 11,
            color: '#9ca3af', // text-gray-400
            align: 'top',
            strokeWidth: 0
        },
        smooth: {
            type: 'continuous',
            roundness: 0.5
        }
    },
    physics: {
        forceAtlas2Based: {
            gravitationalConstant: -100,
            centralGravity: 0.005,
            springLength: 200,
            springConstant: 0.18
        },
        maxVelocity: 50,
        solver: 'forceAtlas2Based',
        timestep: 0.35,
        stabilization: { iterations: 150 }
    },
    interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true
    },
    groups: {} // We'll populate this dynamically based on entity_type
};

// Colors for different entity types
const typeColors = [
    { background: '#3b82f6', border: '#2563eb' }, // blue
    { background: '#10b981', border: '#059669' }, // green
    { background: '#8b5cf6', border: '#7c3aed' }, // purple
    { background: '#f59e0b', border: '#d97706' }, // amber
    { background: '#ec4899', border: '#db2777' }, // pink
    { background: '#06b6d4', border: '#0891b2' }, // cyan
    { background: '#ef4444', border: '#dc2626' }, // red
];

/**
 * Initialize
 */
async function init() {
    setupEventListeners();
    await fetchNamespaces();
}

/**
 * Event Listeners
 */
function setupEventListeners() {
    nsSelect.addEventListener('change', (e) => {
        currentNamespaceId = e.target.value;
        if (currentNamespaceId) {
            closeInspector();
            fetchGraph();
        }
    });

    refreshBtn.addEventListener('click', () => {
        if (currentNamespaceId) fetchGraph();
    });

    autoRefreshToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            if (currentNamespaceId) {
                autoRefreshInterval = setInterval(fetchGraph, 3000);
            } else {
                e.target.checked = false;
                alert("Please select a namespace first.");
            }
        } else {
            clearInterval(autoRefreshInterval);
        }
    });

    closeInspectorBtn.addEventListener('click', closeInspector);

    deleteBtn.addEventListener('click', async () => {
        if (!selectedElement || !currentNamespaceId) return;
        
        const isNode = selectedElement.type === 'node';
        const type = isNode ? 'entities' : 'relationships';
        const msg = isNode ? 
            "Delete this entity and all its connections?" : 
            "Delete this relationship?";
            
        if (confirm(msg)) {
            try {
                const res = await fetch(`/namespaces/${currentNamespaceId}/${type}/${selectedElement.id}`, {
                    method: 'DELETE',
                    headers: { 'X-API-Key': apiKey }
                });
                
                if (res.ok || res.status === 204) {
                    closeInspector();
                    fetchGraph(); // Reload to reflect deletion
                } else {
                    const data = await res.json();
                    alert(`Error: ${data.detail || 'Could not delete item'}`);
                }
            } catch (err) {
                console.error("Delete failed", err);
                alert("Network error during deletion.");
            }
        }
    });

    apiKeySubmit.addEventListener('click', () => {
        apiKey = apiKeyInput.value.trim();
        localStorage.setItem('synapse_api_key', apiKey);
        apiKeyModal.classList.add('hidden');
        if (currentNamespaceId) fetchGraph();
        else fetchNamespaces();
    });
}

/**
 * Fetch Namespaces for Dropdown
 */
async function fetchNamespaces() {
    try {
        const res = await fetch('/namespaces/', {
            headers: apiKey ? { 'X-API-Key': apiKey } : {}
        });
        
        if (res.status === 401 || res.status === 403) {
            showApiKeyModal();
            return;
        }
        
        if (!res.ok) throw new Error('Failed to fetch namespaces');
        
        const namespaces = await res.json();
        
        // Clear existing (keep placeholder)
        while (nsSelect.options.length > 1) {
            nsSelect.remove(1);
        }
        
        namespaces.forEach(ns => {
            const opt = document.createElement('option');
            opt.value = ns.id;
            opt.textContent = ns.name;
            nsSelect.appendChild(opt);
        });
        
    } catch (err) {
        console.error("Namespace fetch error:", err);
    }
}

/**
 * Fetch Graph Data
 */
async function fetchGraph() {
    if (!currentNamespaceId) return;
    
    // Only show loading overlay on first load
    if (!network) {
        showStatus('Loading Graph...', 'Fetching nodes and edges.', true);
    }

    try {
        const res = await fetch(`/namespaces/${currentNamespaceId}/graph`, {
            headers: apiKey ? { 'X-API-Key': apiKey } : {}
        });
        
        if (res.status === 401 || res.status === 403) {
            showApiKeyModal();
            return;
        }
        
        if (!res.ok) throw new Error('Failed to fetch graph');
        
        const data = await res.json();
        currentGraphData = data;
        
        if (data.nodes.length === 0) {
            showStatus('Empty Graph', 'This namespace has no memories yet.');
            if (network) {
                network.destroy();
                network = null;
            }
        } else {
            hideStatus();
            renderGraph(data);
        }
        
    } catch (err) {
        console.error("Graph fetch error:", err);
        showStatus('Error', 'Failed to load graph data.');
    }
}

/**
 * Render VisNetwork
 */
function renderGraph(data) {
    // Generate dynamic groups based on entity sets
    const groups = {};
    const uniqueTypes = [...new Set(data.nodes.map(n => n.group))].sort();
    
    uniqueTypes.forEach((type, index) => {
        const color = typeColors[index % typeColors.length];
        groups[type] = {
            color: { background: color.background, border: color.border },
            font: { color: '#ffffff' }
        };
    });
    
    // Assign titles (tooltips)
    const nodes = data.nodes.map(n => ({
        ...n,
        title: `Type: ${n.group}`
    }));
    
    const maxWeight = Math.max(...data.edges.map(e => e.weight), 1);
    const edges = data.edges.map(e => ({
        ...e,
        title: `Weight: ${e.weight}`,
        width: 1 + (e.weight / maxWeight) * 4 // Scale width based on weight
    }));
    
    const vizData = { nodes, edges };
    const vizOptions = { ...networkOptions, groups };

    // Update existing network or create new
    if (network) {
        network.setData(vizData);
    } else {
        network = new vis.Network(graphContainer, vizData, vizOptions);
        
        // Setup click events
        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                // Node clicked
                openNodeInspector(params.nodes[0]);
            } else if (params.edges.length > 0) {
                // Edge clicked
                openEdgeInspector(params.edges[0]);
            } else {
                // Canvas clicked
                closeInspector();
            }
        });
    }
    
    // Re-highlight if inspector is open
    if (selectedElement && network) {
        network.selectNodes(selectedElement.type === 'node' ? [selectedElement.id] : []);
        network.selectEdges(selectedElement.type === 'edge' ? [selectedElement.id] : []);
    }
}

/**
 * Status Overlay Helpers
 */
function showStatus(title, message, isLoading = false) {
    statusTitle.textContent = title;
    statusMessage.textContent = message;
    statusSpinner.classList.toggle('hidden', !isLoading);
    statusOverlay.classList.remove('opacity-0', 'pointer-events-none');
    statusOverlay.classList.add('opacity-100', 'pointer-events-auto');
}

function hideStatus() {
    statusOverlay.classList.remove('opacity-100', 'pointer-events-auto');
    statusOverlay.classList.add('opacity-0', 'pointer-events-none');
}

function showApiKeyModal() {
    apiKeyModal.classList.remove('hidden');
    apiKeyInput.value = apiKey;
}

/**
 * Inspector Panel Logic
 */
function openNodeInspector(nodeId) {
    selectedElement = { type: 'node', id: nodeId };
    
    const nodeData = currentGraphData.nodes.find(n => n.id === nodeId);
    if (!nodeData) return;
    
    // UI Updates
    nodeDetails.classList.remove('hidden');
    edgeDetails.classList.add('hidden');
    deleteLabel.textContent = "Node";
    
    document.getElementById('node-name').textContent = nodeData.label;
    document.getElementById('node-type').textContent = nodeData.group;
    
    // Calculate connections
    const connectedEdges = currentGraphData.edges.filter(e => e.from === nodeId || e.to === nodeId);
    const connHtml = connectedEdges.map(e => {
        const isSource = e.from === nodeId;
        const otherId = isSource ? e.to : e.from;
        const otherNode = currentGraphData.nodes.find(n => n.id === otherId);
        
        if (!otherNode) return '';
        
        return `<div class="truncate py-1 ${isSource ? 'text-blue-400' : 'text-purple-400'}">
            ${isSource ? 'Out' : 'In'}: ${e.label} <span class="text-gray-500">→</span> ${otherNode.label}
        </div>`;
    }).join('');
    
    document.getElementById('node-connections').innerHTML = connHtml || '<span class="text-gray-500 italic">No connections</span>';
    
    openPanel();
}

function openEdgeInspector(edgeId) {
    selectedElement = { type: 'edge', id: edgeId };
    
    const edgeData = currentGraphData.edges.find(e => e.id === edgeId);
    if (!edgeData) return;
    
    const sourceNode = currentGraphData.nodes.find(n => n.id === edgeData.from);
    const targetNode = currentGraphData.nodes.find(n => n.id === edgeData.to);
    
    if (!sourceNode || !targetNode) return;
    
    // UI Updates
    nodeDetails.classList.add('hidden');
    edgeDetails.classList.remove('hidden');
    deleteLabel.textContent = "Edge";
    
    document.getElementById('edge-source').textContent = sourceNode.label;
    document.getElementById('edge-target').textContent = targetNode.label;
    document.getElementById('edge-label').textContent = edgeData.label;
    
    const weightCalc = Math.min(100, Math.max(10, (edgeData.weight / 5) * 100)); // Cap for visual bar
    document.getElementById('edge-weight-bar').style.width = `${weightCalc}%`;
    document.getElementById('edge-weight').textContent = edgeData.weight;
    
    // Fetch trace logic
    document.getElementById('trace-loading').classList.remove('hidden');
    document.getElementById('trace-content').classList.add('hidden');
    document.getElementById('trace-error').classList.add('hidden');
    
    fetchEdgeTrace(edgeId);
    
    openPanel();
}

async function fetchEdgeTrace(edgeId) {
    try {
        const res = await fetch(`/namespaces/${currentNamespaceId}/relationships/${edgeId}/trace`, {
            headers: apiKey ? { 'X-API-Key': apiKey } : {}
        });
        
        if (!res.ok) throw new Error('Trace fetch failed');
        
        const trace = await res.json();
        const loadingDiv = document.getElementById('trace-loading');
        
        // If we switched elements while fetching, ignore
        if (selectedElement?.id !== edgeId) return;
        
        loadingDiv.classList.add('hidden');
        
        if (trace.matched_memory_id && trace.content) {
            document.getElementById('trace-content').classList.remove('hidden');
            document.getElementById('trace-memory').textContent = trace.content;
            document.getElementById('trace-score').textContent = `Match Score: ${trace.score}`;
            document.getElementById('trace-explanation').textContent = trace.explanation;
        } else {
            document.getElementById('trace-error').classList.remove('hidden');
        }
        
    } catch (err) {
        console.error("Trace error:", err);
        if (selectedElement?.id === edgeId) {
            document.getElementById('trace-loading').classList.add('hidden');
            document.getElementById('trace-error').classList.remove('hidden');
        }
    }
}

function openPanel() {
    inspectorPanel.classList.remove('translate-x-full');
}

function closeInspector() {
    selectedElement = null;
    inspectorPanel.classList.add('translate-x-full');
    if (network) network.unselectAll();
}

// Boot
document.addEventListener('DOMContentLoaded', init);
