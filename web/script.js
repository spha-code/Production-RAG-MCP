const API_BASE = window.location.origin + '/api';
let uploadQueue = [];
let isUploading = false;
let currentThreadId = 'default';
let responseTimes = [];
let chatLog = [];   // every message shown, for Export Chat

// ========== INITIALIZATION ==========
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    setupFileInputs();
    loadDocumentLibrary();
    setupEventListeners();
    checkBackendConnection();
});

function setupEventListeners() {
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');

    if (chatInput) {
        chatInput.focus();
        // Use 'keydown' for better reliability with modifier keys
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Stop newline from being added
                sendMessage();
            }
        });
    }

    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
}

// ========== BACKEND CONNECTION ==========
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const statusEl = document.getElementById('connectionStatus');
        if (response.ok) {
            statusEl.style.background = '#d4edda';
            statusEl.textContent = '● Connected';
            const data = await response.json();
            if (data.chroma_count !== undefined) {
                document.getElementById('totalChunks').textContent = data.chroma_count;
            }
        }
    } catch (error) {
        const statusEl = document.getElementById('connectionStatus');
        statusEl.style.background = '#fff3cd';
        statusEl.textContent = '● Disconnected';
    }
}

// ========== FILE UPLOAD (CONCURRENT) ==========
function handleFiles(files) {
    const fileArray = Array.from(files);
    if (!fileArray.length) return;

    uploadQueue = [...uploadQueue, ...fileArray];
    displayFileList();
    processUploadQueue();
}

async function processUploadQueue() {
    if (isUploading || !uploadQueue.length) return;
    isUploading = true;

    const activeBatch = [...uploadQueue];
    uploadQueue = []; // Clear for next potential drops

    // Process up to 3 files at a time
    const batchSize = 3;
    for (let i = 0; i < activeBatch.length; i += batchSize) {
        const currentBatch = activeBatch.slice(i, i + batchSize);
        await Promise.all(currentBatch.map((file, index) => uploadSingleFile(file, i + index)));
    }

    isUploading = false;
    loadDocumentLibrary();
}

async function uploadSingleFile(file, index) {
    const statusEl = document.getElementById(`status-${index}`);
    if (statusEl) {
        statusEl.className = 'file-status status-uploading';
        statusEl.textContent = 'Uploading...';
    }

    try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API_BASE}/upload/`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Upload failed');

        if (statusEl) {
            statusEl.className = 'file-status status-success';
            statusEl.textContent = 'Success';
        }
    } catch (err) {
        if (statusEl) {
            statusEl.className = 'file-status status-error';
            statusEl.textContent = 'Error';
        }
    }
}

// ========== CHAT LOGIC ==========
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendButton');
    const query = input.value.trim();

    // Prevent sending empty messages or double-sending
    if (!query || input.disabled) return;

    // UI Feedback: Disable UI
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    addChatMessage(query, 'user');
    showThinkingIndicator();

    const startTime = Date.now();

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, thread_id: currentThreadId })
        });

        if (res.status === 429) {
            addChatMessage("⚠️ API Limit reached. Please wait 60 seconds.", 'bot', 'system');
            return;
        }

        const data = await res.json();

        recordResponseTime(Date.now() - startTime);
        addChatMessage(
            data.answer || "No response received.",
            'bot',
            data.mode || 'general',
            Date.now() - startTime,
            data.sources || [],
            Boolean(data.cached)
        );

    } catch (err) {
        addChatMessage("❌ Backend connection error.", 'bot', 'system');
    } finally {
        // Re-enable UI
        hideThinkingIndicator();
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

// ========== UI RENDERING ==========
function addChatMessage(text, sender, mode = 'general', responseTime = null, sources = [], cached = false) {
    const container = document.getElementById('chatMessages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const cleanText = text.trim();
    chatLog.push({ time: new Date().toISOString(), sender, mode, text: cleanText, cached, sources });

    if (sender === 'bot') {
        updateChatModeIndicator(mode);
        const icon = document.createElement('span');
        icon.className = `mode-indicator mode-${mode === 'rag' ? 'doc' : 'gen'}`;
        icon.textContent = mode === 'rag' ? '📄' : '💬';
        contentDiv.appendChild(icon);
    }

    const textSpan = document.createElement('span');
    textSpan.className = 'unicode-safe';
    textSpan.textContent = cleanText; // Safe from XSS
    contentDiv.appendChild(textSpan);

    if (sender === 'bot' && (sources.length || cached)) {
        contentDiv.appendChild(buildSources(sources, cached));
    }

    if (responseTime) {
        const timeDiv = document.createElement('div');
        timeDiv.className = 'response-time';
        timeDiv.textContent = `${responseTime}ms`;
        contentDiv.appendChild(timeDiv);
    }

    msgDiv.appendChild(contentDiv);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function buildSources(sources, cached) {
    const wrap = document.createElement('div');
    wrap.className = 'message-sources';

    if (sources.length) {
        const label = document.createElement('span');
        label.className = 'sources-label';
        label.textContent = 'Sources:';
        wrap.appendChild(label);

        sources.forEach((source) => {
            const chip = document.createElement('span');
            chip.className = 'source-chip';
            const name = source.file || source.doc_id || 'document';
            const distance = typeof source.distance === 'number'
                ? ` · ${source.distance.toFixed(3)}`
                : '';
            chip.textContent = `${name}${distance}`;
            if (source.chunk) chip.title = source.chunk;
            wrap.appendChild(chip);
        });
    }

    if (cached) {
        const badge = document.createElement('span');
        badge.className = 'cache-badge';
        badge.textContent = '⚡ cached';
        wrap.appendChild(badge);
    }

    return wrap;
}

function updateChatModeIndicator(mode) {
    const el = document.getElementById('chatModeIndicator');
    if (!el) return;
    const isRag = mode === 'rag';
    el.className = `mode-indicator ${isRag ? 'mode-doc' : 'mode-gen'}`;
    el.textContent = isRag ? '📄 Document Mode' : '💬 General Mode';
}

function recordResponseTime(ms) {
    responseTimes.push(ms);
    if (responseTimes.length > 10) responseTimes.shift();
    const avg = Math.round(responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length);
    const label = document.getElementById('avgResponse');
    if (label) label.textContent = `${avg}ms`;
}

// ========== HELPER UI FUNCTIONS ==========
function displayFileList() {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    uploadQueue.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <div class="file-info">
                <div class="file-name">${file.name}</div>
            </div>
            <div class="file-status status-pending" id="status-${index}">Pending</div>
        `;
        fileList.appendChild(item);
    });
}

async function loadDocumentLibrary() {
    const listEl = document.getElementById('docList');
    const docCount = document.getElementById('docCount');
    try {
        const res = await fetch(`${API_BASE}/upload/documents`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const docs = data.documents || [];
        if (docCount) docCount.textContent = docs.length;
        renderDocumentList(docs, listEl);
    } catch (e) {
        console.error("Library load failed", e);
        if (listEl) {
            listEl.innerHTML = `
                <div class="doc-item">
                    <div class="doc-info">
                        <h4>Could not load documents</h4>
                        <p>Check that the backend is running, then refresh the page.</p>
                    </div>
                </div>`;
        }
    } finally {
        updateTotalChunks();
    }
}

function renderDocumentList(docs, container) {
    if (!container) return;
    container.innerHTML = '';

    if (!docs.length) {
        const empty = document.createElement('div');
        empty.className = 'doc-item';
        const info = document.createElement('div');
        info.className = 'doc-info';
        const title = document.createElement('h4');
        title.textContent = 'No documents yet';
        const hint = document.createElement('p');
        hint.textContent = 'Upload a file above to get started';
        info.append(title, hint);
        empty.appendChild(info);
        container.appendChild(empty);
        return;
    }

    for (const doc of docs) {
        const item = document.createElement('div');
        item.className = 'doc-item';

        const info = document.createElement('div');
        info.className = 'doc-info';
        const title = document.createElement('h4');
        title.textContent = doc.filename || doc.document_id;
        const meta = document.createElement('p');
        const chunks = doc.total_chunks ?? 0;
        const size = doc.file_size ? ` · ${formatBytes(doc.file_size)}` : '';
        meta.textContent = `${chunks} chunk${chunks === 1 ? '' : 's'}${size}`;
        info.append(title, meta);

        const actions = document.createElement('div');
        actions.className = 'doc-actions';
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-small';
        deleteBtn.textContent = 'Delete';
        deleteBtn.addEventListener('click', () => deleteDocument(doc.document_id));
        actions.appendChild(deleteBtn);

        item.append(info, actions);
        container.appendChild(item);
    }
}

async function deleteDocument(documentId) {
    if (!documentId || !confirm('Delete this document and its indexed chunks?')) return;
    try {
        const res = await fetch(`${API_BASE}/upload/documents/${encodeURIComponent(documentId)}`, {
            method: 'DELETE'
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        await loadDocumentLibrary();
    } catch (e) {
        console.error("Delete failed", e);
        alert('Could not delete the document. Check the backend logs.');
    }
}

async function updateTotalChunks() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) return;
        const data = await res.json();
        const el = document.getElementById('totalChunks');
        if (el && data.chroma_count !== undefined) el.textContent = data.chroma_count;
    } catch (e) { /* the connection badge already reports outages */ }
}

function formatBytes(bytes) {
    if (!bytes) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
    }
    return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function showThinkingIndicator() { document.getElementById('statusIndicator').style.display = 'flex'; }
function hideThinkingIndicator() { document.getElementById('statusIndicator').style.display = 'none'; }

// Placeholders for Drag & Drop setup (as per previous logic)
function setupDragAndDrop() {
    const zone = document.getElementById('uploadZone');
    if (!zone) return;
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    zone.addEventListener('click', () => document.getElementById('fileInput').click());
}

function setupFileInputs() {
    const input = document.getElementById('fileInput');
    if (input) input.addEventListener('change', (e) => handleFiles(e.target.files));
}
// ========== SYSTEM PANEL ==========
function showSystemOutput(text) {
    const out = document.getElementById('systemOutput');
    out.style.display = 'block';
    out.textContent = text;
}

// Health Check: ask the backend for its status and show the reply.
async function healthCheck() {
    showSystemOutput('Checking backend...');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        const uptime = Math.round(data.uptime || 0);
        showSystemOutput(
            `Status: ${data.status} (HTTP ${res.status})
` +
            `Stored chunks: ${data.chroma_count}
` +
            `Uptime: ${Math.floor(uptime / 3600)}h ${Math.floor(uptime / 60) % 60}m ${uptime % 60}s`
        );
        checkBackendConnection();
    } catch (err) {
        showSystemOutput(`Backend unreachable: ${err.message}`);
    }
}

// Clear Chat: remove all messages except the welcome message.
function clearChat() {
    const container = document.getElementById('chatMessages');
    while (container.children.length > 1) container.lastElementChild.remove();
    chatLog = [];
    currentThreadId = `thread-${Date.now()}`;   // start a fresh server-side history too
    updateChatModeIndicator('general');
    showSystemOutput('Chat cleared.');
}

// Export Chat: download the conversation as a JSON file.
function exportConversation() {
    if (!chatLog.length) {
        showSystemOutput('Nothing to export yet.');
        return;
    }
    const blob = new Blob([JSON.stringify(chatLog, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `chat-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
    showSystemOutput(`Exported ${chatLog.length} message(s).`);
}

// Toggle Debug: show or hide the full details of the last answer.
function toggleDebug() {
    const out = document.getElementById('systemOutput');
    if (out.style.display === 'block') {
        out.style.display = 'none';
        return;
    }
    const lastBot = [...chatLog].reverse().find(m => m.sender === 'bot');
    showSystemOutput(lastBot
        ? `Last answer (debug):
${JSON.stringify(lastBot, null, 2)}`
        : 'Debug: no answers yet.');
}
