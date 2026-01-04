const API_BASE = 'http://127.0.0.1:8000';
let uploadQueue = [];
let isUploading = false;
let currentThreadId = 'default';
let responseTimes = [];

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
        const elapsed = Date.now() - startTime;

        recordResponseTime(elapsed);
        addChatMessage(data.answer || "No response received.", 'bot', 'auto', elapsed);

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
function addChatMessage(text, sender, mode = 'gen', responseTime = null) {
    const container = document.getElementById('chatMessages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    let cleanText = text.trim();
    let detectedMode = mode;

    // Detect RAG vs General mode from prefix emojis
    if (cleanText.startsWith('📄')) {
        detectedMode = 'doc';
        cleanText = cleanText.replace(/^📄\s*/, '');
    } else if (cleanText.startsWith('💬')) {
        detectedMode = 'gen';
        cleanText = cleanText.replace(/^💬\s*/, '');
    }

    if (sender === 'bot') {
        updateChatModeIndicator(detectedMode);
        const icon = document.createElement('span');
        icon.className = `mode-indicator mode-${detectedMode}`;
        icon.textContent = detectedMode === 'doc' ? '📄' : '💬';
        contentDiv.appendChild(icon);
    }

    const textSpan = document.createElement('span');
    textSpan.className = 'unicode-safe';
    textSpan.textContent = cleanText; // Safe from XSS
    contentDiv.appendChild(textSpan);

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

function updateChatModeIndicator(mode) {
    const el = document.getElementById('chatModeIndicator');
    if (el) {
        el.className = `mode-indicator mode-${mode}`;
        el.textContent = mode === 'doc' ? '📄 Document Mode' : '💬 General Mode';
    }
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
    try {
        const res = await fetch(`${API_BASE}/upload/documents`);
        const data = await res.json();
        const docCount = document.getElementById('docCount');
        if (docCount) docCount.textContent = data.total_documents || 0;
    } catch (e) { console.error("Library load failed", e); }
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