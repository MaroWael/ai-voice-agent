// UI Elements
const wsStatus = document.getElementById('ws-status');
const micStatus = document.getElementById('mic-status');
const wsUrlInput = document.getElementById('ws-url');
const btnConnect = document.getElementById('btn-connect');
const btnDisconnect = document.getElementById('btn-disconnect');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnClear = document.getElementById('btn-clear');
const logArea = document.getElementById('log-area');

// State Variables
let ws = null;
let audioContext = null;
let stream = null;
let sourceNode = null;
let processorNode = null;
let isRecording = false;

// Event Listeners
btnConnect.addEventListener('click', connect);
btnDisconnect.addEventListener('click', disconnect);
btnStart.addEventListener('click', startRecording);
btnStop.addEventListener('click', stopRecording);
btnClear.addEventListener('click', clearLogs);

// Logging Helper
function log(message, type = 'system') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    if (type === 'json') {
        try {
            const parsed = JSON.parse(message);
            entry.textContent = JSON.stringify(parsed, null, 2);
        } catch (e) {
            entry.textContent = message;
        }
    } else {
        const time = new Date().toLocaleTimeString();
        entry.textContent = `[${time}] ${message}`;
    }
    
    logArea.appendChild(entry);
    logArea.scrollTop = logArea.scrollHeight;
}

function clearLogs() {
    logArea.innerHTML = '';
    log('Logs cleared.', 'system');
}

// WebSocket Connection Handlers
async function connect() {
    const url = wsUrlInput.value.trim();
    if (!url) {
        log('Error: WebSocket URL cannot be empty.', 'error');
        return;
    }

    log(`Initializing audio context to query local device sample rate...`);
    try {
        // Close previous context if exists
        if (audioContext) {
            await audioContext.close();
        }
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (err) {
        log(`Failed to create AudioContext: ${err.message}`, 'error');
        return;
    }

    const sampleRate = audioContext.sampleRate;
    // Append parameters: using native sample rate, 1 channel, and float32 format
    const fullUrl = `${url}?sample_rate=${sampleRate}&channels=1&format=float32`;
    
    log(`Connecting to ${fullUrl}...`);
    btnConnect.disabled = true;
    wsStatus.textContent = 'Connecting...';
    wsStatus.className = 'status-badge disconnected';

    try {
        ws = new WebSocket(fullUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            log('WebSocket connection established successfully.', 'success');
            wsStatus.textContent = 'Connected';
            wsStatus.className = 'status-badge connected';
            btnDisconnect.disabled = false;
            btnStart.disabled = false;
            wsUrlInput.disabled = true;
        };

        ws.onclose = (event) => {
            log(`WebSocket connection closed (code: ${event.code}, reason: ${event.reason || 'None'}).`, 'system');
            handleDisconnectCleanup();
        };

        ws.onerror = (err) => {
            log('WebSocket connection encountered an error.', 'error');
            console.error('WebSocket Error:', err);
        };

        ws.onmessage = (event) => {
            log('Received message from server:', 'system');
            log(event.data, 'json');
        };

    } catch (err) {
        log(`Connection failed: ${err.message}`, 'error');
        btnConnect.disabled = false;
        wsStatus.textContent = 'Disconnected';
        wsStatus.className = 'status-badge disconnected';
    }
}

async function disconnect() {
    log('Closing connection...');
    if (ws) {
        ws.close();
    }
}

function handleDisconnectCleanup() {
    ws = null;
    wsStatus.textContent = 'Disconnected';
    wsStatus.className = 'status-badge disconnected';
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    btnStart.disabled = true;
    btnStop.disabled = true;
    wsUrlInput.disabled = false;
    
    if (isRecording) {
        stopRecording();
    }
}

// Microphone & Recording Handlers
async function startRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('Error: Cannot start recording, WebSocket is not connected.', 'error');
        return;
    }

    log('Requesting microphone access...');
    try {
        stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
        }
    });
    } catch (err) {
        log(`Microphone permission denied or unavailable: ${err.message}`, 'error');
        return;
    }

    log('Microphone access granted. Starting audio processing...');
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnDisconnect.disabled = true; // Prevent disconnecting while recording is active
    micStatus.textContent = 'Listening';
    micStatus.className = 'status-badge active';
    isRecording = true;

    try {
        await audioContext.resume();
        sourceNode = audioContext.createMediaStreamSource(stream);
        
        // Setup ScriptProcessorNode with buffer size of 2048, 1 input channel, 1 output channel
        processorNode = audioContext.createScriptProcessor(2048, 1, 1);
        
        processorNode.onaudioprocess = (e) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const inputData = e.inputBuffer.getChannelData(0); // float32 array
                // Create a copy of the samples to send to WebSocket
                const samples = new Float32Array(inputData);
                ws.send(samples.buffer);
            }
        };

        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
        log('Recording started. Streaming audio frames to backend...', 'success');
        
    } catch (err) {
        log(`Failed to initialize audio processing pipeline: ${err.message}`, 'error');
        stopRecording();
    }
}

function stopRecording() {
    log('Stopping audio capture...');
    isRecording = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnDisconnect.disabled = false;
    micStatus.textContent = 'Inactive';
    micStatus.className = 'status-badge inactive';

    if (processorNode) {
        processorNode.disconnect();
        processorNode.onaudioprocess = null;
        processorNode = null;
    }

    if (sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    log('Recording stopped. WebSocket connection remains open.', 'system');
}
