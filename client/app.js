/**
 * Voice AI — RAG Customer Service Demo Client
 *
 * Flow:
 *   1. Connect WebSocket to /ws/audio
 *   2. Capture microphone audio (float32, 1ch, native sample rate)
 *   3. Stream raw PCM frames to backend
 *   4. Backend: VAD → STT → RagService → Groq → TTS
 *   5. Receive JSON metadata then binary WAV audio
 *   6. Render transcript, RAG status, response text into debug panel
 *   7. Play WAV audio through Web Audio API + <audio> element
 */

'use strict';

// ── DOM References ────────────────────────────────────────────────────────────

const wsStatus       = document.getElementById('ws-status');
const micStatus      = document.getElementById('mic-status');
const pipelineStatus = document.getElementById('pipeline-status');
const wsUrlInput     = document.getElementById('ws-url');

const btnConnect    = document.getElementById('btn-connect');
const btnDisconnect = document.getElementById('btn-disconnect');
const btnMic        = document.getElementById('btn-mic');
const micLabel      = document.getElementById('mic-label');
const btnClear      = document.getElementById('btn-clear');
const btnClearHistory = document.getElementById('btn-clear-history');
const logArea       = document.getElementById('log-area');

// Debug panel
const transcriptText  = document.getElementById('transcript-text');
const transcriptLang  = document.getElementById('transcript-lang');
const ragStatusBadge  = document.getElementById('rag-status-badge');
const responseText    = document.getElementById('response-text');
const audioStatus     = document.getElementById('audio-status');
const audioPlayer     = document.getElementById('audio-player');
const turnCounter     = document.getElementById('turn-counter');
const latTurn         = document.getElementById('lat-turn');
const latAudio        = document.getElementById('lat-audio');
const latRtt          = document.getElementById('lat-rtt');

// History
const historyArea     = document.getElementById('history-area');

// ── State ─────────────────────────────────────────────────────────────────────

let ws            = null;
let audioContext  = null;
let stream        = null;
let sourceNode    = null;
let processorNode = null;
let isRecording   = false;

// Pending state between JSON message and subsequent binary audio
let pendingMeta   = null;   // last received assistant_response JSON
let turnSegmentStart = null; // monotonic timestamp when segment was sent

let turnCount     = 0;

// ── Event Bindings ────────────────────────────────────────────────────────────

btnConnect.addEventListener('click', connect);
btnDisconnect.addEventListener('click', disconnect);
btnMic.addEventListener('click', toggleRecording);
btnClear.addEventListener('click', () => {
    logArea.innerHTML = '';
    log('Log cleared.', 'system');
});
btnClearHistory.addEventListener('click', () => {
    historyArea.innerHTML = '<div class="history-empty">No turns yet.</div>';
    turnCount = 0;
    turnCounter.textContent = '—';
});

// ── Logging ───────────────────────────────────────────────────────────────────

function log(message, type = 'system') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const time = new Date().toLocaleTimeString();
    entry.textContent = `[${time}] ${message}`;
    logArea.appendChild(entry);
    logArea.scrollTop = logArea.scrollHeight;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

async function connect() {
    const url = wsUrlInput.value.trim();
    if (!url) { log('WebSocket URL is empty.', 'error'); return; }

    // AudioContext must be created (or resumed) from a user gesture
    try {
        if (audioContext) await audioContext.close();
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (err) {
        log(`AudioContext creation failed: ${err.message}`, 'error');
        return;
    }

    const sampleRate = audioContext.sampleRate;
    const fullUrl = `${url}?sample_rate=${sampleRate}&channels=1&format=float32`;

    log(`Connecting → ${fullUrl}`);
    btnConnect.disabled = true;
    setWsStatus('connecting');

    try {
        ws = new WebSocket(fullUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            log('WebSocket connected.', 'success');
            setWsStatus('connected');
            btnDisconnect.disabled = false;
            btnMic.disabled = false;
            wsUrlInput.disabled = true;
        };

        ws.onclose = (e) => {
            log(`WebSocket closed (code ${e.code}).`, 'system');
            handleCleanup();
        };

        ws.onerror = () => {
            log('WebSocket error. Is the server running?', 'error');
        };

        ws.onmessage = handleServerMessage;

    } catch (err) {
        log(`Connection failed: ${err.message}`, 'error');
        btnConnect.disabled = false;
        setWsStatus('disconnected');
    }
}

async function disconnect() {
    log('Disconnecting…');
    if (isRecording) await stopRecording();
    if (ws) ws.close();
}

function handleCleanup() {
    ws = null;
    setWsStatus('disconnected');
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    btnMic.disabled = true;
    wsUrlInput.disabled = false;
    if (isRecording) stopRecording();
    setPipelineStatus('idle');
}

// ── Server Message Handler ─────────────────────────────────────────────────────

async function handleServerMessage(event) {
    // ── JSON metadata ────────────────────────────────────────────────────────
    if (typeof event.data === 'string') {
        let payload;
        try { payload = JSON.parse(event.data); } catch {
            log(`Unparseable server message: ${event.data}`, 'error');
            return;
        }

        if (payload.type === 'assistant_response') {
            pendingMeta = payload;
            turnSegmentStart = Date.now();

            const transcript = payload.transcription || '—';
            const lang       = payload.language || '';
            const action     = payload.response?.action || '—';
            const reason     = payload.response?.reason || '—';
            const message    = payload.response?.message || '—';

            log(`STT → "${transcript}"`, 'transcript');
            log(`RAG → ${reason.toUpperCase()} | ${message.substring(0, 60)}${message.length > 60 ? '…' : ''}`, 'rag');

            // Update debug panel
            transcriptText.textContent = transcript;
            transcriptLang.textContent = lang ? `Language: ${lang}` : '';
            renderRagStatus(reason);
            responseText.textContent = message;
            responseText.setAttribute('dir', 'auto');
            audioStatus.textContent = 'Waiting for audio…';
            audioPlayer.style.display = 'none';
            setPipelineStatus('processing');
        }
        return;
    }

    // ── Binary WAV audio ─────────────────────────────────────────────────────
    if (event.data instanceof ArrayBuffer) {
        const byteLen = event.data.byteLength;
        log(`Audio received: ${formatBytes(byteLen)}`, 'success');

        const rtt = turnSegmentStart ? `${(Date.now() - turnSegmentStart).toLocaleString()} ms` : '—';

        turnCount += 1;
        turnCounter.textContent = `Turn ${turnCount}`;
        latTurn.textContent  = `#${turnCount}`;
        latAudio.textContent = formatBytes(byteLen);
        latRtt.textContent   = rtt;

        audioStatus.textContent = `Playing audio… (${formatBytes(byteLen)})`;
        setPipelineStatus('playing');

        await playWavBuffer(event.data);

        // Add to history
        if (pendingMeta) {
            addHistoryTurn(pendingMeta, byteLen, rtt);
            pendingMeta = null;
        }

        setPipelineStatus('idle');
        audioStatus.textContent = `Last audio: ${formatBytes(byteLen)} — played successfully ✓`;
        return;
    }

    log('Unknown server message type.', 'error');
}

// ── Audio Playback ────────────────────────────────────────────────────────────

async function playWavBuffer(arrayBuffer) {
    // Use <audio> element with Blob URL for reliable WAV playback
    try {
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        const url  = URL.createObjectURL(blob);

        audioPlayer.src = url;
        audioPlayer.style.display = 'block';

        // Auto-play and log when done
        audioPlayer.onended = () => {
            URL.revokeObjectURL(url);
            log('Audio playback finished.', 'system');
        };
        audioPlayer.onerror = (e) => {
            log(`Audio playback error: ${audioPlayer.error?.message || 'unknown'}`, 'error');
        };

        try {
            await audioPlayer.play();
            log('Playing synthesized audio…', 'success');
        } catch (playErr) {
            // Browser autoplay policy — audio element is still shown, user can press play
            log(`Autoplay blocked — use the player below to listen. (${playErr.message})`, 'warn');
        }
    } catch (err) {
        log(`Failed to prepare audio: ${err.message}`, 'error');
    }
}

// ── Recording ─────────────────────────────────────────────────────────────────

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('Cannot record: WebSocket is not connected.', 'error');
        return;
    }

    log('Requesting microphone access…');
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: audioContext.sampleRate,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            }
        });
    } catch (err) {
        log(`Microphone denied: ${err.message}`, 'error');
        return;
    }

    log('Microphone active — streaming to backend…', 'success');
    isRecording = true;
    setMicStatus('active');
    btnMic.classList.add('recording');
    micLabel.textContent = 'Stop Recording';
    btnDisconnect.disabled = true;
    setPipelineStatus('listening');

    await audioContext.resume();
    sourceNode    = audioContext.createMediaStreamSource(stream);
    processorNode = audioContext.createScriptProcessor(2048, 1, 1);

    processorNode.onaudioprocess = (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const samples = new Float32Array(e.inputBuffer.getChannelData(0));
            ws.send(samples.buffer);
        }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);
}

function stopRecording() {
    isRecording = false;
    setMicStatus('inactive');
    btnMic.classList.remove('recording');
    micLabel.textContent = 'Start Recording';
    btnDisconnect.disabled = false;

    if (processorNode) {
        processorNode.disconnect();
        processorNode.onaudioprocess = null;
        processorNode = null;
    }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (stream)     { stream.getTracks().forEach(t => t.stop()); stream = null; }

    log('Recording stopped.', 'system');
    setPipelineStatus('idle');
}

// ── History ───────────────────────────────────────────────────────────────────

function addHistoryTurn(meta, audioBytes, rtt) {
    const empty = historyArea.querySelector('.history-empty');
    if (empty) empty.remove();

    const reason  = meta.response?.reason  || '—';
    const message = meta.response?.message || '—';
    const transcript = meta.transcription || '—';
    const lang    = meta.language || '';

    const item = document.createElement('div');
    item.className = 'history-turn';
    item.innerHTML = `
        <div class="history-turn-header">
            <span class="history-turn-num">Turn ${turnCount}</span>
            <span class="history-rtt">${rtt}</span>
            <span class="history-rag-badge ${ragBadgeClass(reason)}">${reason.toUpperCase()}</span>
        </div>
        <div class="history-q" dir="auto">${escapeHtml(transcript)}<span class="history-lang">${lang ? ` [${lang}]` : ''}</span></div>
        <div class="history-a" dir="auto">${escapeHtml(message)}</div>
    `;
    historyArea.prepend(item);
}

// ── UI Helpers ────────────────────────────────────────────────────────────────

function setWsStatus(state) {
    const labels = { connecting: 'Connecting…', connected: 'Connected', disconnected: 'Disconnected' };
    wsStatus.textContent  = labels[state] || state;
    wsStatus.className    = `status-badge ${state}`;
}

function setMicStatus(state) {
    micStatus.textContent = state === 'active' ? 'Recording' : 'Inactive';
    micStatus.className   = `status-badge ${state === 'active' ? 'active' : 'inactive'}`;
}

function setPipelineStatus(state) {
    const map = {
        idle:       ['Idle',       'inactive'],
        listening:  ['Listening',  'active'],
        processing: ['Processing', 'processing'],
        playing:    ['Playing',    'playing'],
    };
    const [label, cls] = map[state] || ['—', 'inactive'];
    pipelineStatus.textContent = label;
    pipelineStatus.className   = `status-badge ${cls}`;
}

function renderRagStatus(reason) {
    const isSuccess = reason === 'success';
    ragStatusBadge.textContent = reason ? reason.replace(/_/g, ' ').toUpperCase() : '—';
    ragStatusBadge.className   = `rag-status-value ${isSuccess ? 'rag-success' : 'rag-refusal'}`;
}

function ragBadgeClass(reason) {
    return reason === 'success' ? 'badge-success' : 'badge-refusal';
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1024).toFixed(1)} KB`;
}

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
