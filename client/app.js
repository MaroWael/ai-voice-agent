/**
 * Voice AI — Banque Misr Voice Banking Assistant Client
 * Production Quality v6.0
 */

'use strict';

// ── DOM References ────────────────────────────────────────────────────────────

const wsStatus = document.getElementById('ws-status');
const micStatus = document.getElementById('mic-status');
const pipelineStatus = document.getElementById('pipeline-status');
const wsUrlInput = document.getElementById('ws-url');

const btnConnect = document.getElementById('btn-connect');
const btnDisconnect = document.getElementById('btn-disconnect');
const btnMic = document.getElementById('btn-mic');
const micLabel = document.getElementById('mic-label');
const btnClear = document.getElementById('btn-clear');
const btnClearHistory = document.getElementById('btn-clear-history');
const logArea = document.getElementById('log-area');

// Debug panel
const transcriptText = document.getElementById('transcript-text');
const transcriptLang = document.getElementById('transcript-lang');
const ragStatusBadge = document.getElementById('rag-status-badge');
const responseText = document.getElementById('response-text');
const audioStatus = document.getElementById('audio-status');
const audioPlayer = document.getElementById('audio-player');
const turnCounter = document.getElementById('turn-counter');
const latTurn = document.getElementById('lat-turn');
const latAudio = document.getElementById('lat-audio');
const latRtt = document.getElementById('lat-rtt');

// History & Layout
const historyArea = document.getElementById('history-area');
const convArea = document.getElementById('conv-area');
const callHero = document.getElementById('call-hero');
const micHint = document.getElementById('mic-hint');
const connDot = document.getElementById('conn-dot');
const connLabel = document.getElementById('conn-label');

// ── State ─────────────────────────────────────────────────────────────────────

let ws = null;
let audioContext = null;
let stream = null;
let sourceNode = null;
let processorNode = null;
let isRecording = false;

let pendingMeta = null;   // last received assistant_response JSON
let turnSegmentStart = null; // monotonic timestamp when segment was sent
let turnCount = 0;

// ── Event Bindings ────────────────────────────────────────────────────────────

if (btnConnect) btnConnect.addEventListener('click', connect);
if (btnDisconnect) btnDisconnect.addEventListener('click', disconnect);
if (btnMic) btnMic.addEventListener('click', toggleRecording);

if (btnClear) {
    btnClear.addEventListener('click', () => {
        logArea.innerHTML = '';
        log('Log cleared.', 'system');
    });
}

if (btnClearHistory) {
    btnClearHistory.addEventListener('click', () => {
        historyArea.innerHTML = `
            <div class="history-empty">
                <svg class="empty-icon" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M18 2L33 10.5V25.5L18 34L3 25.5V10.5L18 2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
                    <circle cx="18" cy="18" r="5" fill="currentColor" opacity="0.9"/>
                </svg>
                <div class="empty-title">Banque Misr Voice Assistant</div>
                <div class="empty-subtitle">Your conversation will appear here.<br>Ask me about accounts, certificates, transfers, loans, or banking services.</div>
            </div>
        `;
        turnCount = 0;
        if (turnCounter) turnCounter.textContent = '—';
        if (callHero) callHero.style.display = 'flex';
        if (convArea) convArea.style.display = 'none';
    });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function scrollToBottom() {
    if (!historyArea) return;
    requestAnimationFrame(() => {
        historyArea.scrollTo({
            top: historyArea.scrollHeight,
            behavior: 'smooth'
        });
    });
}

function ensureConversationView() {
    if (callHero) callHero.style.display = 'none';
    if (convArea) convArea.style.display = 'flex';
    const emptyState = historyArea.querySelector('.history-empty');
    if (emptyState) emptyState.remove();
}

function updateHint(message) {
    if (micHint) micHint.textContent = message;
}

// ── Logging ───────────────────────────────────────────────────────────────────

function log(message, type = 'system') {
    if (!logArea) return;
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const time = new Date().toLocaleTimeString();
    entry.textContent = `[${time}] ${message}`;
    logArea.appendChild(entry);
    logArea.scrollTop = logArea.scrollHeight;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

async function connect() {
    const url = wsUrlInput ? wsUrlInput.value.trim() : 'ws://127.0.0.1:8000/ws/audio';
    if (!url) { log('WebSocket URL is empty.', 'error'); return; }

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
    if (btnConnect) btnConnect.disabled = true;
    setWsStatus('connecting');

    try {
        ws = new WebSocket(fullUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            log('WebSocket connected.', 'success');
            setWsStatus('connected');
            if (btnDisconnect) btnDisconnect.disabled = false;
            if (btnMic) btnMic.disabled = false;
            if (wsUrlInput) wsUrlInput.disabled = true;
            updateHint('Tap microphone to speak with Banque Misr');
        };

        ws.onclose = (e) => {
            log(`WebSocket closed (code ${e.code}).`, 'system');
            handleCleanup();
        };

        ws.onerror = () => {
            log('WebSocket error. Is the backend server running?', 'error');
        };

        ws.onmessage = handleServerMessage;

    } catch (err) {
        log(`Connection failed: ${err.message}`, 'error');
        if (btnConnect) btnConnect.disabled = false;
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
    if (btnConnect) btnConnect.disabled = false;
    if (btnDisconnect) btnDisconnect.disabled = true;
    if (btnMic) btnMic.disabled = true;
    if (wsUrlInput) wsUrlInput.disabled = false;
    if (isRecording) stopRecording();
    setPipelineStatus('idle');
    updateHint('Connect to start your session');
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

        // ── 1. STT Result ──
        if (payload.type === 'stt_result') {
            const transcript = payload.transcription || 'Voice request';
            const lang = payload.language || '';

            log(`STT → "${transcript}"`, 'transcript');
            if (transcriptText) transcriptText.textContent = transcript;
            if (transcriptLang) transcriptLang.textContent = lang ? `Language: ${lang}` : '';

            ensureConversationView();

            // Render User bubble immediately with full STT transcript (no loading dots)
            const userRow = document.createElement('div');
            userRow.className = 'chat-row row-user';
            userRow.innerHTML = `
                <div class="chat-bubble user-bubble" dir="auto">${escapeHtml(transcript)}</div>
                <div class="avatar-chip user-chip">You</div>
            `;
            historyArea.appendChild(userRow);

            // Create Assistant typing indicator bubble placeholder (🤖 ● ● ●)
            const aiRow = document.createElement('div');
            aiRow.className = 'chat-row row-ai';
            aiRow.id = 'current-ai-placeholder';
            aiRow.innerHTML = `
                <div class="avatar-chip ai-chip" aria-hidden="true">
                    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
                        <path d="M8 1L14 4.5V11.5L8 15L2 11.5V4.5L8 1Z"/>
                        <circle cx="8" cy="8" r="2.5" fill="currentColor"/>
                    </svg>
                </div>
                <div class="chat-bubble ai-bubble" dir="auto">
                    <div class="typing-dots"><span></span><span></span><span></span></div>
                </div>
            `;
            historyArea.appendChild(aiRow);

            setPipelineStatus('processing');
            updateHint('🧠 Thinking...');
            scrollToBottom();
            return;
        }

        // ── 2. Assistant Response (LLM Generated) ──
        if (payload.type === 'assistant_response') {
            pendingMeta = payload;
            turnSegmentStart = Date.now();

            const transcript = payload.transcription || 'Voice request';
            const lang = payload.language || '';
            const reason = payload.response?.reason || '—';
            const message = payload.response?.message || 'I am ready to assist you.';

            log(`RAG → ${reason.toUpperCase()} | ${message.substring(0, 60)}…`, 'rag');

            // Update debug panel
            if (transcriptText) transcriptText.textContent = transcript;
            if (transcriptLang) transcriptLang.textContent = lang ? `Language: ${lang}` : '';
            renderRagStatus(reason);
            if (responseText) {
                responseText.textContent = message;
                responseText.setAttribute('dir', 'auto');
            }

            // Failsafe Mode: If audio is unavailable, display text immediately
            if (payload.has_audio === false) {
                requestAnimationFrame(() => {
                    ensureConversationView();
                    const activeAiPlaceholder = document.getElementById('current-ai-placeholder');
                    if (activeAiPlaceholder) {
                        const activeAiBubble = activeAiPlaceholder.querySelector('.ai-bubble');
                        if (activeAiBubble) {
                            activeAiBubble.innerHTML = escapeHtml(message);
                        }
                        activeAiPlaceholder.removeAttribute('id');
                    } else {
                        const userRow = document.createElement('div');
                        userRow.className = 'chat-row row-user';
                        userRow.innerHTML = `
                            <div class="chat-bubble user-bubble" dir="auto">${escapeHtml(transcript)}</div>
                            <div class="avatar-chip user-chip">You</div>
                        `;
                        historyArea.appendChild(userRow);

                        const aiRow = document.createElement('div');
                        aiRow.className = 'chat-row row-ai';
                        aiRow.innerHTML = `
                            <div class="avatar-chip ai-chip" aria-hidden="true">
                                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
                                    <path d="M8 1L14 4.5V11.5L8 15L2 11.5V4.5L8 1Z"/>
                                    <circle cx="8" cy="8" r="2.5" fill="currentColor"/>
                                </svg>
                            </div>
                            <div class="chat-bubble ai-bubble" dir="auto">${escapeHtml(message)}</div>
                        `;
                        historyArea.appendChild(aiRow);
                    }
                    setPipelineStatus('idle');
                    updateHint('✨ Ready for your next question (Audio unavailable)');
                    scrollToBottom();
                });
            }
            // If has_audio is true, text replacement is held for requestAnimationFrame when binary audio arrives!
            return;
        }

        // ── 3. TTS Started Event ──
        if (payload.type === 'tts_started') {
            log('TTS synthesis started…', 'system');
            return;
        }

        // ── 4. TTS Finished Event ──
        if (payload.type === 'tts_finished') {
            log('TTS synthesis finished.', 'system');
            return;
        }

        // ── 5. TTS Failed Event (Failsafe fallback) ──
        if (payload.type === 'tts_failed') {
            log(`TTS synthesis failed: ${payload.reason || 'Audio unavailable'}`, 'warn');
            requestAnimationFrame(() => {
                const activeAiPlaceholder = document.getElementById('current-ai-placeholder');
                if (activeAiPlaceholder && pendingMeta) {
                    const activeAiBubble = activeAiPlaceholder.querySelector('.ai-bubble');
                    if (activeAiBubble) {
                        const msgText = pendingMeta.response?.message || 'I am ready to assist you.';
                        activeAiBubble.innerHTML = escapeHtml(msgText);
                    }
                    activeAiPlaceholder.removeAttribute('id');
                }
                setPipelineStatus('idle');
                updateHint('✨ Ready for your next question (Audio unavailable)');
                scrollToBottom();
            });
            return;
        }

        return;
    }

    // ── Binary WAV audio (Atomic UI Update: Text Replacement + Audio Start) ──
    if (event.data instanceof ArrayBuffer) {
        const byteLen = event.data.byteLength;
        log(`Audio received: ${formatBytes(byteLen)}`, 'success');

        const rtt = turnSegmentStart ? `${(Date.now() - turnSegmentStart).toLocaleString()} ms` : '—';

        turnCount += 1;
        if (turnCounter) turnCounter.textContent = `Turn ${turnCount}`;
        if (latTurn) latTurn.textContent = `#${turnCount}`;
        if (latAudio) latAudio.textContent = formatBytes(byteLen);
        if (latRtt) latRtt.textContent = rtt;

        // ATOMIC UI UPDATE inside requestAnimationFrame:
        // Replace typing dots with text AND trigger audio playback in the EXACT SAME UI FRAME!
        requestAnimationFrame(() => {
            ensureConversationView();
            const activeAiPlaceholder = document.getElementById('current-ai-placeholder');
            if (activeAiPlaceholder && pendingMeta) {
                const activeAiBubble = activeAiPlaceholder.querySelector('.ai-bubble');
                if (activeAiBubble) {
                    const msgText = pendingMeta.response?.message || 'I am ready to assist you.';
                    activeAiBubble.innerHTML = escapeHtml(msgText);
                }
                activeAiPlaceholder.removeAttribute('id');
            }

            setPipelineStatus('playing');
            updateHint('🔊 Banque Misr AI is speaking...');
            scrollToBottom();

            playWavBuffer(event.data);
            pendingMeta = null;
        });
        return;
    }

    log('Unknown server message type.', 'error');
}

// ── Audio Playback ────────────────────────────────────────────────────────────

async function playWavBuffer(arrayBuffer) {
    try {
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);

        if (audioPlayer) {
            audioPlayer.src = url;
            audioPlayer.style.display = 'block';

            audioPlayer.onended = () => {
                URL.revokeObjectURL(url);
                log('Audio playback finished.', 'system');
                setPipelineStatus('idle');
                updateHint('✨ Ready for your next question');
            };

            try {
                await audioPlayer.play();
                log('Playing synthesized audio…', 'success');
            } catch (playErr) {
                log(`Autoplay blocked (${playErr.message})`, 'warn');
            }
        }
    } catch (err) {
        log(`Failed to play audio: ${err.message}`, 'error');
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

    log('Microphone active — streaming audio…', 'success');
    isRecording = true;
    setMicStatus('active');
    btnMic.classList.add('recording');
    if (micLabel) micLabel.textContent = 'Stop Recording';
    if (btnDisconnect) btnDisconnect.disabled = true;

    setPipelineStatus('listening');
    updateHint('🎤 Listening... Tap microphone when finished');

    await audioContext.resume();
    sourceNode = audioContext.createMediaStreamSource(stream);
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
    if (!isRecording) return;
    isRecording = false;
    setMicStatus('inactive');
    btnMic.classList.remove('recording');
    if (micLabel) micLabel.textContent = 'Start Recording';
    if (btnDisconnect) btnDisconnect.disabled = false;

    if (processorNode) {
        processorNode.disconnect();
        processorNode.onaudioprocess = null;
        processorNode = null;
    }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }

    log('Recording stopped — processing user turn…', 'system');
    setPipelineStatus('processing');
    updateHint('🧠 Thinking...');
}

// ── UI Status Updates ─────────────────────────────────────────────────────────

function setWsStatus(state) {
    const labels = { connecting: 'Connecting…', connected: 'Connected', disconnected: 'Disconnected' };
    if (wsStatus) {
        wsStatus.textContent = labels[state] || state;
        wsStatus.className = `status-badge ${state}`;
    }
    if (connDot) connDot.className = `conn-dot ${state === 'connected' ? 'connected' : state === 'connecting' ? 'connecting' : ''}`;
    if (connLabel) connLabel.textContent = state === 'connected' ? 'Connected to Banque Misr' : labels[state] || state;
}

function setMicStatus(state) {
    if (micStatus) {
        micStatus.textContent = state === 'active' ? 'Recording' : 'Inactive';
        micStatus.className = `status-badge ${state === 'active' ? 'active' : 'inactive'}`;
    }
}

function setPipelineStatus(state) {
    const map = {
        idle: ['Ready', 'inactive'],
        listening: ['Listening', 'active'],
        processing: ['Thinking', 'processing'],
        playing: ['Speaking', 'playing'],
    };
    const [label, cls] = map[state] || ['Ready', 'inactive'];
    if (pipelineStatus) {
        pipelineStatus.textContent = label;
        pipelineStatus.className = `status-badge ${cls}`;
    }
}

function renderRagStatus(reason) {
    if (!ragStatusBadge) return;
    const isSuccess = reason === 'success';
    ragStatusBadge.textContent = reason ? reason.replace(/_/g, ' ').toUpperCase() : '—';
    ragStatusBadge.className = `rag-status-value ${isSuccess ? 'rag-success' : 'rag-refusal'}`;
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
