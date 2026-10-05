import { useState, useRef, useEffect, useCallback } from 'react';
import { API_BASE, WS_URL } from '../config';

const SESSION_ID = 'session_' + Math.random().toString(36).slice(2, 9);
const USER_ID    = 'user_'    + Math.random().toString(36).slice(2, 9);

const THOUGHTS = [
  '⚙️  Kernel initialising…',
  '🔬  Routing to domain agent…',
  '🧬  Analysing molecular structures…',
  '🤖  Synthesising response…',
  '📡  Streaming results…',
];

let currentAudio = null;

// ── VAD CDN config (pinned versions) ─────────────────────────────────────────
const VAD_VERSION    = '0.0.22';
const ONNX_VERSION   = '1.14.0';
const VAD_BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${VAD_VERSION}/dist/bundle.min.js`;
// @ricky0123/vad-web expects onnxruntime-web ESM shim loaded first
const ONNX_URL       = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_VERSION}/dist/ort.wasm.min.js`;

// ── VAD asset sources: self-hosted (public/vad, copied by scripts/copy-vad-assets.mjs)
//    first, jsDelivr CDN as fallback. Tried in order until one works.
const VAD_MODEL = 'v5';   // 'legacy' uses 1536-sample frames, which would make VAD_CONFIG's frame counts 3x too long
const LOCAL_VAD_BASE = `${import.meta.env.BASE_URL}vad/`;
const VAD_SOURCES = [
  {
    name:      'local',
    ortUrl:    `${LOCAL_VAD_BASE}ort.wasm.min.js`,
    bundleUrl: `${LOCAL_VAD_BASE}bundle.min.js`,
    assetBase: LOCAL_VAD_BASE,
    wasmBase:  LOCAL_VAD_BASE,
  },
  {
    name:      'cdn',
    ortUrl:    ONNX_URL,
    bundleUrl: VAD_BUNDLE_URL,
    assetBase: `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${VAD_VERSION}/dist/`,
    wasmBase:  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_VERSION}/dist/`,
  },
];
const VAD_SCRIPT_TIMEOUT_MS = 10000;
const VAD_INIT_TIMEOUT_MS   = 30000;

function withTimeout(promise, ms, label) {
  let t;
  const timeout = new Promise((_, reject) => {
    t = setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(t));
}

// Loads a <script> and verifies it really defined `globalName` (a dev server
// can answer a missing file with index.html, which "loads" but defines nothing).
function loadVadScript(url, globalName) {
  if (window[globalName]) return Promise.resolve();
  return withTimeout(new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url;
    s.onload  = () => (window[globalName] ? resolve() : reject(new Error(`${url} loaded but window.${globalName} is undefined`)));
    s.onerror = () => { s.remove(); reject(new Error(`failed to fetch ${url}`)); };
    document.head.appendChild(s);
  }), VAD_SCRIPT_TIMEOUT_MS, `script ${url}`);
}

// Frame size for Silero VAD = 512 samples at 16 kHz → 32 ms per frame
// redemptionFrames: 700 ms ÷ 32 ms ≈ 22
// preSpeechPadFrames: 300 ms ÷ 32 ms ≈ 9
// minSpeechFrames: 250 ms ÷ 32 ms ≈ 8
const VAD_CONFIG = {
  positiveSpeechThreshold:  0.6,
  negativeSpeechThreshold:  0.35,
  redemptionFrames:         22,
  preSpeechPadFrames:       9,
  minSpeechFrames:          8,
};

// ── Fallback energy VAD constants (used when Silero fails to load) ────────────
const FALLBACK_SILENCE_MS = 1200;

// ── MIME type / format helpers ────────────────────────────────────────────────
const PREFERRED_MIME_TYPES = [
  { mime: 'audio/webm;codecs=opus', ext: 'webm' },
  { mime: 'audio/webm',             ext: 'webm' },
  { mime: 'audio/ogg;codecs=opus',  ext: 'ogg'  },
  { mime: 'audio/ogg',              ext: 'ogg'  },
  { mime: 'audio/mp4',              ext: 'mp4'  },
];

function pickRecorderFormat() {
  for (const { mime, ext } of PREFERRED_MIME_TYPES) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(mime)) {
      return { mime, ext };
    }
  }
  return { mime: '', ext: 'webm' };
}

// ── Energy-based VAD helpers (fallback only) ─────────────────────────────────
function computeAudioMetrics(freqData) {
  let totalSum = 0;
  let speechSum = 0;
  const speechStart = 2;
  const speechEnd = Math.min(20, freqData.length);
  for (let i = 0; i < freqData.length; i++) {
    const val = freqData[i];
    totalSum += val * val;
    if (i >= speechStart && i < speechEnd) speechSum += val * val;
  }
  const totalRms  = Math.sqrt(totalSum / freqData.length);
  const cnt       = speechEnd - speechStart;
  const speechRms = cnt > 0 ? Math.sqrt(speechSum / cnt) : totalRms;
  return { totalRms, speechRms };
}

// ── TTS helpers ───────────────────────────────────────────────────────────────
// playGroqAudio: call ONLY for genuine spoken text responses (e.g. text-chat TTS).
// NEVER call for WebSocket "status" or "thought" control messages — those are
// UI-only labels that must not be synthesised.  Doing so would play English TTS
// through the speaker, the mic would pick it up, Silero VAD would fire onSpeechStart,
// and the barge-in logic would send {type:"interrupt"}, killing the real answer.
async function playGroqAudio(text, soundEnabledRef) {
  if (!soundEnabledRef.current || !text || !text.trim()) return;
  stopTTS();
  try {
    const res = await fetch(`${API_BASE}/audio/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.slice(0, 1000), voice: 'auto' }),
    });
    if (res.ok) {
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      currentAudio = new Audio(url);
      currentAudio.play().catch((e) => {
        console.warn('HTTP TTS autoplay blocked, falling back to SpeechSynthesis:', e);
        if (window.speechSynthesis) {
          const utt = new SpeechSynthesisUtterance(text.slice(0, 800));
          window.speechSynthesis.speak(utt);
        }
      });
      return;
    }
  } catch (err) {
    console.warn('Backend TTS request error, falling back to Web Speech API:', err);
  }
  if (window.speechSynthesis) {
    const utt = new SpeechSynthesisUtterance(text.slice(0, 800));
    utt.rate = 1.0; utt.pitch = 1.0;
    window.speechSynthesis.speak(utt);
  }
}

function stopTTS() {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
}

// ── Float32Array → WAV bytes (16 kHz, mono, 16-bit PCM) ─────────────────────
// Used when the vad-web utils.encodeWAV is unavailable.
function float32ToWav(samples, sampleRate = 16000) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = numChannels * (bitsPerSample / 8);
  const byteRate   = sampleRate * blockAlign;
  const dataSize   = samples.length * blockAlign;
  const buffer     = new ArrayBuffer(44 + dataSize);
  const view       = new DataView(buffer);
  const writeStr   = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  writeStr(0,  'RIFF');
  view.setUint32(4,  36 + dataSize,        true);
  writeStr(8,  'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16,                   true);
  view.setUint16(20, 1,                    true); // PCM
  view.setUint16(22, numChannels,          true);
  view.setUint32(24, sampleRate,           true);
  view.setUint32(28, byteRate,             true);
  view.setUint16(32, blockAlign,           true);
  view.setUint16(34, bitsPerSample,        true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize,             true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return buffer;
}

// ── Message component ─────────────────────────────────────────────────────────
function Message({ role, text, variant }) {
  if (role === 'sys') {
    return (
      <div className="message-row sys">
        <div className="avatar sys">⊙</div>
        <div className={`bubble sys ${variant || ''}`}>{text}</div>
      </div>
    );
  }
  return (
    <div className={`message-row ${role}`}>
      <div className={`avatar ${role}`}>{role === 'ai' ? 'OS' : 'U'}</div>
      <div className={`bubble ${role}`}>{text}</div>
    </div>
  );
}

// ── Voice Overlay ─────────────────────────────────────────────────────────────
function VoiceOverlay({ active, speaking, processing, transcript, status, onStop, waveHeights }) {
  return (
    <div className={`voice-overlay ${active ? 'active' : ''}`}>
      <div className="voice-orb-wrap">
        <div className={`orb-ring ${speaking ? 'speaking' : ''}`} />
        <div className={`orb-ring ${speaking ? 'speaking' : ''}`} />
        <div className={`orb-ring ${speaking ? 'speaking' : ''}`} />
        <div className={`orb-core ${speaking ? 'speaking' : processing ? 'processing' : ''}`}>
          {processing ? '⚙️' : speaking ? '🎤' : '🤖'}
        </div>
      </div>

      <div className="live-waveform">
        {waveHeights.map((h, i) => <span key={i} style={{ height: h + 'px' }} />)}
      </div>

      <div className={`live-transcript ${!transcript ? 'dim' : ''}`}>
        {transcript || 'Speak now…'}
      </div>
      <div className="live-status">{status}</div>
      <button className="overlay-stop-btn" onClick={onStop}>■ Stop Voice</button>
    </div>
  );
}

// ── Main Chat Page ────────────────────────────────────────────────────────────
export default function ChatPage() {
  const [messages, setMessages] = useState([
    { id: 0, role: 'sys', text: '⚡ AI-lixir Scientific OS online — Ask about drug discovery, ADMET, molecular analysis, or biomedical pathways.', variant: '' }
  ]);
  const [inputText,    setInputText]    = useState('');
  const [sending,      setSending]      = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const soundEnabledRef = useRef(true);

  const setSoundEnabledSync = (val) => {
    soundEnabledRef.current = val;
    setSoundEnabled(val);
  };

  // WebSocket state
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef              = useRef(null);
  const reconnectTimerRef  = useRef(null);

  // Voice overlay state
  const [voiceActive,      setVoiceActive]      = useState(false);
  const voiceActiveRef                          = useRef(false);
  const [voiceSpeaking,    setVoiceSpeaking]    = useState(false);
  const [voiceProcessing,  setVoiceProcessing]  = useState(false);
  const [voiceTranscript,  setVoiceTranscript]  = useState('');
  const [voiceStatus,      setVoiceStatus]      = useState('');
  const [waveHeights,      setWaveHeights]      = useState(Array(12).fill(4));

  // ── Silero VAD refs ───────────────────────────────────────────────────────
  const vadRef             = useRef(null);   // MicVAD instance (singleton)
  const vadReadyRef        = useRef(false);  // vad-web script loaded & VAD created
  const vadFailedRef       = useRef(false);  // vad-web failed to load → use fallback
  const vadInitPromiseRef  = useRef(null);   // in-flight Silero init (prevents double init)
  const vadSourceRef       = useRef(null);   // 'local' | 'cdn'
  const vadFailReasonRef   = useRef('');
  const vadModeRef         = useRef(null);   // 'silero' | 'energy' (last reported)
  const bargeInTimerRef    = useRef(null);   // timer for 250ms barge-in confirmation
  const bargeInActiveRef   = useRef(false);  // currently timing a barge-in

  // ── Fallback (energy) VAD refs (used only when Silero unavailable) ─────────
  const audioContextRef    = useRef(null);
  const analyserRef        = useRef(null);
  const waveRafRef         = useRef(null);
  const micStreamRef       = useRef(null);   // kept alive across turns (Silero also uses this)
  const mediaRecorderRef   = useRef(null);   // fallback only
  const speechDetectedRef  = useRef(false);
  const speechStartRef     = useRef(null);
  const silenceStartRef    = useRef(null);
  const isSpeakingRef      = useRef(false);
  const recordingActiveRef = useRef(false);

  // AI streaming refs
  const aiStreamingRef        = useRef(false);
  const audioQueueRef         = useRef([]);
  const isPlayingRef          = useRef(false);
  const aiDoneRef             = useRef(false);
  // Echo-guard: ignore VAD barge-in triggers for 1 s after AI audio starts playing.
  // Prevents the mic from picking up the speaker output and self-interrupting.
  const aiAudioStartTimeRef   = useRef(0);
  // Time the last queued AI audio chunk finished playing (tail guard starts here).
  const aiAudioEndTimeRef     = useRef(0);
  // True when the current VAD speech segment began while the AI was busy
  // (speaking, streaming, or just finished) → treat it as speaker echo, not the user.
  const echoSpeechRef         = useRef(false);
  const ECHO_GUARD_MS         = 1500;
  // Voice barge-in is unreliable on speakers (the mic hears the AI). Keep off unless
  // the user is on headphones.
  const ALLOW_VOICE_BARGE_IN  = false;

  const chatEndRef = useRef(null);
  const nextId     = useRef(1);

  // Keep voiceActiveRef in sync
  const setVoiceActiveSync = (val) => {
    voiceActiveRef.current = val;
    setVoiceActive(val);
  };

  const addMsg = (role, text, variant = '') => {
    const id = nextId.current++;
    setMessages(prev => [...prev, { id, role, text, variant }]);
    return id;
  };
  const updateMsg = (id, text) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, text } : m));
  };

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const startVoiceListeningRef = useRef(null);

  // ── Progressive audio queue playback ────────────────────────────────────────
  const playNextInQueue = useCallback(() => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      aiAudioEndTimeRef.current = Date.now();
      if (aiDoneRef.current) {
        aiDoneRef.current = false;
        setVoiceStatus('Ready');
        if (voiceActiveRef.current && startVoiceListeningRef.current) {
          startVoiceListeningRef.current();
        }
      }
      return;
    }
    isPlayingRef.current = true;
    setVoiceStatus('Speaking…');
    const blob = audioQueueRef.current.shift();
    const url  = URL.createObjectURL(blob);
    // Record when AI audio begins so the echo-guard can suppress VAD for 1 s
    aiAudioStartTimeRef.current = Date.now();
    stopTTS();
    currentAudio = new Audio(url);
    currentAudio.onended = () => { URL.revokeObjectURL(url); playNextInQueue(); };
    currentAudio.onerror = () => { URL.revokeObjectURL(url); playNextInQueue(); };
    currentAudio.play().catch(() => { URL.revokeObjectURL(url); playNextInQueue(); });
  }, []);

  // ── Send audio to server (called by both Silero and fallback paths) ─────────
  const sendAudioToServer = useCallback((wavArrayBuffer, format = 'wav') => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const bytes  = new Uint8Array(wavArrayBuffer);
    const binary = Array.from(bytes).map(b => String.fromCharCode(b)).join('');
    const b64    = btoa(binary);
    wsRef.current.send(JSON.stringify({ type: 'audio_chunk', data: b64, format }));
    wsRef.current.send(JSON.stringify({ type: 'audio_end',   format }));
    setVoiceSpeaking(false);
    setVoiceProcessing(true);
    setVoiceStatus('Transcribing speech…');
    setWaveHeights(Array(12).fill(4));
  }, []);

  // ── Waveform animation (shared, uses analyser when available) ───────────────
  const animateWave = useCallback(() => {
    if (!analyserRef.current || !voiceActiveRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const step    = Math.floor(data.length / 12);
    const heights = Array.from({ length: 12 }, (_, i) => {
      const v = data[i * step] || 0;
      return Math.max(4, (v / 255) * 36);
    });
    setWaveHeights(heights);

    if (voiceActiveRef.current) {
      waveRafRef.current = requestAnimationFrame(animateWave);
    }
  }, []);

  // ── Fallback energy VAD loop (only when vadFailedRef.current === true) ──────
  const fallbackVADLoop = useCallback(() => {
    if (!analyserRef.current || !voiceActiveRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const step    = Math.floor(data.length / 12);
    const heights = Array.from({ length: 12 }, (_, i) => {
      const v = data[i * step] || 0;
      return Math.max(4, (v / 255) * 36);
    });
    setWaveHeights(heights);

    const { speechRms } = computeAudioMetrics(data);

    // Mode A: recording turn
    if (recordingActiveRef.current) {
      const SPEECH_THRESHOLD = 28.0;
      const MIN_SPEECH_MS    = 500;

      if (speechRms > SPEECH_THRESHOLD) {
        if (speechStartRef.current === null) speechStartRef.current = Date.now();
        if (Date.now() - speechStartRef.current >= MIN_SPEECH_MS) {
          speechDetectedRef.current = true;
          if (!isSpeakingRef.current) {
            isSpeakingRef.current = true;
            setVoiceSpeaking(true);
            setVoiceStatus('Listening (speaking)…');
          }
        }
        silenceStartRef.current = null;
      } else {
        speechStartRef.current = null;
        if (isSpeakingRef.current) {
          isSpeakingRef.current = false;
          setVoiceSpeaking(false);
          setVoiceStatus('Listening…');
        }
        if (speechDetectedRef.current) {
          if (silenceStartRef.current === null) {
            silenceStartRef.current = Date.now();
          } else if (Date.now() - silenceStartRef.current >= FALLBACK_SILENCE_MS) {
            console.log('[VAD-fallback] Silence after speech — finalising turn');
            const mr = mediaRecorderRef.current;
            if (mr && mr.state !== 'inactive') {
              recordingActiveRef.current = false;
              const ext = mr._recExt || 'webm';
              mr.onstop = () => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ type: 'audio_end', format: ext }));
                }
              };
              try { mr.stop(); } catch {}
              setVoiceSpeaking(false);
              setVoiceProcessing(true);
              setVoiceStatus('Transcribing speech…');
              setWaveHeights(Array(12).fill(4));
            }
          }
        }
      }
    }
    // Mode B: barge-in while AI speaking
    else {
      const isAIActive = isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current;
      if (isAIActive) {
        const BARGE_IN_THRESHOLD = 35.0;
        const MIN_BARGE_IN_MS    = 600;
        if (speechRms > BARGE_IN_THRESHOLD) {
          if (bargeInTimerRef.current === null) {
            bargeInTimerRef.current = Date.now();
          } else if (Date.now() - bargeInTimerRef.current >= MIN_BARGE_IN_MS) {
            console.log('[VAD-fallback] Barge-in triggered');
            bargeInTimerRef.current = null;
            stopTTS();
            audioQueueRef.current = [];
            isPlayingRef.current  = false;
            aiStreamingRef.current = false;   // we are interrupting; let the new recording stream
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
            }
            if (startVoiceListeningRef.current) startVoiceListeningRef.current();
          }
        } else {
          bargeInTimerRef.current = null;
        }
      }
    }

    if (voiceActiveRef.current) {
      waveRafRef.current = requestAnimationFrame(fallbackVADLoop);
    }
  }, []);

  // ── Report which VAD is active (browser console + server log) ───────────────
  const reportVadMode = useCallback((mode, detail = {}) => {
    if (vadModeRef.current === mode) return;
    vadModeRef.current = mode;
    if (mode === 'silero') {
      console.log(`%c[VAD] ACTIVE: Silero VAD (source=${detail.source}, model=${VAD_MODEL})`,
                  'color:#16a34a;font-weight:bold');
    } else {
      console.warn(`[VAD] ACTIVE: ENERGY fallback VAD — Silero did not load. Reason: ${detail.reason || 'unknown'}`);
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'client_info',
        vad: mode,
        source: detail.source || null,
        model: mode === 'silero' ? VAD_MODEL : null,
        reason: detail.reason || null,
      }));
    }
  }, []);

  // ── Load Silero VAD (self-hosted first, CDN fallback) and create singleton ──
  const initSileroVAD = useCallback((stream) => {
    if (vadReadyRef.current || vadFailedRef.current) return Promise.resolve();
    if (vadInitPromiseRef.current) return vadInitPromiseRef.current;

    const createVad = (src) => window.vad.MicVAD.new({
        stream,
        ...VAD_CONFIG,
        model: VAD_MODEL,               // v5 = 512-sample (32 ms) frames, matches VAD_CONFIG maths
        baseAssetPath:    src.assetBase,
        onnxWASMBasePath: src.wasmBase,

        onSpeechStart: () => {
          console.log('[Silero VAD] onSpeechStart');
          setVoiceSpeaking(true);
          setVoiceStatus('Listening (speaking)…');

          // Barge-in: if AI is active, start 700ms confirmation timer.
          // Echo-guard logic: suppress barge-in if AI audio is CURRENTLY PLAYING
          // (isPlayingRef.current === true) OR within 1 s after it stopped.
          //
          // The old guard used only a 1 s window from the moment the first chunk
          // started.  That was too short for large Arabic responses (3–28 s of audio).
          // With the new rule the guard is "live" for the whole playback duration
          // plus a 1 s silence tail — the exact window during which the speaker
          // output can leak back into the mic despite echoCancellation:true.
          const isAIActive    = isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current;
          // Guard stays up for the whole AI turn (including gaps BETWEEN audio chunks,
          // when isPlayingRef is briefly false) plus a tail after the last chunk ends.
          const echoGuardActive = isAIActive
                               || (Date.now() - aiAudioEndTimeRef.current) < ECHO_GUARD_MS;
          echoSpeechRef.current = echoGuardActive && !ALLOW_VOICE_BARGE_IN;

          if (echoSpeechRef.current) {
            console.log('[Silero VAD] onSpeechStart ignored — AI busy / echo-guard active');
          } else if (isAIActive) {
            bargeInActiveRef.current = true;
            // Duck volume immediately
            if (currentAudio) currentAudio.volume = 0.15;
            bargeInTimerRef.current = setTimeout(() => {
              if (!bargeInActiveRef.current) return; // misfire cancelled it
              console.log('[Silero VAD] Barge-in confirmed — interrupting AI');
              bargeInActiveRef.current = false;
              stopTTS();
              audioQueueRef.current = [];
              isPlayingRef.current  = false;
              aiStreamingRef.current = false;
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
              }
            }, 700);
          }
        },

        onVADMisfire: () => {
          console.log('[Silero VAD] onVADMisfire — cancelling barge-in');
          echoSpeechRef.current = false;
          bargeInActiveRef.current = false;
          if (bargeInTimerRef.current) {
            clearTimeout(bargeInTimerRef.current);
            bargeInTimerRef.current = null;
          }
          // Restore volume
          if (currentAudio) currentAudio.volume = 1.0;
          setVoiceSpeaking(false);
          setVoiceStatus('Listening…');
        },

        onSpeechEnd: (audioFloat32) => {
          // Echo of the AI's own voice: drop it. Sending it would make the backend
          // treat it as a new user turn and cancel the answer that is still speaking.
          if (echoSpeechRef.current) {
            echoSpeechRef.current = false;
            console.log('[Silero VAD] onSpeechEnd dropped — was speaker echo');
            setVoiceSpeaking(false);
            return;
          }
          console.log('[Silero VAD] onSpeechEnd — encoding WAV, samples:', audioFloat32.length);
          setVoiceSpeaking(false);

          // If this was a barge-in, cancel the pending timer (the speech ended normally)
          if (bargeInTimerRef.current) {
            clearTimeout(bargeInTimerRef.current);
            bargeInTimerRef.current = null;
          }
          bargeInActiveRef.current = false;

          // Encode Float32Array → WAV using vad-web utils if available, else manual
          let wavBuffer;
          try {
            const blob = window.vad.utils.encodeWAV(audioFloat32);
            // encodeWAV returns a Blob; convert to ArrayBuffer for base64
            const reader = new FileReader();
            reader.onload = () => { sendAudioToServer(reader.result, 'wav'); };
            reader.readAsArrayBuffer(blob);
            return;
          } catch (e) {
            console.warn('[Silero VAD] utils.encodeWAV failed, using manual encoder:', e);
            wavBuffer = float32ToWav(audioFloat32, 16000);
          }
          sendAudioToServer(wavBuffer, 'wav');
        },
      });

    vadInitPromiseRef.current = (async () => {
      const failures = [];
      for (const src of VAD_SOURCES) {
        try {
          console.log(`[VAD] Loading Silero VAD from ${src.name}: ${src.bundleUrl}`);
          await loadVadScript(src.ortUrl,    'ort');
          await loadVadScript(src.bundleUrl, 'vad');
          if (!window.vad?.MicVAD) throw new Error('window.vad.MicVAD not found after script load');

          const myvad = await withTimeout(createVad(src), VAD_INIT_TIMEOUT_MS, 'MicVAD.new');
          vadRef.current        = myvad;
          vadReadyRef.current   = true;
          vadSourceRef.current  = src.name;
          console.log(`[VAD] Silero VAD initialised from ${src.name}`);
          return;
        } catch (err) {
          const msg = `${src.name}: ${err?.message || err}`;
          failures.push(msg);
          console.warn(`[VAD] Silero load failed (${msg})`);
        }
      }
      vadFailedRef.current     = true;
      vadFailReasonRef.current = failures.join(' | ');
      console.warn('[VAD] All Silero sources failed → energy fallback will be used.', vadFailReasonRef.current);
    })().finally(() => { vadInitPromiseRef.current = null; });

    return vadInitPromiseRef.current;
  }, [sendAudioToServer]);

  // ── Start listening for a voice turn (Silero path) ─────────────────────────
  const startVoiceListening = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    stopTTS();
    aiDoneRef.current          = false;
    speechDetectedRef.current  = false;
    speechStartRef.current     = null;
    silenceStartRef.current    = null;
    bargeInTimerRef.current    = null;
    bargeInActiveRef.current   = false;
    isSpeakingRef.current      = false;
    recordingActiveRef.current = true;

    try {
      // Acquire mic once; reuse for subsequent turns
      if (!micStreamRef.current || !micStreamRef.current.active) {
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation:  true,
            noiseSuppression:  true,
            autoGainControl:   false,   // critical: AGC skews Silero calibration
          }
        });
      }

      // Build AudioContext + Analyser for waveform (reuse across turns)
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      if (!analyserRef.current) {
        const src = audioContextRef.current.createMediaStreamSource(micStreamRef.current);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        src.connect(analyserRef.current);
      }

      setVoiceActiveSync(true);
      setVoiceProcessing(false);
      setVoiceStatus('Listening… (speak now)');

      cancelAnimationFrame(waveRafRef.current);

      if (!vadFailedRef.current) {
        // ── Silero path ────────────────────────────────────────────────────
        await initSileroVAD(micStreamRef.current);

        if (vadReadyRef.current && vadRef.current) {
          reportVadMode('silero', { source: vadSourceRef.current });
          // Resume/start the VAD (it keeps the mic open across turns)
          vadRef.current.start();
          waveRafRef.current = requestAnimationFrame(animateWave);
          return;
        }
        // If initSileroVAD set vadFailedRef, fall through to energy VAD
      }

      // ── Fallback energy VAD path ───────────────────────────────────────
      reportVadMode('energy', { reason: vadFailReasonRef.current || 'Silero VAD unavailable' });
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch {}
      }
      const { mime: recMime, ext: recExt } = pickRecorderFormat();
      const mrOptions = recMime ? { mimeType: recMime } : {};
      const mr = new MediaRecorder(micStreamRef.current, mrOptions);
      mediaRecorderRef.current = mr;
      mr._recExt = recExt;

      mr.ondataavailable = (e) => {
        if (e.data.size === 0 || wsRef.current?.readyState !== WebSocket.OPEN) return;
        // Only stream audio from the CURRENT recorder, and never while the AI is
        // replying: the server would otherwise see mic audio mid-answer.
        if (mediaRecorderRef.current !== mr) return;
        if (isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current) return;
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result?.split(',')[1];
          if (b64 && wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'audio_chunk', data: b64, format: recExt }));
          }
        };
        reader.readAsDataURL(e.data);
      };
      mr.start(250);
      waveRafRef.current = requestAnimationFrame(fallbackVADLoop);

    } catch (err) {
      console.error('[Mic Error]', err);
      recordingActiveRef.current = false;
      addMsg('ai', `⚠️ Microphone error: ${err.message}`, 'error');
    }
  }, [animateWave, fallbackVADLoop, initSileroVAD, reportVadMode]);

  startVoiceListeningRef.current = startVoiceListening;

  // ── WebSocket setup ──────────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) { try { wsRef.current.close(); } catch {} }

    const ws = new WebSocket(`${WS_URL}?session_id=${SESSION_ID}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen  = () => { console.log('[WS] Connected'); setWsConnected(true); };
    ws.onclose = () => {
      console.log('[WS] Disconnected, will reconnect in 3s');
      setWsConnected(false);
      reconnectTimerRef.current = setTimeout(connectWS, 3000);
    };
    ws.onerror = () => setWsConnected(false);

    ws.onmessage = (ev) => {
      // ── Binary frame: TTS audio chunk ──────────────────────────────────
      if (ev.data instanceof ArrayBuffer) {
        if (soundEnabledRef.current) {
          const blob = new Blob([ev.data], { type: 'audio/wav' });
          audioQueueRef.current.push(blob);
          if (!isPlayingRef.current) playNextInQueue();
        }
        return;
      }

      // ── Text frame: JSON control messages ──────────────────────────────
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === 'vad_status') {
        setVoiceSpeaking(msg.speaking);
      } else if (msg.type === 'status') {
        setVoiceStatus(msg.status);
        if (msg.status.includes('Transcribing')) setVoiceProcessing(true);
        // "No speech detected" → restart listening after brief delay
        if (msg.status === 'No speech detected') {
          setVoiceProcessing(false);
          setVoiceStatus('Listening… (speak now)');
        }
      } else if (msg.type === 'thought') {
        // UI-only progress text (backend sends speak:false) — never synthesise via TTS.
        // Calling playGroqAudio here was causing the status string to be spoken via the
        // HTTP /audio/synthesize endpoint, played through speakers, picked up by the mic,
        // and triggering a false barge-in that cut off the real Arabic answer.
        setVoiceStatus(msg.text);
      } else if (msg.type === 'transcript') {
        setVoiceTranscript(msg.text);
        setVoiceStatus('Processing response…');
        setVoiceProcessing(true);
        if (msg.final) addMsg('user', msg.text);
      } else if (msg.type === 'ai_start') {
        stopTTS();
        audioQueueRef.current = [];
        isPlayingRef.current  = false;
        aiStreamingRef.current = true;
        setVoiceStatus('Responding…');
      } else if (msg.type === 'ai_token') {
        aiStreamingRef.current = true;
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last?.role === 'ai' && last?.streaming) {
            return prev.map((m, i) => i === prev.length - 1 ? { ...m, text: m.text + msg.token } : m);
          }
          const id = nextId.current++;
          return [...prev, { id, role: 'ai', text: msg.token, streaming: true }];
        });
      } else if (msg.type === 'ai_done') {
        aiStreamingRef.current = false;
        setVoiceProcessing(false);
        setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m));
        if (isPlayingRef.current || audioQueueRef.current.length > 0) {
          setVoiceStatus('Speaking…');
        } else {
          setVoiceStatus('Ready');
          if (voiceActiveRef.current && startVoiceListeningRef.current) {
            startVoiceListeningRef.current();
          }
        }
        aiDoneRef.current = true;
      } else if (msg.type === 'interrupted') {
        stopTTS();
        audioQueueRef.current  = [];
        isPlayingRef.current   = false;
        aiStreamingRef.current = false;
        setVoiceProcessing(false);
        setVoiceStatus('Interrupted');
      } else if (msg.type === 'error') {
        setVoiceProcessing(false);
        setVoiceStatus('Error');
        addMsg('ai', `⚠️ Voice error: ${msg.message || 'Unknown error'}`, 'error');
        console.error('[WS Error]', msg.message);
      }
    };
  }, [playNextInQueue]);

  useEffect(() => {
    connectWS();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) { try { wsRef.current.close(); } catch {} }
    };
  }, [connectWS]);

  // ── Keepalive pings ──────────────────────────────────────────────────────────
  // Tablets and mobile browsers throttle setInterval to ≥60 s when the tab is
  // backgrounded (Page Visibility API throttling in Safari/Chrome).  A 20 s
  // interval therefore misses the 120 s server timeout window on a sleeping tablet.
  //
  // Fix: use a 10 s interval (gives 12 pings before the 120 s deadline) AND send
  // an immediate ping on visibilitychange so that the connection is refreshed the
  // moment the user lifts the tablet again.
  useEffect(() => {
    const sendPing = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    };

    // Periodic keepalive every 10 s
    const iv = setInterval(sendPing, 10000);

    // Immediate ping when tab regains focus (tablet wake / app switch back)
    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        console.log('[WS] Tab became visible — sending keepalive ping');
        sendPing();
      }
    };
    document.addEventListener('visibilitychange', onVisible);

    return () => {
      clearInterval(iv);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, []);

  // ── Stop voice completely ──────────────────────────────────────────────────
  const stopVoice = () => {
    recordingActiveRef.current = false;
    setVoiceActiveSync(false);
    cancelAnimationFrame(waveRafRef.current);

    // Pause (not destroy) the Silero VAD so we can resume next session
    if (vadRef.current) {
      try { vadRef.current.pause(); } catch {}
    }

    // Stop fallback recorder if running
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') { try { mr.stop(); } catch {} }

    // Stop mic tracks and close audio context
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(t => t.stop());
      micStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    mediaRecorderRef.current = null;

    // Destroy the VAD instance so getUserMedia is released
    if (vadRef.current) {
      try { vadRef.current.destroy(); } catch {}
      vadRef.current    = null;
      vadReadyRef.current = false;
    }
    // Next voice session retries Silero from scratch and re-reports the mode.
    vadFailedRef.current     = false;
    vadFailReasonRef.current = '';
    vadSourceRef.current     = null;
    vadModeRef.current       = null;

    setVoiceSpeaking(false);
    setVoiceProcessing(false);
    setVoiceStatus('');
    setWaveHeights(Array(12).fill(4));
    stopTTS();
    audioQueueRef.current = [];
    isPlayingRef.current  = false;
    aiDoneRef.current     = false;
  };

  // ── Text submit ────────────────────────────────────────────────────────────
  const submitText = async () => {
    const text = inputText.trim();
    if (!text || sending) return;
    setInputText('');
    setSending(true);
    stopTTS();
    addMsg('user', text);

    const thinkId = nextId.current++;
    setMessages(prev => [...prev, { id: thinkId, role: 'ai_think', text: '', thoughts: [THOUGHTS[0]], variant: '' }]);

    let thinkTimer = 0;
    const iv = setInterval(() => {
      thinkTimer++;
      if (thinkTimer < THOUGHTS.length) {
        setMessages(prev => prev.map(m =>
          m.id === thinkId ? { ...m, thoughts: THOUGHTS.slice(0, thinkTimer + 1) } : m
        ));
      }
    }, 900);

    try {
      const res = await fetch(`${API_BASE}/orchestrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: SESSION_ID, user_id: USER_ID, text_input: text }),
      });
      clearInterval(iv);
      setMessages(prev => prev.filter(m => m.id !== thinkId));

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const aiId = addMsg('ai', '');
      setMessages(prev => prev.map(m => m.id === aiId ? { ...m, streaming: true } : m));

      let full = '';
      const reader = res.body.getReader();
      const dec    = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const tok = dec.decode(value, { stream: true });
        full += tok;
        updateMsg(aiId, full);
      }
      setMessages(prev => prev.map(m => m.id === aiId ? { ...m, streaming: false } : m));
    } catch (err) {
      clearInterval(iv);
      setMessages(prev => prev.filter(m => m.id !== thinkId));
      addMsg('ai', `[Error]: ${err.message}`, 'error');
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="chat-layout" style={{ height: 'calc(100dvh - 58px)' }}>
      <VoiceOverlay
        active={voiceActive}
        speaking={voiceSpeaking}
        processing={voiceProcessing}
        transcript={voiceTranscript}
        status={voiceStatus}
        waveHeights={waveHeights}
        onStop={stopVoice}
      />

      {/* Messages */}
      <div className="chat-messages">
        {messages.map(msg => {
          if (msg.role === 'ai_think') {
            return (
              <div key={msg.id} className="message-row ai">
                <div className="avatar ai">OS</div>
                <div className="thought-box">
                  <div className="th-head">Kernel Processing</div>
                  {msg.thoughts.map((t, i) => <div key={i} className="thought-line">{t}</div>)}
                </div>
              </div>
            );
          }
          const isAI   = msg.role === 'ai';
          const isUser = msg.role === 'user';
          const roleClass = isAI ? 'ai' : isUser ? 'user' : 'sys';

          return (
            <div key={msg.id} className={`message-row ${roleClass}`}>
              <div className={`avatar ${roleClass}`}>
                {isAI ? 'OS' : isUser ? 'U' : '⊙'}
              </div>
              <div className={`bubble ${roleClass} ${msg.variant || ''} ${msg.streaming ? 'streaming' : ''}`}>
                {(() => {
                  const imgMatch = msg.text.match(/!\[(.*?)\]\((https:\/\/pubchem\.ncbi\.nlm\.nih\.gov\/rest\/pug\/compound\/.*?\/PNG.*?)\)/);
                  if (imgMatch) {
                    const altText  = imgMatch[1];
                    const imgUrl   = imgMatch[2];
                    const cleanText = msg.text.replace(imgMatch[0], '').trim();
                    return (
                      <>
                        <div className="compound-structure-card">
                          <div className="card-badge">🧪 Molecular Structure</div>
                          <div className="img-wrap">
                            <img src={imgUrl} alt={altText} onError={(e) => { e.target.style.display = 'none'; }} />
                          </div>
                          <div className="card-label">{altText}</div>
                        </div>
                        {cleanText && <div>{cleanText}</div>}
                      </>
                    );
                  }
                  return msg.text;
                })()}

                {msg.text && msg.role !== 'sys' && (
                  <button
                    className="copy-btn"
                    title="Copy message"
                    onClick={(e) => {
                      const btn = e.currentTarget;
                      const cleanText = msg.text.replace(/!\[.*?\]\(.*?\)/g, '').trim();
                      navigator.clipboard.writeText(cleanText);
                      btn.innerText = '✓ Copied';
                      setTimeout(() => { btn.innerText = '📋 Copy'; }, 2000);
                    }}
                  >
                    📋 Copy
                  </button>
                )}
              </div>
            </div>
          );
        })}
        <div ref={chatEndRef} />
      </div>

      {/* Bottom bar */}
      <div className="bottom-bar">
        <div className="input-row">
          <button
            className={`icon-btn ${wsConnected ? 'ws-on' : ''}`}
            title="WebSocket voice channel"
            onClick={voiceActive ? stopVoice : startVoiceListening}
          >
            {voiceActive ? '⏹' : '🎙'}
          </button>

          <button
            className={`icon-btn ${soundEnabled ? 'ws-on' : ''}`}
            title={soundEnabled ? 'Audio response: ON' : 'Audio response: OFF'}
            onClick={() => {
              if (soundEnabled) stopTTS();
              setSoundEnabledSync(!soundEnabled);
            }}
          >
            {soundEnabled ? '🔊' : '🔇'}
          </button>

          <textarea
            className="user-input"
            rows={1}
            placeholder="Ask about drug discovery, ADMET, molecular pathways…"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitText(); } }}
            disabled={sending}
            style={{ maxHeight: '120px', lineHeight: '1.5' }}
          />

          <button className="send-btn" onClick={submitText} disabled={sending || !inputText.trim()}>
            Send ➤
          </button>
        </div>

        <div className="status-bar">
          <div className={`ws-dot ${wsConnected ? 'on' : 'off'}`} />
          <span>{wsConnected ? 'Voice channel connected' : 'Reconnecting…'}</span>
          <span style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}>
            Audio: {soundEnabled ? 'Enabled (Groq Orpheus)' : 'Muted'} | Session: {SESSION_ID}
          </span>
        </div>
      </div>
    </div>
  );
}
