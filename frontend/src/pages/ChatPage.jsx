import { useState, useRef, useEffect, useCallback } from 'react';
import { API_BASE, WS_URL } from '../config';

const SESSION_ID = 'session_' + Math.random().toString(36).slice(2, 9);
const USER_ID    = 'user_' + Math.random().toString(36).slice(2, 9);

const THOUGHTS = [
  '⚙️  Kernel initialising…',
  '🔬  Routing to domain agent…',
  '🧬  Analysing molecular structures…',
  '🤖  Synthesising response…',
  '📡  Streaming results…',
];

let currentAudio = null;

// ── MIME type / format helpers ───────────────────────────────────────────────
// Pick the best audio format the browser's MediaRecorder actually supports.
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

// Compute both total RMS and speech-band (350Hz-3600Hz) RMS from frequency data.
// Speech-band RMS rejects low frequency rumble (fans/AC) and high frequency hiss.
function computeAudioMetrics(freqData) {
  let totalSum = 0;
  let speechSum = 0;
  // Bins 2..20 (~350Hz - ~3600Hz) correspond to human vocal range
  const speechStart = 2;
  const speechEnd = Math.min(20, freqData.length);
  for (let i = 0; i < freqData.length; i++) {
    const val = freqData[i];
    totalSum += val * val;
    if (i >= speechStart && i < speechEnd) {
      speechSum += val * val;
    }
  }
  const totalRms = Math.sqrt(totalSum / freqData.length);
  const speechCount = speechEnd - speechStart;
  const speechRms = speechCount > 0 ? Math.sqrt(speechSum / speechCount) : totalRms;
  return { totalRms, speechRms };
}

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
      const url = URL.createObjectURL(blob);
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
    utt.rate = 1.0;
    utt.pitch = 1.0;
    window.speechSynthesis.speak(utt);
  }
}

function stopTTS() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
}

// ── Message component ────────────────────────────────────────────────────────
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

// ── Voice Overlay ────────────────────────────────────────────────────────────
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

// ── Main Chat Page ───────────────────────────────────────────────────────────
export default function ChatPage() {
  const [messages, setMessages]       = useState([
    { id: 0, role: 'sys', text: '⚡ AI-lixir Scientific OS online — Ask about drug discovery, ADMET, molecular analysis, or biomedical pathways.', variant: '' }
  ]);
  const [inputText, setInputText]     = useState('');
  const [sending, setSending]         = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const soundEnabledRef = useRef(true);

  const setSoundEnabledSync = (val) => {
    soundEnabledRef.current = val;
    setSoundEnabled(val);
  };

  // WebSocket state
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef                         = useRef(null);
  const reconnectTimerRef             = useRef(null);

  // Voice overlay state
  const [voiceActive, setVoiceActive]       = useState(false);
  const voiceActiveRef                      = useRef(false);
  const [voiceSpeaking, setVoiceSpeaking]   = useState(false);
  const [voiceProcessing, setVoiceProcessing] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [voiceStatus, setVoiceStatus]       = useState('');
  const [waveHeights, setWaveHeights]       = useState(Array(12).fill(4));

  // Audio & VAD refs
  const mediaRecorderRef   = useRef(null);
  const audioContextRef    = useRef(null);
  const analyserRef        = useRef(null);
  const waveRafRef         = useRef(null);
  const micStreamRef       = useRef(null);
  const aiStreamingRef     = useRef(false);

  // VAD Auto-endpointing & Barge-in refs
  const speechDetectedRef  = useRef(false);
  const speechStartRef     = useRef(null);
  const silenceStartRef    = useRef(null);
  const bargeInStartRef    = useRef(null);
  const isSpeakingRef      = useRef(false);
  const recordingActiveRef = useRef(false);

  // Progressive audio queue for streaming TTS playback
  const audioQueueRef      = useRef([]);   // Queue of Blob objects (each a complete WAV)
  const isPlayingRef       = useRef(false); // Whether we're currently playing a chunk
  const aiDoneRef          = useRef(false); // Whether ai_done has been received for current turn

  const chatEndRef = useRef(null);
  const nextId     = useRef(1);

  const addMsg = (role, text, variant = '') => {
    const id = nextId.current++;
    setMessages(prev => [...prev, { id, role, text, variant }]);
    return id;
  };
  const updateMsg = (id, text) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, text } : m));
  };

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // Turn sequence counter to prevent cross-turn chunk contamination
  const turnSeqRef               = useRef(0);
  const startVoiceListeningRef   = useRef(null);
  const finalizeVoiceTurnRef     = useRef(null);

  // Keep voiceActiveRef in sync
  const setVoiceActiveSync = (val) => {
    voiceActiveRef.current = val;
    setVoiceActive(val);
  };

  // ── Progressive Audio Queue Playback ─────────────────────────────────────
  // Plays WAV chunks sequentially as they arrive from the server.
  // When the last chunk finishes and ai_done was received, resumes mic.
  const playNextInQueue = useCallback(() => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      // If AI is done generating, resume listening
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
    const url = URL.createObjectURL(blob);
    stopTTS();
    currentAudio = new Audio(url);
    currentAudio.onended = () => {
      URL.revokeObjectURL(url);
      playNextInQueue();
    };
    currentAudio.onerror = () => {
      URL.revokeObjectURL(url);
      playNextInQueue();
    };
    currentAudio.play().catch(() => {
      URL.revokeObjectURL(url);
      playNextInQueue();
    });
  }, []);

  // ── VAD Finalize Turn (Auto-stop when user finishes speaking) ────────────
  const finalizeVoiceTurn = useCallback(() => {
    if (!recordingActiveRef.current) return;
    recordingActiveRef.current = false;

    const mr = mediaRecorderRef.current;
    const currentTurn = turnSeqRef.current;
    if (mr && mr.state !== 'inactive') {
      const ext = mr._recExt || 'webm';
      mr.onstop = () => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'audio_end', turn: currentTurn, format: ext }));
        }
      };
      try { mr.stop(); } catch (_) {}
    } else {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'audio_end', turn: currentTurn, format: 'webm' }));
      }
    }

    setVoiceSpeaking(false);
    setVoiceProcessing(true);
    setVoiceStatus('Transcribing speech…');
    setWaveHeights(Array(12).fill(4));
  }, []);
  finalizeVoiceTurnRef.current = finalizeVoiceTurn;

  // ── Waveform animation & Client-side VAD & Barge-in Monitor ──────────────
  const animateWave = useCallback(() => {
    if (!analyserRef.current || !voiceActiveRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const step = Math.floor(data.length / 12);
    const heights = Array.from({ length: 12 }, (_, i) => {
      const v = data[i * step] || 0;
      return Math.max(4, (v / 255) * 36);
    });
    setWaveHeights(heights);

    const { totalRms, speechRms } = computeAudioMetrics(data);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'vad_energy', rms: totalRms }));
    }

    // ── Mode A: User's turn to speak (recording active) ──
    if (recordingActiveRef.current) {
      const SPEECH_THRESHOLD = 18.0;
      const MIN_SPEECH_MS    = 350;  // Must speak for >350ms to count as real speech
      const SILENCE_TIMEOUT  = 1200; // 1.2s silence after speech triggers auto-stop

      if (speechRms > SPEECH_THRESHOLD) {
        if (speechStartRef.current === null) {
          speechStartRef.current = Date.now();
        }
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
        // If user has spoken at least once in this turn, check silence duration
        if (speechDetectedRef.current) {
          if (silenceStartRef.current === null) {
            silenceStartRef.current = Date.now();
          } else if (Date.now() - silenceStartRef.current >= SILENCE_TIMEOUT) {
            // User finished speaking! Auto-send to server
            console.log('[VAD] Silence detected after speech. Auto-finalizing turn…');
            if (finalizeVoiceTurnRef.current) {
              finalizeVoiceTurnRef.current();
            }
          }
        }
      }
    } 
    // ── Mode B: AI is Thinking or Speaking (Barge-in detection) ──
    else {
      const isAIActive = isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current;
      if (isAIActive) {
        const BARGE_IN_THRESHOLD = 26.0; // Higher threshold to reject noise and speaker leakage
        const MIN_BARGE_IN_MS    = 400;  // Must be sustained speech > 400ms

        if (speechRms > BARGE_IN_THRESHOLD) {
          if (bargeInStartRef.current === null) {
            bargeInStartRef.current = Date.now();
          } else if (Date.now() - bargeInStartRef.current >= MIN_BARGE_IN_MS) {
            // ── Genuine user speech detected while AI was speaking → BARGE IN! ──
            console.log('[VAD] Barge-in triggered! Sustained speech detected during AI output.');
            bargeInStartRef.current = null;
            stopTTS();
            audioQueueRef.current = [];
            isPlayingRef.current = false;
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
            }
            if (startVoiceListeningRef.current) {
              startVoiceListeningRef.current();
            }
          }
        } else {
          bargeInStartRef.current = null;
        }
      }
    }

    if (voiceActiveRef.current) {
      waveRafRef.current = requestAnimationFrame(animateWave);
    }
  }, []);

  // ── Start Listening for a Voice Turn ─────────────────────────────────────
  const startVoiceListening = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    stopTTS();

    // Increment turn sequence for the new utterance
    turnSeqRef.current += 1;

    // Reset VAD & Barge-in state for the new utterance
    speechDetectedRef.current  = false;
    speechStartRef.current     = null;
    silenceStartRef.current    = null;
    bargeInStartRef.current    = null;
    isSpeakingRef.current      = false;
    aiDoneRef.current          = false;
    recordingActiveRef.current = true;

    try {
      let stream = micStreamRef.current;
      if (!stream || !stream.active) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });
        micStreamRef.current = stream;
      }

      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      if (!analyserRef.current) {
        const source = audioContextRef.current.createMediaStreamSource(stream);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        source.connect(analyserRef.current);
      }

      // Stop previous recorder if active
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch (_) {}
      }

      const { mime: recMime, ext: recExt } = pickRecorderFormat();
      const mrOptions = recMime ? { mimeType: recMime } : {};
      const mr = new MediaRecorder(stream, mrOptions);
      mediaRecorderRef.current = mr;
      mr._recExt = recExt;

      mr.ondataavailable = (e) => {
        if (e.data.size === 0 || wsRef.current?.readyState !== WebSocket.OPEN) return;
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result?.split(',')[1];
          if (b64 && wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'audio_chunk',
              data: b64,
              format: recExt
            }));
          }
        };
        reader.readAsDataURL(e.data);
      };

      mr.start(250);
      setVoiceActiveSync(true);
      setVoiceProcessing(false);
      setVoiceStatus('Listening… (speak now)');

      cancelAnimationFrame(waveRafRef.current);
      waveRafRef.current = requestAnimationFrame(animateWave);
    } catch (err) {
      console.error('[Mic Error]', err);
      recordingActiveRef.current = false;
      addMsg('ai', `⚠️ Microphone error: ${err.message}`, 'error');
    }
  }, [animateWave]);

  // Keep ref in sync so playNextInQueue & animateWave can call it without circular deps
  startVoiceListeningRef.current = startVoiceListening;

  // ── WebSocket setup ─────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      try { wsRef.current.close(); } catch (_) {}
    }

    const ws = new WebSocket(`${WS_URL}?session_id=${SESSION_ID}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen  = () => {
      console.log('[WS] Connected');
      setWsConnected(true);
    };
    ws.onclose = () => {
      console.log('[WS] Disconnected, will reconnect in 3s');
      setWsConnected(false);
      reconnectTimerRef.current = setTimeout(connectWS, 3000);
    };
    ws.onerror = () => setWsConnected(false);

    ws.onmessage = (ev) => {
      // ── Binary frame: TTS audio chunk ──────────────────────────────────
      // Each binary frame is a complete WAV for one TTS batch — queue it
      if (ev.data instanceof ArrayBuffer) {
        if (soundEnabledRef.current) {
          const blob = new Blob([ev.data], { type: 'audio/wav' });
          audioQueueRef.current.push(blob);
          // If nothing is currently playing, start playing
          if (!isPlayingRef.current) {
            playNextInQueue();
          }
        }
        return;
      }

      // ── Text frame: JSON control messages ──────────────────────────────
      let msg;
      try { msg = JSON.parse(ev.data); } catch (_) { return; }

      if (msg.type === 'vad_status') {
  setVoiceSpeaking(msg.speaking);
} else if (msg.type === 'status') {
  setVoiceStatus(msg.status);
  if (msg.status.includes('Transcribing')) setVoiceProcessing(true);
} else if (msg.type === 'thought') {
  // Speak thought instantly
  if (soundEnabledRef.current) {
    playGroqAudio(msg.text, soundEnabledRef);
  }
  setVoiceStatus(msg.text);
} else if (msg.type === 'transcript') {
  setVoiceTranscript(msg.text);
  setVoiceStatus('Processing response…');
  setVoiceProcessing(true);
  if (msg.final) addMsg('user', msg.text);
}
  // Speak thought instantly
  if (soundEnabledRef.current) {
    playGroqAudio(msg.text, soundEnabledRef);
  }
  setVoiceStatus(msg.text);
} else if (msg.type === 'transcript') {
        setVoiceSpeaking(msg.speaking);
      } else if (msg.type === 'status') {
        setVoiceStatus(msg.status);
        if (msg.status.includes('Transcribing')) setVoiceProcessing(true);
      } else if (msg.type === 'thought') {
     
        if (soundEnabledRef.current) {
          playGroqAudio(msg.text, soundEnabledRef);
        }
        setVoiceStatus(msg.text);
      } else if (msg.type === 'transcript') {
        setVoiceTranscript(msg.text);
        setVoiceStatus('Processing response…');
        setVoiceProcessing(true);
        if (msg.final) addMsg('user', msg.text);
      } else if (msg.type === 'ai_start') {
        // Clear any old queued audio
        stopTTS();
        audioQueueRef.current = [];
        isPlayingRef.current = false;
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
        // Audio is already playing progressively — just update status
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
        audioQueueRef.current = [];
        isPlayingRef.current = false;
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
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        try { wsRef.current.close(); } catch (_) {}
      }
    };
  }, [connectWS]);

  // Keepalive ping every 20s
  useEffect(() => {
    const iv = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 20000);
    return () => clearInterval(iv);
  }, []);

  // ── Stop Voice Completely ────────────────────────────────────────────────
  const stopVoice = () => {
    recordingActiveRef.current = false;
    setVoiceActiveSync(false);
    cancelAnimationFrame(waveRafRef.current);

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      try { mr.stop(); } catch (_) {}
    }
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
    setVoiceSpeaking(false);
    setVoiceProcessing(false);
    setVoiceStatus('');
    setWaveHeights(Array(12).fill(4));
    stopTTS();
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    aiDoneRef.current = false;
  };

  // ── Text submit ───────────────────────────────────────────────────────────
  const submitText = async () => {
    const text = inputText.trim();
    if (!text || sending) return;
    setInputText('');
    setSending(true);
    stopTTS();
    addMsg('user', text);

    const thinkId = nextId.current++;
    setMessages(prev => [...prev, {
      id: thinkId, role: 'ai_think', text: '', thoughts: [THOUGHTS[0]], variant: ''
    }]);

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
          const isAI = msg.role === 'ai';
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
                    const altText = imgMatch[1];
                    const imgUrl = imgMatch[2];
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
            title={soundEnabled ? "Audio response: ON" : "Audio response: OFF"}
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
