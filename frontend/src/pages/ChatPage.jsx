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
  // Fallback to browser SpeechSynthesis if backend TTS call fails
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
      <button className="overlay-stop-btn" onClick={onStop}>■ Stop</button>
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
  const [ttsActive, setTtsActive]     = useState(false);

  // Keep ref in sync with state so closures always read the latest value
  const setSoundEnabledSync = (val) => {
    soundEnabledRef.current = val;
    setSoundEnabled(val);
  };

  // WebSocket state
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef                         = useRef(null);

  // Voice overlay state
  const [voiceActive, setVoiceActive]       = useState(false);
  const [voiceSpeaking, setVoiceSpeaking]   = useState(false);
  const [voiceProcessing, setVoiceProcessing] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [voiceStatus, setVoiceStatus]       = useState('');
  const [waveHeights, setWaveHeights]       = useState(Array(12).fill(4));

  // Audio refs
  const mediaRecorderRef  = useRef(null);
  const audioContextRef   = useRef(null);
  const analyserRef       = useRef(null);
  const waveRafRef        = useRef(null);
  const aiStreamingRef    = useRef(false);
  const interruptedRef    = useRef(false);

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

  // ── WebSocket setup ─────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`${WS_URL}?session_id=${SESSION_ID}`);
    wsRef.current = ws;

    ws.onopen  = () => setWsConnected(true);
    ws.onclose = () => { setWsConnected(false); setTimeout(connectWS, 3000); };
    ws.onerror = () => setWsConnected(false);

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'vad_status') {
        setVoiceSpeaking(msg.speaking);
      } else if (msg.type === 'status') {
        setVoiceStatus(msg.status);
        if (msg.status.includes('Transcribing')) setVoiceProcessing(true);
      } else if (msg.type === 'transcript') {
        setVoiceTranscript(msg.text);
        setVoiceStatus('Processing response…');
        setVoiceProcessing(true);
        if (msg.final) addMsg('user', msg.text);
      } else if (msg.type === 'ai_token') {
        aiStreamingRef.current = true;
        setVoiceStatus('Responding…');
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last?.role === 'ai' && last?.streaming) {
            return prev.map((m, i) => i === prev.length - 1 ? { ...m, text: m.text + msg.token } : m);
          }
          const id = nextId.current++;
          return [...prev, { id, role: 'ai', text: msg.token, streaming: true }];
        });
      } else if (msg.type === 'ai_audio') {
        // Only play if sound is enabled — check ref (not state) to avoid stale closure
        if (msg.data && soundEnabledRef.current) {
          try {
            stopTTS();
            const binary = atob(msg.data);
            const array = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);
            const blob = new Blob([array.buffer], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            currentAudio = new Audio(url);
            currentAudio.play().catch((playErr) => {
              console.warn('WS ai_audio autoplay blocked, trying SpeechSynthesis fallback:', playErr);
              // We don’t have the original text here so skip synthesis fallback
            });
          } catch (err) {
            console.error('WS audio playback error:', err);
          }
        }
      } else if (msg.type === 'ai_done') {
        aiStreamingRef.current = false;
        setVoiceProcessing(false);
        setVoiceStatus('Ready');
        setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m));
      } else if (msg.type === 'interrupted') {
        setVoiceProcessing(false);
        setVoiceStatus('Interrupted');
      }
    };
  }, []);

  useEffect(() => { connectWS(); return () => wsRef.current?.close(); }, [connectWS]);

  // ── Waveform animation ──────────────────────────────────────────────────
  const animateWave = () => {
    if (!analyserRef.current) { waveRafRef.current = requestAnimationFrame(animateWave); return; }
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const step = Math.floor(data.length / 12);
    const heights = Array.from({ length: 12 }, (_, i) => {
      const v = data[i * step] || 0;
      return Math.max(4, (v / 255) * 36);
    });
    setWaveHeights(heights);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const rms = Math.sqrt(data.reduce((s, v) => s + v * v, 0) / data.length);
      wsRef.current.send(JSON.stringify({ type: 'vad_energy', rms }));
    }
    waveRafRef.current = requestAnimationFrame(animateWave);
  };

  // ── Mic / WS voice start ─────────────────────────────────────────────────
  const startVoice = async () => {
    if (!wsConnected) { addMsg('ai', '⚠️ Voice channel not connected — please wait or reload.', 'error'); return; }
    stopTTS();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      const source  = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);

      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data.size === 0 || wsRef.current?.readyState !== WebSocket.OPEN) return;
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result.split(',')[1];
          wsRef.current.send(JSON.stringify({ type: 'audio_chunk', data: b64, format: 'webm' }));
        };
        reader.readAsDataURL(e.data);
      };

      mr.start(100);
      setVoiceActive(true);
      setVoiceTranscript('');
      setVoiceStatus('Listening…');
      waveRafRef.current = requestAnimationFrame(animateWave);
    } catch (err) {
      addMsg('ai', `⚠️ Microphone error: ${err.message}`, 'error');
    }
  };

  const stopVoice = () => {
    const mr = mediaRecorderRef.current;
    if (mr) {
      // Wait for onstop so the final ondataavailable chunk is flushed BEFORE audio_end
      mr.onstop = () => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'audio_end', format: 'webm' }));
        }
      };
      mr.stop();
      mr.stream?.getTracks().forEach(t => t.stop());
    } else {
      // No recorder running — send audio_end immediately if needed
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'audio_end', format: 'webm' }));
      }
    }
    if (audioContextRef.current) audioContextRef.current.close();
    cancelAnimationFrame(waveRafRef.current);
    setVoiceActive(false);
    setVoiceSpeaking(false);
    setWaveHeights(Array(12).fill(4));
  };

  // ── Text submit ───────────────────────────────────────────────────────────
  const submitText = async () => {
    const text = inputText.trim();
    if (!text || sending) return;
    setInputText('');
    setSending(true);
    stopTTS();
    addMsg('user', text);

    // Show thought box
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
      if (full) playGroqAudio(full, soundEnabledRef);
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

                {/* Copy button */}
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
            onClick={voiceActive ? stopVoice : startVoice}
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
