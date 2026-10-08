import { useState, useRef, useEffect, useCallback } from 'react';
import Message from '../components/Message';
import VoiceOverlay from '../components/VoiceOverlay';
import { useVoiceSocket } from '../hooks/useVoiceSocket';
import { useVoiceSession } from '../hooks/useVoiceSession';
import { preloadVadScripts } from '../services/vadService';
import { streamOrchestrate } from '../services/api';
import { stopTTS } from '../utils/audioUtils';

const SESSION_ID = 'session_' + Math.random().toString(36).slice(2, 9);
const USER_ID    = 'user_'    + Math.random().toString(36).slice(2, 9);

const THOUGHTS = [
  '⚙️  Kernel initialising…',
  '🔬  Routing to domain agent…',
  '🧬  Analysing molecular structures…',
  '🤖  Synthesising response…',
  '📡  Streaming results…',
];

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: 'sys',
      text: '⚡ AI-lixir Scientific OS online — Ask about drug discovery, ADMET, molecular analysis, or biomedical pathways.',
      variant: '',
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [sending, setSending] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const soundEnabledRef = useRef(true);

  const setSoundEnabledSync = (val) => {
    soundEnabledRef.current = val;
    setSoundEnabled(val);
  };

  const chatEndRef = useRef(null);
  const nextId = useRef(1);

  const addMsg = useCallback((role, text, variant = '') => {
    const id = nextId.current++;
    setMessages(prev => [...prev, { id, role, text, variant }]);
    return id;
  }, []);

  const updateMsg = useCallback((id, text) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, text } : m));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const voiceSessionRef = useRef(null);

  // Handle incoming control messages from voice socket
  const handleSocketMessage = useCallback((msg) => {
    const session = voiceSessionRef.current;
    if (!session) return;

    if (msg.type === 'vad_status') {
      // client_info ack — includes speaking:false / ok:true. Do not treat as a
      // speaking event and do not overwrite the overlay status text.
      console.log('[VAD] backend ack', msg.vad, 'ok=', msg.ok);
      return;
    }
    if (typeof msg.speaking === 'boolean') {
      session.setVoiceSpeaking(msg.speaking);
    }
    if (msg.type === 'status') {
      session.setVoiceStatus(msg.status);
      if (msg.status.includes('Transcribing')) session.setVoiceProcessing(true);
      if (msg.status === 'No speech detected') {
        session.setVoiceProcessing(false);
        session.setVoiceStatus('Listening… (speak now)');
      }
    } else if (msg.type === 'thought') {
      session.setVoiceStatus(msg.text);
    } else if (msg.type === 'transcript') {
      session.setVoiceTranscript(msg.text);
      session.setVoiceStatus('Processing response…');
      session.setVoiceProcessing(true);
      if (msg.final) addMsg('user', msg.text);
    } else if (msg.type === 'ai_start') {
      stopTTS();
      session.audioQueueRef.current = [];
      session.isPlayingRef.current = false;
      session.aiStreamingRef.current = true;
      session.setVoiceStatus('Responding…');
    } else if (msg.type === 'ai_token') {
      session.aiStreamingRef.current = true;
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === 'ai' && last?.streaming) {
          return prev.map((m, i) => i === prev.length - 1 ? { ...m, text: m.text + msg.token } : m);
        }
        const id = nextId.current++;
        return [...prev, { id, role: 'ai', text: msg.token, streaming: true }];
      });
    } else if (msg.type === 'ai_done') {
      session.aiStreamingRef.current = false;
      session.setVoiceProcessing(false);
      setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m));
      if (session.isPlayingRef.current || session.audioQueueRef.current.length > 0) {
        // Audio is still playing: playNextInQueue() resumes listening once it drains.
        session.aiDoneRef.current = true;
        session.setVoiceStatus('Speaking…');
      } else {
        // Nothing left to play. Reset the flag BEFORE restarting the mic: it used to be
        // set to true afterwards, which left a stale flag that re-armed listening
        // during the next answer.
        session.aiDoneRef.current = false;
        session.setVoiceStatus('Ready');
        if (session.voiceActiveRef?.current && session.startVoiceListeningRef.current) {
          session.startVoiceListeningRef.current();
        }
      }
    } else if (msg.type === 'interrupted') {
      stopTTS();
      session.audioQueueRef.current = [];
      session.isPlayingRef.current = false;
      session.aiStreamingRef.current = false;
      session.aiDoneRef.current = false;
      session.setVoiceProcessing(false);
      session.setVoiceStatus('Interrupted');
      // Don't leave the cut-off answer bubble spinning forever.
      setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m));
    } else if (msg.type === 'error') {
      session.aiStreamingRef.current = false;
      session.setVoiceProcessing(false);
      setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m));
      session.setVoiceStatus('Error');
      addMsg('ai', `⚠️ Voice error: ${msg.message || 'Unknown error'}`, 'error');
      console.error('[WS Error]', msg.message);
    }
  }, [addMsg]);

  // Handle TTS audio chunk delivered over WebSocket
  const handleBinaryChunk = useCallback((arrayBuffer) => {
    if (soundEnabledRef.current && voiceSessionRef.current) {
      voiceSessionRef.current.queueAudioChunk(arrayBuffer);
    }
  }, []);

  const handleSocketOpen = useCallback(() => {
    voiceSessionRef.current?.resendVadMode?.();
  }, []);

  const voiceSocket = useVoiceSocket({
    sessionId: SESSION_ID,
    onMessage: handleSocketMessage,
    onBinaryChunk: handleBinaryChunk,
    onOpen: handleSocketOpen,
  });

  const voiceSession = useVoiceSession({
    sendJson: voiceSocket.sendJson,
    wsConnected: voiceSocket.wsConnected,
    onError: (err) => addMsg('ai', `⚠️ Microphone error: ${err.message}`, 'error'),
  });

  useEffect(() => {
    voiceSessionRef.current = voiceSession;
  });

  useEffect(() => {
    preloadVadScripts().then((src) => {
      if (src) console.log(`[VAD] scripts preloaded from ${src}`);
    });
  }, []);

  // Text submit handler
  const submitText = async () => {
    const text = inputText.trim();
    if (!text || sending) return;
    setInputText('');
    setSending(true);
    stopTTS();
    addMsg('user', text);

    const thinkId = nextId.current++;
    setMessages(prev => [
      ...prev,
      { id: thinkId, role: 'ai_think', text: '', thoughts: [THOUGHTS[0]], variant: '' },
    ]);

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
      let full = '';
      let aiId = null;

      await streamOrchestrate(text, SESSION_ID, USER_ID, (token) => {
        if (aiId === null) {
          clearInterval(iv);
          setMessages(prev => prev.filter(m => m.id !== thinkId));
          aiId = addMsg('ai', '');
          setMessages(prev => prev.map(m => m.id === aiId ? { ...m, streaming: true } : m));
        }
        full += token;
        updateMsg(aiId, full);
      });

      if (aiId !== null) {
        setMessages(prev => prev.map(m => m.id === aiId ? { ...m, streaming: false } : m));
      } else {
        clearInterval(iv);
        setMessages(prev => prev.filter(m => m.id !== thinkId));
      }
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
        active={voiceSession.voiceActive}
        speaking={voiceSession.voiceSpeaking}
        processing={voiceSession.voiceProcessing}
        transcript={voiceSession.voiceTranscript}
        status={voiceSession.voiceStatus}
        waveHeights={voiceSession.waveHeights}
        onStop={voiceSession.stopVoice}
      />

      {/* Messages */}
      <div className="chat-messages">
        {messages.map(msg => (
          <Message
            key={msg.id}
            role={msg.role}
            text={msg.text}
            variant={msg.variant}
            streaming={msg.streaming}
            thoughts={msg.thoughts}
          />
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Bottom bar */}
      <div className="bottom-bar">
        <div className="input-row">
          <button
            className={`icon-btn ${voiceSocket.wsConnected ? 'ws-on' : ''}`}
            title="WebSocket voice channel"
            onClick={voiceSession.voiceActive ? voiceSession.stopVoice : voiceSession.startVoiceListening}
          >
            {voiceSession.voiceActive ? '⏹' : '🎙'}
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
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitText();
              }
            }}
            disabled={sending}
            style={{ maxHeight: '120px', lineHeight: '1.5' }}
          />

          <button className="send-btn" onClick={submitText} disabled={sending || !inputText.trim()}>
            Send ➤
          </button>
        </div>

        <div className="status-bar">
          <div className={`ws-dot ${voiceSocket.wsConnected ? 'on' : 'off'}`} />
          <span>{voiceSocket.wsConnected ? 'Voice channel connected' : 'Reconnecting…'}</span>
          <span style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}>
            Audio: {soundEnabled ? 'Enabled (Groq Orpheus)' : 'Muted'} | Session: {SESSION_ID}
          </span>
        </div>
      </div>
    </div>
  );
}
