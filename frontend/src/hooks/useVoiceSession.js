import { useState, useRef, useCallback } from 'react';
import {
  initializeSileroVAD,
  armMicVad,
  waitForVadFrames,
  VAD_MODEL,
} from '../services/vadService';
import {
  pickRecorderFormat,
  computeAudioMetrics,
  float32ToWav,
  arrayBufferToBase64,
  stopTTS,
  getActiveAudio,
} from '../utils/audioUtils';

const FALLBACK_SILENCE_MS = 1200;
const ALLOW_VOICE_BARGE_IN = false;

export function useVoiceSession({ sendJson, wsConnected, onSpeechRecorded, onBargeInInterrupt, onError }) {
  const [voiceActive, setVoiceActive] = useState(false);
  const voiceActiveRef = useRef(false);
  const [voiceSpeaking, setVoiceSpeaking] = useState(false);
  const [voiceProcessing, setVoiceProcessing] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [voiceStatus, setVoiceStatus] = useState('');
  const [waveHeights, setWaveHeights] = useState(Array(12).fill(4));
  const [vadMode, setVadMode] = useState(null);

  // Audio queue & playback refs
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const aiStreamingRef = useRef(false);
  const aiDoneRef = useRef(false);
  const aiAudioStartTimeRef = useRef(0);
  const aiAudioEndTimeRef = useRef(0);
  const echoSpeechRef = useRef(false);

  // Silero VAD refs
  const vadRef = useRef(null);
  const vadReadyRef = useRef(false);
  const vadFailedRef = useRef(false);
  const vadInitPromiseRef = useRef(null);
  const vadSourceRef = useRef(null);
  const vadFailReasonRef = useRef('');
  const vadModeRef = useRef(null);
  const lastVadDetailRef = useRef({});
  const frameWatchDoneRef = useRef(false);
  const pendingSendsRef = useRef(0);
  const bargeInTimerRef = useRef(null);
  const bargeInActiveRef = useRef(false);

  // Fallback VAD & Media refs
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const waveRafRef = useRef(null);
  const micStreamRef = useRef(null);
  const analyserStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const speechDetectedRef = useRef(false);
  const speechStartRef = useRef(null);
  const silenceStartRef = useRef(null);
  const isSpeakingRef = useRef(false);
  const recordingActiveRef = useRef(false);

  const startVoiceListeningRef = useRef(null);

  const setVoiceActiveSync = (val) => {
    voiceActiveRef.current = val;
    setVoiceActive(val);
  };

  // Progressive audio playback
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
    if (vadRef.current) {
      try { vadRef.current.pause(); } catch { /* already paused */ }
    }
    const blob = audioQueueRef.current.shift();
    const url = URL.createObjectURL(blob);
    aiAudioStartTimeRef.current = Date.now();
    stopTTS();

    const audio = new Audio(url);
    audio.onended = () => { URL.revokeObjectURL(url); playNextInQueue(); };
    audio.onerror = () => { URL.revokeObjectURL(url); playNextInQueue(); };
    audio.play().catch(() => { URL.revokeObjectURL(url); playNextInQueue(); });
  }, []);

  const queueAudioChunk = useCallback((arrayBuffer) => {
    const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
    audioQueueRef.current.push(blob);
    if (!isPlayingRef.current) {
      playNextInQueue();
    }
  }, [playNextInQueue]);

  const sendAudioToServer = useCallback((wavArrayBuffer, format = 'wav') => {
    if (vadRef.current) {
      try { vadRef.current.pause(); } catch { /* already paused */ }
    }
    const b64 = arrayBufferToBase64(wavArrayBuffer);
    const chunkOk = sendJson({ type: 'audio_chunk', data: b64, format });
    const endOk = sendJson({ type: 'audio_end', format });
    if (!chunkOk || !endOk) {
      console.error('[VAD] Failed to send utterance to backend (socket down)');
    } else {
      console.log(`[VAD] Sent ${format} utterance (${b64.length} b64 chars) → backend`);
    }
    setVoiceSpeaking(false);
    setVoiceProcessing(true);
    setVoiceStatus('Transcribing speech…');
    setWaveHeights(Array(12).fill(4));
    if (onSpeechRecorded) onSpeechRecorded();
  }, [sendJson, onSpeechRecorded]);

  // Waveform animation
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

    if (voiceActiveRef.current) {
      waveRafRef.current = requestAnimationFrame(animateWave);
    }
  }, []);

  // Energy-based fallback VAD loop
  const fallbackVADLoop = useCallback(() => {
    if (!analyserRef.current || !voiceActiveRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(data);
    const step = Math.floor(data.length / 12);
    const heights = Array.from({ length: 12 }, (_, i) => {
      const v = data[i * step] || 0;
      return Math.max(4, (v / 255) * 36);
    });
    setWaveHeights(heights);

    const { speechRms } = computeAudioMetrics(data);

    if (recordingActiveRef.current) {
      const SPEECH_THRESHOLD = 28.0;
      const MIN_SPEECH_MS = 500;

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
            const mr = mediaRecorderRef.current;
            if (mr && mr.state !== 'inactive') {
              recordingActiveRef.current = false;
              const ext = mr._recExt || 'webm';
              mr.onstop = () => {
                const flushEnd = async () => {
                  let spins = 0;
                  while (pendingSendsRef.current > 0 && spins < 100) {
                    await new Promise((r) => setTimeout(r, 20));
                    spins += 1;
                  }
                  sendJson({ type: 'audio_end', format: ext });
                };
                flushEnd();
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
    } else {
      const isAIActive = isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current;
      // The energy detector cannot tell the user's voice from the AI's own voice
      // coming out of the speakers, so voice barge-in is only evaluated when it is
      // explicitly enabled. Without this gate, speaker echo above the threshold
      // sent a spurious {"type":"interrupt"} and cut the answer off mid-sentence
      // even though the user never spoke.
      if (isAIActive && ALLOW_VOICE_BARGE_IN) {
        const BARGE_IN_THRESHOLD = 35.0;
        const MIN_BARGE_IN_MS = 600;
        if (speechRms > BARGE_IN_THRESHOLD) {
          if (bargeInTimerRef.current === null) {
            bargeInTimerRef.current = Date.now();
          } else if (Date.now() - bargeInTimerRef.current >= MIN_BARGE_IN_MS) {
            bargeInTimerRef.current = null;
            stopTTS();
            audioQueueRef.current = [];
            isPlayingRef.current = false;
            aiStreamingRef.current = false;
            sendJson({ type: 'interrupt' });
            if (onBargeInInterrupt) onBargeInInterrupt();
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
  }, [sendJson, onBargeInInterrupt]);

  // Report VAD mode to server and console
  const reportVadMode = useCallback((mode, detail = {}, force = false) => {
    if (!force && vadModeRef.current === mode) return;
    vadModeRef.current = mode;
    lastVadDetailRef.current = detail;
    setVadMode(mode);
    if (mode === 'silero') {
      console.log(`%c[VAD] ACTIVE: Silero VAD (source=${detail.source}, model=${VAD_MODEL})`,
                  'color:#16a34a;font-weight:bold');
    } else {
      console.warn(`[VAD] ACTIVE: ENERGY fallback VAD — Silero did not load. Reason: ${detail.reason || 'unknown'}`);
    }
    sendJson({
      type: 'client_info',
      vad: mode,
      source: detail.source || null,
      model: mode === 'silero' ? VAD_MODEL : null,
      reason: detail.reason || null,
    });
  }, [sendJson]);

  const resendVadMode = useCallback(() => {
    if (!vadModeRef.current) return;
    reportVadMode(vadModeRef.current, lastVadDetailRef.current, true);
  }, [reportVadMode]);

  // Silero VAD initialization
  const initSileroVAD = useCallback((stream) => {
    if (vadReadyRef.current || vadFailedRef.current) return Promise.resolve();
    if (vadInitPromiseRef.current) return vadInitPromiseRef.current;

    vadInitPromiseRef.current = initializeSileroVAD(stream, {
      onSpeechStart: () => {
        const isAIActive = isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current;
        // Only treat speech as echo while the AI is actually outputting audio.
        // A post-playback tail here is wrong: startVoiceListening() runs as soon as
        // TTS drains, so the user's first reply always began inside ECHO_GUARD_MS
        // and onSpeechEnd dropped the whole utterance.
        if (isAIActive && !ALLOW_VOICE_BARGE_IN) {
          echoSpeechRef.current = true;
          console.log('[Silero VAD] onSpeechStart ignored — AI busy / echo-guard active');
          return;
        }
        echoSpeechRef.current = false;

        setVoiceSpeaking(true);
        setVoiceStatus('Listening (speaking)…');

        if (isAIActive) {
          bargeInActiveRef.current = true;
          const currAudio = getActiveAudio();
          if (currAudio) currAudio.volume = 0.15;
          bargeInTimerRef.current = setTimeout(() => {
            if (!bargeInActiveRef.current) return;
            console.log('[Silero VAD] Barge-in confirmed — interrupting AI');
            bargeInActiveRef.current = false;
            stopTTS();
            audioQueueRef.current = [];
            isPlayingRef.current = false;
            aiStreamingRef.current = false;
            sendJson({ type: 'interrupt' });
            if (onBargeInInterrupt) onBargeInInterrupt();
          }, 700);
        }
      },
      onVADMisfire: () => {
        echoSpeechRef.current = false;
        bargeInActiveRef.current = false;
        if (bargeInTimerRef.current) {
          clearTimeout(bargeInTimerRef.current);
          bargeInTimerRef.current = null;
        }
        const currAudio = getActiveAudio();
        if (currAudio) currAudio.volume = 1.0;
        setVoiceSpeaking(false);
        setVoiceStatus('Listening…');
      },
      onSpeechEnd: (audioFloat32) => {
        if (echoSpeechRef.current) {
          echoSpeechRef.current = false;
          console.log('[Silero VAD] onSpeechEnd dropped — was speaker echo');
          setVoiceSpeaking(false);
          return;
        }
        setVoiceSpeaking(false);

        if (bargeInTimerRef.current) {
          clearTimeout(bargeInTimerRef.current);
          bargeInTimerRef.current = null;
        }
        bargeInActiveRef.current = false;

        // encodeWAV() returns an ArrayBuffer (IEEE float WAV by default), not a Blob.
        // FileReader.readAsArrayBuffer() requires a Blob, so that path always threw
        // and we already fell back — send 16-bit PCM directly.
        sendAudioToServer(float32ToWav(audioFloat32, 16000), 'wav');
      },
    })
    .then(({ vad, source }) => {
      vadRef.current = vad;
      vadReadyRef.current = true;
      vadSourceRef.current = source;
    })
    .catch((err) => {
      vadFailedRef.current = true;
      vadFailReasonRef.current = err.message;
    })
    .finally(() => {
      vadInitPromiseRef.current = null;
    });

    return vadInitPromiseRef.current;
  }, [sendJson, sendAudioToServer, onBargeInInterrupt]);

  // Start voice listening
  const startVoiceListening = useCallback(async () => {
    stopTTS();
    aiDoneRef.current = false;
    speechDetectedRef.current = false;
    speechStartRef.current = null;
    silenceStartRef.current = null;
    bargeInTimerRef.current = null;
    bargeInActiveRef.current = false;
    echoSpeechRef.current = false;
    aiAudioEndTimeRef.current = 0;
    isSpeakingRef.current = false;
    recordingActiveRef.current = true;

    try {
      if (!micStreamRef.current || !micStreamRef.current.active) {
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: false,
          },
        });
      }

      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      if (!analyserRef.current) {
        // Clone so the waveform analyser does not share a MediaStreamAudioSource
        // with MicVAD's own AudioContext (some browsers starve the second reader).
        analyserStreamRef.current = micStreamRef.current.clone();
        const src = audioContextRef.current.createMediaStreamSource(analyserStreamRef.current);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        src.connect(analyserRef.current);
      }

      setVoiceActiveSync(true);
      setVoiceProcessing(false);
      setVoiceStatus(wsConnected === false ? 'Connecting voice channel…' : 'Listening… (speak now)');

      cancelAnimationFrame(waveRafRef.current);

      const abandonSilero = (reason) => {
        console.warn(`[VAD] ${reason} — falling back to energy VAD`);
        if (vadRef.current) {
          try { vadRef.current.pause(); } catch { /* already paused */ }
          try { vadRef.current.destroy(); } catch { /* already destroyed */ }
          vadRef.current = null;
        }
        vadReadyRef.current = false;
        vadFailedRef.current = true;
        vadFailReasonRef.current = reason;
      };

      let useSilero = false;
      if (!vadFailedRef.current) {
        await initSileroVAD(micStreamRef.current);

        if (vadReadyRef.current && vadRef.current) {
          const running = await armMicVad(vadRef.current);
          const ctxState = vadRef.current.audioContext?.state;
          if (!running || ctxState === 'interrupted' || ctxState === 'suspended') {
            abandonSilero(`Silero produced no audio frames (AudioContext/worklet: ${ctxState || 'missing'})`);
          } else {
            reportVadMode('silero', { source: vadSourceRef.current });
            vadRef.current.start();
            if (!frameWatchDoneRef.current) {
              frameWatchDoneRef.current = true;
              const gotFrames = await waitForVadFrames(vadRef.current);
              if (!gotFrames && voiceActiveRef.current) {
                abandonSilero('Silero produced no audio frames (AudioContext/worklet)');
              }
            }
            if (vadReadyRef.current && vadRef.current && voiceActiveRef.current) {
              useSilero = true;
              waveRafRef.current = requestAnimationFrame(animateWave);
            }
          }
        }
      }

      if (useSilero || !voiceActiveRef.current) return;

      // Fallback — MediaRecorder does not need Silero's AudioContext.
      reportVadMode('energy', { reason: vadFailReasonRef.current || 'Silero VAD unavailable' });
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch {}
      }
      const { mime: recMime, ext: recExt } = pickRecorderFormat();
      const mr = new MediaRecorder(micStreamRef.current, recMime ? { mimeType: recMime } : {});
      mediaRecorderRef.current = mr;
      mr._recExt = recExt;
      pendingSendsRef.current = 0;

      mr.ondataavailable = (e) => {
        if (e.data.size === 0) return;
        if (mediaRecorderRef.current !== mr) return;
        if (isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current) return;
        const reader = new FileReader();
        pendingSendsRef.current += 1;
        reader.onload = () => {
          try {
            const b64 = reader.result?.split(',')[1];
            if (b64) {
              sendJson({ type: 'audio_chunk', data: b64, format: recExt });
            }
          } finally {
            pendingSendsRef.current = Math.max(0, pendingSendsRef.current - 1);
          }
        };
        reader.onerror = () => {
          pendingSendsRef.current = Math.max(0, pendingSendsRef.current - 1);
        };
        reader.readAsDataURL(e.data);
      };
      mr.start(250);
      waveRafRef.current = requestAnimationFrame(fallbackVADLoop);

    } catch (err) {
      console.error('[Mic Error]', err);
      recordingActiveRef.current = false;
      if (onError) onError(err);
    }
  }, [animateWave, fallbackVADLoop, initSileroVAD, reportVadMode, sendJson, wsConnected, onError]);

  startVoiceListeningRef.current = startVoiceListening;

  // Stop voice completely
  const stopVoice = useCallback(() => {
    recordingActiveRef.current = false;
    setVoiceActiveSync(false);
    cancelAnimationFrame(waveRafRef.current);

    if (vadRef.current) {
      try { vadRef.current.pause(); } catch {}
    }

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') { try { mr.stop(); } catch {} }

    if (analyserStreamRef.current) {
      analyserStreamRef.current.getTracks().forEach(t => t.stop());
      analyserStreamRef.current = null;
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

    if (vadRef.current) {
      try { vadRef.current.destroy(); } catch {}
      vadRef.current = null;
      vadReadyRef.current = false;
    }
    vadFailedRef.current = false;
    vadFailReasonRef.current = '';
    vadSourceRef.current = null;
    vadModeRef.current = null;
    lastVadDetailRef.current = {};
    frameWatchDoneRef.current = false;
    pendingSendsRef.current = 0;
    setVadMode(null);

    setVoiceSpeaking(false);
    setVoiceProcessing(false);
    setVoiceStatus('');
    setWaveHeights(Array(12).fill(4));
    stopTTS();
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    aiDoneRef.current = false;
  }, []);

  return {
    voiceActive,
    voiceSpeaking,
    voiceProcessing,
    voiceTranscript,
    voiceStatus,
    waveHeights,
    setVoiceSpeaking,
    setVoiceProcessing,
    setVoiceTranscript,
    setVoiceStatus,
    vadMode,
    startVoiceListening,
    stopVoice,
    resendVadMode,
    queueAudioChunk,
    aiStreamingRef,
    audioQueueRef,
    isPlayingRef,
    aiDoneRef,
    startVoiceListeningRef,
  };
}
