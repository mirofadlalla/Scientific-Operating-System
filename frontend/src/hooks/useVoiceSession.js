import { useState, useRef, useCallback } from 'react';
import {
  initializeSileroVAD,
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
const ECHO_GUARD_MS = 1500;
const ALLOW_VOICE_BARGE_IN = false;

export function useVoiceSession({ sendJson, onSpeechRecorded, onBargeInInterrupt, onError }) {
  const [voiceActive, setVoiceActive] = useState(false);
  const voiceActiveRef = useRef(false);
  const [voiceSpeaking, setVoiceSpeaking] = useState(false);
  const [voiceProcessing, setVoiceProcessing] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [voiceStatus, setVoiceStatus] = useState('');
  const [waveHeights, setWaveHeights] = useState(Array(12).fill(4));

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
  const bargeInTimerRef = useRef(null);
  const bargeInActiveRef = useRef(false);

  // Fallback VAD & Media refs
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const waveRafRef = useRef(null);
  const micStreamRef = useRef(null);
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
    const b64 = arrayBufferToBase64(wavArrayBuffer);
    sendJson({ type: 'audio_chunk', data: b64, format });
    sendJson({ type: 'audio_end', format });
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
                sendJson({ type: 'audio_end', format: ext });
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
  const reportVadMode = useCallback((mode, detail = {}) => {
    if (vadModeRef.current === mode) return;
    vadModeRef.current = mode;
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

  // Silero VAD initialization
  const initSileroVAD = useCallback((stream) => {
    if (vadReadyRef.current || vadFailedRef.current) return Promise.resolve();
    if (vadInitPromiseRef.current) return vadInitPromiseRef.current;

    vadInitPromiseRef.current = initializeSileroVAD(stream, {
      onSpeechStart: () => {
        const isAIActive = isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current;
        const echoGuardActive = isAIActive || (Date.now() - aiAudioEndTimeRef.current) < ECHO_GUARD_MS;
        echoSpeechRef.current = echoGuardActive && !ALLOW_VOICE_BARGE_IN;

        if (echoSpeechRef.current) {
          // Almost certainly the AI's own voice leaking into the mic: do not touch the
          // UI (no "Listening (speaking)" flicker) and never start a barge-in timer.
          console.log('[Silero VAD] onSpeechStart ignored — AI busy / echo-guard active');
          return;
        }

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

        let wavBuffer;
        try {
          const blob = window.vad.utils.encodeWAV(audioFloat32);
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
        await initSileroVAD(micStreamRef.current);

        if (vadReadyRef.current && vadRef.current) {
          reportVadMode('silero', { source: vadSourceRef.current });
          vadRef.current.start();
          waveRafRef.current = requestAnimationFrame(animateWave);
          return;
        }
      }

      // Fallback
      reportVadMode('energy', { reason: vadFailReasonRef.current || 'Silero VAD unavailable' });
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch {}
      }
      const { mime: recMime, ext: recExt } = pickRecorderFormat();
      const mr = new MediaRecorder(micStreamRef.current, recMime ? { mimeType: recMime } : {});
      mediaRecorderRef.current = mr;
      mr._recExt = recExt;

      mr.ondataavailable = (e) => {
        if (e.data.size === 0) return;
        if (mediaRecorderRef.current !== mr) return;
        if (isPlayingRef.current || audioQueueRef.current.length > 0 || aiStreamingRef.current) return;
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result?.split(',')[1];
          if (b64) {
            sendJson({ type: 'audio_chunk', data: b64, format: recExt });
          }
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
  }, [animateWave, fallbackVADLoop, initSileroVAD, reportVadMode, sendJson, onError]);

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
    startVoiceListening,
    stopVoice,
    queueAudioChunk,
    aiStreamingRef,
    audioQueueRef,
    isPlayingRef,
    aiDoneRef,
    startVoiceListeningRef,
  };
}
