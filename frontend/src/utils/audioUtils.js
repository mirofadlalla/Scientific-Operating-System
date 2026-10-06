/**
 * Audio encoding, format detection, and TTS utility helpers.
 */

import { API_BASE } from '../config';

export const PREFERRED_MIME_TYPES = [
  { mime: 'audio/webm;codecs=opus', ext: 'webm' },
  { mime: 'audio/webm',             ext: 'webm' },
  { mime: 'audio/ogg;codecs=opus',  ext: 'ogg'  },
  { mime: 'audio/ogg',              ext: 'ogg'  },
  { mime: 'audio/mp4',              ext: 'mp4'  },
];

export function pickRecorderFormat() {
  for (const { mime, ext } of PREFERRED_MIME_TYPES) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(mime)) {
      return { mime, ext };
    }
  }
  return { mime: '', ext: 'webm' };
}

/**
 * Energy-based audio metric computation for fallback VAD.
 */
export function computeAudioMetrics(freqData) {
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

/**
 * Converts Float32Array PCM samples to a 16 kHz, mono, 16-bit PCM WAV ArrayBuffer.
 */
export function float32ToWav(samples, sampleRate = 16000) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = numChannels * (bitsPerSample / 8);
  const byteRate   = sampleRate * blockAlign;
  const dataSize   = samples.length * blockAlign;
  const buffer     = new ArrayBuffer(44 + dataSize);
  const view       = new DataView(buffer);
  const writeStr   = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };
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

/**
 * Converts an ArrayBuffer to a base64 encoded string.
 */
export function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

let activeAudioElement = null;

export function stopTTS() {
  if (activeAudioElement) {
    activeAudioElement.pause();
    activeAudioElement = null;
  }
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
}

export function setActiveAudio(audio) {
  activeAudioElement = audio;
}

export function getActiveAudio() {
  return activeAudioElement;
}

/**
 * Synthesizes text to speech using backend endpoint with fallback to browser SpeechSynthesis.
 */
export async function playTextTTS(text, soundEnabled = true) {
  if (!soundEnabled || !text || !text.trim()) return;
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
      const audio = new Audio(url);
      setActiveAudio(audio);
      audio.onended = () => { URL.revokeObjectURL(url); setActiveAudio(null); };
      audio.play().catch((e) => {
        console.warn('Backend TTS autoplay blocked, using SpeechSynthesis fallback:', e);
        if (typeof window !== 'undefined' && window.speechSynthesis) {
          const utt = new SpeechSynthesisUtterance(text.slice(0, 800));
          window.speechSynthesis.speak(utt);
        }
      });
      return;
    }
  } catch (err) {
    console.warn('Backend TTS error, falling back to Web Speech API:', err);
  }
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    const utt = new SpeechSynthesisUtterance(text.slice(0, 800));
    utt.rate = 1.0;
    utt.pitch = 1.0;
    window.speechSynthesis.speak(utt);
  }
}
