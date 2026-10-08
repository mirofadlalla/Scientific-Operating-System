/**
 * Silero VAD script loading and initialization service.
 * Supports self-hosted assets first, falling back to jsDelivr CDN.
 */

export const VAD_VERSION    = '0.0.22';
export const ONNX_VERSION   = '1.14.0';
export const VAD_MODEL      = 'v5';

const LOCAL_VAD_BASE = `${import.meta.env.BASE_URL}vad/`;
export const VAD_SOURCES = [
  {
    name:      'local',
    ortUrl:    `${LOCAL_VAD_BASE}ort.wasm.min.js`,
    bundleUrl: `${LOCAL_VAD_BASE}bundle.min.js`,
    assetBase: LOCAL_VAD_BASE,
    wasmBase:  LOCAL_VAD_BASE,
  },
  {
    name:      'cdn',
    ortUrl:    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_VERSION}/dist/ort.wasm.min.js`,
    bundleUrl: `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${VAD_VERSION}/dist/bundle.min.js`,
    assetBase: `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${VAD_VERSION}/dist/`,
    wasmBase:  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_VERSION}/dist/`,
  },
];

export const VAD_CONFIG = {
  positiveSpeechThreshold:  0.6,
  negativeSpeechThreshold:  0.35,
  redemptionFrames:         22,
  preSpeechPadFrames:       9,
  minSpeechFrames:          8,
};

const VAD_SCRIPT_TIMEOUT_MS = 10000;
const VAD_INIT_TIMEOUT_MS   = 30000;

export function withTimeout(promise, ms, label) {
  let timerId;
  const timeout = new Promise((_, reject) => {
    timerId = setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timerId));
}

export function loadVadScript(url, globalName) {
  if (typeof window !== 'undefined' && window[globalName]) return Promise.resolve();
  return withTimeout(new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url;
    s.onload  = () => (window[globalName] ? resolve() : reject(new Error(`${url} loaded but window.${globalName} is undefined`)));
    s.onerror = () => { s.remove(); reject(new Error(`failed to fetch ${url}`)); };
    document.head.appendChild(s);
  }), VAD_SCRIPT_TIMEOUT_MS, `script ${url}`);
}

/**
 * Initializes Silero VAD on the provided MediaStream.
 * Tries local assets first, then falls back to CDN.
 */
export async function initializeSileroVAD(stream, callbacks) {
  const failures = [];
  for (const src of VAD_SOURCES) {
    try {
      console.log(`[VAD] Loading Silero VAD from ${src.name}: ${src.bundleUrl}`);
      await loadVadScript(src.ortUrl,    'ort');
      await loadVadScript(src.bundleUrl, 'vad');
      if (!window.vad?.MicVAD) throw new Error('window.vad.MicVAD not found after script load');

      const createVad = window.vad.MicVAD.new({
        stream,
        ...VAD_CONFIG,
        model: VAD_MODEL,
        baseAssetPath:    src.assetBase,
        onnxWASMBasePath: src.wasmBase,
        // Without COOP/COEP, onnxruntime's default thread count fails to allocate
        // the Silero session and MicVAD.new throws (or hangs until our timeout).
        ortConfig: (ort) => {
          try {
            if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) {
              ort.env.wasm.numThreads = 1;
            }
          } catch { /* keep library defaults */ }
        },
        onSpeechStart: callbacks.onSpeechStart,
        onVADMisfire:  callbacks.onVADMisfire,
        onSpeechEnd:   callbacks.onSpeechEnd,
      });

      const myvad = await withTimeout(createVad, VAD_INIT_TIMEOUT_MS, 'MicVAD.new');
      await armMicVad(myvad);
      console.log(`[VAD] Silero VAD initialised from ${src.name}`);
      return { vad: myvad, source: src.name };
    } catch (err) {
      const msg = `${src.name}: ${err?.message || err}`;
      failures.push(msg);
      console.warn(`[VAD] Silero load failed (${msg})`);
    }
  }

  const failReason = failures.join(' | ');
  throw new Error(`All Silero sources failed: ${failReason}`);
}

/**
 * MicVAD.new() runs after await getUserMedia(), so its AudioContext is often
 * created outside the user-gesture window and stays "suspended". start() only
 * flips the frame-processor flag — it does not resume the context, so the
 * worklet never sees frames and onSpeechStart never fires.
 *
 * Chrome also skips AudioWorkletNodes that are not in the destination graph;
 * vad-web 0.0.22's worklet path never connects to destination (only the
 * ScriptProcessor fallback does).
 */
export async function armMicVad(myvad) {
  if (!myvad) return;
  const ctx = myvad.audioContext;
  if (ctx && ctx.state === 'suspended') {
    try {
      await ctx.resume();
    } catch (err) {
      console.warn('[VAD] AudioContext.resume() failed', err);
    }
  }
  const node = myvad.audioNodeVAD?.audioNode;
  if (ctx && node && !myvad._destinationArmed) {
    try {
      const mute = ctx.createGain();
      mute.gain.value = 0;
      node.connect(mute);
      mute.connect(ctx.destination);
      myvad._destinationArmed = true;
    } catch (err) {
      console.warn('[VAD] worklet destination connect failed', err);
    }
  }
}
