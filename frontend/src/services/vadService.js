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
  // v5 default is 512; keep it explicit so the worklet and frame processor agree.
  frameSamples:             512,
};

const VAD_SCRIPT_TIMEOUT_MS = 10000;
const VAD_INIT_TIMEOUT_MS   = 30000;
export const VAD_FRAME_WAIT_MS = 1200;

export function withTimeout(promise, ms, label) {
  let timerId;
  const timeout = new Promise((_, reject) => {
    timerId = setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timerId));
}

function unloadVadGlobals() {
  if (typeof document !== 'undefined') {
    document.querySelectorAll('script[data-vad-asset]').forEach((el) => el.remove());
  }
  try { delete window.vad; } catch { window.vad = undefined; }
  try { delete window.ort; } catch { window.ort = undefined; }
}

export function loadVadScript(url, globalName) {
  if (typeof window !== 'undefined' && window[globalName]) return Promise.resolve();
  return withTimeout(new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url;
    s.dataset.vadAsset = '1';
    s.onload  = () => (window[globalName] ? resolve() : reject(new Error(`${url} loaded but window.${globalName} is undefined`)));
    s.onerror = () => { s.remove(); reject(new Error(`failed to fetch ${url}`)); };
    document.head.appendChild(s);
  }), VAD_SCRIPT_TIMEOUT_MS, `script ${url}`);
}

/** Load ORT + Silero bundles early so MicVAD.new is faster on the mic click. */
export async function preloadVadScripts() {
  for (const src of VAD_SOURCES) {
    try {
      await loadVadScript(src.ortUrl, 'ort');
      await loadVadScript(src.bundleUrl, 'vad');
      if (window.vad?.MicVAD) return src.name;
    } catch (err) {
      console.warn(`[VAD] preload failed for ${src.name}:`, err?.message || err);
      unloadVadGlobals();
    }
  }
  return null;
}

function isAudioContextDead(err) {
  return /AudioContext is |produced no audio frames/i.test(String(err?.message || err || ''));
}

/**
 * Initializes Silero VAD on the provided MediaStream.
 * Tries local assets first, then falls back to CDN.
 */
export async function initializeSileroVAD(stream, callbacks) {
  const failures = [];
  for (const src of VAD_SOURCES) {
    let myvad = null;
    try {
      console.log(`[VAD] Loading Silero VAD from ${src.name}: ${src.bundleUrl}`);
      await loadVadScript(src.ortUrl,    'ort');
      await loadVadScript(src.bundleUrl, 'vad');
      if (!window.vad?.MicVAD) throw new Error('window.vad.MicVAD not found after script load');

      const stats = { vad: null, logged: false };
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
        onFrameProcessed: () => {
          if (stats.vad) {
            stats.vad._framesReceived = (stats.vad._framesReceived || 0) + 1;
          }
          if (!stats.logged) {
            stats.logged = true;
            console.log('[VAD] onFrameProcessed — worklet is delivering frames');
          }
          if (typeof callbacks.onFrameProcessed === 'function') callbacks.onFrameProcessed();
        },
        onSpeechStart: callbacks.onSpeechStart,
        onVADMisfire:  callbacks.onVADMisfire,
        onSpeechEnd:   callbacks.onSpeechEnd,
      });

      myvad = await withTimeout(createVad, VAD_INIT_TIMEOUT_MS, 'MicVAD.new');
      myvad._framesReceived = 0;
      stats.vad = myvad;
      const running = await armMicVad(myvad);
      const ctxState = myvad.audioContext?.state;
      if (!running || (ctxState && ctxState !== 'running')) {
        throw new Error(`AudioContext is ${ctxState || 'missing'} after resume — worklet will never see frames`);
      }
      console.log(`[VAD] Silero VAD initialised from ${src.name}`);
      return { vad: myvad, source: src.name };
    } catch (err) {
      const msg = `${src.name}: ${err?.message || err}`;
      failures.push(msg);
      console.warn(`[VAD] Silero load failed (${msg})`);
      if (myvad) {
        try { myvad.destroy(); } catch { /* ignore */ }
      }
      unloadVadGlobals();
      // A suspended/interrupted context is not a source-asset problem; CDN retry
      // would also sit outside the user-gesture window.
      if (isAudioContextDead(err)) break;
    }
  }

  const failReason = failures.join(' | ');
  throw new Error(`All Silero sources failed: ${failReason}`);
}

/**
 * MicVAD.new() runs after await getUserMedia() + script/ONNX fetches, so its
 * AudioContext is often created outside the user-gesture window and stays
 * "suspended" (or iOS "interrupted"). start() only flips the frame-processor
 * flag — it does not resume the context, so the worklet never sees frames and
 * onSpeechStart never fires (frontend looks "ready", backend never gets audio_chunk).
 *
 * Chrome also skips AudioWorkletNodes that are not in the destination graph;
 * vad-web 0.0.22's worklet path never connects to destination (only the
 * ScriptProcessor fallback does).
 */
export async function armMicVad(myvad) {
  if (!myvad) return false;
  const ctx = myvad.audioContext;
  if (ctx && ctx.state !== 'running') {
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
  return !!(ctx && ctx.state === 'running');
}

/** True if Silero has processed at least one mic frame (worklet is alive). */
export function waitForVadFrames(myvad, timeoutMs = VAD_FRAME_WAIT_MS) {
  return new Promise((resolve) => {
    if (myvad?._framesReceived > 0) {
      resolve(true);
      return;
    }
    const started = Date.now();
    const tick = () => {
      if (myvad?._framesReceived > 0) {
        resolve(true);
        return;
      }
      if (Date.now() - started >= timeoutMs) {
        resolve(false);
        return;
      }
      setTimeout(tick, 50);
    };
    tick();
  });
}
