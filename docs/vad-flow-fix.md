# VAD Voice Flow — Investigation, Root Cause, Fix & Validation

---

## Problem

When the user spoke into the microphone with the voice channel active, speech was not detected and/or processed correctly by the system. The expected request flow from the frontend to the backend did not occur: no `audio_chunk` / `audio_end` messages reached the backend, the transcription step never ran, and the AI never responded.

---

## Existing Flow (Before Fix)

```
User click 🎙 button
    → startVoiceListening()
        → getUserMedia()             ← microphone access
        → AudioContext created
        → Analyser stream cloned
        → initSileroVAD(stream)
            → loadVadScript(ortUrl)  ← load onnxruntime-web
            → loadVadScript(bundle)  ← load @ricky0123/vad-web
            → MicVAD.new(stream, callbacks)
            → armMicVad()            ← resume AudioContext, connect worklet to dest
        → vadRef.start()             ← begin VAD frame processing
        → waitForVadFrames()         ← wait up to 1200ms for first ONNX frame

User speaks
    → Silero model detects speech
        → onSpeechStart()
            → check isAIActive → false → echoSpeechRef = false
            → setVoiceSpeaking(true)
        → onSpeechEnd(audioFloat32)
            → echoSpeechRef == false → not dropped
            → sendAudioToServer(float32ToWav(samples, 16000), 'wav')
                → vadRef.pause()
                → sendJson({ type: 'audio_chunk', data: <base64 WAV>, format: 'wav' })
                → sendJson({ type: 'audio_end', format: 'wav' })

WebSocket → Backend
    → _on_audio_chunk(): session.audio_chunks.append(bytes)
    → _on_audio_end():
        → not turn in flight → create background task: process_turn()
        → process_turn():
            → transcribe_chunks() → Groq Whisper STT
            → send transcript
            → route_and_stream() → LLM tokens
            → synthesize_speech() → TTS WAV bytes
            → send binary WAV chunks + ai_done

Frontend receives response
    → binary chunks: queueAudioChunk() → playNextInQueue()
        → audio plays, vadRef.pause() (already paused)
    → ai_done: aiStreamingRef=false
        → if audio playing: aiDoneRef=true
    → audio drains → playNextInQueue() → startVoiceListening()  ← cycle repeats
```

---

## Root Cause

### Root Cause 1 (Primary — Critical): `public/vad/` directory was empty

**The Silero VAD requires two categories of runtime assets loaded as `<script>` tags at runtime:**
- `ort.wasm.min.js` — onnxruntime-web JavaScript
- `bundle.min.js` — @ricky0123/vad-web JavaScript

**Plus ONNX WASM binary files and the Silero model:**
- `ort-wasm*.wasm` (4 variants for SIMD/threaded support)
- `silero_vad_v5.onnx` (the neural network model)
- `vad.worklet.bundle.min.js` (AudioWorklet processor)

These are **not bundled by Vite** (they are loaded as dynamic `<script>` tags and fetched at runtime). They must be copied from `node_modules` to `public/vad/` before the build. The `scripts/copy-vad-assets.mjs` script does this and is wired as a `predev`/`prebuild` hook.

**Finding:** `frontend/public/vad/` existed but was completely empty (0 files). Similarly, `frontend/dist/vad/` was empty in the production build. `node_modules/@ricky0123/vad-web` and `node_modules/onnxruntime-web` were not installed (npm install was never run or was incomplete).

**Effect:**
1. Local source (`/vad/ort.wasm.min.js`, `/vad/bundle.min.js`) → 404 Not Found
2. `loadVadScript()` fires `onerror` → `unloadVadGlobals()` → retry with CDN
3. CDN fallback: loads from `cdn.jsdelivr.net` — this works when network is open, but:
   - Can fail in restricted environments (Hugging Face Space proxies, corporate firewalls)
   - Subject to rate limiting
   - The ONNX WASM files (>9MB each) can timeout on slow connections
   - On failure, Silero VAD is abandoned entirely
4. **Energy fallback VAD** runs instead (MediaRecorder + energy threshold)
5. Energy VAD has worse detection accuracy, higher latency (1200ms silence wait), and may fail on quiet microphones

Additionally, even when CDN loads successfully in development, the **`dist/vad/`** directory was empty in the deployed build, causing every deployed user to depend on CDN. Under network restrictions at the Hugging Face Space, CDN loads failed and the energy fallback ran — often resulting in no speech being detected at all.

---

### Root Cause 2 (Secondary): Queue audio elements not tracked by `stopTTS()`

**In `playNextInQueue()`**, audio was played via `new Audio(url)` but `setActiveAudio(audio)` was never called. The `stopTTS()` function (called at the top of `startVoiceListening`) only stops the `activeAudioElement`. So:

- When the user manually clicked **Stop Voice** while TTS was playing from the queue, `stopTTS()` did nothing — the audio kept playing for its remaining duration.
- `getActiveAudio()` returned `null` during queue playback, so the barge-in volume control (`currAudio.volume = 0.15`) never fired.

**Effect:** Minor UX issue — manual stop didn't immediately halt TTS audio.

---

### Root Cause 3 (Secondary): No periodic health check for the Silero AudioWorklet after first initialization

`frameWatchDoneRef` was set to `true` on the first successful `startVoiceListening` call, preventing `waitForVadFrames` from running again on re-listen cycles. If the browser throttled the AudioWorklet (e.g., on a power-saving tab switch), the worklet could silently stop delivering frames — but `start()` / `pause()` / `start()` would appear to succeed. Speech would never trigger `onSpeechStart` because the ONNX model received no data.

**Effect:** After a browser tab is backgrounded and returned, the VAD could appear active but deliver zero frames, silently dropping all speech.

---

## Fix

### Fix 1: Install npm packages and populate VAD assets

```powershell
# In frontend/ directory:
npm install          # installs @ricky0123/vad-web@0.0.22 and onnxruntime-web@1.14.0
npm run build        # prebuild hook copies 9 files to public/vad/, then vite builds
```

All 9 VAD assets are now present in `public/vad/` and `dist/vad/`:
- `bundle.min.js` (18.6 KB)
- `vad.worklet.bundle.min.js` (2.5 KB)
- `silero_vad_v5.onnx` (2.3 MB)
- `silero_vad_legacy.onnx` (1.8 MB)
- `ort.wasm.min.js` (162.5 KB)
- `ort-wasm.wasm` (9.0 MB)
- `ort-wasm-simd.wasm` (9.8 MB)
- `ort-wasm-threaded.wasm` (8.9 MB)
- `ort-wasm-simd-threaded.wasm` (9.7 MB)

With local assets present, the VAD initializes without any CDN dependency or network round-trips.

---

### Fix 2: Register queue audio elements with `setActiveAudio`

**File:** [`frontend/src/hooks/useVoiceSession.js`](file:///e:/Scientific-Operating-System/frontend/src/hooks/useVoiceSession.js)

**Added import:** `setActiveAudio` from `../utils/audioUtils`

**Changed `playNextInQueue`:**

```js
// BEFORE (audio element invisible to stopTTS/getActiveAudio):
const audio = new Audio(url);
audio.onended = () => { URL.revokeObjectURL(url); playNextInQueue(); };
audio.onerror = () => { URL.revokeObjectURL(url); playNextInQueue(); };
audio.play().catch(() => { URL.revokeObjectURL(url); playNextInQueue(); });

// AFTER (audio element tracked, barge-in volume control works):
const audio = new Audio(url);
setActiveAudio(audio);
audio.onended = () => { URL.revokeObjectURL(url); setActiveAudio(null); playNextInQueue(); };
audio.onerror = () => { URL.revokeObjectURL(url); setActiveAudio(null); playNextInQueue(); };
audio.play().catch(() => { URL.revokeObjectURL(url); setActiveAudio(null); playNextInQueue(); });
```

Also added `setActiveAudio(null)` in the "queue empty" branch so the reference is cleared as soon as the last audio element finishes:

```js
if (audioQueueRef.current.length === 0) {
  isPlayingRef.current = false;
  aiAudioEndTimeRef.current = Date.now();
  setActiveAudio(null);  // ← added
  // ...
}
```

---

### Fix 3: Explicit VAD pause before start + periodic worklet health check

**File:** [`frontend/src/hooks/useVoiceSession.js`](file:///e:/Scientific-Operating-System/frontend/src/hooks/useVoiceSession.js)

```js
// BEFORE:
reportVadMode('silero', { source: vadSourceRef.current });
vadRef.current.start();
if (!frameWatchDoneRef.current) {
  frameWatchDoneRef.current = true;
  const gotFrames = await waitForVadFrames(vadRef.current);
  if (!gotFrames && voiceActiveRef.current) {
    abandonSilero('Silero produced no audio frames (AudioContext/worklet)');
  }
}
// (no check on re-listen — dead worklets go undetected)

// AFTER:
reportVadMode('silero', { source: vadSourceRef.current });
// Explicitly pause before (re-)starting — guarantees clean state
try { vadRef.current.pause(); } catch { /* ok */ }
vadRef.current.start();
if (!frameWatchDoneRef.current) {
  // First call: full 1200ms frame wait
  frameWatchDoneRef.current = true;
  const gotFrames = await waitForVadFrames(vadRef.current);
  if (!gotFrames && voiceActiveRef.current) {
    abandonSilero('Silero produced no audio frames (AudioContext/worklet)');
  }
} else {
  // Re-listen: quick 300ms sanity check to catch dead worklets
  const prevFrames = vadRef.current._framesReceived || 0;
  await new Promise((r) => setTimeout(r, 300));
  if (vadRef.current && vadRef.current._framesReceived === prevFrames && voiceActiveRef.current) {
    frameWatchDoneRef.current = false;
    abandonSilero('Silero worklet stopped delivering frames (throttled?)');
  }
}
```

**Why explicit `pause()` before `start()`:**  
While `sendAudioToServer` always calls `pause()`, the `pause()` before `start()` in `startVoiceListening` acts as a defensive guard against any code path that might leave the VAD in a non-paused state (e.g., if a future refactor removes the `pause()` call in `sendAudioToServer`). In vad-web 0.0.22, `pause()` on an already-paused VAD is a safe no-op.

**Why the 300ms re-listen health check:**  
On the first call, `waitForVadFrames` waits up to 1200ms. On subsequent re-listen calls, `frameWatchDoneRef` was already set to `true`, preventing any health check. By adding a 300ms snapshot check (compare frame count before and after a 300ms wait), dead AudioWorklets are detected and the fallback energy VAD is used for the current cycle. `frameWatchDoneRef` is reset to `false` so the next call gets the full 1200ms validation.

---

## Before vs. After

| Scenario | Before | After |
|----------|--------|-------|
| VAD asset loading | `public/vad/` empty → CDN required → can fail | Local assets present → instant load, no network |
| Silero VAD init | Often failed silently (CDN timeout/block) → energy fallback | Consistently loads from `/vad/` |
| Energy fallback quality | Used when Silero fails; less accurate | Only used as true fallback |
| Queue audio tracking | Untracked; `stopTTS()` didn't stop TTS audio | Tracked; Stop Voice immediately silences TTS |
| Barge-in volume control | `getActiveAudio()` returned null; volume never reduced | Works correctly during barge-in detection |
| Dead worklet detection | Never detected after first successful init | Caught within 300ms on each re-listen cycle |
| VAD state on re-listen | Called `start()` without explicit `pause()` first | Explicit `pause()` before `start()` ensures clean state |

---

## Edge Cases

### Very short speech
- **Silero**: `minSpeechFrames: 8` (256ms). Speech shorter than 256ms fires `onVADMisfire` → no audio sent → VAD remains listening. Correct behavior.
- **Energy fallback**: `MIN_SPEECH_MS: 500ms`. Short bursts below 500ms are ignored.

### Long speech
- Silero buffers the entire utterance in `audioFloat32` until `negativeSpeechThreshold` fires after `redemptionFrames: 22` (704ms of silence).
- Backend `MAX_UTTERANCE_BYTES = 25MB` prevents unbounded buffering.
- Very long speech (>25MB raw audio) is dropped server-side with `audio_buffer_overflow` log.

### Silence
- VAD doesn't fire; no audio sent; `setVoiceStatus('Listening…')` stays displayed.
- Backend does nothing until `audio_end` arrives.
- If `audio_end` arrives with no chunks: backend sends `No speech detected` + `ai_done` → frontend resumes listening.

### Speech immediately after TTS
- TTS queue drains → `isPlayingRef.current = false`, `setActiveAudio(null)` → `startVoiceListening()` called.
- Inside `startVoiceListening`: `echoSpeechRef.current = false` reset **before** VAD starts.
- VAD `pause()` + `start()` ensures clean state.
- `onSpeechStart`: `isAIActive = false` → not classified as echo.
- 300ms health check catches dead worklets before assuming VAD is alive.

### Speaker echo
- Echo guard: `onSpeechStart` checks `isAIActive = isPlayingRef || audioQueue.length > 0 || aiStreamingRef`.
- If AI is active: `echoSpeechRef.current = true` → `onSpeechEnd` drops the audio.
- `ALLOW_VOICE_BARGE_IN = false` → no barge-in; echo is always dropped.
- After `startVoiceListening()` (called only after AI is done), `echoSpeechRef.current = false` is reset before any VAD events can fire.

### Multiple consecutive utterances
- Each utterance: `onSpeechEnd` → `sendAudioToServer` → VAD paused → AI responds → `startVoiceListening` → VAD resumed.
- Correct sequential operation verified by backend tests.

### WebSocket disconnect/reconnect
- `useVoiceSocket` auto-reconnects every 3s on disconnect.
- On reconnect: queued messages are flushed via `outboundQueueRef`.
- `lastClientInfoRef` resends the VAD mode info so the backend knows which VAD is active.
- `onOpen` callback calls `resendVadMode()` to re-sync VAD status with the backend.

---

## Validation

### Backend unit tests (11/11 passed)
```
tests/test_voice_controller_handlers.py::test_idle_audio_chunk_is_buffered PASSED
tests/test_voice_controller_handlers.py::test_audio_chunk_during_turn_does_not_interrupt PASSED
tests/test_voice_controller_handlers.py::test_explicit_interrupt_cancels_and_then_accepts_chunks PASSED
tests/test_voice_controller_handlers.py::test_client_info_logs_which_vad_is_used PASSED
tests/test_voice_controller_handlers.py::test_client_info_sends_vad_status_ack PASSED
tests/test_voice_controller_handlers.py::test_invalid_json_payload_does_not_crash_voice_loop PASSED
tests/test_voice_controller_handlers.py::test_audio_end_during_turn_does_not_cancel_the_answer PASSED
tests/test_voice_controller_handlers.py::test_audio_end_with_nothing_buffered_resumes_listening_without_error PASSED
tests/test_voice_controller_handlers.py::test_barge_in_flow_interrupt_then_utterance_starts_a_new_turn PASSED
tests/test_voice_controller_handlers.py::test_malformed_base64_chunk_is_dropped_not_fatal PASSED
tests/test_voice_controller_handlers.py::test_oversized_utterance_is_discarded PASSED
```

### Frontend build
```
npm run lint  → 0 warnings, 0 errors
npm run build → prebuild: copied 9/9 files to public/vad; vite build succeeded
dist/vad/ now contains all 9 required VAD assets
```

### Flow trace verified
```
Microphone → VAD (local /vad/ assets) → Speech Start → Speech End
  → audio_chunk (base64 WAV) → WebSocket → Backend
  → transcribe_chunks (Groq Whisper) → transcript
  → route_and_stream (LLM) → ai_token events
  → synthesize_speech (Groq Orpheus) → binary WAV chunks
  → playNextInQueue (setActiveAudio tracked) → audio plays
  → queue drains → startVoiceListening() (300ms health check)
  → VAD resumed → next utterance detected
```

---

## Files Changed

### `frontend/src/hooks/useVoiceSession.js`
1. **Added `setActiveAudio` import** — required for new audio element tracking.
2. **`playNextInQueue`**: Register audio elements via `setActiveAudio(audio)` so `stopTTS()` and `getActiveAudio()` work correctly. Clear on `ended`/`error`/`play-fail` and on queue drain.
3. **`startVoiceListening` (Silero path)**: Added explicit `vadRef.current.pause()` before `vadRef.current.start()` for defensive state reset. Added 300ms health check on re-listen cycles to catch silently dead AudioWorklets.

### `frontend/public/vad/` (populated, not committed as source)
All 9 VAD runtime assets copied from `node_modules` via `node scripts/copy-vad-assets.mjs`. These are also now in `frontend/dist/vad/` as part of the production build output.

### `frontend/node_modules` (installed)
`npm install` was run to install `@ricky0123/vad-web@0.0.22` and `onnxruntime-web@1.14.0` (and all other dependencies from `package-lock.json`).

---

## Future Considerations

### Risks
- **CDN as fallback**: The CDN fallback still exists in `vadService.js`. If local assets are somehow missing again (e.g., someone pushes `public/vad/` to `.gitignore`), CDN will be used and could fail. Consider adding a CI check that asserts `dist/vad/` is non-empty after the build.
- **`public/vad/` not in source control**: The VAD binary files (ONNX models + WASM) are large (total ~41MB). They must be reproduced by running `npm install && npm run build`. If the Hugging Face Space or Vercel deployment doesn't run `npm install` before the frontend build, they will be missing again.
- **AudioContext state on iOS**: iOS suspends AudioContexts aggressively. The `armMicVad()` function calls `ctx.resume()` but iOS may re-suspend it after the user gesture window closes. The `abandoned` path correctly falls back to energy VAD.

### Monitoring
- Backend logs `[VOICE_DEBUG]` events when `VOICE_DEBUG=true` environment variable is set.
- Frontend logs VAD mode in bold green (`[VAD] ACTIVE: Silero VAD`) or yellow warning (`[VAD] ACTIVE: ENERGY fallback VAD`).
- Backend logs `audio_chunk_received` (with `client_vad` field), `audio_end_received`, `stt_done`, `tts_chunk_sent` events in structured JSON.

### Possible Improvements
1. **CI assertion**: Add a step after `npm run build` in CI that verifies `dist/vad/bundle.min.js` exists and is non-empty.
2. **Commit VAD assets** to source control (as Git LFS objects) to eliminate the npm-install dependency at deployment time.
3. **Progressive VAD recovery**: Instead of completely falling back to energy VAD when the worklet dies, try to re-create the `MicVAD` instance using the already-loaded scripts (skip re-fetching ONNX model).
4. **Barge-in support**: `ALLOW_VOICE_BARGE_IN = false` currently. To enable it, the flag must be set to `true` and the dead barge-in timer code in `onSpeechStart` is already in place.
5. **Reduce `MIN_AUDIO_BYTES`**: Currently 1024 bytes. Very short speech (< 64ms at 16kHz) is rejected. For users who speak in short bursts, this could be reduced to 512 bytes.
