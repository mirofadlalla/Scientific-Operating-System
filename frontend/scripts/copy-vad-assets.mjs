// Copies the Silero VAD + onnxruntime-web runtime files into public/vad so the
// browser loads them from your own origin (no CDN dependency).
// Runs automatically before `npm run dev` and `npm run build`.
import { cpSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out  = resolve(root, 'public/vad');
const vad  = resolve(root, 'node_modules/@ricky0123/vad-web/dist');
const ort  = resolve(root, 'node_modules/onnxruntime-web/dist');

const files = [
  [vad, 'bundle.min.js'],
  [vad, 'vad.worklet.bundle.min.js'],
  [vad, 'silero_vad_v5.onnx'],
  [vad, 'silero_vad_legacy.onnx'],
  [ort, 'ort.wasm.min.js'],
  [ort, 'ort-wasm.wasm'],
  [ort, 'ort-wasm-simd.wasm'],
];

mkdirSync(out, { recursive: true });
let missing = 0;
for (const [dir, name] of files) {
  const src = resolve(dir, name);
  if (!existsSync(src)) { console.warn(`[vad-assets] missing ${src}`); missing++; continue; }
  cpSync(src, resolve(out, name));
}
console.log(`[vad-assets] copied ${files.length - missing}/${files.length} files to public/vad`);
