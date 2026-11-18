// open-body-tracker browser demo: YOLOX person detector + 3D pose model, both on ONNX Runtime Web (WebGPU, WASM fallback).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs';

// Models come from Hugging Face; when served from localhost, ./models/ is tried first.
const HF = 'https://huggingface.co/Yurup/open-body-tracker/resolve/main/';
// float16 weights (190 MB) when the GPU supports shader-f16, otherwise float32 (380 MB).
const POSE_MODELS = { f16: 'obt-h3c-fp16.onnx', f32: 'obt-h3c.onnx' };
const CROP_W = 192, CROP_H = 256, PADDING = 1.25, DET_SIZE = 416, DET_THRESHOLD = 0.3;
const HIPS = [11, 12];
const EDGES = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [15, 17], [15, 19], [17, 18], [16, 20], [16, 22], [20, 21]];
const LEFT = new Set([5, 7, 9, 11, 13, 15, 17, 18, 19]), RIGHT = new Set([6, 8, 10, 12, 14, 16, 20, 21, 22]);
const colour = (a, b) => (LEFT.has(a) && LEFT.has(b) ? '#2b8cea' : RIGHT.has(a) && RIGHT.has(b) ? '#f58c28' : '#c8c8c8');

const $ = (id) => document.getElementById(id);
const status = (text) => { $('status').textContent = text; };
const video = $('video');
let backend = null, pose = null, detector = null, detectorName = null, running = false, generation = 0;

// ------------------------------------------------------------ model loading

async function modelBytes(name) {
  const cache = 'caches' in self ? await caches.open('open-body-tracker-v1') : null;
  const local = ['localhost', '127.0.0.1'].includes(location.hostname);
  for (const url of local ? [`./models/${name}`, HF + name] : [HF + name]) {
    const hit = cache && await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const response = await fetch(url).catch(() => null);
    if (!response || !response.ok) continue;
    const total = Number(response.headers.get('content-length')) || 0;
    const reader = response.body.getReader(), parts = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); loaded += value.length;
      status(`Downloading ${name}: ${(loaded / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ''} MB`);
    }
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const part of parts) { bytes.set(part, offset); offset += part.length; }
    if (cache && url.startsWith('http')) await cache.put(url, new Response(bytes)).catch(() => {});
    return bytes;
  }
  throw new Error(`could not download ${name}`);
}

async function session(name) {
  const bytes = await modelBytes(name);
  status(`Preparing ${name} (${backend})…`);
  return ort.InferenceSession.create(bytes, { executionProviders: [backend], graphOptimizationLevel: 'all' });
}

// ------------------------------------------------------------ person detector (official YOLOX ONNX)

const detCanvas = new OffscreenCanvas(DET_SIZE, DET_SIZE), detCtx = detCanvas.getContext('2d', { willReadFrequently: true });
const grid = [];
for (const stride of [8, 16, 32]) {
  const cells = DET_SIZE / stride;
  for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) grid.push([x, y, stride]);
}

async function detect(width, height) {
  const ratio = Math.min(DET_SIZE / width, DET_SIZE / height);
  detCtx.fillStyle = 'rgb(114,114,114)';
  detCtx.fillRect(0, 0, DET_SIZE, DET_SIZE);
  detCtx.drawImage(video, 0, 0, width * ratio, height * ratio);
  const rgba = detCtx.getImageData(0, 0, DET_SIZE, DET_SIZE).data, plane = DET_SIZE * DET_SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) { // BGR, 0-255, no normalisation
    input[i] = rgba[i * 4 + 2]; input[plane + i] = rgba[i * 4 + 1]; input[2 * plane + i] = rgba[i * 4];
  }
  const feeds = { [detector.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, DET_SIZE, DET_SIZE]) };
  const out = (await detector.run(feeds))[detector.outputNames[0]].data;
  let best = -1, bestScore = DET_THRESHOLD;
  for (let i = 0; i < grid.length; i++) {
    const score = out[i * 85 + 4] * out[i * 85 + 5];
    if (score > bestScore) { bestScore = score; best = i; }
  }
  if (best < 0) return null;
  const [gx, gy, stride] = grid[best], o = best * 85;
  const cx = (out[o] + gx) * stride / ratio, cy = (out[o + 1] + gy) * stride / ratio;
  const w = Math.exp(out[o + 2]) * stride / ratio, h = Math.exp(out[o + 3]) * stride / ratio;
  const left = Math.max(0, cx - w / 2), top = Math.max(0, cy - h / 2);
  const right = Math.min(width, cx + w / 2), bottom = Math.min(height, cy + h / 2);
  return right > left && bottom > top ? [left, top, right - left, bottom - top] : null;
}

// ------------------------------------------------------------ pose model

const cropCanvas = new OffscreenCanvas(CROP_W, CROP_H), cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

async function estimate(box) {
  // Same crop as the Python code: keep aspect, 25% margin, black outside the frame.
  const [x, y, w0, h0] = box;
  const w = Math.max(w0, h0 * CROP_W / CROP_H) * PADDING, h = w * CROP_H / CROP_W;
  const sx = x + w0 / 2 - w / 2, sy = y + h0 / 2 - h / 2;
  cropCtx.fillStyle = '#000';
  cropCtx.fillRect(0, 0, CROP_W, CROP_H);
  cropCtx.setTransform(CROP_W / w, 0, 0, CROP_H / h, -sx * CROP_W / w, -sy * CROP_H / h);
  cropCtx.drawImage(video, 0, 0);
  cropCtx.setTransform(1, 0, 0, 1, 0, 0);
  const rgba = cropCtx.getImageData(0, 0, CROP_W, CROP_H).data, plane = CROP_W * CROP_H;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) { input[i] = rgba[i * 4]; input[plane + i] = rgba[i * 4 + 1]; input[2 * plane + i] = rgba[i * 4 + 2]; }
  const out = await pose.run({ crop: new ort.Tensor('float32', input, [1, 3, CROP_H, CROP_W]) });
  const xy = [], xyz = [], score = Array.from(out.score.data), s = w / CROP_W;
  for (let k = 0; k < 23; k++) {
    xy.push([sx + out.xy.data[k * 2] * s, sy + out.xy.data[k * 2 + 1] * s]);
    xyz.push([out.xyz.data[k * 3], out.xyz.data[k * 3 + 1], out.xyz.data[k * 3 + 2]]);
  }
  const root = [0, 1, 2].map((i) => (xyz[HIPS[0]][i] + xyz[HIPS[1]][i]) / 2);
  return { box, xy, score, xyz: xyz.map((p) => p.map((v, i) => v - root[i])) };
}

// Box from the previous pose: confident points (>= 0.3, at least 8), 15% margin per side.
function boxFromPose(result, width, height) {
  const good = result.xy.filter((_, k) => result.score[k] >= 0.3);
  if (good.length < 8) return null;
  let [l, t, r, b] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const [x, y] of good) { l = Math.min(l, x); t = Math.min(t, y); r = Math.max(r, x); b = Math.max(b, y); }
  const mx = (r - l) * 0.15, my = (b - t) * 0.15;
  l = Math.max(0, l - mx); t = Math.max(0, t - my); r = Math.min(width, r + mx); b = Math.min(height, b + my);
  return Math.min(r - l, b - t) >= 8 ? [l, t, r - l, b - t] : null;
}

// One Euro filter (Casiez et al. 2012) over a flat array.
class OneEuro {
  constructor(minCutoff, beta) { this.minCutoff = minCutoff; this.beta = beta; this.reset(); }
  reset() { this.value = null; }
  static alpha(cutoff, dt) { return 1 / (1 + 1 / (2 * Math.PI * cutoff) / dt); }
  filter(value, t) {
    if (!this.value) { this.value = value.slice(); this.speed = value.map(() => 0); this.t = t; return this.value; }
    const dt = Math.max(t - this.t, 1e-3); this.t = t;
    const ad = OneEuro.alpha(1, dt);
    this.value = this.value.map((prev, i) => {
      this.speed[i] = ad * (value[i] - prev) / dt + (1 - ad) * this.speed[i];
      const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.speed[i]), dt);
      return a * value[i] + (1 - a) * prev;
    });
    return this.value;
  }
}
const filters = { xy: new OneEuro(0.65, 0.006), xyz: new OneEuro(0.65, 0.5) };

// ------------------------------------------------------------ drawing

let yaw = 0.5, pitch = 0.15;
function draw(result, box, source) {
  const ctx = $('image').getContext('2d');
  ctx.drawImage(video, 0, 0, $('image').width, $('image').height);
  const sk = $('skeleton').getContext('2d'), size = $('skeleton').width;
  sk.fillStyle = '#18181b'; sk.fillRect(0, 0, size, size);
  if (!result) return;
  const scale = $('image').width / video.videoWidth;
  ctx.lineWidth = 1; ctx.setLineDash(source === 'tracker' ? [4, 4] : []);
  ctx.strokeStyle = source === 'tracker' ? '#50c850' : '#ffd400';
  ctx.strokeRect(box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale);
  ctx.setLineDash([]); ctx.lineWidth = 3;
  for (const [a, b] of EDGES) {
    if (Math.min(result.score[a], result.score[b]) < 0.3) continue;
    ctx.strokeStyle = colour(a, b); ctx.beginPath();
    ctx.moveTo(result.xy[a][0] * scale, result.xy[a][1] * scale); ctx.lineTo(result.xy[b][0] * scale, result.xy[b][1] * scale); ctx.stroke();
  }
  // 3D: camera frame (x right, y down, z away), rotated by yaw/pitch, 2.4 m across the view.
  const [cy, sy, cp, sp] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch)];
  const project = ([x, y, z]) => {
    const x1 = cy * x - sy * z, z1 = sy * x + cy * z;
    return [size / 2 + x1 * size / 2.4, size / 2 + (cp * y - sp * z1) * size / 2.4];
  };
  sk.strokeStyle = '#3a3a40'; sk.lineWidth = 1; sk.beginPath();
  for (let i = -1; i <= 1; i += 0.25) {  // floor grid 1 m below the hips
    const a = project([i, 1, -1]), b = project([i, 1, 1]), c = project([-1, 1, i]), d = project([1, 1, i]);
    sk.moveTo(...a); sk.lineTo(...b); sk.moveTo(...c); sk.lineTo(...d);
  }
  sk.stroke(); sk.lineWidth = 3;
  for (const [a, b] of EDGES) {
    sk.strokeStyle = colour(a, b); sk.beginPath(); sk.moveTo(...project(result.xyz[a])); sk.lineTo(...project(result.xyz[b])); sk.stroke();
  }
}

let drag = null;
$('skeleton').addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; e.target.setPointerCapture(e.pointerId); });
$('skeleton').addEventListener('pointermove', (e) => {
  if (!drag) return;
  yaw += (e.clientX - drag[0]) * 0.01; pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * 0.01));
  drag = [e.clientX, e.clientY];
});
$('skeleton').addEventListener('pointerup', () => { drag = null; });

// ------------------------------------------------------------ main loop

async function loop(id) {
  let tracked = null, detectedAt = -1e9, frames = 0, detections = 0, last = performance.now(), fps = 0;
  while (running && id === generation) {
    await new Promise((resolve) => video.requestVideoFrameCallback(resolve));
    const now = performance.now(), width = video.videoWidth, height = video.videoHeight;
    let box = tracked, source = 'tracker';
    if (!$('track').checked || !box || now - detectedAt >= 1000) {
      box = await detect(width, height); detectedAt = now; source = 'detector'; detections++;
    }
    let result = box ? await estimate(box) : null;
    tracked = result && $('track').checked ? boxFromPose(result, width, height) : null;
    if (result && $('smooth').checked) {
      const t = now / 1000, xy = filters.xy.filter(result.xy.flat(), t), xyz = filters.xyz.filter(result.xyz.flat(), t);
      result = { ...result, xy: result.xy.map((_, k) => [xy[2 * k], xy[2 * k + 1]]), xyz: result.xyz.map((_, k) => [xyz[3 * k], xyz[3 * k + 1], xyz[3 * k + 2]]) };
    } else { filters.xy.reset(); filters.xyz.reset(); }
    draw(result, box, source);
    window.openBodyTracker = { result, box, source, backend }; // for debugging and tests
    frames++; fps = 0.9 * fps + 0.1 * (1000 / Math.max(1, now - last)); last = now;
    if (frames % 15 === 0) $('stats').textContent = `${fps.toFixed(0)} FPS · ${backend} · detector ran on ${(100 * detections / frames).toFixed(0)}% of frames`;
    if (id === generation) status(result ? 'Running' : 'No person found');
  }
}

async function start() {
  running = false;
  if (!backend) {
    const adapter = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
    backend = adapter ? 'webgpu' : 'wasm';
    pose = await session(adapter?.features.has('shader-f16') ? POSE_MODELS.f16 : POSE_MODELS.f32);
  }
  if (detectorName !== $('detector').value) { detector = await session($('detector').value); detectorName = $('detector').value; }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: $('camera').value ? { exact: $('camera').value } : undefined, width: 640, height: 480 }, audio: false });
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = stream;
  await video.play();
  $('image').width = 640; $('image').height = Math.round(640 * video.videoHeight / video.videoWidth);
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const current = stream.getVideoTracks()[0].getSettings().deviceId;
  $('camera').replaceChildren(...devices.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId, false, d.deviceId === current)));
  filters.xy.reset(); filters.xyz.reset();
  running = true;
  const id = ++generation;
  if (backend === 'wasm') status('WebGPU is not available; running on the CPU (slow).');
  loop(id).catch((error) => { running = false; status(`Error: ${error.message}`); });
}

$('controls').addEventListener('submit', (e) => { e.preventDefault(); start().catch((error) => status(`Error: ${error.message}`)); });
for (const id of ['camera', 'detector']) $(id).addEventListener('change', () => { if (running) start().catch((error) => status(`Error: ${error.message}`)); });
