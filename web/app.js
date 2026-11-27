// open-body-tracker browser demo: YOLOX person detector + 3D pose model, both on ONNX Runtime Web (WebGPU, WASM fallback).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs';
import { flipDecode, flipOutputs, mirrorBatch } from './flip.js';
import { BodyAnchor } from './anchor.js';

// Models come from Hugging Face; when served from localhost, ./models/ is tried first.
const HF = 'https://huggingface.co/Yurup/open-body-tracker/resolve/main/';
// float16 weights when the GPU supports shader-f16, otherwise float32. See docs/MODELS.md.
// Named after the release tag (docs/MODELS.md); OLD_NAMES maps saved settings. v0.2.0 has the same weights as v0.3.0.
// INT8 models run on the CPU (WASM) only; `int8` names the one used automatically when WebGPU is not available.
// ray: v0.4.0 models give 3D in the crop's ray frame (turned back to camera axes in estimate()) and flip-test by
// averaging their decoded outputs.
const POSE_MODELS = {
  'v0.4.0-small': { f16: 'obt-body-v0.4.0-small-fp16.onnx', f32: 'obt-body-v0.4.0-small.onnx', ray: true, int8: 'v0.4.0-small-int8' },
  'v0.4.0': { f16: 'obt-body-v0.4.0-fp16.onnx', f32: 'obt-body-v0.4.0.onnx', ray: true, int8: 'v0.4.0-int8' },
  'v0.4.0-small-int8': { f16: 'obt-body-v0.4.0-small-int8.onnx', f32: 'obt-body-v0.4.0-small-int8.onnx', ray: true, cpu: true },
  'v0.4.0-int8': { f16: 'obt-body-v0.4.0-int8.onnx', f32: 'obt-body-v0.4.0-int8.onnx', ray: true, cpu: true },
  'v0.3.0': { f16: 'obt-body-v0.3.0-fp16.onnx', f32: 'obt-body-v0.3.0.onnx', int8: 'v0.3.0-int8' },
  'v0.3.0-int8': { f16: 'obt-body-v0.3.0-int8.onnx', f32: 'obt-body-v0.3.0-int8.onnx', cpu: true },
  'v0.1.0': { f16: 'obt-body-v0.1.0-fp16.onnx', f32: 'obt-body-v0.1.0.onnx' },
};
const OLD_NAMES = { v2: 'v0.3.0', v1: 'v0.1.0' };
// Camera focal length assumed when turning ray-frame 3D into camera axes: a 65-degree horizontal field of view
// (src/open_body_tracker/geometry.py).
const DEFAULT_FOV = 65;
// WASM uses several threads only in a cross-origin isolated page (coi-serviceworker.js sets that up on GitHub Pages).
const cpuThreads = () => (self.crossOriginIsolated ? `${ort.env.wasm.numThreads || 'auto'} threads` : '1 thread');

const CROP_W = 192, CROP_H = 256, PADDING = 1.25, DET_SIZE = 416;
const HIPS = [11, 12];
// Joint 23 is drawn only: the neck, midway between the shoulders.
const NECK = 23;
const EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [0, NECK], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [15, 17], [15, 19], [17, 18], [16, 20], [16, 22], [20, 21]];
const LEFT = new Set([1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19]), RIGHT = new Set([2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22]);
const colour = (a, b) => (LEFT.has(a) && LEFT.has(b) ? '#2b8cea' : RIGHT.has(a) && RIGHT.has(b) ? '#f58c28' : '#c8c8c8');

const $ = (id) => document.getElementById(id);
const status = (text) => { $('status').textContent = text; };
const video = $('video');
// Live settings (read every frame), remembered in this browser.
const SETTINGS = ['model', 'flip', 'detector', 'schedule', 'personThreshold', 'jointThreshold', 'smoothing', 'origin', 'view'];
const setting = (id) => $(id).value, threshold = (id) => Number($(id).value);
let backend = null, pose = null, poseName = null, poseBackend = null, poseRay = false, f16 = false, detector = null, detectorName = null, running = false;

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

async function session(name, provider = backend) {
  const bytes = await modelBytes(name);
  status(`Preparing ${name} (${provider})…`);
  return ort.InferenceSession.create(bytes, { executionProviders: [provider], graphOptimizationLevel: 'all' });
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
  let best = -1, bestScore = threshold('personThreshold');
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
  const out = setting('flip') === 'on' ? await flipTest(input, plane) : await plain(input);
  const xy = [], xyz = [], score = Array.from(out.score), s = w / CROP_W;
  const turn = poseRay ? rayToCamera(sx + 96 * s, sy + 128 * s) : null;
  for (let k = 0; k < 23; k++) {
    xy.push([sx + out.xy[k * 2] * s, sy + out.xy[k * 2 + 1] * s]);
    const p = [out.xyz[k * 3], out.xyz[k * 3 + 1], out.xyz[k * 3 + 2]];
    xyz.push(turn ? [0, 1, 2].map((j) => p[0] * turn[0][j] + p[1] * turn[1][j] + p[2] * turn[2][j]) : p);
  }
  const root = [0, 1, 2].map((i) => (xyz[HIPS[0]][i] + xyz[HIPS[1]][i]) / 2);
  return { box, xy, score, xyz: xyz.map((p) => p.map((v, i) => v - root[i])) };
}

// Ray frame -> camera axes (src/open_body_tracker/geometry.py): R turns the ray through the crop centre (u, v) onto +z
// by the smallest rotation; a ray-frame point p (row vector) becomes p R in camera axes.
function rayToCamera(u, v) {
  const f = video.videoWidth / 2 / Math.tan(DEFAULT_FOV * Math.PI / 360);
  let r = [(u - video.videoWidth / 2) / f, (v - video.videoHeight / 2) / f, 1];
  const n = Math.hypot(...r); r = r.map((c) => c / n);
  const a = [r[1], -r[0], 0], sin = Math.hypot(a[0], a[1]), cos = r[2];  // a = r x z
  if (sin < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const K = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], k = (1 - cos) / (sin * sin);
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + K[i][j] + k * [0, 1, 2].reduce((t, m) => t + K[i][m] * K[m][j], 0)));
}

async function plain(input) {
  const out = await pose.run({ crop: new ort.Tensor('float32', input, [1, 3, CROP_H, CROP_W]) });
  return { xy: out.xy.data, score: out.score.data, xyz: out.xyz.data };
}

// Flip test: the crop and its mirror in one batch, averaged (web/flip.js). About 2x slower.
async function flipTest(input, plane) {
  const out = await pose.run({ crop: new ort.Tensor('float32', mirrorBatch(input, plane, CROP_W, CROP_H), [2, 3, CROP_H, CROP_W]) });
  return poseRay ? flipOutputs(out, CROP_W) : flipDecode(out);
}

// Box from the previous pose: points scoring at least TRACK_SCORE (at least 8), 15% margin per side; as BoxTracker in
// pipeline.py. Fixed, unlike the display threshold: a stricter box (score 3, 5 points) was not steadier in replays and
// called the person detector about twice as often.
const TRACK_SCORE = 0.3;
function boxFromPose(result, width, height) {
  const good = result.xy.filter((_, k) => result.score[k] >= TRACK_SCORE);
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
// [min cutoff (Hz), beta for 2D (per px/s), beta for 3D (per m/s)]: lower = steadier, more delay.
const SMOOTHING = { light: [1.5, 0.01, 1.0], medium: [0.65, 0.006, 0.5], strong: [0.25, 0.003, 0.25] };
const filters = { xy: new OneEuro(0.65, 0.006), xyz: new OneEuro(0.65, 0.5) };
const anchor = new BodyAnchor();
function setSmoothing(level) {
  if (!SMOOTHING[level]) return;
  const [minCutoff, beta2d, beta3d] = SMOOTHING[level];
  Object.assign(filters.xy, { minCutoff, beta: beta2d });
  Object.assign(filters.xyz, { minCutoff, beta: beta3d });
}

// ------------------------------------------------------------ drawing

let yaw = 0.5, pitch = 0.15;
const mirrored = () => setting('view') === 'mirror';
function draw(result, box, source) {
  const ctx = $('image').getContext('2d');
  ctx.drawImage(video, 0, 0, $('image').width, $('image').height);
  const sk = $('skeleton').getContext('2d'), size = $('skeleton').width;
  sk.fillStyle = '#18181b'; sk.fillRect(0, 0, size, size);
  if (!result) return;
  const mid = (p) => p[5].map((v, i) => (v + p[6][i]) / 2);
  result = { ...result, xy: [...result.xy, mid(result.xy)], xyz: [...result.xyz, mid(result.xyz)],
    score: [...result.score, Math.min(result.score[5], result.score[6])] };
  const scale = $('image').width / video.videoWidth;
  ctx.lineWidth = 1; ctx.setLineDash(source === 'tracker' ? [4, 4] : []);
  ctx.strokeStyle = source === 'tracker' ? '#50c850' : '#ffd400';
  ctx.strokeRect(box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale);
  ctx.setLineDash([]); ctx.lineWidth = 3;
  const confident = (a, b) => Math.min(result.score[a], result.score[b]) >= threshold('jointThreshold');
  for (const [a, b] of EDGES) {
    if (!confident(a, b)) continue;
    ctx.strokeStyle = colour(a, b); ctx.beginPath();
    ctx.moveTo(result.xy[a][0] * scale, result.xy[a][1] * scale); ctx.lineTo(result.xy[b][0] * scale, result.xy[b][1] * scale); ctx.stroke();
  }
  for (let k = 0; k < 5; k++) {  // face points: nose, eyes, ears
    if (result.score[k] < threshold('jointThreshold')) continue;
    ctx.fillStyle = colour(k, k); ctx.beginPath(); ctx.arc(result.xy[k][0] * scale, result.xy[k][1] * scale, 3.5, 0, 2 * Math.PI); ctx.fill();
  }
  // 3D: camera frame (x right, y down, z away), rotated by yaw/pitch, 2.4 m across the view; mirrored like the image.
  const [cy, sy, cp, sp] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch)], m = mirrored() ? -1 : 1;
  const project = ([x, y, z]) => {
    const x1 = cy * x - sy * z, z1 = sy * x + cy * z;
    return [size / 2 + m * x1 * size / 2.4, size / 2 + (cp * y - sp * z1) * size / 2.4];
  };
  sk.strokeStyle = '#3a3a40'; sk.lineWidth = 1; sk.beginPath();
  for (let i = -1; i <= 1; i += 0.25) {  // floor grid 1 m below the hips
    const a = project([i, 1, -1]), b = project([i, 1, 1]), c = project([-1, 1, i]), d = project([1, 1, i]);
    sk.moveTo(...a); sk.lineTo(...b); sk.moveTo(...c); sk.lineTo(...d);
  }
  sk.stroke(); sk.lineWidth = 3;
  for (const [a, b] of EDGES) {  // 3D joints are always predicted; uncertain ones are drawn faint
    sk.globalAlpha = confident(a, b) ? 1 : 0.25;
    sk.strokeStyle = colour(a, b); sk.beginPath(); sk.moveTo(...project(result.xyz[a])); sk.lineTo(...project(result.xyz[b])); sk.stroke();
  }
  sk.globalAlpha = 1;
}

let drag = null;
$('skeleton').addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; e.target.setPointerCapture(e.pointerId); });
$('skeleton').addEventListener('pointermove', (e) => {
  if (!drag) return;
  yaw += (mirrored() ? -1 : 1) * (e.clientX - drag[0]) * 0.01; pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * 0.01));
  drag = [e.clientX, e.clientY];
});
$('skeleton').addEventListener('pointerup', () => { drag = null; });

// ------------------------------------------------------------ main loop

// ONNX Runtime Web must not run two inferences at once, so a running loop is always halted (and awaited) before the
// camera, detector or models change.
let current = null;  // { halt(), done }

async function loop(halted) {
  let tracked = null, detectedAt = -1e9, frames = 0, detections = 0, last = performance.now(), fps = 0;
  for (;;) {
    const frame = await Promise.race([new Promise((resolve) => video.requestVideoFrameCallback(() => resolve(true))), halted]);
    if (!frame) return;
    const now = performance.now(), width = video.videoWidth, height = video.videoHeight;
    let box = tracked, source = 'tracker';
    if (!box || now - detectedAt >= threshold('schedule')) {
      box = await detect(width, height); detectedAt = now; source = 'detector'; detections++;
    }
    let result = box ? await estimate(box) : null;
    tracked = result ? boxFromPose(result, width, height) : null;
    if (result && setting('origin') === 'visible') result = { ...result, xyz: anchor.apply(result.xyz, result.score, now / 1000) };
    else anchor.reset();
    if (result && setting('smoothing') !== 'off') {
      const t = now / 1000, xy = filters.xy.filter(result.xy.flat(), t), xyz = filters.xyz.filter(result.xyz.flat(), t);
      result = { ...result, xy: result.xy.map((_, k) => [xy[2 * k], xy[2 * k + 1]]), xyz: result.xyz.map((_, k) => [xyz[3 * k], xyz[3 * k + 1], xyz[3 * k + 2]]) };
    } else { filters.xy.reset(); filters.xyz.reset(); }
    draw(result, box, source);
    window.openBodyTracker = { result, box, source, backend, poseName, poseBackend }; // for debugging and tests
    frames++; fps = 0.9 * fps + 0.1 * (1000 / Math.max(1, now - last)); last = now;
    if (frames % 15 === 0) $('stats').textContent = `${fps.toFixed(0)} FPS · pose ${poseBackend === 'wasm' ? `CPU (${cpuThreads()})` : poseBackend} · detector ran on ${(100 * detections / frames).toFixed(0)}% of frames`;
    status(result ? 'Running' : 'No person found');
  }
}

async function halt() {
  if (!current) return;
  const { halt: signal, done } = current;
  current = null; running = false;
  signal(); await done;
}

async function stop(message = 'Stopped.') {
  await halt();
  $('start').disabled = false;
  if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
  $('start').textContent = 'Start';
  $('stats').textContent = '';
  status(message);
}

async function start() {
  $('start').disabled = true;
  await halt();
  if (!backend) {
    const adapter = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
    backend = adapter ? 'webgpu' : 'wasm';
    f16 = Boolean(adapter?.features.has('shader-f16'));
  }
  let choice = $('model').value;
  if (backend === 'wasm' && POSE_MODELS[choice].int8) choice = POSE_MODELS[choice].int8;  // no WebGPU: the faster CPU model
  const model = POSE_MODELS[choice], wanted = model[f16 ? 'f16' : 'f32'];
  poseRay = Boolean(model.ray);
  if (poseName !== wanted) {
    poseBackend = model.cpu ? 'wasm' : backend;
    pose = await session(wanted, poseBackend); poseName = wanted;
  }
  if (detectorName !== $('detector').value) { detector = await session($('detector').value); detectorName = $('detector').value; }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: cameraChoice ? { exact: cameraChoice } : undefined, width: 640, height: 480 }, audio: false });
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = stream;
  await video.play();
  $('image').width = 640; $('image').height = Math.round(640 * video.videoHeight / video.videoWidth);
  await listCameras(stream.getVideoTracks()[0].getSettings().deviceId);
  filters.xy.reset(); filters.xyz.reset(); anchor.reset();
  running = true;
  $('start').disabled = false; $('start').textContent = 'Stop';
  if (backend === 'wasm') status(`WebGPU is not available; running ${wanted} on the CPU (${cpuThreads()}).`);
  else if (poseBackend === 'wasm') status(`Running ${wanted} on the CPU (${cpuThreads()}); WebGPU models are faster here.`);
  let signal;
  const halted = new Promise((resolve) => { signal = () => resolve(false); });
  const done = loop(halted).catch((error) => { current = null; running = false; stop(`Error: ${error.message}`); });
  current = { halt: signal, done };
}

// ------------------------------------------------------------ controls

// Cameras can be chosen before Start. Browsers hide camera names (and sometimes ids) until camera access is allowed.
const ASK = '?ask';
let cameraChoice = '';  // last real choice ('' = default camera)
async function listCameras(selected = cameraChoice) {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput' && d.deviceId);
  const named = devices.length > 0 && devices.every((d) => d.label);
  const options = devices.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId));
  if (!named) options.push(new Option('List all cameras (asks for camera access)…', ASK));
  $('camera').replaceChildren(new Option('Default camera', ''), ...options);
  cameraChoice = devices.some((d) => d.deviceId === selected) ? selected : '';
  $('camera').value = cameraChoice;
}

$('camera').addEventListener('change', async () => {
  if ($('camera').value === ASK) {
    $('camera').value = cameraChoice;
    try {
      (await navigator.mediaDevices.getUserMedia({ video: true, audio: false })).getTracks().forEach((t) => t.stop());
      status('Choose a camera, then press Start.');
    } catch (error) { status(`Camera access was not allowed: ${error.message}`); }
    await listCameras();
    return;
  }
  cameraChoice = $('camera').value;
  if (running) start().catch((error) => stop(`Error: ${error.message}`));
});
navigator.mediaDevices?.addEventListener?.('devicechange', () => listCameras().catch(() => {}));

function showValue(id) {
  const output = document.querySelector(`output[for=${id}]`);
  if (output) output.value = Number($(id).value).toFixed(2);
}
for (const id of SETTINGS) {
  try {
    let saved = localStorage.getItem(`obt.${id}`), select = $(id).options;
    if (id === 'model' && saved in OLD_NAMES) saved = OLD_NAMES[saved];
    if (id === 'jointThreshold' && Number(saved) < 1) saved = null;  // saved on the old 0-0.9 scale
    if (saved !== null && (!select || [...select].some((o) => o.value === saved))) $(id).value = saved;
  } catch {}
  showValue(id);
  $(id).addEventListener('input', () => {
    showValue(id);
    if (id === 'smoothing') setSmoothing(setting('smoothing'));
    if (id === 'view') $('image').classList.toggle('mirrored', mirrored());
    try { localStorage.setItem(`obt.${id}`, $(id).value); } catch {}
  });
}
setSmoothing(setting('smoothing'));
$('image').classList.toggle('mirrored', mirrored());
listCameras().catch(() => {});

$('controls').addEventListener('submit', (e) => {
  e.preventDefault();
  if (running) stop(); else start().catch((error) => stop(`Error: ${error.message}`));
});
for (const id of ['detector', 'model']) {
  $(id).addEventListener('change', () => { if (running) start().catch((error) => stop(`Error: ${error.message}`)); });
}
