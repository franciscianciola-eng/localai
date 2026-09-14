// Inference worker: loads a quantized ONNX chat model with transformers.js and
// streams generated tokens back to the page. Runs fully on-device — the only
// network use is downloading the public model weights once (then cached).
//
// The transformers.js bundle and the ONNX wasm runtime are vendored into this
// repo (vendor/), so the only external host the app needs is huggingface.co
// for the weights themselves. If the vendored wasm can't be served by the
// current host, we retry from the jsdelivr CDN.

import {
  pipeline,
  env,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "./vendor/transformers.min.js";

const VENDOR_URL = new URL("./vendor/", self.location.href).href;
const CDN_DIST = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/";
env.backends.onnx.wasm.wasmPaths = VENDOR_URL;

let generator = null;
let loadedModelId = null;
let stoppingCriteria = new InterruptableStoppingCriteria();

function errText(err) {
  return err?.message || String(err);
}

// Rough classification of a load error, used to decide whether the next
// attempt in the ladder could still succeed.
function classify(err) {
  const m = errText(err).toLowerCase();
  // Memory first: wasm OOM surfaces as "no available backend ... RangeError:
  // Out of memory" on low-RAM devices, which must not read as a backend bug.
  if (
    m.includes("out of memory") ||
    m.includes("bad_alloc") ||
    m.includes("allocation failed") ||
    m.includes("cannot enlarge memory") ||
    m.includes("rangeerror")
  )
    return "memory";
  if (m.includes("simd")) return "simd"; // pre-2021 Chrome, e.g. post-AUE Chromebooks
  if (m.includes("no available backend")) return "backend";
  if (m.includes("could not locate file") || m.includes("404")) return "missing-file";
  if (m.includes("forbidden access to file")) return "blocked-http"; // filtering proxy answering 403
  // A filtering proxy serving a 200 HTML block page instead of model files:
  if (m.includes("unexpected token") || m.includes("not valid json") || m.includes("protobuf parsing failed"))
    return "corrupted";
  if (m.includes("failed to fetch") || m.includes("networkerror") || m.includes("network error") || m.includes("load failed"))
    return "network";
  if (m.includes("quota") || m.includes("storage") || m.includes("browser cache is not available")) return "storage";
  if (m.includes("webgpu") || m.includes("adapter")) return "webgpu";
  return "unknown";
}

async function buildAttempts() {
  const attempts = [];
  let adapter = null;
  try {
    if (globalThis.navigator?.gpu) adapter = await navigator.gpu.requestAdapter();
  } catch {}
  if (adapter) {
    if (adapter.features?.has?.("shader-f16")) attempts.push({ device: "webgpu", dtype: "q4f16" });
    attempts.push({ device: "webgpu", dtype: "q4" });
  }
  // CPU path: q8 (a.k.a. model_quantized) is the best-supported wasm dtype;
  // the rest are fallbacks in case a repo lacks a particular file.
  attempts.push({ device: "wasm", dtype: "q8" });
  attempts.push({ device: "wasm", dtype: "q4" });
  attempts.push({ device: "wasm", dtype: "uint8" });
  attempts.push({ device: "wasm", dtype: "int8" });
  return attempts;
}

function makeProgressReporter(attempt) {
  const files = new Map();
  let lastPost = 0;
  return (info) => {
    if (info.status === "progress" || info.status === "done") {
      files.set(info.file, {
        loaded:
          info.status === "done"
            ? files.get(info.file)?.total ?? info.loaded ?? 0
            : info.loaded ?? 0,
        total: info.total ?? files.get(info.file)?.total ?? 0,
      });
      const now = Date.now();
      if (now - lastPost < 100 && info.status !== "done") return;
      lastPost = now;
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      self.postMessage({
        type: "progress",
        file: info.file,
        loaded,
        total,
        pct: total ? Math.min(100, Math.round((loaded / total) * 100)) : 0,
        attempt,
      });
    }
  };
}

async function probeUrl(url) {
  try {
    const r = await fetch(url, { method: "HEAD", cache: "no-store" });
    return { ok: r.ok, status: r.status };
  } catch (e) {
    return { ok: false, error: errText(e) };
  }
}

// GET + parse: a filtering proxy can answer 200 with an HTML block page, so a
// status check alone would lie — verify the body is the JSON it claims to be.
async function probeJson(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    const out = { ok: r.ok, status: r.status };
    if (r.ok) {
      const text = await r.text();
      try {
        JSON.parse(text);
        out.validJson = true;
      } catch {
        out.validJson = false;
        out.snippet = text.slice(0, 80);
      }
    }
    return out;
  } catch (e) {
    return { ok: false, error: errText(e) };
  }
}

async function runDiagnostics(modelId) {
  const diag = {
    crossOriginIsolated: !!globalThis.crossOriginIsolated,
    threads: navigator.hardwareConcurrency || 1,
    userAgent: navigator.userAgent,
  };
  diag.huggingface = await probeJson(
    (env.remoteHost || "https://huggingface.co/") + `${modelId}/resolve/main/config.json`
  );
  diag.vendorWasm = await probeUrl(VENDOR_URL + "ort-wasm-simd-threaded.jsep.wasm");
  diag.jsdelivr = await probeUrl("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/package.json");
  try {
    if (navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      diag.storage = { usage: est.usage, quota: est.quota };
    }
  } catch {}
  try {
    diag.webgpuAdapter = !!(globalThis.navigator?.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    diag.webgpuAdapter = false;
  }
  return diag;
}

async function load(modelId, opts = {}) {
  if (opts.remoteHost) env.remoteHost = opts.remoteHost; // test hook
  // Page-controlled recovery options. These must be applied in a FRESH worker:
  // onnxruntime latches a failed wasm init, so the page respawns this worker
  // and passes the adjusted configuration instead of retrying in place.
  if (opts.numThreads) env.backends.onnx.wasm.numThreads = opts.numThreads;
  if (opts.wasmSource === "cdn") env.backends.onnx.wasm.wasmPaths = CDN_DIST;

  if (generator && loadedModelId === modelId) {
    self.postMessage({ type: "ready", modelId });
    return;
  }
  if (generator) {
    try {
      await generator.dispose();
    } catch {}
    generator = null;
    loadedModelId = null;
  }

  const attempts = await buildAttempts();
  const failures = [];

  for (const a of attempts) {
    const label =
      (a.device === "webgpu" ? "GPU" : "CPU") + " · " + a.dtype;
    self.postMessage({
      type: "device",
      device: a.device,
      dtype: a.dtype,
      // Mirrors onnxruntime-web's own choice: min(4, ceil(cores/2)) when
      // cross-origin isolated, else single-threaded.
      threads: opts.numThreads
        ? opts.numThreads
        : globalThis.crossOriginIsolated
          ? Math.min(4, Math.ceil((navigator.hardwareConcurrency || 1) / 2))
          : 1,
      crossOriginIsolated: !!globalThis.crossOriginIsolated,
      attempt: label,
    });
    try {
      generator = await pipeline("text-generation", modelId, {
        device: a.device,
        dtype: a.dtype,
        progress_callback: makeProgressReporter(label),
      });
      loadedModelId = modelId;
      self.postMessage({ type: "ready", modelId, device: a.device, dtype: a.dtype });
      return;
    } catch (err) {
      const kind = classify(err);
      failures.push({ device: a.device, dtype: a.dtype, kind, message: errText(err) });

      if (kind === "network" || kind === "corrupted") {
        // The model host is unreachable (or a filter is tampering with its
        // responses); every other attempt needs the same downloads, so stop
        // early instead of hammering a dead or hostile network.
        break;
      }
      if (kind === "memory" && a.device === "wasm") {
        // Every CPU variant of this model is roughly the same size; a smaller
        // model is the only real fix, so don't burn RAM on more attempts.
        break;
      }
      // missing-file / webgpu / backend / unknown → try the next combo.
    }
  }

  const diag = await runDiagnostics(modelId);
  self.postMessage({ type: "load-failed", modelId, failures, diag });
}

async function generate(messages, params = {}) {
  if (!generator) throw new Error("Model not loaded yet");

  stoppingCriteria = new InterruptableStoppingCriteria();
  let text = "";
  let tokenCount = 0;
  let firstTokenAt = 0;
  const startedAt = performance.now();

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      if (!firstTokenAt) firstTokenAt = performance.now();
      text += chunk;
      const elapsed = (performance.now() - firstTokenAt) / 1000;
      self.postMessage({
        type: "token",
        text: chunk,
        tps: elapsed > 0.2 ? tokenCount / elapsed : 0,
      });
    },
    token_callback_function: () => {
      tokenCount += 1;
    },
  });

  await generator(messages, {
    max_new_tokens: params.max_new_tokens ?? 512,
    do_sample: true,
    temperature: params.temperature ?? 0.6,
    top_p: params.top_p ?? 0.9,
    repetition_penalty: params.repetition_penalty ?? 1.1,
    streamer,
    stopping_criteria: stoppingCriteria,
  });

  const totalSec = (performance.now() - startedAt) / 1000;
  const genSec = firstTokenAt ? (performance.now() - firstTokenAt) / 1000 : 0;
  self.postMessage({
    type: "done",
    text,
    stats: {
      tokens: tokenCount,
      seconds: totalSec,
      tps: genSec > 0 ? tokenCount / genSec : 0,
    },
  });
}

self.addEventListener("message", async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await load(msg.modelId, msg.opts);
    } else if (msg.type === "generate") {
      await generate(msg.messages, msg.params);
    } else if (msg.type === "stop") {
      stoppingCriteria.interrupt();
    }
  } catch (err) {
    self.postMessage({ type: "error", message: errText(err) });
  }
});

self.postMessage({ type: "boot" });
