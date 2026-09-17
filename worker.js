// Inference worker: loads a quantized ONNX chat model with transformers.js and
// streams generated tokens back to the page. Runs fully on-device — the only
// network use is downloading the public model weights once (then cached).
//
// The transformers.js bundle and the ONNX wasm runtime are vendored into this
// repo (vendor/). Some static hosts (e.g. raw.githack) refuse to serve the
// 21 MB wasm runtime with a 403, so before loading we probe whether this host
// actually serves it and fall back to the jsdelivr CDN copy when it doesn't —
// otherwise onnxruntime aborts with a bare error code on every backend.

import {
  pipeline,
  env,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "./vendor/transformers.min.js";

const VENDOR_URL = new URL("./vendor/", self.location.href).href;
let CDN_DIST = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/";

let generator = null;
let loadedModelId = null;
let currentConfig = null; // {device, dtype} of the loaded pipeline
let crashGenOnce = false; // test hook: crash the next generation once
let stoppingCriteria = new InterruptableStoppingCriteria();

function errText(err) {
  return err?.message || String(err);
}

// Rough classification of a load error, used to decide whether the next
// attempt in the ladder could still succeed.
function classify(err) {
  const m = errText(err).toLowerCase();
  // Emscripten C++ exceptions surface as a bare number (a pointer) — the
  // signature of a native engine crash, typically a GPU driver failure.
  if (/^\d+$/.test(m.trim()) || m.includes("unreachable") || m.includes("function signature mismatch"))
    return "engine-crash";
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
  // transformers.js maps HTTP 5xx to "... error occurred while trying to load
  // file" — a (usually transient) server-side problem, not a device problem.
  if (m.includes("occurred while trying to load file")) return "network";
  // A filtering proxy serving a 200 HTML block page instead of model files:
  if (m.includes("unexpected token") || m.includes("not valid json") || m.includes("protobuf parsing failed"))
    return "corrupted";
  if (m.includes("failed to fetch") || m.includes("networkerror") || m.includes("network error") || m.includes("load failed"))
    return "network";
  if (m.includes("quota") || m.includes("storage") || m.includes("browser cache is not available")) return "storage";
  if (m.includes("webgpu") || m.includes("adapter")) return "webgpu";
  return "unknown";
}

async function buildAttempts(opts = {}) {
  const attempts = [];
  let adapter = null;
  try {
    if (!opts.forceWasm && globalThis.navigator?.gpu) adapter = await navigator.gpu.requestAdapter();
  } catch {}
  if (adapter) {
    if (adapter.features?.has?.("shader-f16")) attempts.push({ device: "webgpu", dtype: "q4f16" });
    attempts.push({ device: "webgpu", dtype: "q4" });
  }
  // CPU path: q8 (a.k.a. model_quantized) is the best-supported wasm dtype,
  // with q4 as the one fallback — more variants would mean more multi-hundred
  // MB downloads for near-identical odds; a smaller model is the real fix.
  attempts.push({ device: "wasm", dtype: "q8" });
  attempts.push({ device: "wasm", dtype: "q4" });
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

// Some static hosts (e.g. raw.githack) refuse to serve the large (21 MB) wasm
// runtime with a 403, which makes onnxruntime abort on load. A HEAD can be
// forbidden while a GET works, so probe with a 1-byte ranged GET.
async function hostServesVendorWasm(opts) {
  if (opts.vendorBlocked) return false; // test hook
  try {
    const r = await fetch(VENDOR_URL + "ort-wasm-simd-threaded.jsep.wasm", {
      headers: { Range: "bytes=0-0" },
      cache: "no-store",
    });
    if (r.body) { try { await r.body.cancel(); } catch {} }
    return r.ok || r.status === 206;
  } catch {
    return false;
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
  if (opts.wasmCdn) CDN_DIST = opts.wasmCdn; // test hook
  // Page-controlled recovery options. These must be applied in a FRESH worker:
  // onnxruntime latches a failed wasm init, so the page respawns this worker
  // and passes the adjusted configuration instead of retrying in place.
  if (opts.numThreads) env.backends.onnx.wasm.numThreads = opts.numThreads;
  if (opts.testCrashGenerate) crashGenOnce = true; // test hook

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

  // Decide where the wasm runtime comes from BEFORE the first load attempt: a
  // failed wasm init is latched for the worker's lifetime, so a same-origin
  // 403 (host won't serve the big file) must never be attempted. Prefer the
  // vendored same-origin copy; fall back to the CDN the moment it's not served.
  let wasmSource;
  if (opts.wasmSource === "cdn") wasmSource = "cdn";
  else wasmSource = (await hostServesVendorWasm(opts)) ? "vendor" : "cdn";
  env.backends.onnx.wasm.wasmPaths = wasmSource === "cdn" ? CDN_DIST : VENDOR_URL;
  self.postMessage({ type: "wasm-source", source: wasmSource });

  const attempts = await buildAttempts(opts);
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
      if (opts.testCrashLoad) throw 576720528; // test hook: emulate load crash
      generator = await pipeline("text-generation", modelId, {
        device: a.device,
        dtype: a.dtype,
        progress_callback: makeProgressReporter(label),
      });
      loadedModelId = modelId;
      currentConfig = { device: a.device, dtype: a.dtype };
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

  try {
    if (crashGenOnce) {
      crashGenOnce = false;
      throw 576720528; // test hook: emulate a raw Emscripten engine crash
    }
    await generator(messages, {
      max_new_tokens: params.max_new_tokens ?? 512,
      do_sample: true,
      temperature: params.temperature ?? 0.6,
      top_p: params.top_p ?? 0.9,
      repetition_penalty: params.repetition_penalty ?? 1.1,
      streamer,
      stopping_criteria: stoppingCriteria,
    });
  } catch (err) {
    // Inference crashed (typically a native/GPU engine failure). Report with
    // context so the page can reload on a safer configuration and regenerate.
    self.postMessage({
      type: "generate-failed",
      message: errText(err),
      kind: classify(err),
      device: currentConfig?.device,
      dtype: currentConfig?.dtype,
      partial: text,
    });
    return;
  }

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

// A short, greedy, non-streaming completion used for the "should I search?"
// decision — returns the whole text in one message instead of streaming to UI.
async function runClassify(messages, id, maxTokens) {
  let text = "";
  if (!generator) { self.postMessage({ type: "classify-result", id, text: "" }); return; }
  try {
    const out = await generator(messages, {
      max_new_tokens: maxTokens ?? 24,
      do_sample: false,
      repetition_penalty: 1.1,
    });
    const seq = Array.isArray(out) ? out[0] : out;
    const gen = seq?.generated_text;
    if (Array.isArray(gen)) text = gen[gen.length - 1]?.content || "";
    else if (typeof gen === "string") text = gen;
  } catch {}
  self.postMessage({ type: "classify-result", id, text });
}

self.addEventListener("message", async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await load(msg.modelId, msg.opts);
    } else if (msg.type === "generate") {
      await generate(msg.messages, msg.params);
    } else if (msg.type === "classify") {
      await runClassify(msg.messages, msg.id, msg.maxTokens);
    } else if (msg.type === "stop") {
      stoppingCriteria.interrupt();
    }
  } catch (err) {
    if (msg.type === "classify") self.postMessage({ type: "classify-result", id: msg.id, text: "" });
    else self.postMessage({ type: "error", message: errText(err) });
  }
});

self.postMessage({ type: "boot" });
