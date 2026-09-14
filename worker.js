// Inference worker: loads a quantized ONNX chat model with transformers.js and
// streams generated tokens back to the page. Runs fully on-device — the only
// network use is downloading the public model weights once (then cached).

import {
  pipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

let generator = null;
let loadedModelId = null;
let stoppingCriteria = new InterruptableStoppingCriteria();

async function detectDevice() {
  // Prefer WebGPU when the browser exposes a real adapter; q4f16 needs
  // shader-f16 support, otherwise fall back to q4 on the GPU.
  try {
    if (globalThis.navigator?.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const fp16 = adapter.features?.has?.("shader-f16");
        return { device: "webgpu", dtype: fp16 ? "q4f16" : "q4" };
      }
    }
  } catch {
    // fall through to WASM
  }
  return { device: "wasm", dtype: "q8" };
}

function makeProgressReporter() {
  const files = new Map();
  let lastPost = 0;
  return (info) => {
    if (info.status === "progress" || info.status === "done") {
      files.set(info.file, {
        loaded: info.status === "done" ? (files.get(info.file)?.total ?? info.loaded ?? 0) : (info.loaded ?? 0),
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
      });
    }
  };
}

async function load(modelId) {
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

  let { device, dtype } = await detectDevice();
  self.postMessage({
    type: "device",
    device,
    dtype,
    threads: globalThis.crossOriginIsolated ? (navigator.hardwareConcurrency || 1) : 1,
    crossOriginIsolated: !!globalThis.crossOriginIsolated,
  });

  try {
    generator = await pipeline("text-generation", modelId, {
      device,
      dtype,
      progress_callback: makeProgressReporter(),
    });
  } catch (err) {
    if (device === "webgpu") {
      // Some devices advertise WebGPU but fail to compile the model — retry on CPU.
      device = "wasm";
      dtype = "q8";
      self.postMessage({
        type: "device",
        device,
        dtype,
        threads: globalThis.crossOriginIsolated ? (navigator.hardwareConcurrency || 1) : 1,
        crossOriginIsolated: !!globalThis.crossOriginIsolated,
        note: "WebGPU failed, retrying on CPU",
      });
      generator = await pipeline("text-generation", modelId, {
        device,
        dtype,
        progress_callback: makeProgressReporter(),
      });
    } else {
      throw err;
    }
  }

  loadedModelId = modelId;
  self.postMessage({ type: "ready", modelId });
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
      await load(msg.modelId);
    } else if (msg.type === "generate") {
      await generate(msg.messages, msg.params);
    } else if (msg.type === "stop") {
      stoppingCriteria.interrupt();
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err?.message || String(err) });
  }
});

self.postMessage({ type: "boot" });
