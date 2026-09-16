# LocalAI — a private AI chatbot that runs in your browser

A super-efficient local AI chat app built on **public open weights**. It runs
**100% on your device** — no server, no API key, no account, and nothing you
type ever leaves your machine. It's a plain static web page, so it runs
straight from GitHub links and works on a Chromebook.

## ▶ Open it

**→ https://franciscianciola-eng.github.io/localai/**

That's the clean, permanent link — no commit hash, no branch name — and it's
also the most reliable host. It needs a **one-time setup** (below); until then,
this always-works link serves the same app straight from the repo:

- https://raw.githack.com/franciscianciola-eng/localai/claude/eager-cori-r6fqfl/index.html

The **first** open downloads the model once (about 100–500 MB depending on the
model you pick) and caches it in the browser. Every open after that starts
**instantly**, straight from cache.

> **One-time setup for the clean Pages link** (GitHub doesn't let automation
> enable Pages): open the repo's **Settings → Pages**, set **Source** to
> **GitHub Actions**, and that's it. The deploy workflow is already in the
> repo, so it publishes on the next push — every later update deploys
> automatically.

You can also run it locally with any static file server:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

(Opening `index.html` via `file://` won't work — browsers block workers and
model downloads there.)

## If the model won't load

The app tries every backend and quantization it knows, and when all of them
fail it shows a **diagnosis card** that pinpoints the cause — most commonly:

- **Your network blocks `huggingface.co`** (typical on school/work networks).
  The model weights live there; everything else ships with the app itself.
  Load the page once on another network (a phone hotspot works) — the model
  is then cached in the browser and later visits don't need the download.
  Filters that serve an HTML "blocked" page instead of the real files are
  detected too, and the poisoned download cache is cleared automatically.
- **Not enough browser storage** to cache the model — free disk space or use
  the smallest model.
- **The model is too heavy for this device** — the app automatically steps
  down to the next smaller model until one runs.
- **The engine failed or crashed on this device** — the app automatically
  retries on safer settings (CPU instead of GPU, single-threaded), and the
  error card has a **Reset app** button that wipes all stored state for a
  clean start.

The card's *Technical details* section shows exactly what was attempted and
what each probe found — paste it into an issue if you're stuck.

## Models

All models are public, permissively licensed (Apache-2.0) open weights,
downloaded directly from Hugging Face and quantized to 4/8-bit ONNX for speed:

**Small models** (run on any device — WebGPU *or* CPU/WASM):

| Model | Download | Best for |
|---|---|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) *(default)* | ~0.1 GB | Runs on anything — the safe default |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | ~0.3 GB | Best speed/coherence balance |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | ~0.5 GB | Smartest small model |

**Large models** (need **WebGPU**; run on the WebLLM engine only — much smarter,
much larger downloads, and a capable GPU helps):

| Model | Params | Download | Notes |
|---|---|---|---|
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | ~0.9 GB | Solid all-rounder |
| [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it) | 2B | ~1.6 GB | Strong 2B model |
| [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | ~2.3 GB | Closest browser-runnable model to 5B* |

\* There's no widely-available ~5B open model in a browser-ready format;
Phi-3.5-mini (3.8B) is the nearest that runs reliably in-browser. Bigger
options (Qwen2.5-7B, Llama-3.1-8B) exist and can be added, but need a lot of
GPU memory.

Switch models from the dropdown in the header; each is cached after its first
download. The large models load through WebGPU — if your browser has no WebGPU,
the app tells you and points you back to the small models.

## Two engines for maximum compatibility

The app ships **two independent inference engines** and switches between them
automatically so that a failure in one doesn't leave you stuck:

1. **Transformers.js / ONNX Runtime** (default) — runs on WebGPU *or* CPU
   (multithreaded WASM), so it works even on devices without a GPU.
2. **WebLLM / Apache TVM** (automatic fallback) — a completely separate
   WebGPU engine. If the ONNX Runtime engine crashes on a particular
   browser/GPU combination (it can abort inside a native kernel on some
   devices), and WebGPU is available, the app switches to WebLLM
   automatically and remembers the choice for next time.

Both pull the same families of public open weights; WebLLM uses the
[mlc-ai](https://huggingface.co/mlc-ai) MLC-compiled builds.

### WebGPU throttling failsafe (tuned for integrated graphics)

The larger models are built to run on **integrated graphics**, where the usual
failure is the GPU running out of memory or a compute pass tripping the
driver's watchdog and crashing. A browser can't read GPU temperature/power or
change clock speeds, so the app instead **reduces the workload** and backs off
automatically:

- Big models **start in "eco mode"** (capped output length, trimmed chat
  history, light per-token pacing) since integrated graphics is the target.
- If the GPU crashes or **stalls** (a watchdog catches a wedged context), the
  app **escalates**: eco → **low-power mode**, which also switches to the
  **1k-context build** of the model — a much smaller KV cache, the main thing
  that pushes a big model past an integrated GPU's memory.
- If even the lowest setting crashes, it **steps down to the next smaller
  model** and keeps going until one runs.
- The working level is **remembered per device**, and the mid-reply message is
  answered automatically after each back-off. Per-token pacing caps sustained
  GPU utilisation — the closest a web page can get to "easing off the power."

An **external/discrete GPU is used automatically** when one is present (the app
doesn't pin the integrated adapter); the throttling just keeps things safe on
integrated-only machines.

## Why it's fast (even on a Chromebook)

- **WebGPU acceleration** — on Chromebooks/browsers with WebGPU (Chrome 113+,
  most recent ChromeOS devices), inference runs on the GPU with 4-bit
  fp16 weights (`q4f16`), falling back through `q4` and the CPU formats until
  one works.
- **Multithreaded WASM fallback** — everywhere else it runs on the CPU using
  multiple cores. A tiny service worker (`coi-serviceworker.js`) adds the
  COOP/COEP headers GitHub Pages can't set, which unlocks
  `SharedArrayBuffer` and multithreading (one automatic reload on first visit).
- **Self-hosted runtime with CDN fallback** — the inference engine (`vendor/`)
  ships with the app. Some static hosts refuse to serve its 21 MB WebAssembly
  file, so the app probes for that up front and transparently loads the runtime
  from the jsdelivr CDN when the host won't; only the model weights come from
  Hugging Face, once. (GitHub Pages serves the vendored copy fine — it's the
  most reliable place to host this app.)
- **One-time download, permanent cache** — weights are stored in the browser's
  Cache Storage, so reopening the page does no re-downloading.
- **Short prompt window** — the chat keeps a trimmed rolling history so
  generation stays snappy on small hardware.

Status chips in the header show which backend you got (WebGPU vs CPU threads)
and live tokens/second while the model is talking.

## Settings

Open the **⚙️ gear** in the header:

- **Web search** *(off by default)* — an optional "internet sandbox". When on,
  your message is sent to **Wikipedia's public API** (CORS-enabled, no key,
  read-only) to fetch relevant facts, which are handed to the local model as
  grounding, with the sources shown in the chat. This is the **only** feature
  that sends anything off your device, which is why it's off unless you opt in.
  (A keyless, backend-less page can only reach APIs that allow cross-origin
  requests, so Wikipedia is the safe, reliable source here.)
- **Reply length** — Short / Medium / Long output cap.
- **Creativity** — temperature, from Precise to Wild.
- **Performance mode** — Auto (tunes for integrated graphics and backs off on
  crashes), Full, Eco, or Low-power for the WebGPU models.
- **Theme** — System / Light / Dark.
- **Send with Enter** — Enter sends vs. Enter makes a newline.
- **Custom instructions** — a persona/system-prompt addition.
- **Storage** — see cache usage and clear cached models to free space.

All settings are saved in the browser.

## How it works

No build step:

- **`index.html`** — the chat UI. You can always type; messages queue until
  the model is ready.
- **`worker.js`** — a Web Worker running
  [Transformers.js](https://huggingface.co/docs/transformers.js) (ONNX Runtime
  Web under the hood). It walks a ladder of device/quantization combos
  (WebGPU `q4f16` → `q4`, then CPU `q8` → `q4` → `uint8` → `int8`), streams
  tokens back as they're generated, and runs network diagnostics if every
  attempt fails.
- **`coi-serviceworker.js`** — enables cross-origin isolation on static hosts
  for multithreaded WASM. If a load fails while isolation is on, the app
  turns it off and retries in compatibility mode automatically.
- **`vendor/`** — the pinned transformers.js bundle and ONNX wasm runtime
  (from npm, `@huggingface/transformers@3.8.1`), so the app doesn't depend on
  any CDN being reachable.

`.github/workflows/deploy.yml` publishes the page to GitHub Pages on every
push.

## Testing

The repo contains a real end-to-end test that needs no network: a ~300 KB
random-weight llama-shaped ONNX model (`test/fixtures/tiny-llm`, regenerable
with `python3 test/make-fixture.py`) exercises the entire stack — tokenizer,
chat template, KV cache, multithreaded WASM inference, streaming — in headless
Chromium, plus the offline-diagnosis path:

```bash
npm install
npm test          # runs test/e2e.mjs against a local Chromium
```

## Privacy

After the one-time model download from Hugging Face's public CDN, all
inference happens in your browser tab. Conversations live only in the tab
(they survive a refresh, and vanish when the tab closes). Nothing is logged
or transmitted anywhere.
