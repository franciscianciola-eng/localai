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

**Large models** (best with **WebGPU**; smarter, larger downloads):

| Model | Params | Download | Notes |
|---|---|---|---|
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | ~0.9 GB | Runs on WebGPU **or** the CPU (slowly) |
| [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it) | 2B | ~1.6 GB | WebGPU |
| [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | ~2.3 GB | WebGPU · closest browser-runnable model to 5B* |

\* There's no widely-available ~5B open model in a browser-ready format;
Phi-3.5-mini (3.8B) is the nearest that runs reliably in-browser.

**Every model works on a Chromebook**, even without WebGPU: modern Chromebooks
have WebGPU and run all of these; on a Chromebook without it, the small models
and Llama-3.2-1B run on the CPU (slowly), and Gemma/Phi (which have no
CPU-runnable build) **automatically step down to the largest CPU-capable model**
instead of failing. Switch models from the dropdown; each is cached after its
first download.

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

- **Web search** *(off by default)* — an optional internet feature with three
  modes: **Off**, **Smart**, and **Always**.
  - In **Smart** mode the model makes a **conscious decision** before every
    reply about whether a search is actually worth it — a fast heuristic
    catches the obvious cases (greetings, creative writing and math → no
    search; "latest", "today", prices, weather → search), and genuinely
    ambiguous questions are put to the model itself, which answers only
    "NO" or "SEARCH: <query>". This avoids useless searches.
  - When it does search, it queries the **full web via [Jina Reader](https://jina.ai/reader/)**
    **and Wikipedia** (directly), and hands the top results to the local model
    as grounding — with the system prompt telling the model it *has* live web
    results and must use and cite them — and clickable sources shown in the chat.
  - **Why Jina Reader?** A keyless, backend-less browser page can't read
    arbitrary websites directly (browsers block cross-origin reads). Jina Reader
    fetches pages for the browser and is CORS-enabled and keyless (rate-limited).
    A free key from **jina.ai** (pasted in settings) makes web search faster and
    more reliable; Wikipedia always works with no key. This is the **only**
    feature that sends anything off your device, which is why it's off unless you
    opt in.
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

## One file with the model built in — no download at all

**Download:** https://github.com/franciscianciola-eng/localai/releases/download/offline-models/LocalAI-Qwen2.5-0.5B-offline.html

A single HTML file (~400 MB) with the **Qwen2.5 0.5B** model inside it. Double-click
it and chat: nothing downloads, no internet needed, ever. The first open takes a
bit longer while it unpacks the model into the browser; later opens are quick.

- **How:** `build-embedded.mjs` appends the model's files (config, tokenizer,
  weight shards, compiled WebGPU library) to the single-file app as base64
  chunks. On open, the app writes them into the exact Cache Storage entries
  WebLLM would create after a download, so WebLLM loads with no network.
- **Built by GitHub Actions** (`.github/workflows/offline-models.yml`), which
  downloads the weights from Hugging Face and publishes the file as a release.
  To build it yourself: `python3 offline-package/download-models.py --only qwen`
  then `node build-embedded.mjs --models qwen`.
- **Why only one model per file, and why the 0.5B:** a file's size is roughly
  the model's size plus a third (base64). The 0.5B makes a ~400 MB file that a
  Chromebook opens fine; the 1B would be ~950 MB (heavy — may not open on a 4 GB
  Chromebook) and the 2B ~2 GB; all three together would be ~3.3 GB, which a
  browser can't hold in memory to open at all. `build-embedded.mjs --models llama`
  can build the 1B on its own if you have a roomier machine.

## One local file for a Chromebook (`localai-standalone.html`)

There's also a **single self-contained HTML file** you can keep on a Chromebook
and open locally (even via `file://`, no server):

**Download it here:** https://franciscianciola-eng.github.io/localai/localai-standalone.html
— open that link, then **Save Page As → "Webpage, HTML Only"** to get the one
file, or use the "Download the offline single-file version" link on the main app.

It bundles the WebLLM engine inline (so nothing loads from a CDN) and offers the
three requested models — **Qwen2.5 0.5B, Llama 3.2 1B, Gemma 2 2B** — running on
**WebGPU**. It also has:

- **Think mode** (🧠, on by default) — the model reasons step-by-step (which
  makes these small models much more accurate), streaming the reasoning live and
  then collapsing it into a "Thoughts" toggle, leaving a clean answer.
- **Attachments** (📎, drag-drop, or paste) — text/code/data files are read and
  given to the model, and **Word, PowerPoint and Excel** files (`.docx/.pptx/.xlsx`,
  plus OpenDocument) are unzipped in the browser and their text extracted; images
  are shown inline and run through **OCR** so the model can read any text in them.
  Files are judged by their bytes, so binary files are refused rather than fed to
  the model; PDFs and old `.doc/.xls/.ppt` get a clear "how to attach this" note.
  Long files are trimmed so the whole prompt fits the model's context window, and
  a reply that degenerates into gibberish is stopped with an explanation. These
  are text-only models, so images without text are labelled as such.

Notes:

- **First run needs internet once per model** — ~3 GB of weights can't be
  embedded in an HTML file, so each model downloads from the web the first time
  and is then cached in the browser and runs **fully offline** afterward. The
  OCR engine (Tesseract + English data, ~7 MB) *is* inlined, so image
  text-reading needs no network at any point.
- **Needs WebGPU** (Chrome 113+, which nearly all Chromebooks from ~2023 have).
  Unlike the hosted app, this single file can't use the CPU engine, because
  `file://` blocks the Web Workers that path needs — so if a device has no
  WebGPU it shows a clear message and you'd use the hosted app instead. (OCR
  still works: its worker is created from a Blob URL, which `file://` allows.)
- The file is ~16 MB (WebLLM engine + OCR engine inlined). Rebuild it after
  editing the template with `npm run build:standalone` (regenerates
  `localai-standalone.html` from `standalone.template.html` + the vendored
  WebLLM and Tesseract assets).

## A folder with the models on disk (`offline-package/`)

The single file above still downloads each model **into the browser** the first
time. If you'd rather have the model files sitting **in a folder** — to copy to
an air-gapped machine, share on a USB stick, or just keep off the network — use
[`offline-package/`](offline-package/). It's the same WebGPU app pointed at a
local `models/` folder instead of the internet.

```bash
cd offline-package
python3 download-models.py      # once, with internet (~3 GB, or --only qwen for ~0.5 GB)
python3 serve.py                # any time; opens http://localhost:8000 — no internet used
```

(or just `./start.sh` on Linux/macOS/Chromebook, or double-click `start.bat` on
Windows — it does both steps.)

Two honest caveats, both handled by the package:

- **The ~3 GB of weights aren't shipped in the repo** — a git repo is the wrong
  place for gigabytes of binaries, and they can't be embedded in a small file.
  So `download-models.py` fetches them once into `models/` (mirroring the exact
  Hugging Face layout the app requests). Re-running it resumes — it skips what
  you already have.
- **It needs a local server, not `file://`** — browsers refuse to let a
  `file://` page read other local files, so `serve.py` serves the folder at
  `http://localhost` with the right MIME types. Everything still stays on your
  machine.

Rebuild the app after editing its template with `npm run build:offline`
(regenerates `offline-package/index.html` from `offline.template.html` + the
vendored WebLLM bundle). See [`offline-package/README.md`](offline-package/README.md)
for the full walkthrough and troubleshooting.

## Works offline

After the first visit, the whole app works with **no internet connection**:

- A service worker **precaches the app shell** and caches the vendored AI
  runtimes on first use, so the page loads offline.
- **WebLLM is vendored** (not loaded from a CDN), so the WebGPU engine works
  offline too.
- Model weights are cached in the browser after their first download, so any
  model you've already used **runs fully offline** — pick it, chat, done.
- It's a **PWA**: "Install" / "Add to Home Screen" it and launch it like a
  native app with no connection.

What still needs a connection: downloading a model you haven't used yet, and
web search (which is off by default). The app shows an **✈️ offline** chip,
pauses web search while offline, and tells you clearly if you pick a model that
hasn't been downloaded yet.

## Privacy

After the one-time model download from Hugging Face's public CDN, all
inference happens in your browser tab. Conversations live only in the tab
(they survive a refresh, and vanish when the tab closes). Nothing is logged
or transmitted anywhere.
