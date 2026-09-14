# LocalAI — a private AI chatbot that runs in your browser

A super-efficient local AI chat app built on **public open weights**. It runs
**100% on your device** — no server, no API key, no account, and nothing you
type ever leaves your machine. It's a plain static web page, so it runs
straight from GitHub links and works on a Chromebook.

## ▶ Open it

| Link | Notes |
|---|---|
| **https://franciscianciola-eng.github.io/localai/** | GitHub Pages — the main link. Deployed automatically by the included workflow. |
| **https://raw.githack.com/franciscianciola-eng/localai/claude/eager-cori-r6fqfl/index.html** | Works immediately from the branch, no setup at all. |

The **first** open downloads the model once (about 100–500 MB depending on the
model you pick) and caches it in the browser. Every open after that starts
**instantly**, straight from cache.

> If the GitHub Pages link shows a 404: the deploy workflow tries to enable
> Pages automatically, but if your repo settings block that, enable it once
> under **Settings → Pages → Source: GitHub Actions**, then re-run the
> "Deploy to GitHub Pages" workflow.

You can also run it locally with any static file server:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

(Opening `index.html` via `file://` won't work — browsers block workers and
model downloads there.)

## Models

All models are public, permissively licensed (Apache-2.0) open weights,
downloaded directly from Hugging Face and quantized to 4/8-bit ONNX for speed:

| Model | Download | Best for |
|---|---|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | ~0.1 GB | Oldest / slowest Chromebooks |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) *(default)* | ~0.3 GB | Best speed/coherence balance |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | ~0.5 GB | Smartest answers |

Switch models from the dropdown in the header; each is cached after its first
download.

## Why it's fast (even on a Chromebook)

- **WebGPU acceleration** — on Chromebooks/browsers with WebGPU (Chrome 113+,
  most recent ChromeOS devices), inference runs on the GPU with 4-bit
  fp16 weights (`q4f16`).
- **Multithreaded WASM fallback** — everywhere else it runs on the CPU using
  all cores. A tiny service worker (`coi-serviceworker.js`) adds the
  COOP/COEP headers GitHub Pages can't set, which unlocks
  `SharedArrayBuffer` and multithreading (one automatic reload on first visit).
- **Aggressive quantization** — 4-bit weights on GPU, 8-bit on CPU.
- **One-time download, permanent cache** — weights are stored in the browser's
  Cache Storage, so reopening the page does no re-downloading.
- **Short prompt window** — the chat keeps a trimmed rolling history so
  generation stays snappy on small hardware.

Status chips in the header show which backend you got (WebGPU vs CPU threads)
and live tokens/second while the model is talking.

## How it works

Three files, zero build step:

- **`index.html`** — the chat UI.
- **`worker.js`** — a Web Worker that runs
  [Transformers.js](https://huggingface.co/docs/transformers.js) (ONNX Runtime
  Web under the hood) to load and run the model off the main thread, streaming
  tokens back as they're generated.
- **`coi-serviceworker.js`** — enables cross-origin isolation on static hosts
  for multithreaded WASM.

`.github/workflows/deploy.yml` publishes the page to GitHub Pages on every
push.

## Privacy

After the one-time model download from Hugging Face's public CDN, all
inference happens in your browser tab. Conversations are kept in memory only —
close the tab and they're gone. Nothing is logged or transmitted anywhere.
