# LocalAI — offline package

A private AI chat that runs **entirely on your device** and needs **no internet**
once set up. The models live in the `models/` folder right next to `index.html`;
nothing you type ever leaves your computer.

Three models are available in the dropdown:

| Model | Size | Best for |
|-------|------|----------|
| Qwen2.5 **0.5B** | ~0.5 GB | fastest, lightest — great on a Chromebook |
| Llama 3.2 **1B** | ~0.9 GB | balanced |
| Gemma 2 **2B** | ~1.6 GB | smartest, needs a bit more GPU |

---

## The honest part: two one-time steps

Two facts about browsers make a truly "just double-click it" bundle impossible,
so this package handles both for you:

1. **The model weights (~3 GB) can't be shipped inside a small file.** So this
   folder includes a downloader that fetches them once, from Hugging Face, into
   `models/`. You need internet **only for this first step**.
2. **A browser won't let a `file://` page read other local files.** So the app
   is served by a tiny local web server (`serve.py`) at `http://localhost`.
   Everything still stays on your machine — the server only talks to your own
   browser.

After the one-time download, you can go fully offline forever.

---

## Quick start

### Chromebook / Linux / macOS

```bash
cd offline-package
./start.sh
```

That downloads the models the first time (internet needed), then opens the app.
Next time, `./start.sh` just starts the server — no internet used.

> On a Chromebook you need the **Linux** environment turned on
> (Settings → Advanced → Developers → Linux). Then run the commands in the
> Linux Terminal. Python 3 is usually already there; if not:
> `sudo apt update && sudo apt install -y python3`.

### Windows

Double-click **`start.bat`** (or run it in a terminal). Same behavior.

### Do it by hand

```bash
python3 download-models.py     # once, with internet (~3 GB)
python3 serve.py               # any time, opens http://localhost:8000
```

Only want the small model? `python3 download-models.py --only qwen` (~0.5 GB).
See sizes without downloading: `python3 download-models.py --list`.

---

## Requirements

- **Python 3** (any recent version — used only for the downloader and the local
  server; no `pip install` needed).
- **A browser with WebGPU** — Chrome/Edge/Chromebook from ~2023 on (Chrome 113+).
  Check `chrome://gpu` if a model won't start. WebGPU is what lets the model run
  fast on the GPU; there is no CPU fallback in this offline build.
- Disk space for whichever models you download (up to ~3 GB for all three).

---

## How it works

- `index.html` is a single self-contained app. It bundles the
  [WebLLM](https://github.com/mlc-ai/web-llm) (MLC) WebGPU engine and, instead of
  pointing at the internet, points every model URL at `./models/…`.
- `download-models.py` mirrors three public Hugging Face repos into
  `models/<repo>/resolve/main/` — the exact paths the app asks for — plus the
  three compiled WebGPU libraries into `models/libs/`. It's resumable: re-run it
  and it skips whatever you already have.
- `serve.py` is a plain static file server with the correct MIME types
  (`.wasm` → `application/wasm`, ES modules → `text/javascript`).
- The first time you load a model the browser also caches it internally, so even
  the local server isn't hit on later loads.

Nothing here phones home. You can confirm it: after downloading, turn off Wi-Fi
and the app still works.

---

## Troubleshooting

- **"needs setup" / "model files weren't found"** — you haven't run
  `download-models.py` yet, or you opened `index.html` as a `file://` page
  instead of through `serve.py`. Start the server and open the printed
  `http://localhost:8000/index.html`.
- **"needs WebGPU"** — your browser/GPU doesn't expose WebGPU. Update Chrome,
  check `chrome://gpu`, or try a newer device.
- **The 2B model is slow or the tab reloads** — that's an integrated GPU running
  out of headroom. The app automatically backs off (eco → low-power → a smaller
  model); you can also just pick Qwen 0.5B.
- **Download is blocked by a firewall / proxy** — point the downloader at a
  mirror: `python3 download-models.py --endpoint https://hf-mirror.com`.
- **Port 8000 in use** — `python3 serve.py --port 9000`.

---

## What's in this folder

```
offline-package/
├── index.html            the app (self-contained; loads models from ./models/)
├── download-models.py    one-time model downloader (stdlib only)
├── serve.py              local web server (stdlib only)
├── start.sh              one-step launcher (Linux / macOS / Chromebook)
├── start.bat             one-step launcher (Windows)
├── README.md             this file
└── models/               where the downloaded weights + libraries live
```

Models are Apache-2.0 / Llama-3.2 / Gemma licensed by their respective authors;
the WebLLM engine is Apache-2.0.
