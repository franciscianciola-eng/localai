#!/usr/bin/env python3
"""
Download the LocalAI offline models into ./models/ — run this ONCE, on a machine
with internet. After it finishes, the app runs with no internet at all.

  python3 download-models.py            # download all three models (~3 GB)
  python3 download-models.py --only qwen   # just the 0.5B  (smallest, ~0.5 GB)
  python3 download-models.py --only qwen,llama
  python3 download-models.py --list        # show sizes, download nothing
  python3 download-models.py --endpoint https://hf-mirror.com   # use a mirror

It is safe to re-run: files already present (with the right size) are skipped,
so an interrupted download just resumes where it left off.

Nothing here is magic — it mirrors three public Hugging Face repos into
models/<repo>/resolve/main/ (the exact paths the app asks for) and fetches the
three compiled WebGPU model libraries into models/libs/. Stdlib only; no pip.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error, shutil

# --- what to download -------------------------------------------------------
# Each model: the HF repo that holds its weights + config + tokenizer, and the
# compiled WebGPU library (.wasm) that must match those weights. These mirror
# mlc-ai's prebuilt WebLLM config exactly.
MODELS = {
    "qwen":  {
        "repo": "mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
        "lib":  "Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm",
        "note": "Qwen2.5 0.5B — fastest, ~0.5 GB",
    },
    "llama": {
        "repo": "mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
        "lib":  "Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm",
        "note": "Llama 3.2 1B — balanced, ~0.9 GB",
    },
    "gemma": {
        "repo": "mlc-ai/gemma-2-2b-it-q4f16_1-MLC",
        "lib":  "gemma-2-2b-it-q4f16_1_cs1k-webgpu.wasm",
        "note": "Gemma 2 2B — smartest, ~1.6 GB",
    },
}
# The compiled model libraries live in a separate binary repo, versioned by the
# WebLLM runtime this app ships (v0_2_84/base). Same names the app references.
# Overridable with LOCALAI_LIB_BASE for a mirror or a locked-down network.
LIB_BASE = os.environ.get("LOCALAI_LIB_BASE") or \
    "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/"

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
TIMEOUT = 60
UA = {"User-Agent": "localai-offline-downloader/1"}


def endpoint():
    return (os.environ.get("HF_ENDPOINT") or ARGS.endpoint or "https://huggingface.co").rstrip("/")


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024


def http_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        link = r.headers.get("Link", "")
        return json.loads(r.read().decode("utf-8")), link


def list_repo_files(repo):
    """Return [(path, size)] for every file in a repo, via the HF tree API.
    Follows Link-header pagination and reads LFS sizes so resume works."""
    files, page = [], f"{endpoint()}/api/models/{repo}/tree/main?recursive=1&expand=1"
    while page:
        data, link = http_json(page)
        for e in data:
            if e.get("type") == "file":
                size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
                files.append((e["path"], int(size)))
        page = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                page = part[part.find("<") + 1: part.find(">")]
    return files


def download(url, dest, expect=0):
    """Stream url -> dest with retries. Skip if a correctly-sized file exists."""
    if expect and os.path.exists(dest) and os.path.getsize(dest) == expect:
        print(f"    ✓ have {os.path.basename(dest)} ({human(expect)})")
        return expect
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or expect or 0)
                done, t0, lastp = 0, time.time(), -1
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk); done += len(chunk)
                    if total:
                        p = int(done * 100 / total)
                        if p != lastp:
                            speed = done / max(time.time() - t0, 0.001)
                            sys.stdout.write(f"\r    ↓ {os.path.basename(dest)[:42]:42} {p:3d}%  {human(done)}/{human(total)}  {human(speed)}/s   ")
                            sys.stdout.flush(); lastp = p
            if total:
                sys.stdout.write("\n")
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
            wait = 2 ** attempt
            sys.stdout.write(f"\n    ! {type(e).__name__}: {e} — retry {attempt}/5 in {wait}s\n")
            time.sleep(wait)
        finally:
            if os.path.exists(tmp) and not os.path.exists(dest):
                pass  # keep .part for the next attempt's overwrite
    print(f"    ✗ giving up on {url}", file=sys.stderr)
    if os.path.exists(tmp):
        os.remove(tmp)
    raise SystemExit(1)


def plan(keys):
    """Print what will be downloaded and the total size; return the plan."""
    grand, per = 0, []
    for k in keys:
        m = MODELS[k]
        print(f"\n• {m['note']}\n  {m['repo']}")
        try:
            files = list_repo_files(m["repo"])
        except Exception as e:
            print(f"  ! could not list repo ({e}). Check your internet / --endpoint.", file=sys.stderr)
            raise SystemExit(1)
        sub = sum(s for _, s in files)
        for path, size in files:
            print(f"    - {path}  ({human(size)})")
        print(f"    - libs/{m['lib']}  (compiled WebGPU library)")
        print(f"  subtotal: {human(sub)} + 1 library")
        grand += sub; per.append((k, files))
    print(f"\nTotal weights to fetch: {human(grand)} (plus 3 small libraries)\n")
    return per


def main():
    keys = list(MODELS.keys())
    if ARGS.only:
        want = [x.strip().lower() for x in ARGS.only.split(",") if x.strip()]
        bad = [w for w in want if w not in MODELS]
        if bad:
            print(f"Unknown model(s): {', '.join(bad)}. Choose from: {', '.join(MODELS)}", file=sys.stderr)
            raise SystemExit(2)
        keys = want

    print("LocalAI offline model downloader")
    print(f"Endpoint : {endpoint()}")
    print(f"Target   : {MODELS_DIR}")
    per = plan(keys)
    if ARGS.list:
        print("(--list: nothing downloaded)")
        return

    free = shutil.disk_usage(HERE).free
    print(f"Free disk: {human(free)}. Starting download… (safe to Ctrl-C and re-run)\n")

    for k, files in per:
        m = MODELS[k]
        print(f"=== {m['note']} ===")
        base = f"{endpoint()}/{m['repo']}/resolve/main/"
        for path, size in files:
            download(base + path, os.path.join(MODELS_DIR, m["repo"], "resolve", "main", *path.split("/")), size)
        # compiled WebGPU library
        download(LIB_BASE + m["lib"], os.path.join(MODELS_DIR, "libs", m["lib"]))
        print()

    print("✅ Done. All requested models are in ./models/")
    print("Next: start the local server and open the app —")
    print("    python3 serve.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download LocalAI offline models into ./models/")
    ap.add_argument("--only", help="comma-separated subset: qwen,llama,gemma")
    ap.add_argument("--list", action="store_true", help="list files and sizes, download nothing")
    ap.add_argument("--endpoint", default="", help="Hugging Face endpoint (default https://huggingface.co; or set HF_ENDPOINT)")
    ARGS = ap.parse_args()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — re-run to resume where it left off.")
        raise SystemExit(130)
