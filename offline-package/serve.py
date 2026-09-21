#!/usr/bin/env python3
"""
Serve the LocalAI offline app on http://localhost so the browser can load the
model files from ./models/. (Opening index.html directly as a file:// page does
NOT work — browsers block file:// pages from reading other local files.)

  python3 serve.py            # serve on http://localhost:8000 and open a browser
  python3 serve.py --port 9000
  python3 serve.py --no-open  # don't launch a browser

Stop it with Ctrl-C. Everything stays on your machine; this server only talks to
your own browser.
"""
import argparse, functools, http.server, os, socket, sys, threading, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    # Make sure the browser gets the right types: ES modules must be
    # text/javascript, and the WebGPU libraries must be application/wasm.
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
        ".html": "text/html",
        ".svg": "image/svg+xml",
        "": "application/octet-stream",  # params_shard_* have no extension
    }

    def end_headers(self):
        # Long-cache the immutable model files so a reload is instant.
        p = self.path.split("?")[0]
        if "/models/" in p:
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Quieter: only note errors, not every 200.
        try:
            if args and str(args[1]).startswith(("4", "5")):
                sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Serve the LocalAI offline app locally.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(HERE, "index.html")):
        print("! index.html not found next to serve.py — are you in the offline-package folder?", file=sys.stderr)
        raise SystemExit(1)
    models = os.path.join(HERE, "models")
    have_models = os.path.isdir(models) and any(
        os.path.isdir(os.path.join(models, d)) for d in (os.listdir(models) if os.path.isdir(models) else [])
    )
    if not have_models:
        print("! The models/ folder looks empty. Run this first:\n    python3 download-models.py\n", file=sys.stderr)

    handler = functools.partial(Handler, directory=HERE)
    try:
        httpd = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        print(f"! Could not bind {args.host}:{args.port} ({e}). Try --port 9000.", file=sys.stderr)
        raise SystemExit(1)

    url = f"http://localhost:{args.port}/index.html"
    print("LocalAI offline server")
    print(f"  Serving : {HERE}")
    print(f"  Open    : {url}")
    print("  Stop    : Ctrl-C")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
