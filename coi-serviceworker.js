// Service worker with two jobs:
//
// 1) Cross-origin isolation: injects Cross-Origin-Opener-Policy /
//    Cross-Origin-Embedder-Policy headers on same-origin responses so the page
//    becomes cross-origin isolated on static hosts (GitHub Pages) that can't set
//    headers. This unlocks SharedArrayBuffer for multithreaded WASM inference.
//
// 2) Offline support: precaches the small app shell on install and serves every
//    same-origin file cache-first (caching new ones as they're fetched), so the
//    app — and the vendored AI runtimes — load with no network. Model weights
//    are cross-origin and cached separately by transformers.js / WebLLM, so once
//    a model has been downloaded once it also runs offline.
//
// Bump SHELL_VERSION whenever the vendored runtime or this file changes so old
// caches are cleared and a fresh online visit repopulates them.
const SHELL_VERSION = "v14";
const CACHE = "localai-shell-" + SHELL_VERSION;

// Small, always-needed files precached on install. The big files (the 21 MB
// wasm and the 6 MB web-llm bundle) are cached on first fetch instead, so the
// first visit isn't blocked downloading them before anything works.
const PRECACHE = [
  "./",
  "./index.html",
  "./worker.js",
  "./coi-serviceworker.js",
  "./manifest.webmanifest",
  "./vendor/transformers.min.js",
];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE).then((cache) =>
      // Add individually so one missing file doesn't abort the whole precache.
      Promise.all(PRECACHE.map((u) => cache.add(u).catch(() => {})))
    )
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE && k.startsWith("localai-shell-")).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Re-wrap a response with the isolation headers (needed on every same-origin
// response, cached or fresh, for crossOriginIsolated to hold).
function withIsolation(response) {
  if (!response || response.status === 0 || response.type === "opaque" || response.type === "opaqueredirect") {
    return response;
  }
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Embedder-Policy", "credentialless");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;

  const url = new URL(request.url);
  // Cross-origin (model weights, web search, CDNs): don't touch — the browser
  // and the libraries' own Cache API handle these.
  if (url.origin !== self.location.origin) return;
  // Only GET is cacheable.
  if (request.method !== "GET") return;

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);

      // Navigations: network-first so the app updates when online, falling back
      // to the cached page when offline.
      if (request.mode === "navigate") {
        try {
          const net = await fetch(request);
          cache.put(request, net.clone()).catch(() => {});
          return withIsolation(net);
        } catch {
          const cached = (await cache.match(request)) || (await cache.match("./index.html")) || (await cache.match("./"));
          if (cached) return withIsolation(cached);
          return new Response("You are offline and this page isn't cached yet.", { status: 503, headers: { "content-type": "text/plain" } });
        }
      }

      // Everything else same-origin (scripts, wasm, worker): cache-first, then
      // network (caching the result), so the vendored runtime works offline.
      const hit = await cache.match(request);
      if (hit) return withIsolation(hit);
      try {
        const net = await fetch(request);
        if (net && net.ok) cache.put(request, net.clone()).catch(() => {});
        return withIsolation(net);
      } catch (e) {
        const fallback = await cache.match(request);
        if (fallback) return withIsolation(fallback);
        throw e;
      }
    })()
  );
});
