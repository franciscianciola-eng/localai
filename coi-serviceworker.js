// Injects Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy headers on
// same-origin responses so the page becomes cross-origin isolated on static
// hosts (like GitHub Pages) that can't set headers. This unlocks
// SharedArrayBuffer, which lets the WASM inference backend use all CPU cores.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  // Only rewrite same-origin responses (the page and its scripts); cross-origin
  // model/CDN downloads go straight to the network untouched.
  if (new URL(request.url).origin !== self.location.origin) return;

  event.respondWith(
    (async () => {
      const response = await fetch(request);
      if (response.status === 0 || response.type === "opaque" || response.type === "opaqueredirect") {
        return response;
      }
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Embedder-Policy", "credentialless");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    })()
  );
});
