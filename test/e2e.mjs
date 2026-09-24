// End-to-end test: serves the app + the tiny fixture model locally, drives a
// real chat round-trip in headless Chromium (tokenize → chat template → ONNX
// inference on WASM → streamed tokens), and checks the offline diagnosis path.
//
// Usage: npm i playwright-core (or have it available) then:
//   node test/e2e.mjs [path-to-chromium]
import { chromium } from "playwright-core";
import http from "http";
import { readFile } from "fs/promises";
import { writeFileSync, mkdtempSync, mkdirSync, readFileSync } from "fs";
import { randomBytes, createHash } from "crypto";
import { execFileSync } from "child_process";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const executablePath = process.argv[2] || process.env.CHROMIUM || "/opt/pw-browsers/chromium";
const PORT = 8934;

const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream",
  ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json",
};

const server = http.createServer(async (req, res) => {
  let p = decodeURIComponent(req.url.split("?")[0]);
  if (p === "/") p = "/index.html";
  // Fake Hugging Face hub: /hub/<model>/resolve/<rev>/<file> -> test/fixtures/<model>/<file>
  // Simulated school filter: answers every model-file request with an HTML
  // block page and HTTP 200 (the cache-poisoning worst case).
  if (p.startsWith("/hubblock/")) {
    res.writeHead(200, { "content-type": "text/html", "access-control-allow-origin": "*" });
    return res.end("<!DOCTYPE html><html><body>Blocked by NetFilter</body></html>");
  }
  // Fake hub serves the tiny fixture for ANY requested model id, so tests can
  // exercise the app's real model registry (step-down, defaults) offline.
  // A stand-in Wikipedia MediaWiki API for the web-search test.
  if (p === "/wikiapi") {
    res.writeHead(200, { "content-type": "application/json", "access-control-allow-origin": "*" });
    return res.end(JSON.stringify({
      query: { pages: { "1": { index: 1, title: "Photosynthesis", extract: "CANARYTOKEN123 Plants make food from light." } } },
    }));
  }
  // A stand-in Jina Reader returning canned markdown for the full-web path.
  if (p.startsWith("/reader/")) {
    res.writeHead(200, { "content-type": "text/plain", "access-control-allow-origin": "*" });
    return res.end("Search results\n\n[Big News Today](https://example.com/news) WEBCANARY42 the latest headline from the web.\n\n[Other](https://example.org/x) another snippet.");
  }
  // A stand-in CDN mirror serving the vendored runtime files (used to test the
  // fallback when the host itself refuses to serve the big wasm).
  if (p.startsWith("/cdnmirror/")) {
    const f = path.join(root, "vendor", p.slice("/cdnmirror/".length));
    try {
      const data = await readFile(f);
      res.writeHead(200, {
        "content-type": types[path.extname(f)] || "application/octet-stream",
        "access-control-allow-origin": "*",
      });
      return res.end(data);
    } catch { res.writeHead(404); return res.end("nf"); }
  }
  const hub = p.match(/^\/hub\/(.+?)\/resolve\/[^/]+\/(.+)$/);
  const file = hub
    ? path.join(root, "test", "fixtures", "tiny-llm", hub[2])
    : path.join(root, p);
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  try {
    const data = await readFile(file);
    res.writeHead(200, {
      "content-type": types[path.extname(file)] || "application/octet-stream",
      "access-control-allow-origin": "*",
    });
    res.end(data);
  } catch {
    res.writeHead(404); res.end("not found: " + p);
  }
});
await new Promise((r) => server.listen(PORT, r));

const browser = await chromium.launch({ executablePath, args: ["--no-sandbox"] });
let failures = 0;
const check = (name, cond, extra = "") => {
  console.log((cond ? "PASS" : "FAIL") + "  " + name + (cond ? "" : "  " + extra));
  if (!cond) failures++;
};

// ---------- Test 1: full chat round-trip on the tiny fixture model ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

  const url = `http://localhost:${PORT}/?cpu=1&modelId=tiny-llm&hub=http://localhost:${PORT}/hub/`;
  await page.goto(url, { waitUntil: "load" });

  // Type immediately — exercises "typing always works" + the message queue
  // while the model is still loading (and survives the COI reload if any).
  await page.fill("#input", "Say something!");
  await page.press("#input", "Enter");

  // Wait for a completed assistant reply (its .meta stats footer appears).
  const replied = await page
    .waitForSelector(".msg.bot .bubble .meta", { timeout: 90000 })
    .then(() => true)
    .catch(() => false);
  check("assistant reply completed", replied, logs.slice(-8).join(" | "));

  if (replied) {
    const meta = await page.textContent(".msg.bot .bubble .meta");
    check("reply has token stats", /\d+ tokens/.test(meta), meta);
    const chip = await page.textContent("#deviceChip");
    check("device chip shows backend", /CPU|WebGPU/.test(chip), chip);
    console.log("  device:", chip, "| stats:", meta);
    const iso = await page.evaluate(() => ({
      coi: window.crossOriginIsolated,
      threads: navigator.hardwareConcurrency,
    }));
    console.log("  crossOriginIsolated:", iso.coi, "| cores:", iso.threads);
    const userBubble = await page.textContent(".msg.user .bubble");
    check("queued user message rendered", userBubble.includes("Say something!"), userBubble);

    // Second round-trip on the now-ready model (normal send path).
    await page.fill("#input", "And again?");
    await page.click("#sendBtn");
    const replied2 = await page
      .waitForFunction(() => document.querySelectorAll(".msg.bot .bubble .meta").length >= 2, { timeout: 60000 })
      .then(() => true)
      .catch(() => false);
    check("second reply completed", replied2, logs.slice(-8).join(" | "));
  }
  await page.screenshot({ path: path.join(root, "..", "e2e-chat.png") }).catch(() => {});
  await page.screenshot({ path: "/tmp/claude-0/-home-user-localai/1195d466-a8fe-5a3e-93e1-f9839c588b96/scratchpad/e2e-chat.png" }).catch(() => {});
  await page.close();
}

// ---------- Test 2: unreachable model host -> diagnosis card, typing alive ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  // Default models point at huggingface.co, which this environment blocks —
  // exactly the school-network scenario the diagnosis exists for.
  await page.goto(`http://localhost:${PORT}/?cpu=1&nogpu=1`, { waitUntil: "load" });
  const cardShown = await page
    .waitForSelector(".card.error", { timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  check("diagnosis card appears when weights host unreachable", cardShown);
  if (cardShown) {
    const text = await page.textContent(".card.error");
    check("diagnosis names huggingface.co blockage", /huggingface\.co/.test(text), text.slice(0, 200));
    check("retry button present", await page.isVisible("#retryBtn"));
    await page.fill("#input", "still typing fine");
    const val = await page.inputValue("#input");
    check("typing still works after failure", val === "still typing fine");
    check("send button enabled with text", !(await page.isDisabled("#sendBtn")));
  }
  await page.screenshot({ path: "/tmp/claude-0/-home-user-localai/1195d466-a8fe-5a3e-93e1-f9839c588b96/scratchpad/e2e-error.png" }).catch(() => {});
  await page.close();
}

// ---------- Test 3: engine crash mid-generation -> auto-recover + regenerate ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  await page.goto(
    `http://localhost:${PORT}/?cpu=1&nogpu=1&modelId=tiny-llm&hub=http://localhost:${PORT}/hub/&crashgen=1`,
    { waitUntil: "load" }
  );
  await page.fill("#input", "Crash then recover");
  await page.press("#input", "Enter");
  const replied = await page
    .waitForSelector(".msg.bot .bubble .meta", { timeout: 120000 })
    .then(() => true)
    .catch(() => false);
  check("reply completes after engine crash recovery", replied);
  if (replied) {
    const botCount = await page.locator(".msg.bot").count();
    check("crashed partial reply was discarded (one bot message)", botCount === 1, "count=" + botCount);
    const bodyText = await page.textContent("#messages");
    check("no error text shown for recovered crash", !bodyText.includes("⚠️"), bodyText.slice(0, 200));
    const forced = await page.evaluate(() => localStorage.getItem("localai-force-wasm"));
    check("CPU crash does not blacklist the GPU", forced === null, String(forced));
  }
  await page.close();
}

// ---------- Test 4: host blocks the vendored wasm -> load from the CDN ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  // vendorblocked=1 simulates the host's 403 on the big wasm; wasmcdn points the
  // CDN fallback at a local mirror so the whole thing runs offline.
  await page.goto(
    `http://localhost:${PORT}/?cpu=1&nogpu=1&modelId=tiny-llm&hub=http://localhost:${PORT}/hub/` +
    `&vendorblocked=1&wasmcdn=http://localhost:${PORT}/cdnmirror/`,
    { waitUntil: "load" }
  );
  await page.fill("#input", "Load from the CDN please");
  await page.press("#input", "Enter");
  const replied = await page
    .waitForSelector(".msg.bot .bubble .meta", { timeout: 90000 })
    .then(() => true)
    .catch(() => false);
  check("model loads via CDN when host blocks the vendored wasm", replied, logs.slice(-6).join(" | "));
  if (replied) {
    const src = await page.evaluate(() => window.__lastWasmSource);
    check("runtime source reported as CDN", src === "cdn", String(src));
  }
  await page.close();
}

// ---------- Test 5: WebLLM engine plumbing (mock) loads and streams ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const logs = [];
  page.on("pageerror", (e) => logs.push("PAGEERR " + e.message));
  // forcewebllm routes to the WebLLM engine; webllmmock swaps in a fake module.
  await page.goto(`http://localhost:${PORT}/?forcewebllm=1&webllmmock=1`, { waitUntil: "load" });
  await page.fill("#input", "hi webgpu engine");
  await page.press("#input", "Enter");
  const replied = await page
    .waitForSelector(".msg.bot .bubble .meta", { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  check("WebLLM engine loads and completes a reply", replied, logs.join(" | "));
  if (replied) {
    const chip = await page.textContent("#deviceChip");
    check("device chip shows WebLLM", /WebLLM/.test(chip), chip);
    const text = await page.textContent(".msg.bot .bubble");
    check("WebLLM streamed the reply text", text.includes("WebGPU engine"), text.slice(0, 80));
    const eng = await page.evaluate(() => window.__engine);
    check("engine reported as webllm", eng === "webllm", String(eng));
  }
  check("no page errors in WebLLM path", logs.length === 0, logs.join(" | "));
  await page.close();
}

// ---------- Test 6: a GPU-only model routes to WebLLM when WebGPU is present ----------
{
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  await ctx.addInitScript(() => { try { localStorage.setItem("localai-model", "phi35-mini"); } catch {} });
  const page = await ctx.newPage();
  // webllmmock makes hasWebGPU() true and swaps in the fake WebLLM module.
  await page.goto(`http://localhost:${PORT}/?webllmmock=1`, { waitUntil: "load" });
  await page.fill("#input", "hello big model");
  await page.press("#input", "Enter");
  const replied = await page
    .waitForSelector(".msg.bot .bubble .meta", { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  check("GPU-only model routes to WebLLM and replies", replied);
  if (replied) {
    const eng = await page.evaluate(() => window.__engine);
    check("GPU-only model used the webllm engine", eng === "webllm", String(eng));
  }
  await ctx.close();
}

// ---------- Test 7: a GPU-only model without WebGPU steps down to a CPU model ----------
{
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  await ctx.addInitScript(() => { try { localStorage.setItem("localai-model", "gemma2-2b"); } catch {} });
  const page = await ctx.newPage();
  // nogpu forces "no WebGPU": Gemma (WebGPU-only) can't run, so the app should
  // step down to the largest CPU-capable model (Llama 1B) and load it. hub
  // serves the tiny fixture for any model id so it loads offline.
  await page.goto(`http://localhost:${PORT}/?nogpu=1&cpu=1&hub=http://localhost:${PORT}/hub/`, { waitUntil: "load" });
  const settled = await page
    .waitForFunction(() =>
      document.getElementById("modelSelect").value === "llama32-1b" &&
      document.getElementById("loader").hidden &&
      !document.querySelector(".card.error"), { timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  check("no-WebGPU big model steps down to a CPU-capable model", settled);
  await ctx.close();
}

// ---------- Test 8: WebGPU crash mid-reply -> throttle down and recover ----------
{
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  await ctx.addInitScript(() => {
    try { localStorage.setItem("localai-model", "phi35-mini"); } catch {}
    window.__mockGenCrashes = 1; // first generation attempt "crashes" the GPU
  });
  const page = await ctx.newPage();
  await page.goto(`http://localhost:${PORT}/?webllmmock=1`, { waitUntil: "load" });
  await page.fill("#input", "run the big model");
  await page.press("#input", "Enter");
  const replied = await page
    .waitForSelector(".msg.bot .bubble .meta", { timeout: 40000 })
    .then(() => true)
    .catch(() => false);
  check("recovers and replies after a WebGPU crash mid-reply", replied);
  if (replied) {
    const thr = await page.evaluate(() => window.__throttle);
    check("throttle level escalated after the crash", thr >= 1, "throttle=" + thr);
    const chip = await page.textContent("#deviceChip");
    check("device chip shows the reduced-load mode", /eco|low-power/i.test(chip), chip);
    const model = await page.evaluate(() => window.__mockModelId);
    check("still on the same model (throttled, not stepped down)", /Phi-3\.5/i.test(model), model);
  }
  await ctx.close();
}

// ---------- Test 9: big model that won't load at all -> steps down after backing off ----------
{
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  await ctx.addInitScript(() => {
    try { localStorage.setItem("localai-model", "llama32-1b"); } catch {}
    // The big models won't load at all; the small ones (Qwen/SmolLM2) load fine.
    window.__mockLoadCrashPattern = "Llama-3\\.2-1B|gemma|Phi";
  });
  const page = await ctx.newPage();
  await page.goto(`http://localhost:${PORT}/?webllmmock=1`, { waitUntil: "load" });
  // It should back off through throttle levels, step down through the models,
  // and finally land on a smaller model that loads.
  const settled = await page
    .waitForFunction(() =>
      document.getElementById("loader").hidden &&
      !document.querySelector(".card.error") &&
      window.__mockModelId, { timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  check("un-loadable big model backs off and steps down to a working model", settled);
  if (settled) {
    const loaded = await page.evaluate(() => window.__mockModelId);
    check("landed on a smaller model that loads", /Qwen2\.5-0\.5B|SmolLM2/i.test(loaded), loaded);
  }
  await ctx.close();
}

// ---------- Test 10: settings panel opens, toggles persist, theme applies ----------
{
  // Isolated context so the settings written here don't leak into other tests.
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  const page = await ctx.newPage();
  await page.goto(`http://localhost:${PORT}/?cpu=1&nogpu=1&modelId=tiny-llm&hub=http://localhost:${PORT}/hub/`, { waitUntil: "load" });
  await page.click("#settingsBtn");
  check("settings modal opens", await page.isVisible(".modal"));
  // Set web search to "always" and confirm it persists.
  await page.click('#setSearchMode button[data-v="always"]');
  const persisted = await page.evaluate(() => JSON.parse(localStorage.getItem("localai-settings") || "{}").searchMode);
  check("web search mode persists", persisted === "always", String(persisted));
  check("header shows the web-search indicator", await page.isVisible("#searchNote"));
  // Dark theme applies immediately.
  await page.click('#setTheme button[data-v="dark"]');
  const theme = await page.getAttribute("html", "data-theme");
  check("dark theme applied", theme === "dark", String(theme));
  await page.click("#settingsClose");
  check("settings modal closes", !(await page.isVisible(".modal")));
  await ctx.close();
}

// ---------- Test 11: "always" mode injects full-web + Wikipedia context ----------
{
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  await ctx.addInitScript(() => {
    try { localStorage.setItem("localai-settings", JSON.stringify({ searchMode: "always" })); } catch {}
  });
  const page = await ctx.newPage();
  await page.goto(`http://localhost:${PORT}/?forcewebllm=1&webllmmock=1` +
    `&searchapi=http://localhost:${PORT}/wikiapi&readerbase=http://localhost:${PORT}/reader/`, { waitUntil: "load" });
  await page.fill("#input", "how do plants eat");
  await page.press("#input", "Enter");
  const replied = await page
    .waitForSelector(".msg.bot .bubble .meta", { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  check("reply completes with web search on", replied);
  if (replied) {
    const msgs = await page.evaluate(() => JSON.stringify(window.__mockLastMessages || []));
    check("full-web (Jina Reader) context injected", msgs.includes("WEBCANARY42"), msgs.slice(0, 160));
    check("Wikipedia context also injected", msgs.includes("CANARYTOKEN123"));
    check("results are labeled WEB SEARCH RESULTS for the model", msgs.includes("WEB SEARCH RESULTS"));
    check("system prompt tells the model it has a web search tool", /web search tool/i.test(msgs));
    const src = await page.textContent("#messages");
    check("search sources shown in the chat", /Searched the web/s.test(src));
  }
  await ctx.close();
}

// ---------- Test 11b: "smart" mode makes a conscious decision (no useless search) ----------
{
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  await ctx.addInitScript(() => {
    try { localStorage.setItem("localai-settings", JSON.stringify({ searchMode: "smart" })); } catch {}
  });
  const page = await ctx.newPage();
  await page.goto(`http://localhost:${PORT}/?forcewebllm=1&webllmmock=1` +
    `&searchapi=http://localhost:${PORT}/wikiapi&readerbase=http://localhost:${PORT}/reader/`, { waitUntil: "load" });

  // A greeting: the heuristic says no search — nothing should be fetched.
  await page.fill("#input", "hey there");
  await page.press("#input", "Enter");
  await page.waitForSelector(".msg.bot .bubble .meta", { timeout: 30000 }).catch(() => {});
  let msgs = await page.evaluate(() => JSON.stringify(window.__mockLastMessages || []));
  check("smart mode does NOT search a greeting", !msgs.includes("WEBCANARY42") && !msgs.includes("CANARYTOKEN123"));
  check("no search-sources line for the greeting", !(await page.textContent("#messages")).includes("Searched the web"));

  // A clearly time-sensitive question: the heuristic says search.
  await page.fill("#input", "what is the latest news on mars");
  await page.press("#input", "Enter");
  await page.waitForFunction(() => document.querySelectorAll(".msg.bot .bubble .meta").length >= 2, { timeout: 30000 }).catch(() => {});
  msgs = await page.evaluate(() => JSON.stringify(window.__mockLastMessages || []));
  check("smart mode DOES search a time-sensitive question", msgs.includes("WEBCANARY42") || msgs.includes("CANARYTOKEN123"), msgs.slice(0, 160));
  await ctx.close();
}

// ---------- Test 12: heavy model crashes at load -> auto step-down to smaller ----------
{
  const context = await browser.newContext({ viewport: { width: 900, height: 700 } });
  await context.addInitScript(() => {
    try { localStorage.setItem("localai-model", "smollm2-360m"); } catch {}
  });
  const page = await context.newPage();
  await page.goto(
    `http://localhost:${PORT}/?cpu=1&nogpu=1&hub=http://localhost:${PORT}/hub/&crashload=1`,
    { waitUntil: "load" }
  );
  // Every load attempt for the selected 360M model crashes (test hook); the
  // app should step down to 135M automatically and load it successfully.
  const ready = await page
    .waitForFunction(() =>
      document.getElementById("modelSelect").value === "smollm2-135m" &&
      document.getElementById("loader").hidden &&
      !document.querySelector(".card.error"), { timeout: 90000 })
    .then(() => true)
    .catch(() => false);
  check("app steps down to the smallest model and loads it", ready);
  if (ready) {
    await page.fill("#input", "Does the smaller model talk?");
    await page.click("#sendBtn");
    const replied = await page
      .waitForSelector(".msg.bot .bubble .meta", { timeout: 60000 })
      .then(() => true)
      .catch(() => false);
    check("stepped-down model generates a reply", replied);
  }
  await context.close();
}

// ---------- Test 5: filter serving a 200 HTML block page -> detected ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  await page.goto(
    `http://localhost:${PORT}/?cpu=1&nogpu=1&modelId=tiny-llm&hub=http://localhost:${PORT}/hubblock/`,
    { waitUntil: "load" }
  );
  const cardShown = await page
    .waitForSelector(".card.error", { timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  check("error card appears for HTML block page", cardShown);
  if (cardShown) {
    const text = await page.textContent(".card.error");
    check("diagnosis identifies the block page", /block page/.test(text), text.slice(0, 250));
    check("cache-clear retry was attempted", /"cacheCleared":\s*true/.test(text));
  }
  await page.close();
}

// ---------- Test 13: works offline — app shell + cached model load with no network ----------
{
  const ctx = await browser.newContext({ viewport: { width: 900, height: 700 } });
  const page = await ctx.newPage();
  const url = `http://localhost:${PORT}/?cpu=1&nogpu=1&modelId=tiny-llm&hub=http://localhost:${PORT}/hub/`;
  // First visit ONLINE: caches the app shell, the vendored runtime, and the model.
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => navigator.serviceWorker && navigator.serviceWorker.controller, { timeout: 20000 }).catch(() => {});
  await page.waitForFunction(() =>
    document.getElementById("loader").hidden && /CPU|WebGPU/.test(document.getElementById("deviceChip").textContent),
    { timeout: 90000 }).catch(() => {});
  await page.fill("#input", "cache everything please");
  await page.press("#input", "Enter");
  const onlineReplied = await page.waitForSelector(".msg.bot .bubble .meta", { timeout: 90000 }).then(() => true).catch(() => false);
  check("online: model generated (and everything is now cached)", onlineReplied);

  // GO OFFLINE and reload — nothing may touch the network now.
  await ctx.setOffline(true);
  await page.reload({ waitUntil: "load" }).catch(() => {});
  const shellLoaded = await page.waitForSelector("#modelSelect", { timeout: 20000 }).then(() => true).catch(() => false);
  check("offline: the app shell loads from cache", shellLoaded);
  check("offline indicator is shown", await page.isVisible("#offlineChip"));
  // The cached model should still load and generate with no network.
  const ready = await page.waitForFunction(() =>
    document.getElementById("loader").hidden && !document.querySelector(".card.error") &&
    /CPU|WebGPU/.test(document.getElementById("deviceChip").textContent),
    { timeout: 90000 }).then(() => true).catch(() => false);
  check("offline: cached model loads with no network", ready);
  if (ready) {
    const before = await page.locator(".msg.bot .bubble .meta").count();
    await page.fill("#input", "answer me while offline");
    await page.click("#sendBtn");
    const offlineReplied = await page.waitForFunction(
      (n) => document.querySelectorAll(".msg.bot .bubble .meta").length > n, before, { timeout: 90000 }
    ).then(() => true).catch(() => false);
    check("offline: model generates a fresh reply with no network", offlineReplied);
  }
  await ctx.close();
}

// ---------- Test 14: the standalone single-file build is valid and runs ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const logs = [];
  page.on("pageerror", (e) => logs.push("PAGEERR " + e.message));
  // Run as the dedicated app window (window.name marker) so it loads the app,
  // not the pop-out launcher.
  await page.addInitScript(() => { window.name = "localai-app"; });
  await page.goto(`http://localhost:${PORT}/localai-standalone.html`, { waitUntil: "load" });
  await page.waitForTimeout(2500);
  const models = await page.evaluate(() => [...document.getElementById("modelSelect").options].map((o) => o.value));
  check("standalone offers exactly the 3 requested models", JSON.stringify(models) === JSON.stringify(["qwen25-05b", "llama32-1b", "gemma2-2b"]), JSON.stringify(models));
  check("standalone inline module runs without page errors", logs.length === 0, logs.join(" | "));
  // Headless has no real WebGPU, so it should show the clear WebGPU-needed message.
  const err = await page.evaluate(() => document.querySelector(".card.error")?.textContent || "");
  check("standalone gates on WebGPU with a clear message", /needs WebGPU/i.test(err), err.slice(0, 120));
  await page.close();
}

// ---------- Test 14b: the standalone pops out into its own window ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const popups = [];
  page.on("popup", (p) => popups.push(p));
  await page.goto(`http://localhost:${PORT}/localai-standalone.html`, { waitUntil: "load" });
  await page.waitForTimeout(800);
  // A normal tab shows the launcher (not the app itself).
  const hasLauncher = await page.$("#openWin");
  const appGone = await page.evaluate(() => !document.getElementById("modelSelect"));
  check("standalone shows a separate-window launcher in a normal tab", !!hasLauncher && appGone);
  // Clicking it opens a real separate window that runs the app.
  await page.click("#openWin").catch(() => {});
  await page.waitForTimeout(1500);
  check("standalone launcher opens a separate window", popups.length >= 1, "popups=" + popups.length);
  if (popups.length) {
    const popup = popups[0];
    await popup.waitForTimeout(2000);
    const popModels = await popup.evaluate(() => [...(document.getElementById("modelSelect")?.options || [])].map((o) => o.value));
    check("the separate window runs the app (3 models)", JSON.stringify(popModels) === JSON.stringify(["qwen25-05b", "llama32-1b", "gemma2-2b"]), JSON.stringify(popModels));
    await popup.close();
  }
  await page.close();
}

// ---------- Test 14c: standalone "Think" mode hides reasoning, shows clean answer ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const errs = [];
  page.on("pageerror", (e) => errs.push(String(e)));
  await page.addInitScript(() => { window.name = "localai-app"; });
  await page.goto(`http://localhost:${PORT}/localai-standalone.html?thinktest=1`, { waitUntil: "load" });
  await page.waitForTimeout(1500);
  const r = await page.evaluate(() => {
    const T = window.__thinkTest; if (!T) return { err: "no hook" };
    T.setThinking(true);
    const raw = "<think>ball=x, bat=x+1, so 2x+1=1.10 → x=0.05</think>The ball costs 5 cents.";
    const out = {
      answer: T.splitThink(raw).answer.trim(),
      finished: T.replyHTML(raw, false),
      thinking: T.replyHTML("<think>working it out", true),
      unclosed: T.replyHTML("<think>no close tag here", false),
    };
    T.setThinking(false);
    out.off = T.replyHTML(raw, false);
    return out;
  });
  check("think: reasoning is parsed out of the answer", r.answer === "The ball costs 5 cents.", r.answer);
  check("think: finished reply shows a Thoughts toggle + clean answer, reasoning hidden",
    /🧠 Thoughts/.test(r.finished) && /The ball costs 5 cents\./.test(r.finished) && /think-body" hidden/.test(r.finished));
  check("think: reasoning streams live while thinking (visible, answer not leaked)",
    /Thinking…/.test(r.thinking) && /working it out/.test(r.thinking) && !/think-body" hidden/.test(r.thinking) && !/costs/.test(r.thinking));
  check("think: an unclosed <think> is revealed as the answer (nothing lost)",
    /no close tag here/.test(r.unclosed) && !/Thoughts/.test(r.unclosed));
  check("think: with Think off, the answer renders plainly (no toggle)",
    /5 cents/.test(r.off) && !/Thoughts/.test(r.off));
  check("think: no page errors", errs.filter((e) => !/webgpu/i.test(e)).length === 0, errs.join(" | "));
  await page.close();
}

// ---------- Test 14d: attachments — text files read, images OCR'd, fed to model ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const errs = [];
  page.on("pageerror", (e) => errs.push(String(e)));
  // App window + a stub OCR engine so the image path is deterministic offline.
  await page.addInitScript(() => {
    window.name = "localai-app";
    window.Tesseract = { createWorker: async () => ({ recognize: async () => ({ data: { text: "INVOICE TOTAL 1234" } }), terminate: async () => {} }) };
  });
  await page.goto(`http://localhost:${PORT}/localai-standalone.html?thinktest=1`, { waitUntil: "load" });
  await page.waitForTimeout(1200);

  // Pure-function checks.
  const u = await page.evaluate(() => {
    const T = window.__thinkTest;
    return {
      txtA: T.isTextFile("notes.md", ""), txtB: T.isTextFile("x", "text/plain"), notImg: T.isTextFile("p.png", "image/png"),
      fileContent: T.buildModelContent("Summarize", [{ kind: "file", name: "a.txt", mime: "text/plain", text: "HELLO CONTENT" }]),
      imgNoText: T.buildModelContent("what is this", [{ kind: "image", name: "p.png", text: "" }]),
    };
  });
  check("attach: text-file detection", u.txtA && u.txtB && !u.notImg);
  check("attach: a text file's contents are put into the model message",
    /a\.txt/.test(u.fileContent) && /HELLO CONTENT/.test(u.fileContent) && /Summarize/.test(u.fileContent));
  check("attach: an unreadable image tells the model it can't see images",
    /cannot see/i.test(u.imgNoText) && /p\.png/.test(u.imgNoText));

  // Real file inputs: a text file and an image (OCR via the stub).
  const tmp = mkdtempSync(path.join(os.tmpdir(), "att-"));
  const txtPath = path.join(tmp, "greeting.txt");
  writeFileSync(txtPath, "Hello from a text file. The secret code is BANANA.");
  const pngPath = path.join(tmp, "pic.png");
  writeFileSync(pngPath, Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "base64"));

  await page.setInputFiles("#fileInput", txtPath);
  await page.setInputFiles("#fileInput", pngPath);
  // Wait until both attachments finish processing.
  let atts = [];
  for (let i = 0; i < 40; i++) {
    atts = await page.evaluate(() => window.__thinkTest.getAttachments());
    if (atts.length === 2 && atts.every((a) => a.status === "ready")) break;
    await page.waitForTimeout(150);
  }
  const txt = atts.find((a) => a.kind === "file"), img = atts.find((a) => a.kind === "image");
  check("attach: the text file is read", !!txt && /BANANA/.test(txt.text || ""), JSON.stringify(txt));
  check("attach: the image is OCR'd into text", !!img && (img.text || "") === "INVOICE TOTAL 1234", JSON.stringify(img));

  // Sending shows both in the user's bubble (model isn't loaded, so it just queues).
  await page.fill("#input", "what do these say?");
  await page.click("#sendBtn");
  await page.waitForTimeout(300);
  const bubble = await page.$eval(".msg.user .bubble", (el) => el.innerHTML).catch(() => "");
  check("attach: sent message shows the image thumbnail and file chip + text",
    /att-strip/.test(bubble) && /<img/.test(bubble) && /greeting\.txt/.test(bubble) && /what do these say\?/.test(bubble));
  check("attach: the attachment tray clears after sending",
    (await page.evaluate(() => window.__thinkTest.getAttachments())).length === 0);
  check("attach: no page errors", errs.filter((e) => !/webgpu/i.test(e)).length === 0, errs.join(" | "));
  await page.close();
}

// ---------- Test 14e: OCR engine is inlined and reads image text with NO network ----------
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  await page.addInitScript(() => { window.name = "localai-app"; });
  // Allow only the local test server (the app itself); block everything else so
  // a stray CDN fetch would fail the OCR — proving it's truly offline.
  await page.route("**/*", (r) => (r.request().url().startsWith(`http://localhost:${PORT}`) ? r.continue() : r.abort()));
  await page.goto(`http://localhost:${PORT}/localai-standalone.html?thinktest=1`, { waitUntil: "load" });
  await page.waitForTimeout(800);
  const inlined = await page.evaluate(() => window.__thinkTest.ocrInlined());
  check("attach: OCR engine is bundled into the file (not a CDN)", inlined === true);
  let text = "";
  try {
    text = await page.evaluate(async () => {
      const c = document.createElement("canvas"); c.width = 760; c.height = 130;
      const g = c.getContext("2d"); g.fillStyle = "#fff"; g.fillRect(0, 0, c.width, c.height);
      g.fillStyle = "#000"; g.font = "38px Georgia"; g.fillText("Grocery total is 42 dollars", 20, 82);
      return await window.__thinkTest.ocrDataUrl(c.toDataURL("image/png"));
    });
  } catch (e) { text = "ERROR " + e; }
  check("attach: offline OCR reads the image text (network blocked)", /Grocery total is 42/i.test(text || ""), JSON.stringify((text || "").slice(0, 80)));
  await page.close();
}

// ---------- Test 14g: a model BUILT INTO the file loads with no download ----------
// Builds a synthetic model in the download layout, embeds it with
// build-embedded.mjs (tiny parts, to exercise chunking), then opens the result
// from file:// with every network request blocked and checks that the files land
// byte-identical in the caches WebLLM reads — and that WebLLM agrees it's ready.
{
  const tmp = mkdtempSync(path.join(os.tmpdir(), "embed-"));
  const repo = path.join(tmp, "models", "mlc-ai", "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", "resolve", "main");
  mkdirSync(repo, { recursive: true }); mkdirSync(path.join(tmp, "models", "libs"), { recursive: true });
  const s0 = randomBytes(2 * 1024 * 1024 + 321), s1 = randomBytes(900 * 1024 + 5), lib = randomBytes(150 * 1024);
  writeFileSync(path.join(repo, "mlc-chat-config.json"), JSON.stringify({ tokenizer_files: ["tokenizer.json", "vocab.json"] }));
  writeFileSync(path.join(repo, "tensor-cache.json"), JSON.stringify({ metadata: {}, records: [{ dataPath: "params_shard_0.bin", records: [] }, { dataPath: "params_shard_1.bin", records: [] }] }));
  writeFileSync(path.join(repo, "params_shard_0.bin"), s0); writeFileSync(path.join(repo, "params_shard_1.bin"), s1);
  writeFileSync(path.join(repo, "tokenizer.json"), JSON.stringify({ model: { type: "BPE" } }));
  writeFileSync(path.join(tmp, "models", "libs", "Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm"), lib);
  const out = path.join(tmp, "embedded.html");
  execFileSync(process.execPath, ["--no-warnings", path.join(root, "build-embedded.mjs"), "--models", "qwen",
    "--models-dir", path.join(tmp, "models"), "--part-mb", "1", "--out", out], { stdio: "ignore" });
  const hub = "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC/resolve/main/";
  const sha = (b) => createHash("sha256").update(b).digest("hex");
  const want = {
    [hub + "params_shard_0.bin"]: ["webllm/model", sha(s0)],
    [hub + "params_shard_1.bin"]: ["webllm/model", sha(s1)],
    [hub + "mlc-chat-config.json"]: ["webllm/config", sha(readFileSync(path.join(repo, "mlc-chat-config.json")))],
    "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm": ["webllm/wasm", sha(lib)],
  };

  const ctx = await browser.newContext();
  const net = [], errs = [];
  const openIt = async () => {
    const page = await ctx.newPage();
    page.on("pageerror", (e) => errs.push(String(e)));
    await page.addInitScript(() => { window.name = "localai-app"; });
    await page.route("**/*", (r) => { const u = r.request().url(); if (u.startsWith("file:")) return r.continue(); net.push(u); return r.abort(); });
    await page.goto("file://" + out + "?thinktest=1", { waitUntil: "load" });
    await page.waitForTimeout(1000);
    return page;
  };
  let page = await openIt();
  const info = await page.evaluate(() => ({
    e: window.__thinkTest.embedInfo(), left: window.__thinkTest.payloadLeft(),
    label: document.querySelector("#modelSelect").selectedOptions[0].textContent,
    hero: document.getElementById("heroText").textContent,
  }));
  check("embed: the file carries a manifest for the built-in model", !!info.e && info.e.models[0] === "qwen25-05b" && info.e.files.length === 6, JSON.stringify(info.e && info.e.files.length));
  check("embed: model data is chunked into parts inside the page", info.left > 5, "parts=" + info.left);
  check("embed: opens on the built-in model, labelled 'built in'", /Qwen2\.5/.test(info.label) && /built in ✓/.test(info.label), info.label);
  check("embed: the page says the model is built in (nothing to download)", /built into this file/.test(info.hero));
  await page.evaluate(() => window.__thinkTest.importEmbedded());
  const got = await page.evaluate(async (want) => {
    const out = {};
    for (const [url, [name]] of Object.entries(want)) {
      const r = await (await caches.open(name)).match(url);
      out[url] = r ? [...new Uint8Array(await crypto.subtle.digest("SHA-256", await r.arrayBuffer()))].map((b) => b.toString(16).padStart(2, "0")).join("") : null;
    }
    return { out, left: window.__thinkTest.payloadLeft(), ready: await window.__thinkTest.webllmHasModel("Qwen2.5-0.5B-Instruct-q4f16_1-MLC") };
  }, want);
  check("embed: every file is unpacked byte-identical into the cache WebLLM reads",
    Object.entries(want).every(([u, [, h]]) => got.out[u] === h), JSON.stringify(got.out).slice(0, 160));
  check("embed: WebLLM's own hasModelInCache() reports the model ready", got.ready === true);
  check("embed: the embedded copy is dropped from the page after unpacking", got.left === 0);
  await page.close();
  page = await openIt();   // reopen: already unpacked, so it's a quick no-op
  await page.evaluate(() => window.__thinkTest.importEmbedded());
  check("embed: reopening is instant and still ready", await page.evaluate(() => window.__thinkTest.webllmHasModel("Qwen2.5-0.5B-Instruct-q4f16_1-MLC")));
  check("embed: zero network requests were attempted", net.length === 0, net.slice(0, 2).join(" "));
  check("embed: no page errors", errs.filter((e) => !/webgpu/i.test(e)).length === 0, errs.join(" | "));
  await ctx.close();
}

// ---------- Test 15: the offline-package build is valid and runs ----------
// Served over http (not file://), with no models present. Catches inline-bundle
// identifier collisions (e.g. a top-level `lib`) and confirms the local-appConfig
// app boots and gates cleanly.
{
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const logs = [];
  page.on("pageerror", (e) => logs.push("PAGEERR " + e.message));
  await page.goto(`http://localhost:${PORT}/offline-package/index.html`, { waitUntil: "load" });
  await page.waitForTimeout(2500);
  const models = await page.evaluate(() => [...document.getElementById("modelSelect").options].map((o) => o.value));
  check("offline package offers exactly the 3 requested models", JSON.stringify(models) === JSON.stringify(["qwen25-05b", "llama32-1b", "gemma2-2b"]), JSON.stringify(models));
  check("offline package inline module runs without page errors", logs.length === 0, logs.join(" | "));
  // No WebGPU in headless → clear WebGPU-needed message (checked before the models fetch).
  const err = await page.evaluate(() => document.querySelector(".card.error")?.textContent || "");
  check("offline package gates on WebGPU with a clear message", /needs WebGPU/i.test(err), err.slice(0, 120));
  await page.close();
}

await browser.close();
server.close();
console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL TESTS PASSED");
process.exit(failures ? 1 : 0);
