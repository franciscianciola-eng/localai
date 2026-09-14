// End-to-end test: serves the app + the tiny fixture model locally, drives a
// real chat round-trip in headless Chromium (tokenize → chat template → ONNX
// inference on WASM → streamed tokens), and checks the offline diagnosis path.
//
// Usage: npm i playwright-core (or have it available) then:
//   node test/e2e.mjs [path-to-chromium]
import { chromium } from "playwright-core";
import http from "http";
import { readFile } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const executablePath = process.argv[2] || process.env.CHROMIUM || "/opt/pw-browsers/chromium";
const PORT = 8934;

const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream",
};

const server = http.createServer(async (req, res) => {
  let p = decodeURIComponent(req.url.split("?")[0]);
  if (p === "/") p = "/index.html";
  // Fake Hugging Face hub: /hub/<model>/resolve/<rev>/<file> -> test/fixtures/<model>/<file>
  const hub = p.match(/^\/hub\/([^/]+)\/resolve\/[^/]+\/(.+)$/);
  const file = hub
    ? path.join(root, "test", "fixtures", hub[1], hub[2])
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

  const url = `http://localhost:${PORT}/?modelId=tiny-llm&hub=http://localhost:${PORT}/hub/`;
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
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "load" });
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

await browser.close();
server.close();
console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL TESTS PASSED");
process.exit(failures ? 1 : 0);
