// Builds localai-standalone.html: a single self-contained HTML file that inlines
// the vendored WebLLM bundle AND the Tesseract OCR engine (engine + English
// data), so it runs from file:// on a Chromebook with no external scripts and
// reads text from images fully offline. Run: node build-standalone.mjs
import { readFile, writeFile } from "fs/promises";
import { fileURLToPath } from "url";
import path from "path";

const dir = path.dirname(fileURLToPath(import.meta.url));
const tpl = await readFile(path.join(dir, "standalone.template.html"), "utf8");
const bundle = await readFile(path.join(dir, "vendor", "web-llm.js"), "utf8");

// A literal </script (any case) in the inlined-raw WebLLM bundle would close the
// inline <script> element and break the page. (The OCR assets are base64, so
// they can't contain that sequence.) The current bundle has none.
if (/<\/script/i.test(bundle)) {
  throw new Error("web-llm bundle contains a </script> sequence — cannot inline safely");
}
const marker = "/*__WEBLLM_BUNDLE__*/";
if (!tpl.includes(marker)) throw new Error("template is missing the bundle marker");

// The OCR engine, inlined as base64 (decoded + run at OCR time, in a worker).
const tessDir = path.join(dir, "vendor", "tesseract");
const tessB64 = async (f) => (await readFile(path.join(tessDir, f))).toString("base64");
const tess = {
  "__TESS_MAIN_B64__": await tessB64("tesseract.min.js"),
  "__TESS_WORKER_B64__": await tessB64("worker.min.js"),
  "__TESS_CORE_B64__": await tessB64("tesseract-core-simd-lstm.wasm.js"),
  "__TESS_LANG_B64__": await tessB64("eng.traineddata.gz"),
};

// Function replacers so `$` sequences in the payloads aren't treated as special.
let out = tpl.replace(marker, () => bundle);
for (const [placeholder, b64] of Object.entries(tess)) {
  if (!out.includes(placeholder)) throw new Error("template is missing placeholder " + placeholder);
  out = out.replace(placeholder, () => b64);
}
await writeFile(path.join(dir, "localai-standalone.html"), out);
console.log("wrote localai-standalone.html (" + (out.length / 1048576).toFixed(1) + " MB)");
