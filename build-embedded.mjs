// Builds a single HTML file with a model BUILT IN, so it needs no download at
// all. It takes localai-standalone.html and appends the model's files (config,
// tokenizer, weight shards, compiled WebGPU library) as base64 chunks inside
// inert <script> elements, plus a manifest. On open, the app unpacks them into
// the same browser caches WebLLM fills after a download, keyed by the same URLs,
// so WebLLM loads the model without touching the network.
//
//   node build-embedded.mjs --models qwen
//        [--models-dir offline-package/models] [--in localai-standalone.html]
//        [--out dist/LocalAI-Qwen2.5-0.5B-offline.html] [--part-mb 24]
//
// The models dir uses the layout offline-package/download-models.py writes:
//   <dir>/mlc-ai/<model>/resolve/main/<files>   and   <dir>/libs/<lib>.wasm
import { readFile, stat, mkdir } from "fs/promises";
import { createWriteStream } from "fs";
import { once } from "events";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const dir = path.dirname(fileURLToPath(import.meta.url));

const CATALOG = {
  qwen: {
    key: "qwen25-05b", name: "Qwen2.5-0.5B", ids: ["Qwen2.5-0.5B-Instruct-q4f16_1-MLC"], apache: true,
    notice: "Qwen2.5-0.5B-Instruct © Alibaba Cloud, used under the Apache License 2.0 (a copy is included in this file).",
  },
  llama: {
    key: "llama32-1b", name: "Llama-3.2-1B", ids: ["Llama-3.2-1B-Instruct-q4f16_1-MLC"],
    notice: "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Built with Llama.",
  },
  gemma: {
    key: "gemma2-2b", name: "Gemma-2-2B", ids: ["gemma-2-2b-it-q4f16_1-MLC", "gemma-2-2b-it-q4f16_1-MLC-1k"],
    notice: "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.",
  },
};

const arg = (name, def) => { const i = process.argv.indexOf("--" + name); return i > 0 ? process.argv[i + 1] : def; };
const want = arg("models", "qwen").split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
for (const w of want) if (!CATALOG[w]) throw new Error(`unknown model "${w}" (choose from ${Object.keys(CATALOG).join(", ")})`);
const modelsDir = path.resolve(arg("models-dir", path.join(dir, "offline-package", "models")));
const input = path.resolve(arg("in", path.join(dir, "localai-standalone.html")));
const partBytes = Math.max(1, Number(arg("part-mb", "24"))) * 1024 * 1024;
const out = path.resolve(arg("out", path.join(dir, "dist", `LocalAI-${want.map((w) => CATALOG[w].name).join("+")}-offline.html`)));

// The exact URLs WebLLM looks up come from its own prebuilt config.
const { prebuiltAppConfig } = await import(pathToFileURL(path.join(dir, "vendor", "web-llm.js")).href);
const cleanModelUrl = (u) => { u += u.endsWith("/") ? "" : "/"; if (!/.+\/resolve\/.+\//.test(u)) u += "resolve/main/"; return u; };

// Collect every file WebLLM reads for these models, with its cache + key URL.
const files = [], seen = new Set();
const add = async (f) => {
  const k = f.cache + " " + f.url;
  if (seen.has(k)) return;
  seen.add(k);
  f.size = (await stat(f.local).catch(() => { throw new Error("missing model file: " + f.local); })).size;
  files.push(f);
};
for (const w of want) {
  for (const id of CATALOG[w].ids) {
    const rec = prebuiltAppConfig.model_list.find((r) => r.model_id === id);
    if (!rec) throw new Error("WebLLM has no model record for " + id);
    const base = cleanModelUrl(rec.model);                                   // …/resolve/main/
    const repo = new URL(rec.model).pathname.replace(/^\/+|\/+$/g, "");       // mlc-ai/<model>
    const local = (p) => path.join(modelsDir, ...repo.split("/"), "resolve", "main", ...p.split("/"));

    const cfg = JSON.parse(await readFile(local("mlc-chat-config.json"), "utf8"));
    await add({ cache: "webllm/config", url: new URL("mlc-chat-config.json", base).href, type: "application/json", local: local("mlc-chat-config.json") });

    const tc = JSON.parse(await readFile(local("tensor-cache.json"), "utf8"));
    await add({ cache: "webllm/model", url: new URL("tensor-cache.json", base).href, type: "application/json", local: local("tensor-cache.json") });
    for (const r of tc.records) {
      await add({ cache: "webllm/model", url: new URL(r.dataPath, base).href, type: "application/octet-stream", local: local(r.dataPath) });
    }

    const tf = cfg.tokenizer_files || [];
    const tok = tf.includes("tokenizer.json") ? "tokenizer.json" : tf.includes("tokenizer.model") ? "tokenizer.model" : null;
    if (!tok) throw new Error(id + ": config lists no tokenizer.json or tokenizer.model");
    await add({ cache: "webllm/model", url: new URL(tok, base).href, type: tok.endsWith(".json") ? "application/json" : "application/octet-stream", local: local(tok) });

    await add({ cache: "webllm/wasm", url: rec.model_lib, type: "application/wasm", local: path.join(modelsDir, "libs", path.basename(new URL(rec.model_lib).pathname)) });
  }
}

const html = await readFile(input, "utf8");
const cut = html.lastIndexOf("</body>");
if (cut < 0) throw new Error("input has no </body>: " + input);

const manifest = {
  version: 1,
  models: [...new Set(want.map((w) => CATALOG[w].key))],
  notice: want.map((w) => CATALOG[w].notice).join(" "),
  files: files.map((f) => ({ cache: f.cache, url: f.url, type: f.type, size: f.size, parts: Math.max(1, Math.ceil(f.size / partBytes)) })),
};
const attr = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

await mkdir(path.dirname(out), { recursive: true });
const ws = createWriteStream(out);
const write = async (s) => { if (!ws.write(s)) await once(ws, "drain"); };
await write(html.slice(0, cut));
await write('\n<script type="application/json" id="localaiEmbedded">' + JSON.stringify(manifest).replace(/</g, "\\u003c") + "</script>\n");
if (want.some((w) => CATALOG[w].apache)) {
  const lic = await readFile(path.join(dir, "vendor", "licenses", "Apache-2.0.txt"), "utf8");
  await write("<!-- Apache License 2.0 (applies to the built-in Qwen2.5 model weights)\n\n" + lic.replace(/--/g, "- -") + "\n-->\n");
}
for (const f of files) {
  const buf = await readFile(f.local);
  const n = Math.max(1, Math.ceil(buf.length / partBytes));
  for (let p = 0; p < n; p++) {
    const part = buf.subarray(p * partBytes, Math.min(buf.length, (p + 1) * partBytes));
    await write('<script type="application/x-localai-file" data-url="' + attr(f.url) + '" data-part="' + p + '">');
    await write(part.toString("base64"));
    await write("</script>\n");
  }
}
await write(html.slice(cut));
ws.end();
await once(ws, "finish");

const model = files.reduce((s, f) => s + f.size, 0);
const size = (await stat(out)).size;
console.log(`built in: ${want.join(", ")} — ${files.length} files, ${(model / 1048576).toFixed(1)} MB of model data`);
console.log(`wrote ${path.relative(process.cwd(), out)} (${(size / 1048576).toFixed(1)} MB)`);
if (size > 2 * 1024 ** 3) console.warn("warning: over 2 GiB — too big for a GitHub release asset");
else if (size > 1.2 * 1024 ** 3) console.warn("warning: over ~1.2 GB — likely too big to open on a Chromebook");
