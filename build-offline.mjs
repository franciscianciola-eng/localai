// Builds offline-package/index.html: a single self-contained HTML app that
// loads its models from the sibling ./models/ folder (populated once by
// download-models.py) and makes no network calls of its own.
// Run: node build-offline.mjs
import { readFile, writeFile, mkdir } from "fs/promises";
import { fileURLToPath } from "url";
import path from "path";

const dir = path.dirname(fileURLToPath(import.meta.url));
const tpl = await readFile(path.join(dir, "offline.template.html"), "utf8");
const bundle = await readFile(path.join(dir, "vendor", "web-llm.js"), "utf8");

// A literal </script (any case) inside the bundle would prematurely close the
// inline <script> element and break the page. The current bundle has none.
if (/<\/script/i.test(bundle)) {
  throw new Error("web-llm bundle contains a </script> sequence — cannot inline safely");
}
const marker = "/*__WEBLLM_BUNDLE__*/";
if (!tpl.includes(marker)) throw new Error("template is missing the bundle marker");

// Function replacer so `$` sequences in the bundle aren't treated as special.
const out = tpl.replace(marker, () => bundle);
await mkdir(path.join(dir, "offline-package"), { recursive: true });
await writeFile(path.join(dir, "offline-package", "index.html"), out);
console.log("wrote offline-package/index.html (" + (out.length / 1048576).toFixed(1) + " MB)");
