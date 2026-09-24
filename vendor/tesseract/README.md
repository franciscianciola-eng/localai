# Vendored Tesseract.js (for fully-offline OCR)

These files are inlined into `localai-standalone.html` by `build-standalone.mjs`
so the single-file app can read text from images with **no network at all**.

- `tesseract.min.js`, `worker.min.js` — Tesseract.js v5.1.1 (Apache-2.0)
- `tesseract-core-simd-lstm.wasm.js` — tesseract.js-core v5.1.1 (Apache-2.0),
  a SIMD + LSTM build with the WebAssembly embedded (single-file).
- `eng.traineddata.gz` — English model, `4.0.0_best_int` (Apache-2.0),
  from @tesseract.js-data/eng.

Upstream: https://github.com/naptha/tesseract.js
