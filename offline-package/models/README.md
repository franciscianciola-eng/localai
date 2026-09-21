# models/ — the pre-downloaded model files go here

This folder starts almost empty. Run the downloader once (from the parent
folder, with internet) and it fills in:

```
models/
├── mlc-ai/
│   ├── Qwen2.5-0.5B-Instruct-q4f16_1-MLC/resolve/main/…   (weights, config, tokenizer)
│   ├── Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/…
│   └── gemma-2-2b-it-q4f16_1-MLC/resolve/main/…
└── libs/
    ├── Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm       (compiled WebGPU library)
    ├── Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm
    └── gemma-2-2b-it-q4f16_1_cs1k-webgpu.wasm
```

That mirrors the exact paths the app requests, so no configuration is needed —
put the files here and it just works. Total size is roughly **3 GB** for all
three models (about 0.5 GB if you only download Qwen with `--only qwen`).

To get them:

```bash
python3 download-models.py          # all three
python3 download-models.py --only qwen   # just the smallest
```
