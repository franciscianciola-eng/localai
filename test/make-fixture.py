#!/usr/bin/env python3
"""Generate a tiny random llama-shaped ONNX model + tokenizer for e2e tests.

The graph mimics the interface transformers.js expects from an
optimum-exported decoder model (input_ids / attention_mask / position_ids /
past_key_values.*, outputs logits / present.*) so the whole browser stack —
tokenizer, chat template, KV cache plumbing, streaming, stopping — can be
exercised offline with a ~200 KB model. The weights are random; the output is
gibberish by design. Requires: pip install onnx numpy

Usage: python3 test/make-fixture.py
Writes: test/fixtures/tiny-llm/
"""

import json
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "tiny-llm")
VOCAB = 260  # 256 byte-level tokens + 3 specials + 1 spare
HIDDEN = 32
KV_HEADS = 2
HEAD_DIM = 16

IM_START, IM_END, EOT = 256, 257, 258


def bytes_to_unicode():
    """GPT-2's byte<->unicode map (the printable stand-ins used in vocab keys)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def build_tokenizer():
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    specials = {"<|im_start|>": IM_START, "<|im_end|>": IM_END, "<|endoftext|>": EOT}
    added = [
        {
            "id": tid,
            "content": tok,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for tok, tid in specials.items()
    ]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {**vocab, **specials},
            "merges": [],
        },
    }
    with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)

    chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer_config = {
        "tokenizer_class": "GPT2Tokenizer",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "model_max_length": 1024,
        "clean_up_tokenization_spaces": False,
        "chat_template": chat_template,
    }
    with open(os.path.join(OUT, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)


def build_model():
    rng = np.random.default_rng(42)
    emb_w = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32) * 0.1
    head_w = rng.standard_normal((HIDDEN, VOCAB)).astype(np.float32) * 0.1

    f32, i64 = TensorProto.FLOAT, TensorProto.INT64
    inputs = [
        helper.make_tensor_value_info("input_ids", i64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("attention_mask", i64, ["batch_size", "total_sequence_length"]),
        helper.make_tensor_value_info("position_ids", i64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info(
            "past_key_values.0.key", f32, ["batch_size", KV_HEADS, "past_sequence_length", HEAD_DIM]
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.value", f32, ["batch_size", KV_HEADS, "past_sequence_length", HEAD_DIM]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", f32, ["batch_size", "sequence_length", VOCAB]),
        helper.make_tensor_value_info(
            "present.0.key", f32, ["batch_size", KV_HEADS, "total_sequence_length", HEAD_DIM]
        ),
        helper.make_tensor_value_info(
            "present.0.value", f32, ["batch_size", KV_HEADS, "total_sequence_length", HEAD_DIM]
        ),
    ]
    initializers = [
        helper.make_tensor("emb_w", f32, emb_w.shape, emb_w.flatten()),
        helper.make_tensor("head_w", f32, head_w.shape, head_w.flatten()),
        helper.make_tensor("kv_shape", i64, [4], np.array([0, 0, KV_HEADS, HEAD_DIM], dtype=np.int64)),
        helper.make_tensor("zero_f", f32, [], np.array([0.0], dtype=np.float32)),
        helper.make_tensor("half_f", f32, [], np.array([0.5], dtype=np.float32)),
    ]
    # attention_mask and position_ids feed a zero-valued bias so every graph
    # input is genuinely consumed (some runtimes warn on unused inputs).
    nodes = [
        helper.make_node("Gather", ["emb_w", "input_ids"], ["emb"], axis=0),
        helper.make_node("MatMul", ["emb", "head_w"], ["logits_raw"]),
        helper.make_node("Cast", ["position_ids"], ["pos_f"], to=f32),
        helper.make_node("ReduceSum", ["pos_f"], ["pos_sum"], keepdims=0),
        helper.make_node("Cast", ["attention_mask"], ["att_f"], to=f32),
        helper.make_node("ReduceSum", ["att_f"], ["att_sum"], keepdims=0),
        helper.make_node("Add", ["pos_sum", "att_sum"], ["aux_sum"]),
        helper.make_node("Mul", ["aux_sum", "zero_f"], ["zero_bias"]),
        helper.make_node("Add", ["logits_raw", "zero_bias"], ["logits"]),
        helper.make_node("Reshape", ["emb", "kv_shape"], ["kv"]),
        helper.make_node("Transpose", ["kv"], ["kv_t"], perm=[0, 2, 1, 3]),
        helper.make_node("Concat", ["past_key_values.0.key", "kv_t"], ["present.0.key"], axis=2),
        helper.make_node("Mul", ["kv_t", "half_f"], ["kv_v"]),
        helper.make_node("Concat", ["past_key_values.0.value", "kv_v"], ["present.0.value"], axis=2),
    ]
    graph = helper.make_graph(nodes, "tiny_llm", inputs, outputs, initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    onnx.checker.check_model(model)

    onnx_dir = os.path.join(OUT, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    main = os.path.join(onnx_dir, "model.onnx")
    onnx.save(model, main)
    # The same fp32 graph stands in for every quantization variant the app's
    # dtype ladder may request — dtype only selects the file to download.
    for alias in ["model_quantized.onnx", "model_q4.onnx", "model_int8.onnx", "model_uint8.onnx"]:
        shutil.copyfile(main, os.path.join(onnx_dir, alias))


def build_configs():
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": KV_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 1024,
        "bos_token_id": EOT,
        "eos_token_id": IM_END,
        "pad_token_id": EOT,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.40.0",
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(OUT, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": EOT, "eos_token_id": IM_END, "pad_token_id": EOT}, f, indent=2)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_tokenizer()
    build_model()
    build_configs()
    total = sum(
        os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs
    )
    print(f"fixture written to {OUT} ({total / 1024:.0f} KB)")
