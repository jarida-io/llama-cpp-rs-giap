# Native changes in this fork

`llama-cpp-2` and `llama-cpp-sys-2` here are the published 0.1.160 crates, with llama.cpp moved past
what 0.1.160 vendors (`d81235049`, 2026-10-05) to `b11597` (`cb55ffe6e2`, 2026-10-13). That brings
EmbeddingGemma 2 (#30054), the padded no-cache attention (#30416) and pooled graphs that skip the
logits buffer and the LM head (#30255, #30445), which the Pond's in-process embedder needs.

The sources under `llama.cpp/` are upstream's, selected by the crate's own rule: its Cargo
`include` patterns plus every `README.md`. That rule reproduces 0.1.160's vendored tree exactly, so a
later bump is the same selection from a newer upstream commit, then the native changes below.

| Where | Change | Origin |
|---|---|---|
| `llama-cpp-2/src/openai.rs`, `model.rs`, `wrapper_oai.*` | The OpenAI-compatible chat template and response parser that goose's engine calls | 0.1.146, re-applied |
| `llama-cpp-2/src/context.rs` | `LlamaContext::synchronize` | this fork |
| `llama-cpp-2/src/context/params/get_set.rs` | `LlamaContextParams::with_n_outputs_max` | this fork |
| `llama-cpp-2/src/model/params.rs` | `LlamaModelParams::with_no_host` | this fork |
| `llama.cpp/ggml/CMakeLists.txt`, `llama-cpp-sys-2/build.rs` | Flash Attention compiles the `q8_0-q4_0` asymmetric vector kernel without requiring `GGML_CUDA_FA_ALL_QUANTS` | this fork |
| `llama.cpp/ggml/src/ggml-alloc.c` | A failed reservation forgets the layout it recorded | this fork; upstream `b11597` does not have it |
| `llama.cpp/ggml/src/ggml-cuda/mmvq.cu`, `ggml-cuda.cu` | Jetson Orin (cc 8.7) gets its own mat-vec table (4 rows per block; 2 warps up to batch 3, then 1), stays out of upstream's `small_k` mode, and skips the fused gate/up kernel for quantized weights. Rows past the matrix are clamped before the K loop | this fork |

Upstream now carries the three allocation fixes this fork used to (`911f6cdc8a`, `5b335f413e` and
`2f3fd025`'s `gf_res_prev_active`), so they are no longer patched here.

## Why the allocation change

`ggml_gallocr_reserve_n_impl` records the graph's layout, then allocates the buffers. When an
allocation fails it returns `false` with the buffer `NULL` and the layout still recorded. The next
attempt at a graph of the same shape (the next decode, the next chat turn) finds that layout, skips
reallocation and writes through the `NULL` buffer. Clearing the layout makes that attempt reserve
again: it succeeds if memory has been freed meanwhile and returns `false` if not. On an 8 GB Jetson
shared by the language model and Whisper, a compute buffer with no room is an ordinary event.

`tests/graph_reservation_failure.rs` reproduces it on the CPU backend with a graph no allocator can
satisfy, attempting it three times. Run it against new sources before dropping the change.

## Why the Orin mat-vec table

On an Orin Nano 8GB the batch-1 quantized mat-vec read weights at about 52 GB/s against the board's
~82 GB/s ceiling: four warps shared each short row, and nothing reused the activations across rows.
llama-bench, gemma-4-E4B-it-qat (Q4_0), f16 KV, flash attention on: tg128 19.9 -> 28.8 tok/s, and
the MTP verify batches (pp 2/5/8 at depth 2048) 35/73/83 -> 50/94/122. Those figures were measured on
the August sources; re-measure them on the Orin after any bump.

`build.rs` emits `rerun-if-changed` for everything under `llama.cpp/src`, `ggml/src` and `common`
and builds from the source tree, so an edit here is compiled (only the edited object rebuilds).
Headers under `ggml/include` and `include/` are not watched.
