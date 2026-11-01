# Native changes in this fork

`llama-cpp-sys-2` here is the published 0.1.156 crate with its llama.cpp sources committed
directly (no submodule; upstream llama.cpp `e79e4bf660`, 2026-08-13). Besides the OpenAI-compatible
chat surface that `4550a5c` re-applies and `llama_synchronize` (`ad6e4b8`), the native sources carry
four changes that keep the Pond alive when the GPU has no room for a compute buffer, which on an
8 GB Jetson shared by the LLM and Whisper is an ordinary event, not an edge case, and four backports
for EmbeddingGemma 2 (see below).

| File | Change | Origin |
|---|---|---|
| `llama.cpp/ggml/src/ggml-backend.cpp` | `ggml_backend_sched_alloc_splits` returns `false` when `ggml_gallocr_reserve_n` fails | upstream `911f6cdc8a` (#26070, 2026-09-18), unchanged |
| `llama.cpp/tools/mtmd/clip.cpp` | `clip_encode` returns `false` when `ggml_backend_sched_alloc_graph` fails | upstream `5b335f413e` (#28149, 2026-09-18), unchanged |
| `llama.cpp/ggml/src/ggml-alloc.c` | a failed reservation forgets the layout it recorded | this fork; upstream master (2026-10-05) does not have it |
| `llama.cpp/src/llama-context.cpp` | `process_ubatch` forgets a graph it could not build or allocate | this fork; upstream `2f3fd025` (2026-09-16) fixes the same with `gf_res_prev_active` |

### EmbeddingGemma 2, backported from upstream

The Pond's memory embedder needs EmbeddingGemma 2, which upstream added in October. Moving these
sources to October would also move llama.cpp's chat API, which the OpenAI-compatible surface above
is written against, so the four commits below are applied to these August sources instead.

| Commit | Change | Here |
|---|---|---|
| `4fbc76dec5` (#30054) | The `gemma-embedding2` architecture | Unchanged; the converter and test hunks are outside the vendored tree |
| `358a420ec4` (#30416) | The no-cache attention pads its KV dimension to 256 | Calls this tree's `build_attn_mha`, which has no `n_kv_tail` argument; without the `clef` model, which postdates it |
| `ed019cd8b9` (#30255) | No logits buffer when the reserved graph has no logits | Unchanged |
| `cb55ffe6e2` (#30445) | `build_logits()` adds the LM head centrally, and leaves it out of pooled embeddings | Without twelve models that postdate these sources; the `gemma4`, `eagle3`, `qwen3` and `lfm2` hunks applied by hand at their older offsets |

Drop them when these sources move past upstream `cb55ffe6e2` (2026-10-13). Every graph here is built
through `llama_model::build_graph`, MTP drafts included, so the central LM head reaches all of them.

## Why

`ggml_gallocr_reserve_n_impl` records the graph's layout, then allocates the buffers. When an
allocation fails it returns `false` with the buffer `NULL` and the layout still recorded.

- The scheduler ignored that `false`. `ggml_gallocr_alloc_graph` found a matching layout, skipped
  reallocation and wrote through the `NULL` buffer: `SIGSEGV` on the first failure.
- With only the upstream fix, the first failure returns an error, but the next attempt at a graph of
  the same shape (the next decode, the next chat turn) found the same stale layout and crashed the
  same way. The third change clears the layout, so that attempt reserves again: it succeeds if memory
  has been freed meanwhile and returns `false` if not.
- One level up, `llama_context::process_ubatch` keeps the previous graph to reuse when the next
  ubatch has the same shape. Building records the graph's params before allocating, and a failed
  allocation left them recorded. llama.cpp then drops the failed ubatch from the KV cache, so a retry
  has the same shape, passes `can_reuse`, skips allocation and aborts in `set_inputs` ("tensor buffer
  not set"). The fourth change resets the graph on both failure branches, so the retry rebuilds and
  allocates again. The CPU test below calls the scheduler directly and cannot reach this path.
- `clip_encode` ignored the allocation result and went on to set inputs, aborting the process when an
  image encode ran out of memory.

A failed decode now reaches Rust as `DecodeError::Unknown(-2)` (llama.cpp's `process_ubatch` sets
`GGML_STATUS_ALLOC_FAILED`). The log shows `ggml_gallocr_reserve_n_impl: failed to allocate ...
buffer`, then `ggml_backend_sched_alloc_splits: failed to reserve graph buffers`.

`tests/graph_reservation_failure.rs` reproduces all of this on the CPU backend with a graph no
allocator can satisfy, attempting it three times. Without the upstream change it crashes on the
first attempt; with only that change, on the second.

## When to drop

Rebasing onto llama-cpp-rs 0.1.157 or later brings llama.cpp `26394b4e67` (2026-09-21), which
contains both upstream commits and `2f3fd025`; drop those three rows then. Keep the `ggml-alloc.c` change and the test
until upstream clears a failed layout itself: run the test against the new sources first.

`build.rs` emits `rerun-if-changed` for everything under `llama.cpp/src`, `ggml/src` and `common`
and builds from the source tree, so an edit here is compiled (only the edited object rebuilds).
Headers under `ggml/include` and `include/` are not watched.
