//! A graph whose compute buffer cannot be allocated must be refused on every attempt.
//!
//! See `POND-PATCH.md`. Without the upstream fix the first `ggml_backend_sched_alloc_graph`
//! dereferenced a NULL buffer; with it alone, the second attempt at the same graph did, because
//! the failed reservation kept a layout that named the missing buffer. Either way the process
//! died with `SIGSEGV`, so this test crashes rather than fails when a patch is missing.
//!
//! The CPU backend is enough: no allocator hands out a petabyte, so the failure is the same one
//! an exhausted GPU produces.

use llama_cpp_sys_2::*;
use std::ptr;

#[test]
fn an_unallocatable_graph_is_refused_on_every_attempt() {
    unsafe {
        let ctx = ggml_init(ggml_init_params {
            mem_size: 16 * 1024 * 1024,
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        });
        assert!(!ctx.is_null());

        // 2^48 f32 elements: a petabyte per tensor.
        let elements = 1i64 << 48;
        let a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        let b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        ggml_set_input(a);
        ggml_set_input(b);
        let sum = ggml_add(ctx, a, b);
        ggml_set_output(sum);
        let graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum);

        let mut cpu = ggml_backend_cpu_init();
        assert!(!cpu.is_null());
        let sched = ggml_backend_sched_new(&mut cpu, ptr::null_mut(), 1, 2048, false, true);
        assert!(!sched.is_null());

        for attempt in 1..=3 {
            assert!(
                !ggml_backend_sched_alloc_graph(sched, graph),
                "attempt {attempt} claims to have allocated a petabyte"
            );
            ggml_backend_sched_reset(sched);
        }

        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }
}
