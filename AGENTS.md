# AGENTS.md — ML-in-C

Neural network in C11 with automatic differentiation via computational graphs.
Trains MNIST digit classification: 784 → 16 → 16 (residual) → 10. ~91% test
accuracy after 3 epochs, ~10s runtime.

## Layout

| File | Contents |
|------|----------|
| `main.c` | Entry point: load data, build model, train, predict |
| `types.h` | Type aliases: `u32 u64 i64 b8 b32 f32` |
| `matrix.h/.c` | `matrix` type + math ops, activations, gradient kernels |
| `model.h/.c` | Autodiff graph (`model_var`), compilation, training loop |
| `mnist.h/.c` | MNIST loading/one-hot, `draw_mnist_digit`, `create_mnist_model` |
| `data/` | MNIST `.mat` files (raw f32, gitignored) |
| `scripts/mnist.py` | Regenerates the `.mat` files |
| `docs/graph.md` | Worked example: the graph with a numeric forward/backward pass |

## Build & Run

```sh
cmake --build cmake-build-debug
./cmake-build-debug/ml_c    # run from project root (relative data/ paths)
```

## Key Concepts

- `matrix` — `{ u32 rows, cols; f32 *data; }`, row-major. `mat_create` / `mat_free`.
- `model_var` — graph node: `val`, `grad`, `op`, `inputs[2]`. Flags mark
  `PARAMETER`, `INPUT`, `OUTPUT`, `DESIRED_OUTPUT`, `COST`.
- `model_context` — owns a registry of all vars; one `model_destroy()` frees
  the whole graph. Memory is plain `malloc`/`calloc`/`free`, no custom allocator.
- `model_compile()` flattens the graph into execution programs;
  `model_train()` runs mini-batch SGD (batch 50, lr 0.01) with Fisher-Yates shuffle.
- Allocation and file-load failures are caught with `assert`.

## Conventions

- Use `types.h` aliases everywhere — never raw `float` / `uint32_t`.
- C11, must compile with zero warnings under `-Wall -Wextra`.
- Keep it simple: no features beyond what was asked, no speculative abstractions.
