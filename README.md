# ML-in-C

Neural nets in C with autodiff on a computational graph. Classifies MNIST digits
(784 → 16 → 16 → 10, residual connection) at ~91% after 3 epochs, ~10s.

Based on [coding a machine learning library in c from scratch](https://www.youtube.com/watch?v=hL_n_GljC0I).
We kept the graph / backprop idea and reworked the rest: plain `malloc` instead of
the arena, `rand()` instead of PCG, code split into modules.

```sh
cmake --build cmake-build-debug
./cmake-build-debug/ml_c   # from the project root so data/ resolves
```

Files:

- `main.c` — load, train, predict
- `types.h` — `u32`, `f32`, and the other short aliases
- `matrix.h/.c` — matrices and the math (mul, relu, softmax, grads…)
- `model.h/.c` — the graph, compile, forward/backward, SGD
- `mnist.h/.c` — data loading, digit draw, the MNIST net definition
- `data/` — `.mat` files (gitignored); regenerate with `scripts/mnist.py`
- `docs/graph.md` — numeric walkthrough of a forward + backward pass

How it hangs together: every value is a row-major `matrix`. Ops wire into a
graph of `model_var` nodes (value, grad, op, up to two inputs). Compile once
into an execution order, then forward walks it and backprop walks it reverse,
accumulating grads. Training is mini-batch SGD (batch 50, lr 0.01) with a
Fisher-Yates shuffle. One `model_destroy()` frees the whole graph via a node
registry; bad allocs and missing files `assert`.

Use the `types.h` aliases, not raw `float`/`uint32_t`. C11, clean under
`-Wall -Wextra`. Prefer less code over more.
