# ML-in-C

**[github.com/renzoorpelli/machine-learning-c](https://github.com/renzoorpelli/machine-learning-c)**

A minimal machine learning library in C: neural networks with automatic differentiation
via computational graphs. Trains MNIST digit classification (784→16→16→10 with residual connections).

Implementation based on [coding a machine learning library in c from scratch](https://www.youtube.com/watch?v=hL_n_GljC0I), with modifications: plain `malloc`/`free` instead of the arena allocator, stdlib `rand()` instead of PCG, and the code split into modules.

## Pipeline

```
Data → Forward Pass → Predictions → Loss → Backprop → Parameter Update
```

## Quick Reference

| Concept | Type | Role |
|---------|------|------|
| `matrix` | `struct { u32 rows, cols; f32 *data; }` | Core data type, row-major |
| `model_var` | graph node | Holds `val`, `grad`, `op`, `inputs[2]` |
| `model_context` | network container | Owns input, output, cost, compiled programs |
| memory | `malloc`/`calloc` | `mat_free` per matrix; one `model_destroy` frees the whole graph |

## Docs

- [Graph worked example](docs/graph.md) — the MNIST computational graph plus a full numeric forward/backward pass traced through the actual ops

## Usage

```c
mnist_data data = mnist_load("data");

model_context *model = model_create();
create_mnist_model(model);
model_compile(model);

model_training_desc desc = { .epochs=3, .batch_size=50, .learning_rate=0.01f, ... };
model_train(model, &desc);

memcpy(model->input->val->data, test_image, 784 * sizeof(f32));
model_feedforward(model);
u32 predicted = mat_argmax(model->output->val);

mnist_data_free(&data);
model_destroy(model);
```

**Notation**: `θ` = parameters · `η` = learning rate · `·` = matrix multiply · `∂` = partial derivative
