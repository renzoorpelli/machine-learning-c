#include "model.h"

#include <assert.h>
#include <stdlib.h>

#define MV_NUM_INPUTS(op) ((op) <= _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

model_context *model_create(void) {
    model_context *model = calloc(1, sizeof(model_context));
    assert(model != NULL);
    return model;
}

void model_destroy(model_context *model) {
    if (model == NULL) {
        return;
    }

    for (u32 i = 0; i < model->num_vars; i++) {
        model_var *var = model->vars[i];
        mat_free(var->val);
        mat_free(var->grad);
        free(var);
    }
    free(model->vars);
    free(model->forward_prog.vars);
    free(model->cost_prog.vars);
    free(model);
}

model_var *mv_create(model_context *model, u32 rows, u32 cols, u32 flags) {
    // create the computational graph
    model_var *out = calloc(1, sizeof(model_var));
    assert(out != NULL);

    out->op = MV_OP_CREATE;
    out->flags = flags;
    out->val = mat_create(rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        // create gradient vector
        out->grad = mat_create(rows, cols);
    }

    if (model->num_vars == model->vars_capacity) {
        model->vars_capacity = model->vars_capacity == 0 ? 64 : model->vars_capacity * 2;
        model->vars = realloc(model->vars, (u64) model->vars_capacity * sizeof(model_var *));
        assert(model->vars != NULL);
    }
    out->index = model->num_vars;
    model->vars[model->num_vars++] = out;

    if (flags & MV_FLAG_INPUT) {
        model->input = out;
    }
    if (flags & MV_FLAG_OUTPUT) {
        model->output = out;
    }
    if (flags & MV_FLAG_DESIRED_OUTPUT) {
        model->desired_output = out;
    }
    if (flags & MV_FLAG_COST) {
        model->cost = out;
    }

    return out;
}

static model_var *mv_unary_op(model_context *model, model_var *input, u32 rows,
                              u32 cols, u32 flags, model_var_ops op) {
    if (input->flags & MV_FLAG_REQUIRES_GRAD) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var *out = mv_create(model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = input;

    return out;
}

static model_var *mv_binary_op(model_context *model, model_var *a, model_var *b,
                               u32 rows, u32 cols, u32 flags, model_var_ops op) {
    if ((a->flags & MV_FLAG_REQUIRES_GRAD) || (b->flags & MV_FLAG_REQUIRES_GRAD)) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var *out = mv_create(model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

model_var *mv_relu(model_context *model, model_var *input, u32 flags) {
    return mv_unary_op(model, input, input->val->rows, input->val->cols, flags, MV_OP_RELU);
}

model_var *mv_softmax(model_context *model, model_var *input, u32 flags) {
    return mv_unary_op(model, input, input->val->rows, input->val->cols, flags,
                       MV_OP_SOFTMAX);
}

model_var *mv_add(model_context *model, model_var *a, model_var *b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    return mv_binary_op(model, a, b, a->val->rows, a->val->cols, flags, MV_OP_ADD);
}

model_var *mv_matmul(model_context *model, model_var *a, model_var *b, u32 flags) {
    if (a->val->cols != b->val->rows) {
        return NULL;
    }

    return mv_binary_op(model, a, b, a->val->rows, b->val->cols, flags, MV_OP_MATMUL);
}

model_var *mv_cross_entropy(model_context *model, model_var *p, model_var *q, u32 flags) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return NULL;
    }

    return mv_binary_op(model, p, q, p->val->rows, p->val->cols, flags, MV_OP_CROSS_ENTROPY);
}

static model_program model_prog_create(model_context *model, model_var *out_var) {
    // DFS through the graph, producing execution order
    b8 *visited = calloc(model->num_vars, sizeof(b8));
    assert(visited != NULL);
    u32 stack_size = 0;
    u32 out_size = 0;

    model_var **stack = malloc((u64) model->num_vars * sizeof(model_var *));
    model_var **out = malloc((u64) model->num_vars * sizeof(model_var *));
    assert(stack != NULL && out != NULL);

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        model_var *cur = stack[--stack_size];

        if (cur->index >= model->num_vars) {
            continue;
        }

        if (visited[cur->index]) {
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }
            continue;
        }
        visited[cur->index] = true;

        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }

        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for (u32 i = 0; i < num_inputs; i++) {
            model_var *input = cur->inputs[i];

            if (input->index >= model->num_vars || visited[input->index])
                continue;

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size - 1; k++) {
                        stack[k] = stack[k + 1];
                    }
                    stack_size--;
                }
            }
            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }
    model_program prog = {.size = out_size, .vars = malloc((u64) out_size * sizeof(model_var *))};
    assert(prog.vars != NULL);

    memcpy(prog.vars, out, sizeof(model_var *) * out_size);

    free(visited);
    free(stack);
    free(out);
    return prog;
}

static void model_program_compute(model_program *program) {
    for (u32 i = 0; i < program->size; i++) {
        model_var *cur = program->vars[i];

        model_var *a = cur->inputs[0];
        model_var *b = cur->inputs[1];

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE:
                break;

            case _MV_OP_UNARY_START:
                break;

            case MV_OP_RELU: {
                mat_relu(cur->val, a->val);
            }
            break;
            case MV_OP_SOFTMAX: {
                mat_softmax(cur->val, a->val);
            }
            break;

            case _MV_OP_BINARY_START:
                break;

            case MV_OP_ADD: {
                mat_add(cur->val, a->val, b->val);
            }
            break;
            case MV_OP_MATMUL: {
                mat_mul(cur->val, a->val, b->val, 1, 0, 0);
            }
            break;
            case MV_OP_CROSS_ENTROPY: {
                mat_cross_entropy(cur->val, a->val, b->val);
            }
            break;
        }
    }
}

static void model_program_compute_grads(model_program *prog) {
    // clear non-parameter gradients (parameter grads accumulate across the batch)
    for (u32 i = 0; i < prog->size; i++) {
        model_var *cur = prog->vars[i];

        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
            continue;
        }
        if (cur->flags & MV_FLAG_PARAMETER) {
            continue;
        }

        mat_clear(cur->grad);
    }
    mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

    // start going array backwards
    for (i64 i = (i64) prog->size - 1; i >= 0; i--) {
        model_var *cur = prog->vars[i];

        model_var *a = cur->inputs[0];
        model_var *b = cur->inputs[1];

        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        if (num_inputs == 1 && (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
            continue;
        }

        if (num_inputs == 2 && (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
            (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
            continue;
        }

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE:
                break;

            case _MV_OP_UNARY_START:
                break;

            case MV_OP_RELU: {
                if ((a->flags & MV_FLAG_REQUIRES_GRAD) && a->grad != NULL) {
                    mat_relu_add_grad(a->grad, a->val, cur->grad);
                }
            }
            break;
            case MV_OP_SOFTMAX: {
                if ((a->flags & MV_FLAG_REQUIRES_GRAD) && a->grad != NULL) {
                    mat_softmax_add_grad(a->grad, cur->val, cur->grad);
                }
            }
            break;

            case _MV_OP_BINARY_START:
                break;

            case MV_OP_ADD: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(b->grad, b->grad, cur->grad);
                }
            }
            break;
            case MV_OP_MATMUL: {
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                }

                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                }
            }
            break;
            case MV_OP_CROSS_ENTROPY: {
                model_var *p = a;
                model_var *q = b;

                mat_cross_entropy_add_grad(p->grad, q->grad, p->val, q->val, cur->grad);
            }
            break;
        }
    }
}

void model_compile(model_context *model) {
    assert(model->output != NULL);
    assert(model->cost != NULL);

    model->forward_prog = model_prog_create(model, model->output);
    model->cost_prog = model_prog_create(model, model->cost);
}

void model_feedforward(model_context *model) { model_program_compute(&model->forward_prog); }

void model_train(model_context *model, const model_training_desc *training_desc) {
    matrix *train_images = training_desc->train_images;
    matrix *train_labels = training_desc->train_labels;
    matrix *test_images = training_desc->test_images;
    matrix *test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches = num_examples / training_desc->batch_size;

    u32 *training_order = malloc((u64) num_examples * sizeof(u32));
    assert(training_order != NULL);
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        // Fisher-Yates shuffle
        for (u32 i = num_examples - 1; i > 0; i--) {
            u32 j = rand() % (i + 1);

            u32 tmp = training_order[j];
            training_order[j] = training_order[i];
            training_order[i] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var *cur = model->cost_prog.vars[i];

                if (cur->flags & MV_FLAG_PARAMETER) {
                    mat_clear(cur->grad);
                }
            }

            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < training_desc->batch_size; i++) {
                u32 order_index = batch * training_desc->batch_size + i;
                u32 index = training_order[order_index];

                memcpy(model->input->val->data, train_images->data + index * input_size,
                       sizeof(f32) * input_size);

                memcpy(model->desired_output->val->data, train_labels->data + index * output_size,
                       sizeof(f32) * output_size);

                model_program_compute(&model->cost_prog);
                model_program_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);
            }
            avg_cost /= (f32) training_desc->batch_size;

            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var *cur = model->cost_prog.vars[i];

                if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
                    continue;
                }

                mat_scale(cur->grad, training_desc->learning_rate / training_desc->batch_size);
                mat_sub(cur->val, cur->val, cur->grad);
            }

            printf("Epoch %2d / %2d, Batch %4d / %4d, Average Cost: %.4f\r", epoch + 1,
                   training_desc->epochs, batch + 1, num_batches, avg_cost);
            fflush(stdout);
        }
        printf("\n");

        u32 num_correct = 0;
        f32 avg_cost = 0;
        for (u32 i = 0; i < num_tests; i++) {
            memcpy(model->input->val->data, test_images->data + i * input_size, sizeof(f32) * input_size);

            memcpy(model->desired_output->val->data, test_labels->data + i * output_size,
                   sizeof(f32) * output_size);

            model_program_compute(&model->cost_prog);

            avg_cost += mat_sum(model->cost->val);
            num_correct += mat_argmax(model->output->val) == mat_argmax(model->desired_output->val);
        }

        avg_cost /= (f32) num_tests;
        printf("Test Completed. Accuracy: %5d / %d (%.1f%%), Average Cost: %.4f\n", num_correct,
               num_tests, (f32) num_correct / num_tests * 100.0f, avg_cost);
    }

    free(training_order);
}
