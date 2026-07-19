#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

typedef enum {
    MV_FLAG_NONE = 0,

    MV_FLAG_REQUIRES_GRAD = (1 << 0),
    MV_FLAG_PARAMETER = (1 << 2),
    MV_FLAG_INPUT = (1 << 3),
    MV_FLAG_OUTPUT = (1 << 4),
    MV_FLAG_DESIRED_OUTPUT = (1 << 5),
    MV_FLAG_COST = (1 << 6)
} model_var_flags;

typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,
    _MV_OP_UNARY_START,
    MV_OP_RELU,
    MV_OP_SOFTMAX,
    _MV_OP_BINARY_START,
    MV_OP_ADD,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY
} model_var_ops;

#define MODEL_MAX_INPUTS 2

typedef struct model_var {
    u32 index;
    u32 flags;

    matrix *val;
    matrix *grad;

    model_var_ops op;
    struct model_var *inputs[MODEL_MAX_INPUTS];
} model_var;

typedef struct {
    model_var **vars;
    u32 size;
} model_program;

typedef struct {
    u32 num_vars;
    u32 vars_capacity;
    model_var **vars;

    model_var *input;
    model_var *output;
    model_var *desired_output;
    model_var *cost;

    model_program forward_prog;
    model_program cost_prog;
} model_context;

typedef struct {
    matrix *train_images;
    matrix *train_labels;
    matrix *test_images;
    matrix *test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

model_context *model_create(void);
void model_destroy(model_context *model);

model_var *mv_create(model_context *model, u32 rows, u32 cols, u32 flags);
model_var *mv_relu(model_context *model, model_var *input, u32 flags);
model_var *mv_softmax(model_context *model, model_var *input, u32 flags);
model_var *mv_add(model_context *model, model_var *a, model_var *b, u32 flags);
model_var *mv_matmul(model_context *model, model_var *a, model_var *b, u32 flags);
model_var *mv_cross_entropy(model_context *model, model_var *p, model_var *q, u32 flags);

void model_compile(model_context *model);
void model_feedforward(model_context *model);
void model_train(model_context *model, const model_training_desc *training_desc);

#endif
