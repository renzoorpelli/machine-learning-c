#ifndef MATRIX_H
#define MATRIX_H

#include "types.h"

typedef struct {
    u32 rows, cols;
    f32 *data;
} matrix;

matrix *mat_create(u32 rows, u32 cols);
void mat_free(matrix *mat);
matrix *mat_load(u32 rows, u32 cols, const char *filename);
void mat_clear(matrix *mat);
f32 mat_sum(const matrix *mat);
u32 mat_argmax(const matrix *mat);
void mat_fill(matrix *mat, f32 value);
void mat_fill_rand(matrix *mat, f32 lower, f32 upper);
void mat_scale(matrix *mat, f32 scale);
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b32 zero_out, b8 transpose_a, b8 transpose_b);
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in);
b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q);
b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out, const matrix *grad);
b32 mat_cross_entropy_add_grad(matrix *p_grad, matrix *q_grad, const matrix *p, const matrix *q, const matrix *grad);

#endif
