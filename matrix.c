#include "matrix.h"

#include <assert.h>
#include <stdlib.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

matrix *mat_create(u32 rows, u32 cols) {
    matrix *mat = malloc(sizeof(matrix));
    assert(mat != NULL);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = calloc((u64) rows * cols, sizeof(f32));
    assert(mat->data != NULL);

    return mat;
}

void mat_free(matrix *mat) {
    if (mat == NULL) {
        return;
    }
    free(mat->data);
    free(mat);
}

matrix *mat_load(u32 rows, u32 cols, const char *filename) {
    matrix *mat = mat_create(rows, cols);
    FILE *f = fopen(filename, "rb");
    assert(f != NULL);

    fseek(f, 0, SEEK_END);
    u64 size = (u64) ftell(f);
    fseek(f, 0, SEEK_SET);

    u64 expected = sizeof(f32) * (u64) rows * cols;
    assert(size >= expected);
    size_t nread = fread(mat->data, 1, expected, f);
    assert(nread == expected);

    fclose(f);
    return mat;
}

void mat_clear(matrix *mat) { memset(mat->data, 0, sizeof(f32) * (u64) mat->rows * mat->cols); }

void mat_fill(matrix *mat, f32 value) {
    u64 size = (u64) mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) {
        mat->data[i] = value;
    }
}

void mat_fill_rand(matrix *mat, f32 lower, f32 upper) {
    u64 size = (u64) mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) {
        mat->data[i] = ((f32) rand() / (f32) RAND_MAX) * (upper - lower) + lower;
    }
}

void mat_scale(matrix *mat, f32 scale) {
    u64 size = (u64) mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(const matrix *mat) {
    u64 size = (u64) mat->rows * mat->cols;
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }
    return sum;
}

u32 mat_argmax(const matrix *mat) {
    u64 size = (u64) mat->rows * mat->cols;

    u32 max_i = 0;
    for (u64 i = 0; i < size; i++) {
        if (mat->data[i] > mat->data[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

b32 mat_add(matrix *out, const matrix *a, const matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->cols != a->cols || out->rows != a->rows) {
        return false;
    }
    u64 size = (u64) out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

b32 mat_sub(matrix *out, const matrix *a, const matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->cols != a->cols || out->rows != a->rows) {
        return false;
    }
    u64 size = (u64) out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return true;
}

static void mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += a->data[k + i * a->cols] * b->data[j + k * b->cols];
            }
        }
    }
}

static void mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] += a->data[k + i * a->cols] * b->data[k + j * b->cols];
            }
        }
    }
}

static void mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 i = 0; i < out->rows; i++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += a->data[i + k * a->cols] * b->data[j + k * b->cols];
            }
        }
    }
}

static void mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] += a->data[i + k * a->cols] * b->data[k + j * b->cols];
            }
        }
    }
}

b32 mat_mul(
    matrix *out, const matrix *a, const matrix *b,
    b32 zero_out, b8 transpose_a, b8 transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;

    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows) {
        return false;
    }
    if (out->rows != a_rows || out->cols != b_cols) {
        return false;
    }

    if (zero_out) {
        mat_clear(out);
    }
    u32 transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0b00: {
            mat_mul_nn(out, a, b);
        }
        break;
        case 0b01: {
            mat_mul_nt(out, a, b);
        }
        break;
        case 0b10: {
            mat_mul_tn(out, a, b);
        }
        break;
        case 0b11: {
            mat_mul_tt(out, a, b);
        }
        break;
    }
    return true;
}

b32 mat_relu(matrix *out, const matrix *in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64) out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

b32 mat_softmax(matrix *out, const matrix *in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    u64 size = (u64) out->rows * out->cols;
    f32 sum = 0.0f;

    // subtract max for numerical stability (expf overflows on large logits)
    f32 max = in->data[0];
    for (u64 i = 1; i < size; i++) {
        if (in->data[i] > max) {
            max = in->data[i];
        }
    }

    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i] - max);
        sum += out->data[i];
    }

    mat_scale(out, 1.0f / sum);

    return true;
}

b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q) {
    if (p->rows != q->rows || p->cols != q->cols) {
        return false;
    }
    if (out->rows != p->rows || out->cols != p->cols) {
        return false;
    }
    u64 size = (u64) out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q->data[i]);
    }
    return true;
}

b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    if (out->rows != grad->rows || out->cols != grad->cols) {
        return false;
    }

    u64 size = (u64) out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }

    return true;
}

b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out, const matrix *grad) {
    // softmax i = e^q i / SUMj E^qj
    if (out == NULL || softmax_out == NULL || grad == NULL) {
        return false;
    }
    if (softmax_out->rows != 1 && softmax_out->cols != 1) {
        return false;
    }

    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    matrix *jacobian = mat_create(size, size);

    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            jacobian->data[j + i * size] = softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
        }
    }
    b32 result = mat_mul(out, jacobian, grad, 0, 0, 0);

    mat_free(jacobian);
    return result;
}

b32 mat_cross_entropy_add_grad(
    matrix *p_grad, matrix *q_grad, const matrix *p,
    const matrix *q, const matrix *grad
) {
    if (p->rows != q->rows || p->cols != q->cols) {
        return false;
    }

    u64 size = (u64) p->rows * p->cols;
    if (p_grad != NULL) {
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) {
            return false;
        };

        for (u64 i = 0; i < size; i++) {
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
        }
    }

    if (q_grad != NULL) {
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
            return false;
        };

        for (u64 i = 0; i < size; i++) {
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
        }
    }
    return true;
}