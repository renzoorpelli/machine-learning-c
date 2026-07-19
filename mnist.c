#include "mnist.h"

#include <assert.h>

mnist_data mnist_load(const char *dir) {
    char path[256];
    mnist_data data;

    snprintf(path, sizeof(path), "%s/train_images.mat", dir);
    data.train_images = mat_load(MNIST_NUM_TRAIN, MNIST_IMAGE_SIZE, path);
    snprintf(path, sizeof(path), "%s/test_images.mat", dir);
    data.test_images = mat_load(MNIST_NUM_TEST, MNIST_IMAGE_SIZE, path);

    data.train_labels = mat_create(MNIST_NUM_TRAIN, MNIST_NUM_CLASSES);
    data.test_labels = mat_create(MNIST_NUM_TEST, MNIST_NUM_CLASSES);

    snprintf(path, sizeof(path), "%s/train_labels.mat", dir);
    matrix *train_labels_file = mat_load(MNIST_NUM_TRAIN, 1, path);
    snprintf(path, sizeof(path), "%s/test_labels.mat", dir);
    matrix *test_labels_file = mat_load(MNIST_NUM_TEST, 1, path);

    for (u32 i = 0; i < MNIST_NUM_TRAIN; i++) {
        u32 num = (u32) train_labels_file->data[i];
        if (num < MNIST_NUM_CLASSES)
            data.train_labels->data[i * MNIST_NUM_CLASSES + num] = 1.0f;
    }

    for (u32 i = 0; i < MNIST_NUM_TEST; i++) {
        u32 num = (u32) test_labels_file->data[i];
        if (num < MNIST_NUM_CLASSES)
            data.test_labels->data[i * MNIST_NUM_CLASSES + num] = 1.0f;
    }

    mat_free(train_labels_file);
    mat_free(test_labels_file);

    return data;
}

void mnist_data_free(mnist_data *data) {
    mat_free(data->train_images);
    mat_free(data->train_labels);
    mat_free(data->test_images);
    mat_free(data->test_labels);
}

void draw_mnist_digit(const f32 *data) {
    for (u32 y = 0; y < 28; y++) {
        for (u32 x = 0; x < 28; x++) {
            f32 num = data[x + y * 28];
            u32 col = 232 + (u32) (num * 24);
            printf("\x1b[48;5;%dm   ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

void create_mnist_model(model_context *model) {
    model_var *input = mv_create(model, MNIST_IMAGE_SIZE, 1, MV_FLAG_INPUT);

    model_var *W0 = mv_create(model, 16, MNIST_IMAGE_SIZE, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var *W1 = mv_create(model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var *W2 = mv_create(model, MNIST_NUM_CLASSES, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    f32 bound0 = sqrtf(6.0f / (MNIST_IMAGE_SIZE + 16));
    f32 bound1 = sqrtf(6.0f / (16 + 16));
    f32 bound2 = sqrtf(6.0f / (16 + MNIST_NUM_CLASSES));

    mat_fill_rand(W0->val, -bound0, bound0);
    mat_fill_rand(W1->val, -bound1, bound1);
    mat_fill_rand(W2->val, -bound2, bound2);

    model_var *b0 = mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var *b1 = mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var *b2 = mv_create(model, MNIST_NUM_CLASSES, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    model_var *z0_a = mv_matmul(model, W0, input, 0);
    model_var *z0_b = mv_add(model, z0_a, b0, 0);
    model_var *a0 = mv_relu(model, z0_b, 0);
    assert(z0_a != NULL && z0_b != NULL && a0 != NULL);

    model_var *z1_a = mv_matmul(model, W1, a0, 0);
    model_var *z1_b = mv_add(model, z1_a, b1, 0);
    model_var *z1_c = mv_relu(model, z1_b, 0);
    model_var *a1 = mv_add(model, a0, z1_c, 0);
    assert(z1_a != NULL && z1_b != NULL && z1_c != NULL && a1 != NULL);

    model_var *z2_a = mv_matmul(model, W2, a1, 0);
    model_var *z2_b = mv_add(model, z2_a, b2, 0);
    model_var *output = mv_softmax(model, z2_b, MV_FLAG_OUTPUT);
    assert(z2_a != NULL && z2_b != NULL && output != NULL);

    model_var *y = mv_create(model, MNIST_NUM_CLASSES, 1, MV_FLAG_DESIRED_OUTPUT);

    model_var *cost = mv_cross_entropy(model, y, output, MV_FLAG_COST);
    assert(cost != NULL);
}
