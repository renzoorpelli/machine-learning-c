#ifndef MNIST_H
#define MNIST_H

#include "model.h"

enum {
    MNIST_NUM_TRAIN = 60000,
    MNIST_NUM_TEST = 10000,
    MNIST_IMAGE_SIZE = 784,
    MNIST_NUM_CLASSES = 10
};

typedef struct {
    matrix *train_images;
    matrix *train_labels;
    matrix *test_images;
    matrix *test_labels;
} mnist_data;

mnist_data mnist_load(const char *dir);
void mnist_data_free(mnist_data *data);
void draw_mnist_digit(const f32 *data);
void create_mnist_model(model_context *model);

#endif
