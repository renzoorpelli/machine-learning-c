#include "types.h"
#include "matrix.h"
#include "model.h"
#include "mnist.h"

#include <stdlib.h>
#include <time.h>

int main(void) {
    srand((unsigned) time(NULL));

    mnist_data data = mnist_load("data");

    draw_mnist_digit(data.train_images->data);

    // create the model
    model_context *model = model_create();
    create_mnist_model(model);
    model_compile(model);

    const model_training_desc training_desc = {
        .train_images = data.train_images,
        .train_labels = data.train_labels,
        .test_images = data.test_images,
        .test_labels = data.test_labels,
        .epochs = 3,
        .batch_size = 50,
        .learning_rate = 0.01f
    };
    model_train(model, &training_desc);

    memcpy(model->input->val->data, data.train_images->data, sizeof(f32) * MNIST_IMAGE_SIZE);

    model_feedforward(model);
    printf("post-training output: \n");
    for (u32 i = 0; i < MNIST_NUM_CLASSES; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");

    mnist_data_free(&data);
    model_destroy(model);
    return 0;
}
