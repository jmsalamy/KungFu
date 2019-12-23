import argparse

import tensorflow as tf
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)

parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='sync-sgd',
                    help='available options: sync-sgd, async-sgd, sma')
args = parser.parse_args()

DATASET_SIZE = 60000


def load_data():

    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % current_rank())
    print(len(mnist_images))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0,
                 tf.float32), tf.cast(mnist_labels, tf.int64)))

    dataset = dataset.repeat()

    # smaller dataset for quick testing
    smaller_dataset = dataset.take(1000)
    train_dataset = smaller_dataset.take(800).batch(128)
    test_dataset = smaller_dataset.skip(800).batch(128)
    return train_dataset, test_dataset


def build_model():

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return mnist_model


def build_optimizer():
    # KungFu: adjust learning rate based on number of GPUs.
    # opt = tf.keras.optimizers.SGD(0.001 * current_cluster_size())
    opt = tf.compat.v1.train.AdamOptimizer(0.001 * current_cluster_size())

    # KungFu: wrap tf.compat.v1.train.Optimizer.
    if args.kf_optimizer == 'sync-sgd':
        opt = SynchronousSGDOptimizer(opt)
    elif args.kf_optimizer == 'async-sgd':
        opt = PairAveragingOptimizer(opt)
    elif args.kf_optimizer == 'sma':
        opt = SynchronousAveragingOptimizer(opt)
    else:
        raise RuntimeError('Unknown KungFu optimizer')

    return opt


# def test_model(model, opt, dataset):
#     model.compile(optimizer=opt,
#                   loss=tf.losses.SparseCategoricalCrossentropy(),
#                   metrics=[tf.metrics.SparseCategoricalAccuracy()])
#     test_metrics = model.evaluate(dataset, verbose=0)
#     # print metrics
#     loss_index = 0
#     accuracy_index = 1
#     print('test accuracy: %f' % test_metrics[accuracy_index])
#     print('test loss : %f' % test_metrics[loss_index])


@tf.function
def training_step(mnist_model, opt, images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss = tf.losses.SparseCategoricalCrossentropy()
        loss_value = loss(labels, probs)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # KungFu: broadcast is done after the first gradient step to ensure optimizer initialization.
    if first_batch:
        print("COMES HERE--------------")
        from kungfu.tensorflow.initializer import broadcast_variables
        broadcast_variables(mnist_model.variables)
        broadcast_variables(opt.variables())
        

    return probs, loss_value


if __name__ == "__main__":
    train_dataset, test_dataset = load_data()
    opt = build_optimizer()
    mnist_model = build_model()

     # Prepare the metrics.
    train_acc_metric = tf.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.metrics.SparseCategoricalAccuracy()

    # KungFu: adjust number of steps based on number of GPUs.
    for batch, (images, labels) in enumerate(
            train_dataset.take(800 // current_cluster_size())):
        probs, loss_value = training_step(
            mnist_model, opt, images, labels, batch == 0)

        train_acc_metric(labels, probs)

        if batch % 10 == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))
            train_acc = train_acc_metric.result()
            print('Training acc over epoch: %s' % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in test_dataset:
        val_logits = mnist_model(x_batch_val)
        # Update val metrics
        val_acc_metric(y_batch_val, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc: %s' % (float(val_acc),))

   
