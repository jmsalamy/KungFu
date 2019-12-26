import argparse
import datetime
import tensorflow as tf
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.ops import reshape_strategy
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)


parser = argparse.ArgumentParser(description='KungFu mnist example.')
parser.add_argument('--kf-optimizer',
                    type=str,
                    default='sync-sgd',
                    help='available options: sync-sgd, async-sgd, sma')
parser.add_argument('--name',
                    type=str,
                    required=True
                    help='name this experiement run for Tensorboard logging')
args = parser.parse_args()

DATASET_SIZE = 10000
TRAIN_VAL_SPLIT = 0.8
NUM_EPOCHS = 20
BATCH_SIZE = 100
# adjust number of steps based on number of workers
NUM_STEPS = (DATASET_SIZE // BATCH_SIZE) // current_cluster_size()


def load_data():

    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % current_rank())
    print(len(mnist_images))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0,
                 tf.float32), tf.cast(mnist_labels, tf.int64)))

    # smaller dataset for quick testing
    smaller_dataset = dataset.take(DATASET_SIZE)
    split = int(DATASET_SIZE*TRAIN_VAL_SPLIT)
    train_dataset = smaller_dataset.take(split).batch(BATCH_SIZE)
    test_dataset = smaller_dataset.skip(split).batch(BATCH_SIZE)
    return train_dataset, test_dataset


def build_model():

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
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
    best_val_acc = 0

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"tensorboard-logs/{args.name}/{time}"
    summary_writer = tf.summary.create_file_writer(
        log_dir, flush_millis=10000)

    step = 0
    with summary_writer.as_default():
        for epoch in range(NUM_EPOCHS):
            print('Start of epoch %d' % (epoch+1,))

            for batch, (images, labels) in enumerate(
                    train_dataset.take(NUM_STEPS)):
                probs, loss_value = training_step(
                    mnist_model, opt, images, labels, batch == 0)

                step += 1
                # print(f"batch number here is {batch}")
                # update training metric
                train_acc_metric(labels, probs)

                # Log loss metric every 10th step only on the 0th worker
                if step % 10 == 0 and current_rank() == 0:
                    print('Training step #%d\tLoss: %.6f' %
                          (step, loss_value))
                    print('Training acc : %s' %
                          float(train_acc_metric.result()))
                    tf.summary.scalar(
                        'training-loss', loss_value, step=step)
                    tf.summary.scalar('training-accuracy',
                                      float(train_acc_metric.result()), step=step)
                    summary_writer.flush()
                    

            # Display metric at the end of each epoch
            train_acc = train_acc_metric.result()
            print('Training acc over epoch: %s' % (float(train_acc),))
            # Reset training metric
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in test_dataset:
                val_logits = mnist_model(x_batch_val)
                # Update val metrics
                val_acc_metric(y_batch_val, val_logits)

            val_acc = val_acc_metric.result()
            tf.summary.scalar('val_accuracy', float(val_acc), step=step)
            summary_writer.flush()
            
            best_val_acc = max(val_acc, best_val_acc)
            val_acc_metric.reset_states()
            print(
                f"VALIDATION ACCURACY : worker {current_rank()} | epoch  {epoch+1} | val_acc {float(val_acc)})")
            print(
                f"BEST VAL ACC OVER TRAINING: worker {current_rank()}, | best_val_acc {best_val_acc}")
