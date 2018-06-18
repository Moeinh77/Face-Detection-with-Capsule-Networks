from os.path import join

import tensorflow as tf

from caps import layers, losses
from data import mnist
from utils import evaluate, reset, new_experiment, Logger

BATCH_SIZE = 24
NUM_EPOCHS = 2


def model():

    X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name='X')
    y = tf.placeholder(shape=[None], dtype=tf.int64, name='y')
    y_one_hot = tf.one_hot(y, depth=10, name='y_one_hot')
    
    conv1 = tf.layers.conv2d(
        X, 
        filters=256,
        kernel_size=9,
        strides=1,
        padding="valid",
        name="conv1"
    )

    primaryCaps = layers.primaryCaps(
        conv1,
        caps=32,
        dims=8,
        kernel_size=9,
        strides=2,
        name="primaryCaps"
    )

    digitCaps = layers.denseCaps(
        primaryCaps,
        caps=10,
        dims=16
    )

    probabilities = layers.norm(digitCaps, axis=-1, name="probabilities")
    margin_loss = losses.margin_loss(y_one_hot, probabilities, name="margin_loss")

    predictions = tf.argmax(probabilities, axis=-1, name="predictions")

    with tf.variable_scope('accuracy'):
        correct = tf.equal(y, predictions, name='correct')
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    with tf.variable_scope('reconstructions'):
        mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
        
        reconstruction_targets = tf.cond(mask_with_labels,
                                        lambda: y,
                                        lambda: predictions,
                                        name="reconstruction_targets")

        reconstructions = layers.denseDecoder(digitCaps, reconstruction_targets, [512, 1024, 28*28], name="reconstructions")

    reconstruction_loss = losses.reconstruction_loss(X, reconstructions, name="reconstruction_loss")

    alpha = tf.constant(0.0005, name='alpha')
    loss = margin_loss + alpha * reconstruction_loss

    return [X, y], [loss, accuracy]


def train_hook(session, iteration, iterations, loss, accuracy):
    return '\rTrain | Iteration {}/{} ({:.1f}%) | Batch Accuracy: {:.4f}% Loss {:.6f}'.format(
        iteration, 
        iterations,
        iteration * 100 / iterations,
        accuracy * 100,
        loss
    )


def validation_hook(session, iteration, iterations, loss, accuracy):
    return '\rValidation | Iteration {}/{} ({:.1f}%) | Batch Accuracy: {:.4f}% Loss {:.6f}'.format(
        iteration, 
        iterations,
        iteration * 100 / iterations,
        accuracy * 100,
        loss
    )


if __name__ == '__main__':

    # Get data
    train_data, val_data, _ = mnist.load_data()
    
    # Build model
    reset()
    inputs, [loss, accuracy] = model()

    # Optimize
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss, name="train_step")

    # Log progress
    logdir = new_experiment(root_logdir=join('./experiments', 'mnist'))
    checkpoint_path = join(logdir, 'parameters')

    logger = Logger(logdir)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()

        best_val_loss = np.infty

        for epoch in range(num_epochs):
            print('[Epoch {}/{}]'.format(epoch + 1, num_epochs))

            # TRAINING PHASE
            train_loss, train_accuracy = evaluate(
                session=sess,
                inputs=inputs,
                dataset=train_data,
                train_step=train_step,
                batch_size=batch_size,
                feed_dict={mask_with_labels: True},
                metrics=[loss, accuracy],
                report_every=10,
                report_hook=train_hook
            )

            print(
                '\rTrain | Average Accuracy: {:.4f}% Loss: {:.6f}'.format(
                    train_accuracy * 100,
                    train_loss,
                ),
                end=" " * 50 + "\n"
            )

            logger.add('train_loss', train_loss, epoch)
            logger.add('train_accuracy', train_accuracy, epoch)

            # TRAINING PHASE
            val_loss, val_accuracy = evaluate(
                session=sess,
                inputs=inputs,
                dataset=val_data,
                batch_size=batch_size,
                metrics=[loss, accuracy],
                report_every=10,
                report_hook=validation_hook
            )

            print(
                '\rValidation | Average Accuracy: {:.4f}% Loss: {:.6f}'.format(
                    val_accuracy * 100,
                    val_loss,
                ),
                end=" " * 50 + "\n"
            )

            logger.add('val_loss', train_loss, epoch)
            logger.add('val_accuracy', train_accuracy, epoch)

            if val_loss < best_val_loss:
                saver.save(sess, checkpoint_path)
                best_val_loss = val_loss

        logger.write()
