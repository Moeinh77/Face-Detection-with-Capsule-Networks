import tensorflow as tf

from utils import reset, evaluate
from data import wider_face
from data.transforms import AffineWarp, Elongate
from yolophem import models
from caps import utils

## SETTINGS

OVERFIT_SAMPLES = 1
MODEL = models.naive
CONFIG = models.config_v1
PARAMS = { 'feature_size': 4096 }
STEP_SIZE = 1e-3
NUM_EPOCHS = 1000

if __name__ == '__main__':

    # Get the data    
    train, _, _ = wider_face.load_data()
    data = train.batch(OVERFIT_SAMPLES, [AffineWarp(448, 448), Elongate()])
    
    images, labels = next(data)

    # Build the model
    reset()
    [X, y], _, loss = MODEL(CONFIG, **PARAMS)

    feed_dict = {
        X: images,
        y: labels
    }

    # Minimize loss
    optimizer = tf.train.AdamOptimizer(STEP_SIZE)
    train_step = optimizer.minimize(loss, name='train_step')

    # Run the graph
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()

        for epoch in range(NUM_EPOCHS):

            # Run a train step           
            sess.run(train_step, feed_dict=feed_dict)

            # Evaluate loss
            epoch_loss = sess.run([loss], feed_dict=feed_dict)

            print('[{}/{}] Loss: {}'.format(epoch+1, NUM_EPOCHS, epoch_loss))
