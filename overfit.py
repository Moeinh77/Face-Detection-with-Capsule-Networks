import tensorflow as tf

from utils import reset, evaluate
from data import wider_face
from data.transforms import AffineWarp, Elongate, Standardize
from yolophem import models
from caps import utils

from gradient_checkpointing import memory_saving_gradients

# Monkey-patch tf.gradients to memory_saving_gradients
tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

## SETTINGS

OVERFIT_SAMPLES = 8
MODEL = models.naive
CONFIG = models.config_small
PARAMS = { 'feature_size': 512 }
STEP_SIZE = 1e-2
NUM_EPOCHS = 1000

if __name__ == '__main__':

    # Get the data    
    train, _, _ = wider_face.load_data()
    data = train.batch(OVERFIT_SAMPLES, [
        AffineWarp(CONFIG['image_size'], CONFIG['image_size']), 
        Standardize(0, 255),
        Elongate()
        ]
    )
    
    images, labels = next(data)

    # Build the model
    reset()
    [X, y], [predictions, confidences], loss = MODEL(CONFIG, **PARAMS)

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
        print('Start session')
        init.run()
        print('Initialized')

        ys, preds = sess.run([y, predictions], feed_dict=feed_dict)
        print(ys.shape)
        print(preds.shape)

        #for epoch in range(NUM_EPOCHS):

            # Run a train step           
            #sess.run(train_step, feed_dict=feed_dict)

            # Evaluate loss
            #epoch_loss = sess.run([loss], feed_dict=feed_dict)

            #print('[{}/{}] Loss: {}'.format(epoch+1, NUM_EPOCHS, epoch_loss))
