from datetime import datetime
from os import makedirs
from os.path import join, isfile
import pickle

import tensorflow as tf
import numpy as np

from utils import reset
from data import wider_face
from data.transforms import AffineWarp, Elongate, Standardize
from caps import utils 
from yolophem import models

from gradient_checkpointing import memory_saving_gradients

# Monkey-patch tf.gradients to memory_saving_gradients
tf.__dict__['gradients'] = memory_saving_gradients.gradients_memory


## SETTINGS

BATCH_SIZE = 16
MODEL = models.adapted
CONFIG = models.config_small
PARAMS = {} #{ 'feature_size': 512 }
STEP_SIZE = 1e-3
NUM_EPOCHS = 50
REPORT_EVERY = 10
EXPERIMENT = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')

def report(mode, iteration, num_iterations, loss):
    print(
        '\r[{} | {}/{} ({:.1f}%)] Batch Loss: {:.6f}'.format(
            mode,
            iteration + 1,
            num_iterations,
            (iteration + 1) * 100 / num_iterations,
            loss
        ),
        end=''
    )


class Logger:
    
    def __init__(self, logdir):
        self.logfile = join(logdir, 'log.pickle')
        self.log = {}

        if isfile(self.logfile):
            self.log = pickle.load(open(self.logfile, 'rb'))


    def add(self, mode, timestamp, statistic):
        if not mode in self.log:
            self.log[mode] = []

        self.log[mode].append((timestamp, statistic))


    def write(self):
        pickle.dump(self.log, open(self.logfile, 'wb'))


if __name__ == '__main__':

    # Get the data
    train, val, _ = wider_face.load_data()

    # Build the model
    reset()
    [X, y], [predictions, confidences], loss = MODEL(CONFIG, **PARAMS)
    
    # Minimize loss
    optimizer = tf.train.AdamOptimizer(STEP_SIZE)
    train_step = optimizer.minimize(loss, name='train_step')
    
    # Capture trainign prorestraining
    logdir = join('./experiments', EXPERIMENT)
    makedirs(logdir)
    
    checkpoint_path = join(logdir, 'parameters')

    saver = tf.train.Saver()
    logger = Logger(logdir)

    # Run the graph
    init = tf.global_variables_initializer()

    train_iterations = int(np.ceil(len(train) / BATCH_SIZE))
    val_iterations = int(np.ceil(len(val) / BATCH_SIZE))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init.run()
        print('Initialized')

        best_val_loss = np.infty

        for epoch in range(NUM_EPOCHS):
            print('[Epoch {}/{}]'.format(epoch + 1, NUM_EPOCHS))

            #################### TRAIN #####################
            train_batches = train.batch(
                BATCH_SIZE, 
                [
                    AffineWarp(CONFIG['image_size'], CONFIG['image_size']),
                    Standardize(0, 255),
                    Elongate()
                ]
            )

            train_losses = []
            for iteration, (images, labels) in enumerate(train_batches):
                feed_dict = { X: images, y: labels }

                # Perform a training step
                sess.run(train_step, feed_dict=feed_dict)

                # Report Progress
                if (iteration + 1) % REPORT_EVERY == 0:
                    train_loss = sess.run(loss, feed_dict=feed_dict)
                    train_losses.append(train_loss)

                    report('TRAIN', iteration, train_iterations, train_loss)          
                
            # Log the last batch loss for 
            epoch_train_loss = np.mean(train_losses)
            
            logger.add('TRAIN', epoch + 1, epoch_train_loss)
            logger.write()
            
            print(
                '\r[TRAIN] Loss: {:.6f}'.format(epoch_train_loss), 
                end=' ' * 50 + '\n'
            )
            ###############################################


            ################## VALIDATE ###################

            val_batches = val.batch(
                BATCH_SIZE, 
                [
                    AffineWarp(CONFIG['image_size'], CONFIG['image_size']),
                    Standardize(0, 255),
                    Elongate()
                ]
            )

            val_losses = []

            for iteration, (images, labels) in enumerate(val_batches):
                feed_dict = { X: images, y: labels }

                val_loss = sess.run(loss, feed_dict=feed_dict)
                val_losses.append(val_loss)

                # Report Progress
                if (iteration + 1) % REPORT_EVERY == 0:
                    report('VAL', iteration, val_iterations, val_loss)          
                
            # Log and report Epoch loss
            epoch_val_loss = np.mean(val_losses)

            if epoch_val_loss < best_val_loss:
                saver.save(sess, checkpoint_path)
                best_val_loss = epoch_val_loss

            logger.add('VAL', epoch + 1, epoch_val_loss)
            logger.write()
            
            print(
                '\r[VAL] Loss: {:.6f}'.format(epoch_val_loss), 
                end=' ' * 50 + '\n'
            )
            ###############################################
