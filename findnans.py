import tensorflow as tf
import numpy as np

from utils import reset
from data import wider_face
from data.transforms import AffineWarp, Elongate, Standardize
from yolophem import models

train, val, _ = wider_face.load_data()

config = models.config_small
img_size = config['image_size']

train_samples = train.batch(
    1, 
    [
        AffineWarp(img_size, img_size), 
        Standardize(0, 255),
        Elongate()
    ]
)

reset()
[X, y], [predictions, confidences], loss = models.naive(
    models.config_small, 
    feature_size=512
)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    init.run()

    for idx, (images, labels) in enumerate(train_samples):
        
        l = sess.run(loss, feed_dict={X:images, y:labels})
        
        print('\r{}/{}'.format(idx, len(train)), end='')
        if np.isnan(l):
            print('Hit!')
