import gzip
import struct
import os.path
from urllib.request import urlretrieve

import numpy as np

TRAIN_IMAGES_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

TEST_IMAGES_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def load_data(datasets_dir='datasets', force=False):

    data_file = os.path.join(datasets_dir, 'mnist.npy')
    
    if not os.path.exists(data_file) or force:
        # Download data files
        train_images_path = _download(TRAIN_IMAGES_URL, datasets_dir)
        train_labels_path = _download(TRAIN_LABELS_URL, datasets_dir)
        test_images_path = _download(TEST_IMAGES_URL, datasets_dir)
        test_labels_path = _download(TEST_LABELS_URL, datasets_dir)
        
        # Extract and parse into Numpy arrays
        X_raw = _extract_images(train_images_path)
        y_raw = _extract_labels(train_labels_path)

        X_val = X_raw[:5000]
        y_val = y_raw[:5000]

        X_train = X_raw[5000:]
        y_train = y_raw[5000:]

        X_test = _extract_images(test_images_path)
        y_test = _extract_labels(test_labels_path)

        # Cleanup
        os.remove(train_images_path)
        os.remove(train_labels_path)
        os.remove(test_images_path)
        os.remove(test_labels_path)

        # Save data to disk
        np.save(data_file, 
            [(X_train, y_train), (X_val, y_val), (X_test, y_test)])
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = np.load(data_file)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        

def _download(url, datasets_dir):
    
    # Extract filename from URL
    _, filename = url.rsplit('/', 1)

    # Download the file
    download_path = os.path.join(datasets_dir, filename)
    urlretrieve(url, download_path)

    return download_path


def _extract_images(path):
    with gzip.open(path, 'rb') as f:
        _, length, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(length, rows*cols)

    return images


def _extract_labels(path):
    with gzip.open(path, 'rb') as f:
        _, length = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels
