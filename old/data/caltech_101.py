import os
import os.path
import pickle
import shutil
import tarfile
from urllib.request import urlretrieve

from PIL import Image
import numpy as np


DATASET_URL = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
IMAGE_SCALE = 64

def load_data(datasets_path='datasets', force=False):

    data_path = os.path.join(datasets_path, 'caltech_101.pickle')
    cache_path = os.path.join(datasets_path, '.cache')

    if not os.path.exists(data_path) or force:
        
        # Extract filename from URL
        _, filename = DATASET_URL.rsplit('/', 1)

        # Cache raw data archive
        download_path = os.path.join(cache_path, filename)
        if not os.path.exists(download_path):
            # Download the file
            print('Downloading... ', end='')
            urlretrieve(DATASET_URL, download_path)
            print('Done!')
        
        # Extract .tar.gz archive
        print('Extracting... ', end='')
        extract_path, _ = download_path.rsplit('.tar.gz')

        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(cache_path)

        print('Done!')
        
        faces_path = os.path.join(extract_path, 'Faces_easy')
        motorbikes_path = os.path.join(extract_path, 'Motorbikes')

        faces_images = _load_images(faces_path)
        faces_labels = np.zeros(len(faces_images))

        motorbikes_images = _load_images(motorbikes_path)
        motorbikes_labels = np.ones(len(motorbikes_images))

        X_raw = np.r_[faces_images, motorbikes_images]
        y_raw = np.r_[faces_labels, motorbikes_labels]

        shuffle_index = np.random.permutation(len(y_raw))
        X_shuffled = X_raw[shuffle_index]
        y_shuffled = y_raw[shuffle_index]

        X_test = X_shuffled[:100]
        y_test = y_shuffled[:100]

        X_val = X_shuffled[100:150]
        y_val = y_shuffled[100:150]

        X_train = X_shuffled[150:]
        y_train = y_shuffled[150:]

        shutil.rmtree(extract_path)
        pickle.dump(((X_train, y_train), (X_val, y_val), (X_test, y_test)), open(data_path, 'wb'))
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = pickle.load(open(data_path, 'rb'))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test) 


def _load_images(path):

    images = []

    for image_file in os.listdir(path):
        image_path = os.path.join(path, image_file)

        image = Image.open(image_path)

        # Convert to grayscale
        image = image.convert('L')

        # Center crop to quadratic size
        image = _center_crop(image)

        # rescale to homogeneous size
        image = image.resize((IMAGE_SCALE, IMAGE_SCALE))

        # convert to numpy array
        image_np = np.array(image)
        image_np = image_np.reshape(-1)

        images.append(image_np)

    images_np = np.array(images)
    return images_np

def _center_crop(image):
    width, height = image.size
    crop_size = min(width, height)

    left = int((width - crop_size) / 2)
    right = int((width + crop_size) / 2)
    top = int((height - crop_size) / 2)
    bottom = int((height + crop_size) / 2)

    return image.crop([left, top, right, bottom])
