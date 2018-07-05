from os.path import join
import pickle

import numpy as np
from PIL import Image


def load_data(dataset_root='datasets', img_size=64, num_train=2992, num_val=800, num_test=500):
    assert num_train + num_val + num_test <= 4292

    data_path = join(dataset_root, 'wider_face')

    # Lead Metadata
    metadata_path = join(data_path, 'metadata_single.pickle')
    img_paths, faces = pickle.load(open(metadata_path, 'rb'))

    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(img_paths))
    img_paths_shuffled = img_paths[shuffle_idx]
    faces_shuffled = faces[shuffle_idx]

    train = Dataset(
        img_paths_shuffled[:num_train], 
        faces_shuffled[:num_train],
        img_size
    )

    val = Dataset(
        img_paths_shuffled[num_train:num_val], 
        faces_shuffled[num_train:num_val],
        img_size
    )

    test = Dataset(
        img_paths_shuffled[num_val:num_test],
        faces_shuffled[num_val:num_test],
        img_size
    )

    return train, val, test


class Dataset:

    def __init__(self, img_paths, faces, img_size):
        self.img_paths = img_paths
        self.faces = faces
        self.img_size = img_size


    def batch(self, batch_size):
        index = 0

        while True:
            if index + batch_size >= len(self.img_paths):
                yield self._load(index,None)
                return
            else:
                next_index = index + batch_size
                yield self._load(index, next_index)

                index = next_index


    def _load(self, from_idx, to_idx):

        images = []
        labels = []

        if to_idx:
            img_paths = self.img_paths[from_idx:to_idx]
            faces = self.faces[from_idx:to_idx]
        else:
            img_paths = self.img_paths[from_idx:]
            faces = self.faces[from_idx:]

        for path, face in zip(img_paths, faces):

            image = Image.open(path)

            # Center-crop image to shorter edge
            width, height = image.size
            crop_size = min(width, height)

            left = (width - crop_size) // 2
            right = (width + crop_size) // 2
            top = (height - crop_size) // 2
            bottom = (height + crop_size) // 2

            image_cropped = image.crop((left, top, right, bottom))

            # Copy face to prevent altering the original (in case its loaded
            # several times)
            face_cropped = np.copy(face)
            face_cropped[0] -= left
            face_cropped[1] -= top

            # Rescale to 128 x 128
            image_scaled = image_cropped.resize((self.img_size, self.img_size))
            face_scaled = face_cropped * (self.img_size / crop_size)

            # Normalize to [0,1]
            image_npy = np.array(image_scaled)
            image_npy_normalized = (image_npy - 255) / 255

            images.append(image_npy_normalized)
            labels.append(face_scaled)

        return np.array(images), np.array(labels)

