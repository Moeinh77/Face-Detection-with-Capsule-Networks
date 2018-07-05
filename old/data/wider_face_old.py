from  os.path import join
import pickle

import numpy as np
from PIL import Image




def load_data(dataset_root='datasets'):
    data_path = join(dataset_root, 'wider_face')
    
    # Load Metadata
    metadata_path = join(data_path, 'metadata.pickle')
    metadata = pickle.load(open(metadata_path, 'rb'))

    # Load Datasets
    partitions = ['train', 'val', 'test']
    datasets = { p: Dataset(join(data_path, p), metadata[p]) for p in partitions }

    return datasets


class Dataset:

    def __init__(self, dataset_path, metadata):
        self.dataset_path = dataset_path
        self.metadata = metadata
    
    
    def batch(self, batch_size):
        
        # Shuffle the dataset
        shuffle_idx = np.random.permutation(len(self.metadata.img_paths))
        img_paths = self.metadata.img_paths[shuffle_idx]

        # Iterate over the dataset
        index = 0

        while True:
            if index + batch_size >= len(img_paths):
                yield self._load(img_paths[index:])
                return
            else:
                next_index = index + batch_size
                yield self._load(img_paths[index:next_index])
                
                index = next_index


    def _load(self, img_paths):
    
        images = []
        faces = []

        for img_path in img_paths:
            image = Image.open(join(self.dataset_path, img_path))
            contained_faces = self.metadata.faces[img_path]

            images.append(image)
            faces.append(contained_faces)

        return images, faces

