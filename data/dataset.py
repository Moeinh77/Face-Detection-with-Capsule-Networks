import numpy as np
from PIL import Image


class Dataset:

    def __init__(self):
        pass

    def batch(self, batch_size, transforms=[]):
        #permutation = np.random.permutation(len(self))
        #print(permutation)
        
        batch_indices = np.arange(len(self))

        index = 0

        while True:
            if index + batch_size >= len(self):
                #images, labels = self._load(index, None)
                images, labels = self._load(batch_indices[index:])                

                for transform in transforms:
                    images, labels = transform(images, labels)

                yield images, labels
                return
            else:
                next_index = index + batch_size
                #images, labels = self._load(index, next_index)
                images, labels = self._load(batch_indices[index:next_index])

                for transform in transforms:
                    images, labels = transform(images, labels)

                yield images, labels
                index = next_index

    def _load(self, indices):
        pass



class InMemoryDataset(Dataset):

    def __init__(self, images, labels):
        super(InMemoryDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def _load(self, start, end):
        if end:
            images = self.images[start:end]
            labels = self.labels[start:end]
        else:
            images = self.images[start:]
            labels = self.labels[start:]

        return images, labels


class StreamDataset(Dataset):

    def __init__(self, img_paths, labels):
        super(StreamDataset, self).__init__()
        self.img_paths = np.array(img_paths)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.img_paths)

    def _load(self, indices):
        img_paths = self.img_paths[indices]
        labels = self.labels[indices]

        images = [np.array(Image.open(p)) for p in img_paths]
        return images, labels
