from os import mkdir
from os.path import join
from shutil import rmtree

import requests
import matplotlib.pyplot as plt

# DISCLAIMER
# Parts have been adapted from 
# https://github.com/the-house-of-black-and-white/morghulis

GOOGLE_DRIVE_URL = 'https://docs.google.com/uc?export=download'
CHUNK_SIZE = 32768

def force_dir(path):
    try:
        rmtree(path)
    except:
        pass

    mkdir(path)


def download_from_webserver(url, destination):
    _, filename = url.rsplit('/', 1)
    download_path = join(destination, filename)

    response = requests.get(url, stream=True)
    save_response_content(response, download_path)

    return download_path


def download_from_google_drive(drive_id, destination):
    session = requests.Session()
    
    response = session.get(
        GOOGLE_DRIVE_URL, 
        params={'id': drive_id}, 
        stream=True
    )
        
    # Confirm download warning (if sent)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(
                GOOGLE_DRIVE_URL,
                params={'id': drive_id, 'confirm': value},
                stream=True
            )
                
        break
        
    save_response_content(response, destination)


def save_response_content(response, destination):
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# TODO is this still relevant?
def peek(dataset, num_samples=5):

    # Get the first num_samples iterms from the dataset
    X, y = next(dataset.batch(num_samples))

    plt.figure(figsize=(num_samples * 2, 3))

    for index in range(num_samples):
        plt.subplot(1, num_samples, index + 1)

        sample, label = X[index], y[index]
        size_x, size_y, _ = sample.shape

        plt.imshow(sample.reshape(size_x, size_y), cmap="binary")
        plt.title(label)
        plt.axis("off")

    plt.show()


class InMemoryDataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y


    def batch(self, batch_size):
        index = 0

        while True:
            if index + batch_size >= len(self.X):
                yield self.X[index:], self.y[index:]
                return
            else:
                next_index = index + batch_size
                yield self.X[index:next_index], self.y[index:next_index]

                index = next_index


    def __len__(self):
        return len(self.X)

