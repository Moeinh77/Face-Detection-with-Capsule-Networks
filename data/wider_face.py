import pickle
from collections import namedtuple
from os.path import exists, join
from zipfile import ZipFile

from data.dataset import StreamDataset
from data.utils import (
    force_dir, 
    download_from_google_drive, 
    download_from_webserver
)


# Google Drive IDs/URLs for WIDER Face
TRAIN_DATA = 'WIDER_train.zip', '0B6eKvaijfFUDQUUwd21EckhUbWs'
VAL_DATA = 'WIDER_val.zip', '0B6eKvaijfFUDd3dIRmpvSk8tLUk'
TEST_DATA = 'WIDER_test.zip', '0B6eKvaijfFUDbW4tdGpaYjgzZkU'
ANNOTATIONS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
EVAL_TOOLS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip'

# Filenames for train/val labels and test images
TRAIN_ANNOTATIONS = 'wider_face_split/wider_face_train_bbx_gt.txt'
VAL_ANNOTATIONS = 'wider_face_split/wider_face_val_bbx_gt.txt'
TEST_FILELIST = 'wider_face_split/wider_face_test_filelist.txt'

Face = namedtuple('Face', 
    [
        'x', 'y', 
        'width', 'height', 
        'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose'
    ]
)

def load_data(datasets_path='datasets'):
    data_path = join(datasets_path, 'wider_face')
    metadata_path = join(data_path, 'metadata.pickle')

    maybe_download(datasets_path)
    maybe_preprocess(datasets_path)

    metadata = pickle.load(open(metadata_path, 'rb'))

    return [
        StreamDataset(*metadata['train']),
        StreamDataset(*metadata['val']),
        StreamDataset(*metadata['test'])
    ]


def maybe_download(datasets_path='datasets', force=False):
    data_path = join(datasets_path, 'wider_face')

    if not exists(data_path) or force:
        force_dir(data_path)
    
        # Download images
        for name, drive_id in (TRAIN_DATA, VAL_DATA, TEST_DATA):
            print('Downloading {}...'.format(name), end='')

            zip_path = join(data_path, name)
            download_from_google_drive(drive_id, zip_path)
        
            with ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(data_path)

            print('Done')

        # Download bbox annotations and eval tools
        print('Downloading annotations and eval tools...', end='')
        for url in (ANNOTATIONS_URL, EVAL_TOOLS_URL):
            zip_path = download_from_webserver(url, data_path)

            with ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(data_path)
            
        print('Done')


def maybe_preprocess(datasets_path='datasets', force=False):
    data_path = join(datasets_path, 'wider_face')
    metadata_path = join(data_path, 'metadata.pickle')

    train_file_path = join(data_path, TRAIN_ANNOTATIONS)
    val_file_path = join(data_path, VAL_ANNOTATIONS)
    test_file_path = join(data_path, TEST_FILELIST)

    # Parse train/val image paths and labels
    if not exists(metadata_path) or force:
        train_metadata = _preprocess_annotations(train_file_path, join(data_path, 'WIDER_train', 'images'))
        val_metadata = _preprocess_annotations(val_file_path, join(data_path, 'WIDER_val', 'images'))

        # Parse test image paths
        with open(test_file_path) as f:
            test_img_files = [l.strip('\n') for l in f.readlines()]
            test_img_paths = [join(data_path, 'WIDER_test', 'images', f) for f in test_img_files]
            test_faces = [None for _ in test_img_paths]

        test_metadata = (test_img_paths, test_faces)

        metadata = {
            'train': train_metadata,
            'val': val_metadata,
            'test': test_metadata
        }

        pickle.dump(metadata, open(metadata_path, 'wb'))


def _preprocess_annotations(annotations_file_path, source_path):

    img_paths = []
    faces = []

    with open(annotations_file_path) as f:
        
        while True:
            # Read the next image path
            img_path = f.readline().strip('\n')

            # Check if EOF is reached
            if img_path == '':
                break

            img_faces = []
            num_faces = int(f.readline())

            # Read annotations for each face
            for _ in range(num_faces):
                face_str = f.readline().strip(' \n')

                # Parse the entire face (but only save the bounding box for now)
                face = Face(*map(int, face_str.split(' ')))
                img_faces.append((face.x, face.y, face.width, face.height))

            img_paths.append(join(source_path, img_path))
            faces.append(img_faces)

    return img_paths, faces

