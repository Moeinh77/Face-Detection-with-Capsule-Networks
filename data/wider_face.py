from os.path import exists, join
from zipfile import Zipfile

from data.utils import (
    force_dir, 
    download_from_google_drive, 
    download_from_webserver
)


TRAIN_DATA = 'WIDER_train.zip', '0B6eKvaijfFUDQUUwd21EckhUbWs'
VAL_DATA = 'WIDER_val.zip', '0B6eKvaijfFUDd3dIRmpvSk8tLUk'
TEST_DATA = 'WIDER_test.zip', '0B6eKvaijfFUDbW4tdGpaYjgzZkU'
ANNOTATIONS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
EVAL_TOOLS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip'


def maybe_download(datasets_path='datasets', force=False):
    data_path = join(datasets_path, 'wider_face')

    if not exists(data_path) or force:
        utils.force_dir(data_path)
    
        # Download images
        for name, drive_id in (TRAIN_DATA, VAL_DATA, TEST_DATA):
            print('Downloading {}...'.format(name), end='')

            zip_path = join(data_path, name)
            download_from_google_drive(drive_id, zip_path)
        
            with Zipfile(zip_path, 'r') as zip_file:
                zip_file.extractall(data_path)

            print('Done')

        # Download bbox annotations and eval tools
        print('Downloading annotations and eval tools...', end='')
        for url in (ANNOTATIONS_URL, EVAL_TOOLS_URL):
            zip_path = download_from_webserver(url, data_path)

            with Zipfile(zip_path, 'r') as zip_file:
                zip_file.extractall(data_path)
            
        print('Done')



