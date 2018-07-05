from os import mkdir
from os.path import join
from shutil import rmtree

import requests

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
