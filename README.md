# Bachelor Thesis

This repository contains code, model parameters and evaluation results for my 
Bachelor's Thesis "Face Detection with Capsule Networks". It is organized into
the following files and subdirectories:

 - `caps/` A TensorFlow implementation of Capsule Networks
 - `yolophem/` Implementation of YOLOphem, our Face Detection architecture
 - `data/` Auxilliary tools for data management
 - `experiments/` model weights and loss curve logs
 - `evaluation/` YOLOphem's predictions on WIDER FACE (Validation and Test set)
 - `train.py` A script to train YOLOphem
 - `overfit.py` A script to let YOLOphem overfit on a small sample of images
 - `Loss Curves.ipynb` A Jypter Notebook that plots loss curves for experiments
 - `Evaluation.ipynb` A Jupyter Notebook that evaluates YOLOphem on WIDER FACE

## Dependencies

Models are written in Python 3.6 using TensorFlow 1.8 and some external
dependencies for pre-processing and data management. Dependencies are managed
with Pipenv; to install run `pipenv install' in the top-level directory of 
your working copy. This will setup a virtual environment and install the 
necessary packages. 


To make the environment accessible inside a Jupyter Notebook, activate the 
virtualenv using `pipenv shell` and run 

```python -m ipykernel install --user --name=Bachelor-Thesis```

**You will still have to manually install TensorFlow on your system**. This is 
a little hacky, but allows running the project on machines both with and without 
TensorFlow GPU support.

We used [gradient checkpointing](https://github.com/openai/gradient-checkpointing)
to train YOLOphem on our hardware. If you want to do so as well, simply copy 
the linked repository into the top-level directory of your working copy and 
uncomment the respective lines in `train.py`.
