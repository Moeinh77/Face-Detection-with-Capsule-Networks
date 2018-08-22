# Face Detection with Capsule Networks

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
the linked repository into the top-level directory and uncomment the respective 
lines in `train.py`.


## Train YOLOphem 

To train YOLOphem on WIDER FACE, simply execute `python train.py`. The dataset 
will be downloaded automatically into a directory called `datasets` (note that
this requires a working internet connection and may take a while). The training
procedure is controlled via global variables in the `train.py` script:

 - `BATCH_SIZE`: the number of training samples used per training step
 - `MODEL`: the YOLOphem variant; either `models.naive` for YOLOphem A or `models.adapted`
for YOLOphem B
 - `CONFIG`: specification of YOLOphem's layers; use `models.config_small` for the
reduced YOLOphem model (which I trained in the thesis), or `models.config_original` for
the larger model (which we couldn't train on our hardware)
 - `PARAMS`: configuration parameters for the model architecture; either `{ 'feature_size': \*some integer\*}` for YOLOphem A or `{}` for YOLOphem B
 - `LEARNING_RATES`: a list of tuples where each tuple contains an int to 
denote a number of epochs and a float to denote a learning rate. During training, the
list is read from left to right; the model is trained with a particular learning for
the corresponding number of epochs.
 - `NUM_EPOCHS`: the total number of epochs, this value must be consistent with the 
numbers specified in `LEARNING_RATES`
 - `REPORT_EVERY`: number of training steps after which the current loss value will
be reported on the command line
 - `EXPERIMENT`: identifier for the training run; this string will be used as the name
for a subdirectory of `experiments` under which the training process will be logged and 
model parameter checkpoints will be saved. If the experiment should already exist, the 
model is instantiated from the saved parameters and training is continued from where it
left off. This allows interrupting training for experiments with a large number of epochs.

The model weights for the YOLOphem variants mentioned in the thesis are stored
under `experiments/yolophem_A_small/` and `experiments/yolophem_B_small/`, 
respectively. 

## Loss Curves

To generate plots of the training and validation loss curves, use the `Loss Curves.ipynb`
Jupyter Notebook. There are two settings:

 - `EXPERIMENT`: identifier for the experiment; see above
 - `OUPTUT`: name of the output file, without file extension


## Model Predictions

To let a trained model compute predictions on either the validation or test set
of WIDER FACE, use the `Evaluation.ipynb` Jupyter Notebook. There are six 
settings:

 - `EXPERIMENT`, `CONFIG`, `MODEL`, `PARAMS`: see above
 - `MODE`: either 'VAL' for the validation set or 'TEST' for the test set
 - `OUTPUT`: path where the computed evaluations should be stored.

Predictions for the YOLOphem variants mentioned in the thesis are stored in the 
`evaluation` directory.
