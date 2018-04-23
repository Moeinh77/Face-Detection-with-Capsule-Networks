# Bachelor Thesis
Notes, LaTeX and code concerning my Bachelor's Thesis on Face Recognition wiht
Capsule Networks.


## Dependencies

Code is written in Python 3.6 and makes extensive use of Jupyter Notebooks.
Python Dependencies are managed with Pipenv. To install dependencies, run
`pipenv install` in the top-level directory of this repository. This will
setup a virtual environment and install

 - numpy
 - scipy
 - matplotlib
 - pandas
 - ipykernel

To make these packages accessible in a Jupyter Notebook, run 

```python -m ipykernel install --user --name=Bachelor-Thesis```

within the activated virtualenv. This will create an IPython Kernel named 
`Bachelor-Thesis` to which all Notebooks in this repo default.

*You will still have to manually install TensorFlow on your system**. This is 
a hack to allow running the project on machines both with and without TensorFlow
GPU support.
