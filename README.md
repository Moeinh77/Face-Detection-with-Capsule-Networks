# Bachelor Thesis
Notes, LaTeX and code concerning my Bachelor's Thesis on post-CNN Computer 
Vision approaches.


## Dependencies

Code is written in Python 3.6 and makes extensive use of Jupyter Notebooks.
Python Dependencies are managed with Pipenv. To install dependencies, run
`pipenv install` in the top-level directory of this repository. This will
setup a virtual environment and install

 - numpy
 - scipy
 - matplotlib
 - pandas
 - torch
 - torchvision
 - ipykernel

To make these packages accessible in a Jupyter Notebook, run 

```python -m ipykernel install --user --name=Bachelor-Thesis```

within the activated virtualenv. This will create an IPython Kernel named 
`Bachelor-Thesis` to which all Notebooks in this repo default. 
