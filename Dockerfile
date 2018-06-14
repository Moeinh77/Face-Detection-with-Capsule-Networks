FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /Bachelor-Thesis

ADD . /Bachelor-Thesis

RUN pip install pipenv
RUN pipenv --python 3 --site-packages
RUN pipenv install
RUN python -m ipykernel install --user --name=Bachelor-Thesis

EXPOSE 8888
