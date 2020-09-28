from python:3.8

WORKDIR /app
RUN apt-get update
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt && python -m pip install .
