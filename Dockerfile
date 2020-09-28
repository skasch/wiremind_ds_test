from python:3.8

RUN apt-get update
RUN python -m pip install --upgrade pip
COPY . /app
WORKDIR /app
RUN python -m pip install -r requirements.txt && python -m pip install .
