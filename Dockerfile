FROM tensorflow/tensorflow:2.11.0-gpu

RUN python3 -m pip install --upgrade pip

RUN apt-get update -y && apt-get install -y git

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

COPY . .

RUN python3 -m pip install deepcelltypes-kit/
