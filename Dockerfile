FROM tensorflow/tensorflow:2.11.0-gpu

RUN python3 -m pip install --upgrade pip

RUN apt-get update -y && \
    apt-get install -y git

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# a hacky way to pass the github token and install the package
COPY env.sh /env.sh
ENV BASH_ENV=/env.sh
RUN python3 -m pip install git+https://$GITHUB_PAT@github.com/xuefei-wang/deepcelltypes-kit.git

COPY . .
