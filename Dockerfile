FROM tensorflow/tensorflow:latest-gpu

RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
