FROM nvidia/cuda:12.1.0-base-ubuntu20.04

COPY ./ ./

RUN apt update
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y python3 python3-pip

RUN python3 -m pip install -U pip wheel cmake

RUN python3 -m pip install -r ./requirements/base.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3344", "--workers", "5"]