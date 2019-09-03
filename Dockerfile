FROM python:3.7
ENV TF_CUDNN_USE_AUTOTUNE=0
RUN apt-get update && apt-get -y install ffmpeg x264 libx264-dev && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
RUN git clone https://github.com/bureaucratic-labs/pose-tensorflow.git pose_tensorflow
RUN touch pose_tensorflow/__init__.py
RUN cd pose_tensorflow/models/mpii && ./download_models.sh && cd -
COPY ./requirements.txt /usr/src/app
RUN pip install -r /usr/src/app/requirements.txt
COPY ./main.py /usr/src/app/pose_tensorflow
WORKDIR /usr/src/app/pose_tensorflow
