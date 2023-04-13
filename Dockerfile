FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get install git gcc -y
RUN apt-get install python3-dev -y
RUN apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg -y

RUN pip install pyaudio --user
RUN apt-get install python-pyaudio -y

RUN : \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN ls

RUN git clone https://github.com/ArthurZucker/PAMAI.git
RUN pip install -r /workspace/PAMAI/requirements.txt
RUN ls

WORKDIR /workspace/PAMAI

