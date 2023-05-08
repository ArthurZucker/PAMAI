FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get install git gcc -y
RUN apt-get install python3.8-dev python3-venv python3.8-venv -y
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 ffmpeg -y
RUN apt-get install python-numpy python-scipy liblapack-dev gfortran -y
RUN apt-get install portaudio19-dev python-pyaudio python3-pyaudio -y

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get install build-essential -y

# RUN git clone https://github.com/ArthurZucker/PAMAI.git
RUN python -m pip install --upgrade pip setuptools wheel 
RUN python -m pip install pamai

WORKDIR /workspace/PAMAI