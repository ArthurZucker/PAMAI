FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get install git gcc -y
RUN apt-get install python3.8-dev python3-venv python3.8-venv -y
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 ffmpeg -y
RUN apt-get install python-numpy python-scipy liblapack-dev gfortran -y

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# RUN pip install pyaudio>=0.2.8
RUN apt-get install portaudio19-dev python-pyaudio python3-pyaudio -y
RUN pip install PyAudio
RUN pip install pybind11
RUN pip install Cython
RUN pip install --upgrade pip; apk add build-base; pip install numpy
RUN pip install certifi
RUN pip install matplotlib
RUN pip install pythran
RUN pip install scipy
RUN pip install wheel
RUN git clone https://github.com/TUT-ARG/sed_vis.git
RUN pip install -e sed_vis

RUN git clone https://github.com/ArthurZucker/PAMAI.git
RUN rm /workspace/PAMAI/requirements.txt
COPY requirements.txt /workspace/PAMAI/requirements.txt
RUN pip install -r /workspace/PAMAI/requirements.txt


WORKDIR /workspace/PAMAI

# RUN python main.py