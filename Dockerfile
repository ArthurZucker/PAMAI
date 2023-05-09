FROM pytorch/pytorch:latest

RUN \
  apt-get update && \
  apt-get -y install \
          git \
          gcc \
          python3.8-dev \
          python3-venv \
          python3.8-venv \
          libasound-dev \
          libportaudio2 \
          libportaudiocpp0 \
          ffmpeg \
          python-numpy \
          python-scipy \
          liblapack-dev \
          gfortran \
          portaudio19-dev \
          python-pyaudio \
          python3-pyaudio \
          build-essential

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN python -m pip install --upgrade pip setuptools wheel 
RUN python -m pip install pamai

WORKDIR /workspace/PAMAI