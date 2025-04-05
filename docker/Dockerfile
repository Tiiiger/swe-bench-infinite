FROM --platform=linux/x86_64 ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y \
  wget \
  git \
  build-essential \
  libffi-dev \
  libtiff-dev \
  python3 \
  python3-pip \
  python-is-python3 \
  jq \
  curl \
  locales \
  locales-all \
  tzdata

COPY apt_install.sh /apt_install.sh
RUN chmod +x /apt_install.sh
RUN /apt_install.sh

# Download and install conda
RUN wget 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh' -O miniforge.sh \
    && bash miniforge.sh -b -p /opt/miniforge

# Add conda to PATH
ENV PATH=/opt/miniforge/bin:$PATH
RUN mamba init --all
RUN conda config --append channels conda-forge

# Setup conda/mamba environment
COPY conda_setup.sh /conda_setup.sh
RUN chmod +x /conda_setup.sh
RUN /conda_setup.sh
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/miniforge/etc/profile.d/conda.sh && conda activate testbed" >> ~/.bashrc

# Git clone the repo
COPY github_setup.sh /github_setup.sh
RUN chmod +x /github_setup.sh
RUN /github_setup.sh

# Run the script
COPY requirements_collection.txt /requirements_collection.txt
# RUN pip install -r requirements_collection.txt

# change directory to the dir testbed
RUN cd testbed

RUN adduser --disabled-password --gecos 'dog' nonroot