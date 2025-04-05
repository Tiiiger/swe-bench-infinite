FROM testbed-base:latest

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
RUN echo "source /opt/miniforge/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate testbed" >> ~/.bashrc

# Git clone the repo
COPY github_setup.sh /github_setup.sh
RUN chmod +x /github_setup.sh
RUN /github_setup.sh

# Install pip requirements
COPY pip_install.sh /pip_install.sh
RUN chmod +x /pip_install.sh
RUN /pip_install.sh

# Execute custom install commands if present
# COPY install_commands.sh /install_commands.sh
# RUN if [ -f /install_commands.sh ]; then chmod +x /install_commands.sh && /install_commands.sh; fi

# change directory to the dir testbed
WORKDIR /testbed
