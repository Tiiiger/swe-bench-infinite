FROM testbed-base:latest

COPY apt_install.sh /apt_install.sh
RUN chmod +x /apt_install.sh
RUN /apt_install.sh

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
