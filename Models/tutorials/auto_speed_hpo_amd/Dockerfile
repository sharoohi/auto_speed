FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    build-essential \
    cmake \
    libjpeg-dev \
    libpng-dev \
    vim \
    time \
    unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/auto_speed

# Copy in the whole Models dir and requirements.txt so we have the entire auto speed codebase
COPY . /app/auto_speed

# Install vision pilot dependencies
RUN cd /app/auto_speed && pip install -r requirements.txt

RUN pip install --upgrade numpy scikit-image

# Install Optuna for running HPO experiments
RUN pip install optuna

ENV PYTHONPATH=/app/auto_speed
