FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/mHRI/bin:$PATH

# downgrade build tools
RUN pip install pip==24.0 wheel==0.38.4 setuptools==65.5.0

COPY libs/ta-rware /app/libs/ta-rware
RUN pip install -e /app/libs/ta-rware

COPY libs/overcooked_ai /app/libs/overcooked_ai
RUN pip install -e /app/libs/overcooked_ai

COPY . /app
RUN pip install -e .
