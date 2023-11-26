FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir data
WORKDIR /data

ARG DEBIAN_FRONTEND=noninteractive
RUN pip uninstall torch -y
RUN pip install torch==2.0.1 -f https://download.pytorch.org/whl/cu118
COPY builder/setup.sh /setup.sh
RUN chmod +x /setup.sh && \
    /setup.sh && \
    rm /setup.sh

# Install Python dependencies (Worker Template)
RUN pip install fastapi==0.99.1

COPY builder/requirements.txt /requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

RUN pip install safetensors==0.3.1 sentencepiece \
        git+https://github.com/winglian/runpod-python.git@fix-generator-check ninja==1.11.1
RUN git clone https://github.com/turboderp/exllama
RUN pip install -r exllama/requirements.txt
RUN pip install git+https://github.com/runpod/runpod-python@a1#egg=runpod --compile

COPY handler.py /data/handler.py
COPY __init.py__ /data/__init__.py

ENV PYTHONPATH=/data/exllama
ENV MODEL_REPO=""
ENV PROMPT_PREFIX=""
ENV PROMPT_SUFFIX=""
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

CMD [ "python", "-m", "handler" ]