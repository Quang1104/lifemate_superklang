# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

ENV PATH="/opt/lifemate/bin:${PATH}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=20261112
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/lifemate
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132

WORKDIR /code
COPY pyproject.toml ./
COPY docker/README.md ./README.md
COPY src ./src
RUN pip install --no-cache-dir '.[model]' 'jupyterlab>=4.4,<5' peft bitsandbytes
COPY --from=planner_weights . /models/planner/
COPY --from=asr_weights . /models/asr/
COPY --chmod=755 docker/predict.sh docker/start_jupyter.sh ./
COPY docker/predict_notebook.ipynb ./
COPY training_code ./training_code
COPY scripts/build_sft.py ./scripts/build_sft.py
COPY research/validation_split.json ./research/validation_split.json
COPY configs/colab_models.json ./configs/colab_models.json

ARG PLANNER_MODE=native
ENV LIFEMATE_PLANNER_MODE=${PLANNER_MODE}
CMD ["/code/predict.sh"]
