# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV PATH="/opt/lifemate/bin:${PATH}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/lifemate
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132

WORKDIR /code
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[model]' 'jupyterlab>=4.4,<5'
COPY --from=planner_weights . /models/planner/
COPY --from=asr_weights . /models/asr/
COPY --chmod=755 docker/predict.sh docker/start_jupyter.sh ./
COPY docker/predict_notebook.ipynb ./
COPY training_code ./training_code
COPY configs/colab_models.json ./configs/colab_models.json

ARG PLANNER_MODE=compiled
ENV LIFEMATE_PLANNER_MODE=${PLANNER_MODE}
CMD ["/code/predict.sh"]
