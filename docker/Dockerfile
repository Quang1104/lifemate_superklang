# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04 AS deps

ENV PATH="/opt/lifemate/bin:${PATH}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=20261112
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/lifemate
# pip cache lives in a BuildKit cache mount (never in the image) so a dropped connection resumes, not restarts.
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 --timeout 60 torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132

WORKDIR /code
COPY pyproject.toml ./
COPY docker/README.md ./README.md
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 --timeout 60 '.[model]' 'jupyterlab>=4.4,<5' peft bitsandbytes
# Everything above is independent of the adapter; `docker build --target deps` pre-builds it.
FROM deps
COPY configs/colab_models.json ./configs/colab_models.json
COPY training_code ./training_code
COPY docker/fetch_models.py ./fetch_models.py
# Pinned base planner + ASR are fetched at build time and the QLoRA adapter is merged in the same
# step, so neither the base checkpoint nor the adapter (a bind mount) ends up in an image layer.
ARG ASR_ROLE=asr
RUN --mount=type=bind,from=adapter,target=/tmp/adapter     HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -B fetch_models.py /tmp/adapter --asr-role "${ASR_ROLE}"
# Sentence embedder for LIFEMATE_EMBED_RETRIEVAL: its own layer, fetched only when the release sets the flag.
ARG LIFEMATE_EMBED_RETRIEVAL=
RUN if [ -n "${LIFEMATE_EMBED_RETRIEVAL}" ]; then HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -B fetch_models.py --embedder /models/embedder; fi
COPY --chmod=755 docker/predict.sh docker/start_jupyter.sh ./
COPY docker/predict_notebook.ipynb ./
COPY scripts/build_sft.py ./scripts/build_sft.py
COPY research/validation_split.json ./research/validation_split.json

ARG PLANNER_MODE=native
ENV LIFEMATE_PLANNER_MODE=${PLANNER_MODE}
# Retrieved TRAIN examples in the native prompt; planner v3 was trained with 2 and must run with 2.
ARG NATIVE_FEWSHOT_K=2
ENV LIFEMATE_NATIVE_FEWSHOT_K=${NATIVE_FEWSHOT_K}
# Feature flags of the chosen planner: it must run with the prompt it was trained with. Empty means off.
# docker/release_flags.py is the allowlist; `release.ps1 -Flags "NAME=1,..."` passes each as a build argument.
ARG LIFEMATE_ARG_NORMALIZE=
ARG LIFEMATE_ASR_FIX=
ARG LIFEMATE_ASR_PROMPT=
ARG LIFEMATE_CONTACT_RETRY=
ARG LIFEMATE_CONTEXT_V4=
ARG LIFEMATE_DATE_TABLE=
ARG LIFEMATE_FEWSHOT_GATE=
ARG LIFEMATE_PROMPT_V2=
ARG LIFEMATE_SMALL_FIXES=
ARG LIFEMATE_TIME_HINTS_V8=
ENV LIFEMATE_ARG_NORMALIZE=${LIFEMATE_ARG_NORMALIZE} \
    LIFEMATE_ASR_FIX=${LIFEMATE_ASR_FIX} \
    LIFEMATE_ASR_PROMPT=${LIFEMATE_ASR_PROMPT} \
    LIFEMATE_CONTACT_RETRY=${LIFEMATE_CONTACT_RETRY} \
    LIFEMATE_CONTEXT_V4=${LIFEMATE_CONTEXT_V4} \
    LIFEMATE_DATE_TABLE=${LIFEMATE_DATE_TABLE} \
    LIFEMATE_EMBED_RETRIEVAL=${LIFEMATE_EMBED_RETRIEVAL} \
    LIFEMATE_FEWSHOT_GATE=${LIFEMATE_FEWSHOT_GATE} \
    LIFEMATE_PROMPT_V2=${LIFEMATE_PROMPT_V2} \
    LIFEMATE_SMALL_FIXES=${LIFEMATE_SMALL_FIXES} \
    LIFEMATE_TIME_HINTS_V8=${LIFEMATE_TIME_HINTS_V8}
CMD ["/code/predict.sh"]
