# LifeMate container inference

Run `/code/predict.sh` with the released private split mounted at `/data/private_test/` and a writable `/result/` mount. The script writes `/result/submission.jsonl` and creates its parent directory. Every private record contains `id`, `tool_calls`, `state_hash`, and integer `time` in milliseconds. Official calls and hashes come from the mounted split's ToolEnv API.

The image uses local weights in `/models/planner/` and `/models/asr/`. Build contexts named `planner_weights` and `asr_weights` must contain only the chosen inference artifacts; use the merged planner checkpoint for a fine-tuned model. Do not include an extra base planner or training checkpoints. Runtime model loading uses local files and offline mode; no model download is performed during inference. The image build itself installs runtime packages and therefore requires its build dependencies to be available.

Planner mode defaults to `native`; `LIFEMATE_PLANNER_MODE` selects the planner mode and `LIFEMATE_PLANNER_MODEL_DIR` selects a local planner directory. Relative-date prompt tables remain opt-in through `LIFEMATE_DATE_TABLE=1`. Keep all inference environment settings identical when comparing script and notebook predictions.

Run `/code/start_jupyter.sh` to start JupyterLab on port `9777`, then open `/code/predict_notebook.ipynb`. Its setup cells identify local input/model paths and the fixed seed; the inference cell invokes the same script and loads the models once in that process. It writes `/result/jupyter_submission.jsonl` and checks that calls and hashes match `/result/submission.jsonl` when that file is available. Runtime measurements can differ between runs.

Python hashing, Python random, NumPy, and PyTorch use seed `20261112`; decoding is greedy with one beam and cuDNN benchmarking disabled. Model loading is reported separately; sample timing starts before audio loading and includes ASR, planning, and the official output export. Actual latency and repeatability on the final GPU are **NOT MEASURED** by the CPU submission audit.

Training implementation is included in `/code/training_code/`, and model revisions are recorded in `/code/configs/colab_models.json`. Reproducibility remains a release TODO: the training README is outdated, the SFT builder and exact split manifest are not included in the image, and the final adapter/merged artifact and exact resume procedure must be documented. Review these before submission.

The official sources conflict on target GPU and CUDA requirements. The Dockerfile currently selects CUDA 13.2; organizer confirmation and an offline clean-container run on the confirmed GPU remain required. Container build, notebook/script parity on real models, image contents/size, archive export, MD5 checksum, and accessible download link are **NOT MEASURED** in this audit.
