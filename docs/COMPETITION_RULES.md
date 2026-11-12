# Verified rules — 2026-11-12 (Asia/Saigon)

Sources: [official portal](https://challenge.zalo.ai/portal/lifemate), its public config API (`name=rule`, `evaluationMetric`, `description`, `submissionGuidance`, `title=1`), and the user's original kit v3.2.0 built 2026-10-24. JSON source snapshots are in `docs/sources/`; kit README/source is at `../datalife/train+public_test`. The Data tab requires portal authentication; the original kit was read instead. No third-party rule summary is treated as authority.

| Constraint | Verified requirement |
|---|---|
| Parameters | LLM strictly <5,000,000,000; all ML components together strictly <6,000,000,000. Quantization does not reduce parameter counts. |
| Context | <=16,384 tokens; cap input plus generated output, including history/schema/tool feedback. |
| Runtime | <=30 seconds per sample; timeout/execution failure scores zero. Include audio reading, ASR and planning; report model load separately. |
| Network | Fully offline inference, no external services. Models/tokenizers must already be in the image. |
| Eligibility | Public data/open models released before 2026-10-26, compatible licenses; pin eligible artifact revisions. Synthetic development data allowed, no external generation services at inference. |
| Hidden data | Never label public/private test to train/tune. Never directly read/copy/parse files under environment directories. |
| Environment | Exactly one ToolEnv per sample; inference API restricted to validate, call, get_tool_calls, get_state_hash. Never get_state/states/Executor/Simulator/clone. |
| History | Already applied to initial state. Interpret conversation, never replay actions. |
| Output | Preserve all official calls, including rejected calls; hash comes exclusively from ToolEnv. |
| Reproducibility | Fix Python/NumPy/PyTorch seeds and deterministic decoding. Nonmatching reruns trigger five-run average by organizer. |
| Deadline | Complete solution due 2026-11-16 23:59 GMT+7. |

## Unresolved official-source conflicts — release blockers

- Rule tab: NVIDIA A30, Xeon Gold 6442Y 16 cores, 64GB RAM. Docker PDF p4: RTX5090 32GB, Xeon Gold6459C 16 cores, 64GB RAM. Do not claim either is conclusively the final machine. Benchmark on the confirmed target before release.
- Docker PDF p4 allows CUDA <=13.2 (explicitly 12.x); p7 says CUDA13.2 mandatory. Scaffold uses13.2, but needs organizer clarification and driver compatibility testing on the actual machine.
- Submission guidance says private-test release 09:00 Nov20; public homepage says11:00. This does not change Nov16 solution deadline.
- Native model context versus configured inference cap interpretation is not explicit. Cap runtime to16384 and document native capacity; seek confirmation if necessary.

## Published score

Read formula image directly (`docs/sources/scoring_formula.png`). Empty gold scores1 iff submitted calls are empty. For nonempty gold: hash mismatch or no matched calls =>0; otherwise P=M/(M+U), R=M/|G|, score=2PR/(P+R). Overall portal score=100*mean(sample score), so project target0.80 corresponds to80/100.

Matching is one-to-one, order-independent, same tool name, all nonfuzzy arguments identical, each fuzzy argument token-F1 strictly >0.6 after lowercase and Unicode NFC. Unmatched submitted calls, including rejected schema calls, count in U. Missing-info is scored normally. Tie-break first ignores state hash and uses call F1, then lower mean latency.

Organizer's executable scorer was not included in the recovered allowlisted kit. Local implementation follows the text with whitespace tokenization and maximum bipartite matching; exact tokenizer/tie-breaking semantics remain to cross-check. README notes free-text and live information apps are removed from canonical hash by official code. Do not reimplement hash for submissions.

## Docker contract

[Official PDF](https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf), downloaded and visually checked pp4/7:

- `/code/predict.sh`, input `/data/private_test/`, output `/result/submission.jsonl`; create output directory.
- Import `zaic_tools` from mounted input split, never the training copy bundled in an image.
- Output `id`, `tool_calls`, `state_hash`, and `time` in milliseconds (PDF examples); kit's minimum schema has the first three.
- Include `predict_notebook.ipynb`, fixed-seed/model-load/input/inference cells, output `/result/jupyter_submission.jsonl` with matching predictions and `time`.
- Include JupyterLab, `/code/start_jupyter.sh` (port9777), README and `training_code/` with reproducibility information. No unnecessary base training weights in image.
- Test offline clean GPU container; export image archive, MD5 checksum, accessible download link. No upload/submission has been performed.

User's stricter no-inspection rule remains active even where kit README permits TRAIN get_state for debugging.
