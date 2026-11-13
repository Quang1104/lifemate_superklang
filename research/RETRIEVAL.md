# Fixed lexical retrieval diagnostic

Dataset: TRAIN held-out 309 samples, seed 20261112, manifest `research/validation_split.json`.
Input: gold TRAIN transcripts (oracle), installed app/tool metadata, public catalog.
Checkpoint: none; deterministic lexical overlap, no learned parameters or label tuning.
Environment: local Windows Python CPU; mean ranking and all closure variants together 4.24 ms/sample.

| Candidate policy | Gold-tool micro recall | All gold tools present (nonempty samples) | Mean tools | Schema character reduction |
|---|---:|---:|---:|---:|
| All installed | 100.00% | 100.00% | 34.37 | 0.00% |
| Top 8 | 67.76% | 53.31% | 8.75 | 71.79% |
| Top 8 + x-ref reads | 80.00% | 67.94% | 10.44 | 67.87% |
| Top 16 | 80.15% | 68.29% | 16.48 | 48.18% |
| Top 16 + x-ref reads | 88.21% | 80.84% | 18.40 | 44.04% |

Recall denominator is 670 per-sample unique gold tool names. Clarification is always
included when available, so nominal top-k can contain k+1 tools. Closure selects
installed read tools whose declared return fields match x-ref namespaces, falling
back to all installed readers in that app when metadata lack a matching producer.
Size means serialized schema characters, not model tokens or measured prompt latency.

**Decision: reject hard pruning for the first baseline.** Even top16 with closure
loses 79 required tool occurrences. Use all installed schemas initially; if model
context requires pruning, investigate better retrieval with an explicit fallback.
This diagnostic establishes neither planner accuracy nor a competition score.

Reproduce:

```powershell
python -m research.retrieval_benchmark --train '<kit>/train/train.jsonl' --catalog '<kit>/train/train_tools_catalog.json' --manifest research/validation_split.json --output research/retrieval_report.json
python -m pytest tests_audit -q
```

Detailed per-sample aggregate measurements are in `research/retrieval_report.json`.
The ranking API accepts only transcript, installed tools, and catalog; labels are
read afterward for diagnostics. Reusing this held-out outcome to tune lexical
weights would require a separate untouched evaluation split.
