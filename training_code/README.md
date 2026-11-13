# Training reproducibility

No fine-tuning has been performed. No trained checkpoint is claimed.

Proposed first baseline uses frozen Qwen3-4B + Whisper-small revisions documented in docs/ARCHITECTURE.md. Weights must be provisioned and measured before release; currently absent. Only inference weights belong in the image.

If training becomes justified by measured errors, record source/revision/license, seed20261112, exact train IDs from research/validation_split.json, preprocessing, optimizer/hyperparameters and output checkpoint SHA256. Exclude held-out IDs, PUBLIC/PRIVATE labels and hidden state. Keep original competition data private per kit README. Do not publish the dataset even if a generic Docker example suggests hosting training data; organizer-specific redistribution terms prevail.
