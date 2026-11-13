# Training reproducibility

No fine-tuning has been performed. No trained checkpoint is claimed.

The Colab benchmark uses frozen Qwen3.5-4B and PhoWhisper-medium or the pinned Qwen3-4B-Instruct-2507 fallback. Checkpoint revisions are in configs/colab_models.json. The inference entry point sets seed 20261112. Only inference weights belong in the image.

If training becomes justified by measured errors, record source, revision, license, seed, exact development IDs from research/validation_split.json, preprocessing, optimizer, hyperparameters and output checkpoint SHA256. Use ASR predictions for training inputs. Keep gold TRAIN transcripts for oracle analysis. Exclude held-out IDs, PUBLIC/PRIVATE labels and hidden state.
