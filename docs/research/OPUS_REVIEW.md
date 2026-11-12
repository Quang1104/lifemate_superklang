# Independent review — Claude Opus 5.5

# Team B: independent ML review of the LifeMate Candidate C design

**Legend:** **[F]** = a fact you supplied. **[H]** = my hypothesis, not yet verified. None of the numbers below were measured.

## 1. Highest-risk correctness issues

1. **The hash gate makes precision on writes matter most.** [F] A state-hash mismatch scores the sample 0. [H] One extra, duplicate or partly failed write can therefore wipe out an otherwise correct sample. To guard against this:
   - Run `validate` before every `call`.
   - Never retry a write that has already run. If the feedback loop wants to retry, it should correct the next action, not repeat the last one.
   - Keep a write ledger keyed on (tool, canonical args).
2. **Treating `ok=true, done=false` as success.** [F] Tools can return this for actions that failed. The executor must treat it as not done. It must not report success, and it must not retry blindly.
3. **Empty-gold samples.** [F] If the gold answer is empty, any predicted call scores 0. You need a calibrated abstain path, measured separately on empty-gold items.
4. **Unknown whether read/lookup calls count.** [H, high risk] If `get_tool_calls` records lookups that gold doesn't contain, then gathering evidence lowers precision and might change the hash. Check this on TRAIN with the official ToolEnv before you design the retrieval-by-lookup strategy.
5. **Exact-match arguments.** [F] Non-fuzzy arguments must match exactly. Dates, times and timezones, relative expressions ("thứ sáu tuần sau"), numbers, enums and IDs should go through deterministic resolvers anchored on `context.current_time`/`timezone`. The LLM should not free-generate them.
6. **ID provenance.** [F] IDs must come from real evidence: tool results seen in this sample, or the provided context. Reject any ID that isn't in the evidence ledger, even if it looks plausible.
7. **History re-execution.** [F] Actions in the history are already applied. Re-issuing one creates a duplicate write and a hash mismatch.
8. **Misreading the schema.** [F] `x-one-of` needs at least one complete argument group and does not require exclusivity. `x-ref` names app collections. [F] Only tools in the installed list are allowed. A validator that enforces exclusivity, or that ignores installation, will reject or allow the wrong calls.
9. **Fuzzy fields and Vietnamese text.** [F] Fuzzy fields pass at token-F1 strictly above 0.6 after lowercase/NFC. [H] Losing diacritics in ASR, or emitting NFD text, could push short strings below that threshold. Normalize all output to NFC. Which fields are fuzzy has to come from the original scorer source; until then it stays unverified.
10. **Retrieval recall.** [H] If schema retrieval drops the gold tool, the planner cannot recover it. Tool schemas also compete for the 16,384-token context.
11. **Determinism and the 30 s budget.** [F] Inference must be offline, deterministic and finish in 30 s per sample. [H] Thinking-mode generation, multi-turn feedback and GPU nondeterminism all threaten this.
12. **Leakage when evaluating.** [H] Few-shot examples or prompt tuning drawn from the dev items will inflate scores. TRAIN may also contain near-duplicate templates, so random splits may overestimate.

## 2. First experiments (three at most)

**Common split, frozen before any experiment:**
- Divide TRAIN (1,545) [F] into dev-train / dev-eval / lockbox at roughly 70/15/15.
- Stratify by: empty-gold vs not, history present, number of gold calls (0 / 1 / ≥2), and primary app.
- Group near-duplicate transcripts (normalized text similarity) into the same partition.
- Commit the manifest of question_ids plus its hash.
- Open the lockbox once, for the final candidate only.
- All scores come from a local re-implementation of the scoring rules, labelled **"local proxy, unverified"**. It uses one fresh official ToolEnv per sample [F].
- [H] How to get the gold hash: use it from the labels if they provide it. Otherwise, replay the gold calls in a separate oracle-analysis process, never in the inference code. Confirm this is allowed first.

**E1: Planner ceiling on oracle transcripts.**
- *Hypothesis:* Candidate C (retrieval → Qwen3-4B, non-thinking, greedy → validation/execution/feedback) is the main bottleneck, separate from ASR.
- *Arms:*
  - (a) full pipeline;
  - (b) gold tool names injected (isolates retrieval);
  - (c) feedback loop disabled.
- *Measure:*
  - proxy score;
  - retrieval recall@k;
  - hash-pass rate;
  - false-action rate on empty-gold items;
  - schema-rejection rate.
- *Accept:* proxy score on oracle transcripts ≥ 0.85. This is a decision threshold, not a prediction; it leaves headroom under the internal 0.80 target for ASR loss.
- *Stop:* if (a) is below 0.60 after two prompt/resolver iterations, or (b) − (a) > 0.10. Then fix retrieval or resolvers before touching ASR or model choice.

**E2: Paired ASR degradation.**
- *Hypothesis:* the drop from oracle to Whisper-small is concentrated in entity, number and time slots, not in tool choice.
- *Measure:*
  - syllable-level WER/CER (Vietnamese is syllable-delimited);
  - slot-level error on the gold argument values;
  - the paired score delta (see §3).
- *Accept:* score drop ≤ 0.05.
- *Stop:* if the drop is > 0.10, prioritize ASR options next (decoding settings, a larger ASR within the 6B aggregate, transcript normalization) ahead of planner tuning.

**E3: Write safety and abstention.**
- *Hypothesis:* gating writes on validation plus a confidence/abstain rule raises the score by cutting hash failures, at an acceptable recall cost.
- *Arms:* no gate / validate-only / validate + abstain threshold. Sweep the threshold on dev-train only and apply it once to dev-eval.
- *Accept:* better mean score on dev-eval, and false-action rate on empty-gold items ≤ 5%.
- *Stop:* if no threshold improves dev-eval, keep the simplest gate.

## 3. Pairing oracle and predicted-ASR runs

- **What stays fixed:** the same question_ids, prompt, retrieval index, model weights, decoding config (greedy, fixed seeds) and code commit. The only change is the transcript source. Each run gets one fresh ToolEnv per sample.
- **What to log per item:** transcript, retrieved tools, calls, validation results, tool returns, hash pass, item score and latency, all keyed by question_id.
- **Analysis:**
  - Per-item difference Δ = score_ASR − score_oracle, with a paired bootstrap 95% CI.
  - McNemar's test on hash-pass/full-match.
  - A 2×2 attribution table:
    - oracle pass, ASR fail → ASR-attributable;
    - oracle fail, ASR fail → planner;
    - oracle fail, ASR pass → noise or luck. Investigate these, since they indicate nondeterminism or brittleness.
- **Error localization:** align the ASR transcript with the gold transcript and check whether each failed argument value is corrupted in the ASR text. That separates "ASR lost the entity" from "planner mis-copied it".
- **Determinism check:** run the oracle arm twice. Any per-item difference is a defect to fix before comparing anything.

## 4. Error taxonomy

Tag each failure with its first cause.

| Code | Class | Examples |
|---|---|---|
| A1/A2 | Abstention | Acted on empty gold / wrongly abstained |
| R1 | Retrieval | Gold tool not among candidates |
| S1/S2 | Tool selection | Wrong tool / tool not installed |
| G1/G2/G3 | Grounding | ID not in ledger; stale or wrong-entity ID; history action re-executed |
| X1 | Argument: exact | Date/time/timezone, number, enum or ID mismatch |
| X2 | Argument: fuzzy | Token-F1 ≤ 0.6 (diacritics, paraphrase, NFC) |
| X3 | Argument: structural | Missing required field, `x-one-of` unmet, bad `x-ref` |
| E1/E2/E3 | Execution | Schema rejection; `done=false` mishandled; duplicate or retried write |
| E4 | Count | Extra or missing calls in multi-call samples |
| H1 | State | Calls matched but hash mismatches (lookup side effects, ordering) |
| ASR1–4 | Speech | Entity/name error; number/time error; diacritic loss; code-switched app/English term |
| B1/B2/B3 | Budget | Timeout; context overflow; nondeterminism |

## 5. Model-selection caveats and runtime recommendation

**Caveats:**
- **Parameter accounting.** [F] Strict limits: LLM < 5B, all models < 6B. Qwen3.5-4B's total, including any vision/audio towers that are loaded, needs an exact audit from the weight files (sum of tensor sizes), not the advertised name. [H] Qwen3-4B, being text-only, is the lower-risk choice for eligibility. Adding Whisper-small (244M [F]) leaves margin, but confirm the exact LLM count is under 5B.
- **Release date and licence.** [F] Both must be checked against the 2026-10-26 cutoff and the permitted-use rules, from archived model cards, for every model and any external data. Record this in docs.
- **Vietnamese quality.** Unknown for both models. Only E1/E2 should decide it; don't rely on reputation.
- **Thinking mode.** [H] Long reasoning traces threaten the 30 s budget and determinism. Default to non-thinking mode with a hard `max_new_tokens`.
- **Whisper.** [H] Force `language="vi"` and greedy decoding, and watch for hallucination on silence and on audio over 30 s.
- **Tokenizer cost.** Vietnamese tokens per schema affects the 16k context. Measure it rather than assume.

**Runtime recommendation:**
- **Local machine:** GTX1650 4GB with a CPU-only torch [F] cannot give meaningful 4B latency. Use it for plumbing, the validator, resolvers, the scorer proxy and ToolEnv tests, with small or mocked models. Mark all latency and accuracy **NOT MEASURED**.
- **Target design [H]:** HF Transformers, bf16, single process, greedy, `torch.use_deterministic_algorithms` where feasible. ASR and LLM both stay resident on the A30. Validate structured output, with one repair turn. Feedback is bounded to about 2 rounds, under a per-stage time budget with a hard deadline that abstains rather than over-runs.
- **Before choosing a model:** get one measured end-to-end latency run on an A30-class GPU.

---
Not checked: original scorer/API sources (so the read-call counting, fuzzy-field list and gold-hash availability are all open), parameter counts, licences, release dates.
Risk: every threshold above is a decision rule I chose, not evidence. Treat the scores from the local scorer re-implementation as unverified until it is checked against the official scorer.
