# Dataset analysis

Generated from explicit permitted paths by `research.data_audit`; no environment traversal/state reads.

## `C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test`

Status: **READY**.

| Metric | TRAIN | PUBLIC_TEST |
|---|---:|---:|
| Samples | 1545 | 634 |
| Unique IDs | 1545 | 634 |
| Installed apps (union) | 45 | 35 |
| Catalog tools | 147 | 122 |
| Samples with history | 1018 | 419 |
| Referenced audio present | 1545 | 634 |
| Referenced audio missing | 0 | 0 |

### TRAIN task complexity

- multi_step_samples: **1065 / 1545 (68.93%)**.
- cross_app_samples: **715 / 1545 (46.28%)**.
- gold_empty_samples: **111 / 1545 (7.18%)**.
- missing_info_samples: **217 / 1545 (14.05%)**.
- Calls per sample: `{"0": 111, "1": 369, "2": 428, "3": 359, "4": 161, "5": 107, "6": 10}`.
- Labeled calls: **3541**; distinct labeled tools: **146**.

Most frequent labeled actions: `contacts.search` 234; `sys.ask_missing_info` 217; `smart_home.light_set` 126; `reminders.create` 115; `bill_payment.list_bills` 112; `smart_home.get_status` 107; `calendar.list_events` 105; `bill_payment.pay` 104; `chat_ott.list_chats` 94; `banking.transfer` 82; `sms.send` 79; `alarm.create` 78; `chat_ott.send_message` 69; `banking.get_balance` 62; `weather.get_forecast` 60.

### Catalog constraints

- train: operation types `{"read/declared": 62, "write/declared": 84, "meta/declared": 1}`.
- TRAIN labeled action types: `{"meta/declared": 217, "read/declared": 1599, "write/declared": 1725}`.
  - `x-ref`: 65 tools, 68 occurrences.
  - `x-one-of`: 6 tools, 6 occurrences.
  - `x-match`: 32 tools, 40 occurrences.
- public_test: operation types `{"read/declared": 52, "write/declared": 69, "meta/declared": 1}`.
  - `x-ref`: 58 tools, 61 occurrences.
  - `x-one-of`: 5 tools, 5 occurrences.
  - `x-match`: 29 tools, 38 occurrences.

### Generalization

Public-only: **34 tools / 11 apps**. `ecommerce`, `event_tickets`, `grocery`, `habit_tracker`, `pantry`, `parking`, `public_transit`, `recipes`, `stocks`, `train_bus_tickets`, `translator`.

Shared tools with changed definitions: **0**. Dynamic schema routing is required; memorizing TRAIN names cannot cover public-test tools.

### Vietnamese time and ASR risk

sáng mai: 129; tối nay: 57; hôm nay: 39; nửa tiếng: 34; tuần sau: 19; chiều mai: 15; hôm qua: 15; cuối tuần: 7; phút nữa: 3; tháng sau: 3; ngày mai: 2; ngày kia: 1. Counts mean samples containing each phrase; phrases may overlap.

Schema heuristics identify contacts, amounts, addresses/locations, times, and device names as sensitive fields. See detailed JSON for tool lists. English code-switching and entity-level ASR errors have **not** been measured. No speech quality is inferred from file presence.

## Method, validation split, and limitations

- Complete app/action distributions, schema extension paths/values, sensitive-field heuristics, and unseen tool names are in `research/data_audit.json`. No sample transcripts or state are copied there.
- Read/write/meta classification uses declared catalog `kind` where available. Fuzzy constraints preserve `x-match`; `x-ref` and `x-one-of` semantics require the official SDK.
- Multi-step means more than one labeled call; cross-app excludes `sys`. Gold-empty is a label category, not an independent semantic judgment.
- Public test uses only IDs, context/app metadata, audio paths, and catalog definitions. Public labels and transcripts are never analyzed.
- `research/validation_split.json`: fixed seed 20261112, 20% target; duplicate audio paths and normalized identical transcripts grouped transitively. Strata use primary action app, cross-app status, call-count bucket, clarification. Manifest records exact coverage gaps; rare groups cannot guarantee coverage. Renamed/paraphrased duplicates are not detected.
- `research/smoke_split.json`: 40 samples chosen from held-out IDs by the same stratification routine. Never use held-out IDs/groups as in-context examples or training rows.
- `research.metrics`: diagnostic exact-call F1 and WER/CER. `research.published_scorer`: published state-gated fuzzy-call metric with maximum one-to-one matching; reference implementation parity is unverified, especially tokenization. Neither synthetic tests nor the audit measure planner accuracy.

## Highest-value follow-up

1. Run all 40 held-out samples through the official ToolEnv; evaluate with separately replayed TRAIN gold hashes.
2. Compare one local planner on the exact same oracle transcripts and ASR predictions, with checkpoint/config and latency recorded.
3. Target real errors in entity references, temporal conversion, or clarification; test generalization on held-out app/tool groups before training.

```powershell
python -m research.data_audit --root '<actual-kit-root>' --output docs/DATASET_ANALYSIS.md --json-output research/data_audit.json
python -m research.validation_split --input '<actual-kit-root>/train/train.jsonl' --output research/validation_split.json --seed 20261112 --fraction 0.2
python -m pytest tests_audit -q
```
