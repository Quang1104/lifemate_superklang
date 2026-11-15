/** CPU-only TRAIN error budget; explicit file allowlist, no ToolEnv or model runs.
 * node research/error_budget.mjs --check
 * After restoring staging permissions: node research/error_budget.mjs --write
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const stage="C:/Users/ADMIN/AppData/Local/Temp/lifemate-errbudget-staging";
const kit="C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test";
const smokeRoot='outputs/colab/smoke_ab_drive/smoke_ab';
const pubRoot='outputs/colab/public_drive/public';
const names=["base","LIFEMATE_ARG_NORMALIZE","LIFEMATE_DATE_TABLE","LIFEMATE_PROMPT_V2","oracle_transcripts","v3","v3_asrft","w_asrft","w_fewshot2"];
const annotate={
  "train_0087": [
    "entity_ids",
    [
      "wrong_args:entity_ids",
      "extra_call"
    ],
    "Sai tài khoản nguồn; thêm open_savings"
  ],
  "train_0131": [
    "asked_when_should_not",
    [
      "asked_when_should_not",
      "wrong_args:names_contacts",
      "wrong_missing_tool"
    ],
    "Anh Tuấn thay Tuấn; hỏi thay gửi SMS"
  ],
  "train_0169": [
    "extra_call",
    [
      "extra_call",
      "wrong_args:enums"
    ],
    "Thử ví thiếu tiền và đọc thêm số dư ngân hàng"
  ],
  "train_0188": [
    "dates_times",
    [
      "wrong_args:dates_times"
    ],
    "13:19 thay 13:55"
  ],
  "train_0229": [
    "entity_ids",
    [
      "wrong_args:entity_ids"
    ],
    "acc_02 thay acc_01"
  ],
  "train_0246": [
    "numbers",
    [
      "wrong_args:numbers"
    ],
    "Thêm floor=2 vào get_status"
  ],
  "train_0280": [
    "entity_ids",
    [
      "wrong_args:entity_ids"
    ],
    "light_bedroom2 thay light_bedroom"
  ],
  "train_0286": [
    "extra_call",
    [
      "extra_call",
      "wrong_args:enums"
    ],
    "Thử ví trước và đọc thêm số dư"
  ],
  "train_0578": [
    "dates_times",
    [
      "wrong_args:dates_times",
      "wrong_args:free_text"
    ],
    "Thiếu time buổi tập; magie thành canxi"
  ],
  "train_0611": [
    "dates_times",
    [
      "wrong_args:dates_times",
      "extra_call"
    ],
    "03/10 thay 04/10; thêm list_events"
  ],
  "train_0665": [
    "names_contacts",
    [
      "wrong_args:names_contacts"
    ],
    "Hạnh thay Cô Hạnh (query khớp chính xác)"
  ],
  "train_0675": [
    "names_contacts",
    [
      "wrong_args:names_contacts",
      "wrong_args:free_text"
    ],
    "Sai người nhận email và nội dung"
  ],
  "train_0695": [
    "wrong_missing_tool",
    [
      "wrong_missing_tool",
      "asked_when_should_not",
      "wrong_args:names_contacts"
    ],
    "contacts.search thay list_chats; hỏi thay gửi Zalo"
  ],
  "train_0707": [
    "dates_times",
    [
      "wrong_args:dates_times",
      "wrong_missing_tool"
    ],
    "20/10 thay 22/10; thiếu update_event"
  ],
  "train_1015": [
    "extra_call",
    [
      "extra_call"
    ],
    "Thêm recent_calls"
  ],
  "train_1095": [
    "wrong_missing_tool",
    [
      "wrong_missing_tool",
      "extra_call"
    ],
    "Gửi ba tin thay đọc chat rồi xoá rule"
  ],
  "train_1161": [
    "extra_call",
    [
      "extra_call"
    ],
    "Gold rỗng; thêm contacts.search"
  ],
  "train_1202": [
    "dates_times",
    [
      "wrong_args:dates_times"
    ],
    "19:00 thay 17:30; thêm end (title vẫn qua fuzzy)"
  ],
  "train_1296": [
    "should_have_asked",
    [
      "should_have_asked",
      "wrong_missing_tool"
    ],
    "Tự chọn máy thay hỏi appliance_id"
  ],
  "train_1360": [
    "extra_call",
    [
      "extra_call"
    ],
    "Gold rỗng; vẫn redeem"
  ],
  "train_1582": [
    "extra_call",
    [
      "extra_call"
    ],
    "Gold rỗng; ba lời gọi ngân hàng"
  ],
  "train_1598": [
    "should_have_asked",
    [
      "should_have_asked",
      "wrong_missing_tool"
    ],
    "Tự chọn kỳ hạn thay hỏi term_months"
  ],
  "train_1645": [
    "stopped_after_read",
    [
      "stopped_after_read",
      "wrong_missing_tool"
    ],
    "Chỉ get_current; thiếu forecast và gửi vị trí"
  ]
};
const labels={"dates_times":"Ngày/giờ","extra_call":"Gọi thừa","entity_ids":"ID thực thể","wrong_missing_tool":"Sai/thiếu tool","should_have_asked":"Đáng lẽ hỏi thiếu","names_contacts":"Tên/liên hệ","asked_when_should_not":"Hỏi không cần","stopped_after_read":"Dừng sau đọc","numbers":"Số/tham số số"};
const fixes={"dates_times":"Dữ liệu TRAIN tương đối/tuần/ngày; prompt giữ mốc và duration, điền time đã nêu.","extra_call":"Ví dụ điều kiện sai → không gọi; kiểm điều kiện trước ghi, bỏ đọc không phục vụ yêu cầu.","entity_ids":"Ví dụ chọn ID theo kết quả đọc và điều kiện số dư; không lấy ID chỉ theo vị trí.","wrong_missing_tool":"Ví dụ TRAIN đọc→rẽ nhánh→ghi đúng app; phân biệt đọc Zalo và gửi tin.","should_have_asked":"Ví dụ tương phản thiếu/đủ appliance_id, term_months; hỏi đúng trường, không tự điền."};
const recordedCommands=[
  {
    "tool": "exec_command",
    "cmd": "Get-Content -LiteralPath 'C:/Users/ADMIN/AppData/Local/Temp/lifemate-errbudget-staging/BRIEF.md'",
    "result": "FAILED before execution: codex-windows-sandbox-setup.exe not found"
  },
  {
    "tool": "Node child_process.execFile",
    "cmd": "git",
    "args": [
      "branch",
      "--show-current"
    ],
    "result": "colab-smoke"
  },
  {
    "tool": "Node child_process.execFile",
    "cmd": "python",
    "args": [
      "--version"
    ],
    "result": "Python 3.12.6"
  },
  {
    "tool": "Node child_process.execFile",
    "cmd": "git",
    "args": [
      "ls-files",
      "outputs",
      "docs/ERROR_BUDGET.md",
      "research/error_budget.py"
    ],
    "result": "empty stdout"
  },
  {
    "cmd": "python",
    "args": [
      "-B",
      "-c",
      "import json, math\nfrom pathlib import Path\nfrom research.data_audit import extract_tools\nfrom research.published_scorer import score_sample\nroot=Path.cwd()\nkit=Path(r'C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test')\ntools=extract_tools(json.loads((kit/'train/train_tools_catalog.json').read_text(encoding='utf-8-sig')))\ngold={r['id']:r for r in map(json.loads,(root/'outputs/gold_smoke.jsonl').read_text(encoding='utf-8-sig').splitlines())}\nmanifest=json.loads((root/'research/smoke_split.json').read_text(encoding='utf-8-sig'))['validation_ids']\nbase=root/'outputs/colab/smoke_ab_drive/smoke_ab'\nnames=['base','LIFEMATE_ARG_NORMALIZE','LIFEMATE_DATE_TABLE','LIFEMATE_PROMPT_V2','oracle_transcripts','v3','v3_asrft','w_asrft','w_fewshot2']\nfor name in names:\n d=json.loads((base/name/'native.scored.json').read_text(encoding='utf-8-sig'))\n rows=list(map(json.loads,(base/name/'native.jsonl').read_text(encoding='utf-8-sig').splitlines()))\n pred={r['id']:r for r in rows}\n assert len(rows)==len(pred)==40 and set(pred)==set(manifest)\n assert {s['id'] for s in d['samples']}==set(manifest)\n for s in d['samples']:\n  assert s['gold']==gold[s['id']]['tool_calls']\n  assert s['predicted']==pred[s['id']]['tool_calls']\n  assert s['state_match']==(gold[s['id']]['state_hash']==pred[s['id']]['state_hash'])\n  actual=score_sample(gold[s['id']],pred[s['id']],tools)['score']\n  assert math.isclose(actual,s['score'],abs_tol=1e-12),(name,s['id'],actual,s['score'])\n assert math.isclose(100*sum(s['score'] for s in d['samples'])/40,d['local_proxy_0_to_100'],abs_tol=1e-10)\n print(name, 'PASS',len(rows),format(d['local_proxy_0_to_100'],'.9f'))\nprint('PASS: 360 saved sample scores reproduced; real saved hashes only; no ToolEnv created')"
    ],
    "code": 0,
    "stdout": "base PASS 40 24.832251082\r\nLIFEMATE_ARG_NORMALIZE PASS 40 24.832251082\r\nLIFEMATE_DATE_TABLE PASS 40 25.059523810\r\nLIFEMATE_PROMPT_V2 PASS 40 24.392857143\r\noracle_transcripts PASS 40 49.916666667\r\nv3 PASS 40 34.583333333\r\nv3_asrft PASS 40 51.726190476\r\nw_asrft PASS 40 44.767857143\r\nw_fewshot2 PASS 40 30.416666667\r\nPASS: 360 saved sample scores reproduced; real saved hashes only; no ToolEnv created\r\n",
    "stderr": ""
  },
  {
    "cmd": "node",
    "args": [
      "research/error_budget.mjs",
      "--check"
    ],
    "code": 0,
    "result": "PASS: 360 sample scores, ID coverage, TRAIN gold agreement, budget sum, public uniqueness, 118-line report"
  },
  {
    "cmd": "node",
    "args": [
      "research/error_budget.mjs",
      "--write"
    ],
    "code": 1,
    "stderr": "STAGING WRITE FAILED: PROGRESS.md: EPERM; REPORT.md: EPERM\n",
    "result": "Repository document written; both staging files failed EPERM"
  }
];
const recordedProgress=[
  {
    "time": "2026-11-15T16:56:49.422Z",
    "step": "Read brief, mandatory docs and rules; inventory explicit output directories",
    "result": "Branch colab-smoke; nine smoke runs plus two older smoke40 runs; three public prediction runs",
    "next": "Review failures"
  },
  {
    "time": "2026-11-15T16:56:49.422Z",
    "step": "Review all 40 v3_asrft samples and compute classes/public statistics",
    "result": "23 nonperfect; 48.273810 points lost; causal ASR and validation309 NOT MEASURED",
    "next": "Verify scores"
  },
  {
    "time": "2026-11-15T16:56:49.422Z",
    "step": "Reproduce saved scores with existing local proxy",
    "result": "360/360 scores PASS; no ToolEnv/replay; raw exports unchanged",
    "next": "Write and verify deliverables"
  },
  {
    "time": "2026-11-15T16:59:31.121Z",
    "step": "Save and validate repository deliverables",
    "result": "ERROR_BUDGET.md 118 lines; research/error_budget.mjs --check PASS",
    "next": "Write staging deliverables"
  },
  {
    "time": "2026-11-15T16:59:31.121Z",
    "step": "Attempt staging delivery with normal script command",
    "result": "PROGRESS.md and REPORT.md both EPERM; analysis and report generator preserved",
    "next": "Restore staging write access, then rerun --write"
  }
];
const verifyCode="import json, math\nfrom pathlib import Path\nfrom research.data_audit import extract_tools\nfrom research.published_scorer import score_sample\nroot=Path.cwd()\nkit=Path(r'C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test')\ntools=extract_tools(json.loads((kit/'train/train_tools_catalog.json').read_text(encoding='utf-8-sig')))\ngold={r['id']:r for r in map(json.loads,(root/'outputs/gold_smoke.jsonl').read_text(encoding='utf-8-sig').splitlines())}\nmanifest=json.loads((root/'research/smoke_split.json').read_text(encoding='utf-8-sig'))['validation_ids']\nbase=root/'outputs/colab/smoke_ab_drive/smoke_ab'\nnames=['base','LIFEMATE_ARG_NORMALIZE','LIFEMATE_DATE_TABLE','LIFEMATE_PROMPT_V2','oracle_transcripts','v3','v3_asrft','w_asrft','w_fewshot2']\nfor name in names:\n d=json.loads((base/name/'native.scored.json').read_text(encoding='utf-8-sig'))\n rows=list(map(json.loads,(base/name/'native.jsonl').read_text(encoding='utf-8-sig').splitlines()))\n pred={r['id']:r for r in rows}\n assert len(rows)==len(pred)==40 and set(pred)==set(manifest)\n assert {s['id'] for s in d['samples']}==set(manifest)\n for s in d['samples']:\n  assert s['gold']==gold[s['id']]['tool_calls']\n  assert s['predicted']==pred[s['id']]['tool_calls']\n  assert s['state_match']==(gold[s['id']]['state_hash']==pred[s['id']]['state_hash'])\n  actual=score_sample(gold[s['id']],pred[s['id']],tools)['score']\n  assert math.isclose(actual,s['score'],abs_tol=1e-12),(name,s['id'],actual,s['score'])\n assert math.isclose(100*sum(s['score'] for s in d['samples'])/40,d['local_proxy_0_to_100'],abs_tol=1e-10)\n print(name, 'PASS',len(rows),format(d['local_proxy_0_to_100'],'.9f'))\nprint('PASS: 360 saved sample scores reproduced; real saved hashes only; no ToolEnv created')";
const stable=v=>JSON.stringify(v===undefined?'__missing__':v,(_,x)=>x&&typeof x==='object'&&!Array.isArray(x)?Object.fromEntries(Object.entries(x).sort(([a],[b])=>a.localeCompare(b))):x);
const stats=(rows,cat)=>{var z={samples:rows.length,calls:0,empty:0,ask:0,last_read:0,read_only:0,duplicate_samples:0,unknown_tool_calls:0,arg_kinds:{},enums:0,bad_enums:0,numbers:0,bad_numbers:0,date_values:0,bad_date_shapes:0};for(var r of rows){var cs=r.tool_calls;z.calls+=cs.length;z.empty+=cs.length===0;z.ask+=cs.some(c=>c.name==='sys.ask_missing_info');z.last_read+=cat[cs.at(-1)?.name]?.kind==='read';z.read_only+=cs.length>0&&cs.every(c=>cat[c.name]?.kind==='read');z.duplicate_samples+=new Set(cs.map(stable)).size<cs.length;for(var c of cs){var t=cat[c.name];if(!t){z.unknown_tool_calls++;continue;}for(var [k,v] of Object.entries(c.arguments)){var p=t.parameters?.properties?.[k]??{};var kind=p.enum?'enums':p.type==='number'||p.type==='integer'?'numbers': /(^|_)(date|time|start|end|date_from|date_to)$/.test(k)?'dates_times':p['x-ref']?'entity_ids':/contact|recipient|assignee|(^to$)|(^cc$)|(^bcc$)|(^query$)/.test(k)?'names_contacts':p['x-match']==='fuzzy'?'free_text':'other';z.arg_kinds[kind]=(z.arg_kinds[kind]??0)+1;if(p.enum){z.enums++;z.bad_enums+=!p.enum.some(e=>stable(e)===stable(v));}if(['number','integer'].includes(p.type)){z.numbers++;z.bad_numbers+=typeof v!=='number'||!Number.isFinite(v)||(p.type==='integer'&&!Number.isInteger(v));}if(kind==='dates_times'&&typeof v==='string'){z.date_values++;z.bad_date_shapes+=!(/^(\d{4}-\d{2}-\d{2})(T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)?)?$|^\d{2}:\d{2}(:\d{2})?$/.test(v));}}}}z.calls_per_sample=z.calls/z.samples;return z;};
const recompute=(samples)=>{var gs={},ov={};for(var s of samples){if(s.score===1)continue;var [g,ts]=annotate[s.id];for(var [map,keys] of [[gs,[g]],[ov,ts]])for(var k of keys){(map[k]??={ids:[],loss:0}).ids.push(s.id);map[k].loss+=(1-s.score)*2.5;}}return {groups:gs,overlap:ov};};
const renderFull=(a)=>{var lines=['# Ngân sách lỗi LifeMate — 15/11/2026','', '**Kết luận:** run lưu cục bộ tốt nhất `v3_asrft`: **51.726190/100**, mất **48.273810 điểm**; 17/40 trọn điểm, 23/40 chưa trọn. Đây là proxy TRAIN, không phải điểm portal.','CPU, chỉ phân tích tệp; không chạy model/ToolEnv, không đọc môi trường hay nhãn public. Không có đo GPU mới.','## Nguồn và cách tính','- Smoke: `outputs/colab/smoke_ab_drive/smoke_ab/<run>/native.{jsonl,report.json,scored.json}`; cả ba tệp có đủ ở chín run dưới.','- `research/smoke_split.json`: 40 validation ID; `research/validation_split.json`: 309 validation ID. TRAIN: `../datalife/train+public_test/train/train.jsonl`.','- Đọc `AGENTS.md`, `docs/{COMPETITION_RULES,PUBLIC_GAP,NEXT_PLAN}.md`, `scripts/score_run.py` và scorer phụ thuộc.','- Điểm mẫu trong tệp là 0–1; bảng đổi thành 0–100. Mất điểm = Σ(100 − điểm mẫu)/40. Nhãn chính được gán tay theo khác biệt lời gọi, không phải phân rã nhân quả.','- Mỗi mẫu lỗi có một nhãn chính để tổng không trùng. Nhãn phụ có thể chồng lấp; trần = sửa hoàn toàn mọi lỗi của các mẫu thuộc nhóm, không phải hiệu quả riêng của một sửa đổi.','- Fuzzy theo `research/published_scorer.py`: NFC/lower/whitespace, F1 > 0.6; hash thật từ export đã lưu. Đối chiếu scorer chính thức: chưa xác minh.','', '| Run | Proxy /100 |','|---|---:|',...Object.entries(a.runs).map(([n,r])=>'| '+n+' | '+r.local_proxy_0_to_100.toFixed(6)+' |'),'','Không có v4/v5 trong các thư mục đã kiểm; số v4 trong NEXT_PLAN không có tệp chạy kèm tại đây. Hai run phụ `outputs/colab/planner_sft/smoke40_sft{154,50}_x3` có 40 mẫu, điểm 24.832251 / 21.809524; TIME_SCALE=3, không chứng minh đạt 30 giây.','**Validation309: NOT MEASURED.** Chỉ có export gold309; không có run dự đoán309 trong cây output đã kiểm. Hai control ở `outputs/` đều 40 mẫu. Không chấm gold như dự đoán.','## Xếp hạng nhãn chính và 5 hướng sửa','| Hạng | Nhãn chính | Mẫu | Điểm mất / trần smoke | Sửa dữ liệu/prompt |','|---|---|---:|---:|---|',...Object.entries(a.groups).sort((x,y)=>y[1].loss-x[1].loss).map(([k,v],i)=>'| '+(i<5?i+1:'—')+' | '+labels[k]+' | '+v.ids.length+' | '+v.loss.toFixed(6)+' | '+(fixes[k]??'Ngoài top 5')+' |'),'','Hai nhóm 5 điểm đồng hạng. Trần các nhóm chính cộng đúng 48.273810; nhiều mẫu có lỗi phụ nên sửa riêng nguyên nhân chính có thể chưa tăng điểm. Trần tăng public / khả năng lên 70: **NOT MEASURED**.','## Từng mẫu smoke (điểm /100)','Nhãn trong bảng là nhãn chính; chi tiết nêu lỗi phụ. O = điểm run `oracle_transcripts`; **ASR-caused của từng mẫu = NOT MEASURED** vì oracle khác planner.','| ID | Điểm | Nhãn chính / bằng chứng | O |','|---|---:|---|---:|',...a.best.samples.map(s=>'| '+s.id+' | '+(s.score*100).toFixed(4)+' | '+(annotate[s.id]?labels[annotate[s.id][0]]+': '+annotate[s.id][2]:'Đạt')+' | '+(a.oracle[s.id].score*100).toFixed(4)+' |'),'','## Nhãn phụ chồng lấp: không cộng các dòng','| Loại quan sát | Điểm mất trên tập mẫu | ID (bỏ tiền tố train_) |','|---|---:|---|',...Object.entries(a.overlap).map(([k,v])=>'| '+k+' | '+v.loss.toFixed(6)+' | '+v.ids.map(id=>id.slice(6)).join(', ')+' |'),'','Sai args gồm ngày/giờ, tên/liên hệ, số, enum, văn bản và ID. Enum ở đây là chọn sai nguồn thanh toán, không phải giá trị ngoài schema; số là thêm bộ lọc floor, không phải nghe sai lượng. Văn bản title ở 1202 qua fuzzy, không tính là lỗi văn bản.','**ASR:** oracle dùng sft154/train_oracle (49.916667), v3_asrft dùng sft_v3/asr_ft (51.726190). Chênh lệch +15.065476 ở 9 mẫu, −16.875000 ở 8 mẫu, ròng −1.809524 là đối chiếu có nhiễu planner, không phải ngân sách ASR.','`w_asrft` là ASR fine-tune, không phải gold; cùng tên artifact sft154 với oracle nhưng chưa chứng minh cùng prompt/cờ. Chênh +5.148810 của oracle chỉ là mô tả. Transcript trong scored là TRAIN gold, không phải bản ASR; không dùng nó để kết luận ASR sai.','## Public: chỉ thống kê dự đoán, không quy lỗi','Nguồn: `outputs/colab/public_drive/public/{sft154,v3,v3_asrft}/submission.jsonl`. TRAIN gold: toàn bộ1545 và tập fit1236 (= bỏ validation309), chỉ đọc `label.tool_calls`.','| Tập | N | Calls / mẫu | Rỗng | Có hỏi | Cuối là read | Chỉ read | Mẫu trùng call |','|---|---:|---:|---:|---:|---:|---:|---:|',...Object.entries(a.stats).map(([n,v])=>'| '+n+' | '+v.samples+' | '+v.calls_per_sample.toFixed(4)+' | '+v.empty+' | '+v.ask+' | '+v.last_read+' | '+v.read_only+' | '+v.duplicate_samples+' |'),'','Có hỏi = ít nhất một sys.ask_missing_info/mẫu; cuối read = catalog kind=read ở call cuối; chỉ read = mọi call đều read, không rỗng. Đây là tín hiệu quan sát, không chứng minh dừng sai hay hỏi sai.','| Tập | Ngày/giờ: sai hình dạng/tổng | Số: sai kiểu/tổng | Enum: ngoài tập/tổng |','|---|---:|---:|---:|',...Object.entries(a.stats).map(([n,v])=>'| '+n+' | '+v.bad_date_shapes+'/'+v.date_values+' | '+v.bad_numbers+'/'+v.numbers+' | '+v.bad_enums+'/'+v.enums+' |'),'','Ngày/giờ kiểm hình dạng YYYY-MM-DD, ISO datetime hoặc HH:MM[:SS] trên khóa ngày/giờ; không kiểm nghĩa/ngày lịch. Số và enum theo schema catalog. Không phát hiện tool ngoài catalog ở các tập trên; chưa đo app được cài vì không đọc đầu vào public.','Sai/thiếu tool, gọi thừa về nghĩa, sai ngày/tên/số/văn bản/ID, đáng lẽ hỏi, hỏi không cần, dừng sai và ASR-caused trên public: **NOT MEASURED**. Chỉ đếm calls, lặp, hỏi, đọc và hình dạng; không suy nhãn từ chênh phân bố.','## Kiểm chứng và tái lập','`node research/error_budget.mjs --check` kiểm 360 điểm bằng proxy hiện có; kiểm ID40, lời gọi/hash với export thật, trung bình, tổng ngân sách, gold TRAIN và nhãn chính; không replay.','`node research/error_budget.mjs --write` viết tài liệu và thử viết PROGRESS.md/REPORT.md vào staging; lỗi quyền ghi được báo rõ, không đổi log chính thức.','Bản này không sửa src/config/log, không commit, không train/download. Hardware/CUDA còn mâu thuẫn như COMPETITION_RULES; không chọn ngầm một cấu hình.'];return lines.join('\n')+'\n';};
const renderDoc=a=>{var text=renderFull(a);var start=text.indexOf('| Run |');var end=text.indexOf('\n\n',start);return text.slice(0,start)+'Run (proxy): '+Object.entries(a.runs).map(([n,r])=>'`'+n+'` '+r.local_proxy_0_to_100.toFixed(6)).join('; ')+text.slice(end);};
const readJ=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const readL=async p=>(await fs.readFile(p,'utf8')).trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
const runs={};
for(const n of names) runs[n]=await readJ(path.join(repo,smokeRoot,n,'native.scored.json'));
const bestName=Object.keys(runs).filter(n=>runs[n].source.transcript_source==='asr').sort((x,y)=>runs[y].local_proxy_0_to_100-runs[x].local_proxy_0_to_100)[0];
assert.equal(bestName,'v3_asrft','Review annotations if best run changes');
const best=runs[bestName];
const smoke=await readJ(path.join(repo,'research/smoke_split.json'));
const validation=await readJ(path.join(repo,'research/validation_split.json'));
const train=(await readL(path.join(kit,'train/train.jsonl'))).map(r=>({id:r.id,tool_calls:r.label.tool_calls}));
const trainIndex=Object.fromEntries(train.map(r=>[r.id,r]));
const tc=await readJ(path.join(kit,'train/train_tools_catalog.json'));
const pc=await readJ(path.join(kit,'public_test/public_test_tools_catalog.json'));
const toolsT=Object.fromEntries(tc.apps.flatMap(a=>a.tools).map(t=>[t.name,t]));
const toolsP={...toolsT,...Object.fromEntries(pc.apps.flatMap(a=>a.tools).map(t=>[t.name,t]))};
const pubs={};
for(const n of ['sft154','v3','v3_asrft']) pubs[n]=await readL(path.join(repo,pubRoot,n,'submission.jsonl'));
const allStats={TRAIN_all:stats(train,toolsT),TRAIN_fit1236:stats(train.filter(r=>!validation.validation_ids.includes(r.id)),toolsT)};
for(const [n,rows] of Object.entries(pubs)){
 assert.equal(new Set(rows.map(r=>r.id)).size,rows.length,'Duplicate public IDs');
 assert.equal(rows.length,634,'Public coverage changed; review report');
 allStats[n]=stats(rows,toolsP);
}
assert.equal(best.samples.length,40);
assert.deepEqual(best.samples.map(s=>s.id).sort(),[...smoke.validation_ids].sort());
assert.equal(new Set(train.map(r=>r.id)).size,train.length);
assert.equal(validation.validation_ids.length,309);
assert.deepEqual(Object.keys(annotate).sort(),best.samples.filter(s=>s.score<1).map(s=>s.id).sort());
for(const s of best.samples){
 assert.ok(s.score>=0&&s.score<=1);
 assert.equal(stable(s.gold),stable(trainIndex[s.id].tool_calls),'TRAIN gold mismatch '+s.id);
}
const oracle=Object.fromEntries(runs.oracle_transcripts.samples.map(s=>[s.id,s]));
const analysis={runs,best,oracle,...recompute(best.samples),stats:allStats};
const totalLoss=Object.values(analysis.groups).reduce((n,g)=>n+g.loss,0);
assert.ok(Math.abs(totalLoss-(100-best.local_proxy_0_to_100))<1e-10,'Primary loss total');
const document=renderDoc(analysis);
assert.ok(document.trimEnd().split('\n').length<=120,'Document exceeds 120 lines');
const verification=execFileSync('python',['-B','-c',verifyCode],{cwd:repo,encoding:'utf8'});
const checkedAt=new Date().toISOString();
const progress=[...recordedProgress,{time:checkedAt,step:'Run repeatable analysis and verification',result:'360 scores reproduced; budget sums; public IDs unique; document <=120 lines',next:'Write requested artifacts'}].map(e=>[e.time,e.step,e.result,e.next].join(' | ')).join('\n')+'\n';
const report=[
 '# REPORT — LifeMate error budget','',
 'Analysis complete. The original session could not write staging files: EPERM. Confirm successful writes before claiming delivery.',
 'Artifacts: docs/ERROR_BUDGET.md (Vietnamese, '+document.trimEnd().split('\n').length+' lines); research/error_budget.mjs (new). No commits, changes to existing source/config, official logs, downloads or inference.',
 'Best saved run: '+bestName+' '+best.local_proxy_0_to_100.toFixed(9)+'/100; 23/40 nonperfect; loss '+totalLoss.toFixed(9)+'.',
 'Top five primary buckets: dates/times 12.5; extra calls 8.898809524; entity IDs 7.5; wrong/missing tool 5; should-have-asked 5. Ceilings require fully fixing every tagged sample; not public uplift forecasts.',
 'Validation309 predictions: NOT MEASURED. Gold-only export exists; checked model runs and controls have 40 IDs. ASR-caused: NOT MEASURED; oracle planner differs. Public correctness and public improvement ceiling: NOT MEASURED.',
 'Public prediction counts: sft154/v3/v3_asrft each 634; calls 1180/1532/1556; asking samples 43/62/60; last-read 96/92/85. TRAIN gold all1545 and fit1236 distinguished.',
 'Scope: explicit TRAIN JSONL, train/public catalogs, existing smoke exports/scores/reports, prediction-only public JSONL and documented repo files. No *_environment, state.json, public/private labels, hidden state, replay or ToolEnv. No new GPU measurement; hardware/CUDA conflicts unresolved.',
 'Labels are reviewed call differences: one primary bucket plus overlapping secondary tags. Assignments embedded here; all40 samples and oracle comparison in ERROR_BUDGET.md.',
 '', '## Exact commands',
 'Executable + argument arrays below are exact; working directory: '+repo,
 '~~~json',JSON.stringify(recordedCommands,null,2),'~~~',
 'Repeatable command: node research/error_budget.mjs --check (includes exact Python invocation above).',
 '~~~text',verification.trim(),'~~~',
 'Other operations: explicit fs.readFile/fs.readdir via Node; fs.writeFile for new repository files. Staging fs.writeFile failed EPERM twice; apply_patch failed to write PROGRESS.md. Shell failed before execution: missing sandbox setup helper. No permission bypass attempted.',
 '', '## Computed results',
 '~~~json',JSON.stringify({primary:analysis.groups,overlapping:analysis.overlap,publicAndTrain:analysis.stats},null,2),'~~~',
 '', 'Recovery after restoring staging writes: node research/error_budget.mjs --write. View report without writing: node research/error_budget.mjs --report.'
].join('\n')+'\n';
if(process.argv.includes('--write')){
 await fs.writeFile(path.join(repo,'docs/ERROR_BUDGET.md'),document);
 const failures=[];
 for(const [name,text] of [['PROGRESS.md',progress],['REPORT.md',report]]){
  try {await fs.writeFile(path.join(stage,name),text);console.log('WROTE '+path.join(stage,name));}
  catch(e){failures.push(name+': '+e.code);}
 }
 if(failures.length){console.error('STAGING WRITE FAILED: '+failures.join('; '));process.exitCode=1;}
}
if(process.argv.includes('--report'))console.log(report);
else console.log(JSON.stringify({best:bestName,score:best.local_proxy_0_to_100,totalLoss,primary:analysis.groups,overlapping:analysis.overlap,stats:analysis.stats,docLines:document.trimEnd().split('\n').length,verification:verification.trim(),checks:'PASS'},null,2));
