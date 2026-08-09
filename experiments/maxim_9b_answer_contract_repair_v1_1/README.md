# 9B answer-contract repair v1.1

This is an explicitly post-score-motivated development experiment. Historical aggregate and residual outcomes were already known before its design. It makes no blind or preregistered claim, and its candidate must remain unscored and unevaluated until separately authorized.

Both task-ID-free arms preserve every pinned source-union row and every image-judge row as exact base JSONL line bytes. On deterministic rows outside the source union, both do nothing when the top-level answer already satisfies the scalar answer contract.

The strict arm can replace an invalid answer only when exact parsing of the outer response object exposes exactly one strict scalar under a top-level `final_answer` key. The separately labeled exploratory arm is also explicitly post-score-motivated. It requires exactly one top-level field of the outer response object whose key is `final_answer`, `answer`, `choice`, or `result`, and whose value is a scalar of at most 64 characters. Exact JSON parsing preserves duplicate keys. The only exploratory malformed-outer repair accepts a missing final outer `}` while still requiring every member key and value to be a complete JSON token. String and container values are atomic in both arms: their contents are never unescaped or rescanned, so key-like text inside `reasoning` or any other value cannot become a candidate. The exploratory selected scalar may discard only unmatched leading closing or trailing opening square/curly brace debris. Multiple outer keys, ambiguity, parse failure, or any bound hit preserves the base row. Task allowlists are forbidden in both arms.

This v1.1 experiment supersedes the unevaluated v1 exploratory parser after independent audit found that global unescaping followed by regex could expose key-like text inside a reasoning string. The predecessor freeze and output are preserved byte-for-byte and hash-pinned by its `SUPERSEDED.json` record.

The same frozen rules are materialized independently on two pinned 9B bases: the final source solver and the audited selector v1.2 primary solver. The second variant is labeled `on_v1_2_primary_240`; the number is historical context only and is never read or recomputed by this experiment. Both variants remain unscored.

The only permitted sequence is:

```powershell
python experiments/maxim_9b_answer_contract_repair_v1_1/answer_contract_repair_v1_1.py --write-rule-freeze
python experiments/maxim_9b_answer_contract_repair_v1_1/answer_contract_repair_v1_1.py --build-candidate
python experiments/maxim_9b_answer_contract_repair_v1_1/answer_contract_repair_v1_1.py --verify-output
```

No command in this experiment reads benchmark answers, references, scores, correctness fields, judge verdicts, or evaluation artifacts.
