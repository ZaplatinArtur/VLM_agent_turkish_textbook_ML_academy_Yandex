# 9B answer-contract repair v1

This is an explicitly post-score-motivated development experiment. Historical aggregate and residual outcomes were already known before its design. It makes no blind or preregistered claim, and its candidate must remain unscored and unevaluated until separately authorized.

Both task-ID-free arms preserve every pinned source-union row and every image-judge row as exact base JSONL line bytes. On deterministic rows outside the source union, both do nothing when the top-level answer already satisfies the scalar answer contract.

The strict arm can replace an invalid answer only when bounded exact nested-JSON parsing of `raw_response` exposes exactly one distinct strict value under an exact `final_answer` key. The separately labeled exploratory arm is also explicitly post-score-motivated. It requires exactly one syntactically explicit quoted key from `final_answer`, `answer`, `choice`, or `result`, immediately followed by one JSON scalar of at most 64 characters. It permits at most two structural unescape layers and can discard only unmatched leading closing or trailing opening square/curly brace debris. It never mines free reasoning or unkeyed numbers. Multiple key occurrences, ambiguity, parse failure, or any bound hit preserves the base row. Task allowlists are forbidden in both arms.

The only permitted sequence is:

```powershell
python experiments/maxim_9b_answer_contract_repair_v1/answer_contract_repair_v1.py --write-rule-freeze
python experiments/maxim_9b_answer_contract_repair_v1/answer_contract_repair_v1.py --build-candidate
python experiments/maxim_9b_answer_contract_repair_v1/answer_contract_repair_v1.py --verify-output
```

No command in this experiment reads benchmark answers, references, scores, correctness fields, judge verdicts, or evaluation artifacts.
