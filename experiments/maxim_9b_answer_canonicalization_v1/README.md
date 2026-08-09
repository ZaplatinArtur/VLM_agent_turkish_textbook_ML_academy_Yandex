# 9B answer canonicalization v1

This is a post-score-motivated development experiment. The historical 240/274 result and prior task outcomes were known before the rules were designed. The experiment is neither blind nor preregistered. Its rules are frozen only with respect to future evaluation of these new arms.

Runtime never reads benchmark references, scores, correctness, judge verdicts, or historical task outcomes. A one-time design step projected the pinned benchmark into an allowlisted artifact containing only ordered task identity, observable question text, `answer_type`, and subject. The projection excludes reference answers and solutions, images, grade, score, judge, and outcome fields.

Explicit JSON recovery is not implemented here. That responsibility belongs to the separate `maxim_9b_answer_contract_repair_v1_1` successor, whose parser operates only on actual outer response members. This experiment does not read or parse `raw_response`, and it never changes an absent or structurally invalid top-level answer.

The primary arm converts a whole one-codepoint choice answer only when Unicode NFKC maps it exactly to ASCII A–E. The nested exploratory choice arm additionally permits a frozen, small Greek/Cyrillic glyph map for A, B, C, and E; D is deliberately omitted. Both require observable `answer_type=choice`.

The fraction/percent arm is separately labeled exploratory. It acts only on `short_text` answers and only when observable question text contains a frozen multilingual phrase that explicitly requests percentage form or fraction form. A bare percent symbol, an image-only placeholder, subject, or historical outcome is never enough. Fraction-to-percent conversion must be exact, terminating, and require no more than six decimal places.

All arms are task-ID-free. They can modify only deterministic rows outside the pinned source union. Every one of the 156 source-union rows and every one of the 97 image-judge rows remains the exact original base JSONL line bytes. No scorer or gold artifact is edited.

The frozen build sequence is projection, rule freeze, candidate materialization, and output verification. Candidate outputs remain unscored until a separate evaluation wave is authorized.
