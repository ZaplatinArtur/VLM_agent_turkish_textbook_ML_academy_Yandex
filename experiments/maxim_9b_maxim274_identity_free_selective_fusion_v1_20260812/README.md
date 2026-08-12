# Maxim274 identity-free selective fusion V1

This is a conservative pre-outcome fusion of the audited 251/274 no-ID
candidate and the pending generic Qwen3.5-9B candidate.

The selector sees only public `answer_type`, the baseline answer, and the
generic prediction's content/structural contract. It never sees task ID,
row number, subject, route, filename/hash, gold, outcome, score, or partial
generic run state. IDs are used after selection only for output alignment.

Rule: keep the 18 certified no-ID answers. On the other 256 rows, keep base251
for every malformed/failed generic prediction and for every parseable baseline
answer. A structurally valid generic answer may replace base251 only when the
baseline answer is syntactically unparseable under the public answer type.

The corrected V1.1 freeze exposes at most 10/256 generic rows to a switch;
246/256 are held on base251 regardless of model disagreement. This protects
against transport/schema failures and sharply limits semantic regression risk,
but it does **not** prove non-regression or improvement over 251: a syntactically
valid generic answer can still be wrong, and an unparseable baseline answer can
still receive credit from a judge. The safe reported portfolio therefore keeps
the already frozen aggregate best-of-two selector as the guaranteed floor.

Run tests:

`python -B -m unittest test_selective_fusion.py`

After candidate completion and an independent PASS audit, run
`selective_fusion.py` with exact freeze, audit, completion and prediction SHAs.

The initial V1 freeze is retained as invalid lineage: its narrative risk count
said 9 while its census correctly contained 10, and its structural validator
received the generic row's alignment key. V1.1 removes identity before the
selector and derives the risk bound directly from the frozen census.
