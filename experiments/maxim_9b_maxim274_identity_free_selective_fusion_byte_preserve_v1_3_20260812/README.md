# Maxim274 identity-free selective fusion — byte-preserving V1.3

This immutable successor changes serialization, not selection. It imports the
exact audited V1.1 branch decisions. Every baseline-selected row is copied as
the complete original base251 JSONL bytes; each generic-selected row is copied
as the complete original V1.1 JSONL bytes. There are no new branch parameters,
thresholds, task-ID rules, or semantic tunables.

The inherited census is 272 baseline rows and two generic rows. Postdecision
alignment proves both generic rows are outside the fixed image97 partition.
Consequently all 97 image rows, full parsed objects, raw JSONL bytes, and the
frozen judge adapter's `candidate_text(...).encode("utf-8")` bytes match
base251 exactly. Its already-audited image97 judge can therefore be reused
without an API call or transferring a verdict to changed candidate content.

Chronology disclosure: the V1.1 rule and independent audit predate candidate
atomic completion, but this V1.2 serialization successor was built after an
existing postscore artifact was accidentally displayed to its builder. No
per-row outcome was used: V1.3 has no decision freedom and mechanically imports
the immutable V1.1 result. This is not represented as a clean-room build.

V1.2 is preserved as a withheld freeze. Its composer did not verify its own
frozen implementation hashes at runtime, and its scorer did not fully bind the
completion or recompute the non-image payload. V1.3 closes both gaps: composer
and scorer verify all frozen implementation descriptors, and the scorer
recomputes all 274 output bytes before it may parse private scoring inputs.

Workflow:

1. `python -B -m unittest test_byte_preserve.py`
2. `python -B prepare_freeze.py`
3. Obtain an independent PASS audit using `INDEPENDENT_AUDIT_TEMPLATE.json`.
4. Run `compose.py` with the exact freeze and audit SHA-256 values.
5. Optionally run `score_reusing_base251_judge.py` with exact completion and
   solver hashes plus the explicit `--execute-private-score` flag. It needs no
   API key and persists only aggregate score data. Success is at least 250/274.
