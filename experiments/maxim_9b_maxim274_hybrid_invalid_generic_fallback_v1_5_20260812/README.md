# Maxim274 invalid-generic fallback V1.5

V1.5 keeps the frozen V1.4 rule exactly: preserve the 18 certified no-ID
answers, use a structurally valid generic answer, and otherwise copy the exact
raw base240 row bytes. It supersedes V1.4 before use because V1.4's direct
`os.open` descriptors are text-mode on Windows and can translate LF to CRLF,
contradicting its exact-raw-byte claim even though its rule and audit passed.

V1.5 uses binary-exclusive writes, checks exact post-write bytes and SHA,
requires 274 terminal-LF rows with no carriage returns, and rechecks every
invalid fallback row byte-for-byte after writing. Its tests include the
hostile Windows newline case. V1.4 remains immutable and is retained as
truthful PASS-but-superseded-preuse lineage.
