# Human evaluation interface

## Implemented workflow

The main pointwise screen follows the useful part of LMArena's comparison pattern while remaining reference-aware:

- the task image is shown at full usable width;
- the expected answer and the agent answer are adjacent, equal-priority panes;
- image-only gold references can be enlarged without leaving the task;
- setup identity is locked hidden until the primary label is complete; it can be revealed only when revisiting an already completed item for diagnostics;
- the annotator assigns a 0–4 score, strict final-answer/reasoning/completeness flags, error tags, confidence, and a short rationale;
- annotations autosave, export to CSV/JSONL, and remain addressable by `(task_id, setup)`;
- keyboard shortcuts support fast labeling.

The Arena mode shows candidate A and B side by side. Pair order is deterministic-randomized, the gold answer is collapsed, and mirrored pairs can be generated to estimate position bias. The saved annotation retains the hidden side-to-setup mapping for later analysis.

## Why gold and candidate stay next to each other

The judge's job is reference-aware correctness, not preference or eloquence. Vertical separation makes annotators rely on memory and especially hurts tasks with several subanswers or annotated diagrams. Adjacent panes reduce visual context switching and make omissions obvious. On narrow displays the same panes stack rather than shrink to unreadable columns.

## Additional implemented workflows

The same interface now has two independent workspaces:

1. Gold transcription shows the source reference beside structured fields for transcription, acceptable equivalents, required subanswers, quality, and notes. Records are keyed only by `task_id`, so the gold is never copied into three setup labels.
2. Adjudication keeps expected and candidate answers adjacent, then shows the human and LLM-judge verdicts side by side. The reviewer can accept either verdict, provide a custom 0–4 score, or exclude the item, identify the issue source, and record a rationale. The queue includes every score/strict disagreement, judge error, low-confidence verdict, or reference flag, plus a stable agreement control sample.

Each workspace has separate progress, status filters, JSONL/CSV exports, persistence, and keyboard flow. Hidden elements use the native `hidden` contract so modes cannot visually leak into one another.

## Recommended next interface increments

1. Reveal retrieval traces only after the primary correctness vote is locked. This allows a second, diagnostic label—retrieval useful, unused, misleading, or exact-solution—without biasing answer-quality scoring.
2. Add a three-column task matrix showing `no_tools`, `web_search`, and `textbook_retrieval` only for post-hoc error analysis. Primary labels should still be collected blind and one answer at a time.
3. Display disagreement heatmaps and completion by raw subject/answer type to steer calibration coverage.

The key separation is: first score the answer blind, then inspect how the agent reached it. Combining these stages would systematically favor verbose tool traces and retrieval-heavy setups.
