# -*- coding: utf-8 -*-
"""Чистка уже собранного корпуса ÖdevJet от шаблонных строк интерфейса.

Пересборка скрейпингом — сутки, поэтому чистим имеющийся JSONL на месте
тем же фильтром, что теперь стоит в скрейпере (odevjet_boilerplate).
Читает и пишет как .jsonl, так и .jsonl.gz (по расширению).

    python scripts/clean_corpus.py data/corpus_backup/odevjet_corpus.jsonl.gz \
                                   data/corpus_backup/odevjet_corpus.clean.jsonl.gz
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odevjet_boilerplate import clean_text  # noqa: E402


def _open(path: Path, mode: str):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 2
    src, dst = Path(args[0]), Path(args[1])
    min_chars = 0
    for a in sys.argv[1:]:
        if a.startswith("--min-chars="):
            min_chars = int(a.split("=", 1)[1])

    pages = written = short = 0
    lines_before = lines_after = 0
    with _open(src, "rt") as fin, _open(dst, "wt") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            before = rec.get("content") or ""
            after = clean_text(before)
            pages += 1
            lines_before += sum(1 for x in before.split("\n") if x.strip())
            lines_after += sum(1 for x in after.split("\n") if x.strip())
            if len(after.strip()) < 100:
                short += 1
            if len(after.strip()) < min_chars:
                continue  # страница без содержания: под навигацией не было текста
            rec["content"] = after
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    removed = lines_before - lines_after
    print(f"страниц прочитано {pages}, записано {written}")
    print(f"строк было {lines_before}, стало {lines_after} "
          f"(убрано {removed}, {removed/max(lines_before,1):.1%})")
    print(f"страниц короче 100 символов после чистки: {short} "
          f"({short/max(pages,1):.1%}) — содержания на них нет, "
          f"отсечь можно флагом --min-chars=N")
    print(f"записано: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
