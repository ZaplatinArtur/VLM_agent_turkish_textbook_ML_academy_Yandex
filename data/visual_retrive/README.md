# visual_retrive data

Corpus root for the visual textbook retriever.

## Layout

```
data/visual_retrive/
  catalog/
    books.jsonl              # one row per book (slug, url, grade, subject, pages)
    pages_index.jsonl        # one row per page URL
    discover_summary.json
  books/
    {book_slug}/
      pages/{NNNN}.jpg       # textbook page images
      answers/
        {NNNN}.json          # answer metadata
        {NNNN}.txt           # text solution (if any)
        {NNNN}.webp|.jpg     # answer image(s) (if any)
```

## Scrape

From the repo root (so `src/` is importable):

```powershell
$env:PYTHONPATH = "src"

# 1) Catalog of all books + pages (~216 books, ~47k pages)
python -m visual_retrive.scripts.discover_catalog

# 2) Textbook page images (/download/sayfalar/...)
python -m visual_retrive.scripts.scrape_pages

# 3) Answers: text (`text-solution-content`) and/or images (/download/cevaplar/)
python -m visual_retrive.scripts.scrape_answers

# Or everything:
python -m visual_retrive.scripts.scrape_all
```

Smoke (one book / few pages):

```powershell
$env:PYTHONPATH = "src"
python -m visual_retrive.scripts.scrape_all --max-books 1 --max-pages 5 --from-book-pages-only
```

Already-downloaded images under `data/books/` are reused when present.

## Train queries + fine-tuning

```powershell
$env:PYTHONPATH = "src"

# 1) Page bundles manifest
python -m visual_retrive.scripts.build_manifest

# 2a) Offline heuristic queries (no LLM)
python -m visual_retrive.scripts.generate_train_queries --mode heuristic --write-splits

# 2b) LLM queries via vLLM / OpenRouter
# $env:MLA_VLLM_BASE_URL = "http://localhost:8000/v1"
# $env:MLA_MODEL_NAME = "Qwen/Qwen3.5-9B"
python -m visual_retrive.scripts.generate_train_queries --mode llm --workers 4 --write-splits

# 3) Cheap text embedder fine-tune (query ↔ answer_text)
python -m visual_retrive.scripts.train_text_retrieval `
  --pairs data/visual_retrive/catalog/train_queries.jsonl `
  --output data/visual_retrive/models/text_embedder_ft `
  --epochs 1

# 4) ColQwen2 LoRA on GPU server (needs colpali-engine)
python -m visual_retrive.scripts.train_colqwen_lora `
  --pairs data/visual_retrive/catalog/train_queries.jsonl `
  --data-root data/visual_retrive `
  --output data/visual_retrive/models/colqwen2_lora_tr `
  --batch-size 1 --grad-accum 8 --epochs 1
```

Outputs:

- `catalog/page_bundles.jsonl` — one row per page
- `catalog/train_queries.jsonl` — `(query, positive_page_id, hard_negative_page_ids, ...)`
- `catalog/train_splits/{train,val,test}.jsonl`
- `models/...` — fine-tuned checkpoints
