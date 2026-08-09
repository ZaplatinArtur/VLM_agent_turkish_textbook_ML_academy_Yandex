# Контракт честного 9B comparison

UI не содержит чисел для 9B milestones в коде. Он принимает один тонкий manifest,
проверяет SHA-256 всех указанных файлов, task set, evaluator semantics и model
closure, и только после этого показывает семь результатов.

## Comparison manifest v2

Допустимы абсолютные пути и пути относительно каталога manifest. Каждый descriptor
имеет ровно два поля: `path` и SHA-256 содержимого файла `sha256`.

```json
{
  "schema_version": "vlm-9b-milestone-comparison-v2",
  "model": "Qwen/Qwen3.5-9B",
  "benchmark": {
    "path": ".../validation_274.jsonl",
    "sha256": "<64 lowercase hex>"
  },
  "milestones": [
    {
      "milestone_id": "page_rag_9b",
      "adapter": "normalized_v2",
      "aggregate": {"path": ".../aggregate.json", "sha256": "<sha>"}
    },
    {
      "milestone_id": "no_tools_9b",
      "adapter": "normalized_v2",
      "aggregate": {"path": ".../aggregate.json", "sha256": "<sha>"}
    },
    {
      "milestone_id": "query_active_crop_v2_9b",
      "adapter": "normalized_v2",
      "aggregate": {"path": ".../aggregate.json", "sha256": "<sha>"}
    },
    {
      "milestone_id": "source_v1_rebase_9b",
      "adapter": "maxim_9b_source_replay_aggregate_v1",
      "aggregate": {"path": ".../source_v1_aggregate/aggregate.json", "sha256": "<sha>"}
    },
    {
      "milestone_id": "source_v3_rebase_9b",
      "adapter": "maxim_9b_source_replay_aggregate_v1",
      "aggregate": {"path": ".../source_v3_aggregate/aggregate.json", "sha256": "<sha>"}
    },
    {
      "milestone_id": "source_v6_rebase_9b",
      "adapter": "maxim_9b_source_replay_aggregate_v1",
      "aggregate": {"path": ".../source_v6_aggregate/aggregate.json", "sha256": "<sha>"}
    },
    {
      "milestone_id": "source_v7_rebase_9b",
      "adapter": "maxim_9b_source_replay_aggregate_v1",
      "aggregate": {"path": ".../source_v7_aggregate/aggregate.json", "sha256": "<sha>"}
    }
  ]
}
```

Порядок и набор `milestone_id` фиксированы. `Source V2`, `V4` и `V5` остаются в
раскрываемой provenance timeline и не подменяют семь основных точек.

## Native source adapter

`maxim_9b_source_replay_aggregate_v1` читает существующий native aggregate со
`schema_version = maxim-9b-source-replay-aggregate-v1`; переносить его метрики в
отдельный UI JSON не надо. Обязательные top-level поля:

```text
schema_version, created_at_utc, label, reporting_status,
model_selection_status, anchor, benchmark, final_solver, final_image_judge,
score, scorer, certificate_bundle, stages, stage_counts,
upstream_generation_model_closure, answer_origin_closure,
inherited_27b_outputs, gold_access_during_generation,
gold_access_during_postgeneration_score, protocol, overall, slices,
evaluator_split, comparison_vs_page_rag, comparison_vs_anchor,
comparison_vs_adjacent, comparisons, content_projection,
content_projection_sha256, source_union, final_origin_counts
```

Loader проверяет, а не просто отображает:

- hash native aggregate из comparison wrapper;
- benchmark, anchor, final solver, final image judge, score и scorer;
- profiles, resolver/composition/judge manifests, candidates, decisions, все
  stage solvers и все certificate bundles;
- канонический `content_projection_sha256` и source-union projection SHA;
- одинаковый task set и знаменатель; уникальные `task_id`;
- все anchor, stage solver и final solver rows имеют `model = Qwen/Qwen3.5-9B`
  и не содержат
  gold/reference fields. Явное row-level `generation.gold_access` при наличии
  обязано быть `false`; отсутствие исторически не существовавшего поля допустимо
  только при точном SHA pin anchor и byte-preserving passthrough либо strong
  certified source decision. Top-level aggregate и все resolver/composition
  manifests при этом обязаны явно подтверждать gold/outcome access `false`;
- `inherited_27b_outputs = false`, upstream generation model closure ровно
  `["Qwen/Qwen3.5-9B"]`; source origin не является моделью. Поэтому
  `answer_origin_closure` — exact отсортированный список ненулевых ключей
  `final_origin_counts`: passthrough, official-source confirmation и
  deterministic official-source replacement;
- replacements — это ровно строки, где final отличается от ActiveCrop anchor,
  есть `official_source_override` и strong accepted certificate; confirmations и
  passthrough замыкают знаменатель;
- score task outcomes, subject slices, deterministic/image split и финальные
  image verdicts; model-free verdict допустим только для deterministic official
  source certificate, остальные judge rows обязаны быть Qwen3.5-9B;
- каждый stage judge manifest обязан хранить stage-local массив
  `source_adjudicated_image_rows` и четыре честных счётчика:
  `stage_source_adjudicated_image_rows_count`,
  `copied_base_judge_rows_byte_identical`,
  `cumulative_source_adjudicated_image_rows_count`,
  `cumulative_original_9b_judge_rows_count`. Loader пересчитывает их по output
  JSONL, проверяет цепочку immediate base → stage output и сравнивает
  `original_9b` именно с исходным ActiveCrop judge на уровне байтов. Поле
  `copied_9b_judge_rows_byte_identical` запрещено как вводящее в заблуждение:
  копия immediate base после первого source stage уже может быть source-adjudicated;
- stage-local records отделяют происхождение verdict от действия над ответом:
  `verdict_origin = deterministic_official_source_adjudication`, а
  `stage_answer_action` имеет ровно одно из значений
  `keep_immediate_base_confirmed_by_source` или
  `replace_immediate_base_with_source`. Loader сверяет action по immediate-base и
  composed solver. Поэтому source confirmation не называется replacement;
- глобальные replacement/confirmation counts берутся отдельно из
  `final_origin_counts` и source-union projection, а не из evaluator records;
- `comparison_vs_page_rag` не смешивается с source gain. UI использует только
  явно размеченные `comparison_vs_anchor` и `comparison_vs_adjacent`.

Любой stale stage SHA закрывает весь comparison fail-closed: карточка не получает
score и default dataset не переключается на 9B.

## Normalized v2 для трёх неоднородных legacy milestones

Page RAG, no-tools и ActiveCrop исторически имеют разные наборы артефактов. Для
них допустим маленький adapter aggregate со
`schema_version = vlm-9b-milestone-aggregate-v2`. Его exact поля:

```text
schema_version, milestone_id, model, pipeline, provenance_status,
bound_before_score, caveats, provenance_manifests, artifacts,
certificate_absence_reason, benchmark_sha256, metrics, model_closure,
source_union, comparisons, evaluator, final_origin_counts
```

`artifacts` содержит `solver`, `raw_solver`, `score`, `judge`, `certificates`.
`solver`, `raw_solver`, `score` и `judge` — `{path, sha256}`; элементы массивов certificates и
provenance_manifests — `{role, path, sha256}`. Для этих трёх milestones certificates
пусты, `certificate_absence_reason` непустой, а source union строго нулевой.

`solver` — нормализованное представление для UI, а `raw_solver` — точный исходный
solver до добавления `final_origin`. Loader независимо проверяет SHA и доказывает
построчно, что единственное разрешённое отличие — это добавленный `final_origin`.
Поэтому native source-replay может ссылаться на ActiveCrop только через проверенный
`raw_solver.sha256`; произвольный alias или SHA нормализованного файла закрывает
весь comparison fail-closed.

Статусы provenance фиксированы:

- Page RAG — `historical_output_control`; `bound_before_score = null`, caveat
  обязателен, baseline lock можно приложить как provenance manifest;
- no-tools — `matched_judge_replay_partial_generation_provenance`;
  `bound_before_score = null`, caveat обязателен, отсутствие immutable generation
  manifest не маскируется;
- ActiveCrop — `preregistered_gold_blind`; `bound_before_score = true` и хотя бы
  один SHA-pinned preregistration/composition manifest обязателен.

В normalized v2 `model_closure` является только generation-model closure и имеет
exact поля `expected_model`, `checked_rows`,
`matching_rows`, `foreign_models`. Все solver rows пересканируются; одного 27B row
достаточно для отказа. `evaluator` имеет exact поля `semantics`,
`deterministic_rows`, `image_rows`, `source_certified_image_rows`,
`model_judged_image_rows`, `judge_model`. `final_origin_counts` имеет exact поля
`model_anchor`, `deterministic_source_replacement`, `unknown` и для честного 9B
comparison требует `unknown = 0`.

## Dataset switch

`--dataset auto` выбирает новый trace только после полного успешного чтения всех
семи milestones. Без manifest или при любом mismatch открывается явно подписанный
`archived-27b-v7` reference. Явный `--dataset nine-b-v7` без валидного полного
comparison завершается ошибкой; смешанного 27B trace + 9B score режима нет.

Отдельный display-only join локальных изображений не меняет это правило. Он
ограничен корнем, выведенным из уже SHA-проверенного benchmark path (или тем же
явно проверенным artifact root), запрещает absolute/`..` locators и связывает
файл только по точному task id. Archived solver answers, judge verdicts и
provenance rows через этот путь не импортируются; наличие изображения не влияет
на метрику.

## Перенос frozen bundle между checkout

Существующий absolute path всегда имеет приоритет и обязан совпасть с записанным
SHA. Portable rebase включается только когда этот исходный absolute path больше
не существует. Loader извлекает ровно один suffix, начинающийся с allowlisted
top-level `reports`, `configs`, `artifacts` или `scripts`, и ищет repo root не
более чем в восьми родительских каталогах от comparison/aggregate. Кандидат
обязан остаться внутри найденного root и пройти тот же exact SHA check.

Неизвестный или повторяющийся top-level prefix, `..`, выход через symlink,
отсутствующий кандидат и несколько подходящих кандидатов завершают загрузку
fail-closed. Frozen JSON при переносе не переписывается, а hash authority не
ослабляется.
