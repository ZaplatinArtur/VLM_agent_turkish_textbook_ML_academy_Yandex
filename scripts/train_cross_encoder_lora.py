"""LoRA-дообучение кросс-энкодера bge-reranker-v2-m3 на нашем корпусе.

Зачем именно он: на синтетике кросс-энкодер уверенно лучший (85.3 hit@1 против
77.0 у сильнейшего би-энкодера), а на реальных задачах проигрывает m3 (78.1
против 81.3) — то есть на настоящих запросах он опускает верный первый результат
вниз. Так ведёт себя реранкер, обученный на чужом домене: bge учили на вебе, а у
нас турецкие учебники, где одну тему дословно повторяют девять изданий.

Обучение групповое, а не попарное: на каждый запрос модель видит позитив и все
его негативы разом, и учится ставить позитив выше — это softmax по группе,
ровно та задача, которую реранкер решает в проде. Поточечная BCE учила бы
абсолютному порогу, который нам не нужен.

Разбиение train/val — по странице-источнику, а не по строкам: запросы с одной
страницы почти дубликаты, и при случайном разбиении они утекли бы в валидацию.

Volta (sm_70) не умеет bfloat16, поэтому только fp16 с GradScaler.

    python scripts/train_cross_encoder_lora.py \
        --pairs data/train/cross_encoder/pairs.jsonl \
        --output data/models/bge-reranker-v2
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from paths import DATA_DIR

BASE_MODEL = "BAAI/bge-reranker-v2-m3"
# Уровни сложности запроса. Старые 2422 пары метки не имеют — они делались одним
# промптом «назови правило со страницы», это лёгкий конец, отсюда и умолчание.
LEVELS = ("kolay", "orta", "zor")


def load_groups(path: Path, max_negatives: int):
    groups = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        negatives = [n for n in row.get("negatives", []) if n.strip()]
        if not row.get("positive", "").strip() or not negatives:
            continue
        groups.append({
            "query": row["query"],
            "passages": [row["positive"]] + negatives[:max_negatives],
            "page": row["source_chunk_id"],
            "difficulty": row.get("difficulty", "kolay"),
        })
    return groups


def split_by_page(groups, val_fraction, seed):
    """Валидация отделяется по страницам: запросы с одной страницы — почти дубли."""
    pages = sorted({g["page"] for g in groups})
    random.Random(seed).shuffle(pages)
    val_pages = set(pages[: max(1, int(len(pages) * val_fraction))])
    train = [g for g in groups if g["page"] not in val_pages]
    val = [g for g in groups if g["page"] in val_pages]
    return train, val


def encode(tokenizer, groups, device, max_length):
    pairs = [(g["query"], passage) for g in groups for passage in g["passages"]]
    batch = tokenizer([q for q, _ in pairs], [p for _, p in pairs],
                      padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def group_loss(logits, sizes, torch):
    """Softmax внутри каждой группы: позитив у неё всегда первый.

    Группы разной длины: у четверти запросов судья отверг меньше четырёх
    кандидатов, и негативов вышло 1-3. Считать softmax по каждой отдельно
    дешевле, чем терять эти 26% данных ради одинаковых батчей.
    """
    losses = []
    offset = 0
    for size in sizes:
        scores = logits[offset: offset + size].view(1, size)
        target = torch.zeros(1, dtype=torch.long, device=scores.device)
        losses.append(torch.nn.functional.cross_entropy(scores, target))
        offset += size
    return torch.stack(losses).mean()


def positive_ranks(logits, sizes):
    """Позиция позитива внутри каждой группы, считая с единицы."""
    ranks, offset = [], 0
    for size in sizes:
        scores = logits[offset: offset + size]
        ranks.append(int((scores > scores[0]).sum().item()) + 1)
        offset += size
    return ranks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LoRA для кросс-энкодера")
    parser.add_argument("--pairs", type=Path, default=DATA_DIR / "train" / "cross_encoder" / "pairs.jsonl")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "models" / "bge-reranker-v2")
    parser.add_argument("--negatives", type=int, default=4, help="негативов на запрос в группе")
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--accum", type=int, default=2, help="шагов накопления градиента")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--hard-fraction", type=float, default=1.0,
                        help="доля сложных групп (orta, zor) в обучении: чтобы измерить, "
                             "растёт ли качество на них от добавления таких же данных")
    parser.add_argument("--train-levels", nargs="+", default=list(LEVELS),
                        help="какие уровни оставить в обучении: лёгкие сделаны старым "
                             "промптом и могут тянуть модель к дословному совпадению")
    parser.add_argument("--lora-targets", nargs="+", default=["query", "key", "value"],
                        help="какие матрицы обучать")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=0,
                        help="складывать чекпоинт каждые N шагов: валидация построена "
                             "той же конструкцией, что и обучение, поэтому лучшую точку "
                             "по ней приходится перепроверять замером на эталоне")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=50,
                        help="как часто мерить валидацию — по ней строятся графики")
    parser.add_argument("--patience", type=int, default=6,
                        help="сколько замеров подряд без улучшения терпеть до остановки")
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    groups = load_groups(args.pairs, args.negatives)
    train_groups, val_groups = split_by_page(groups, args.val_fraction, args.seed)
    # Прореживаем только обучающую часть и только сложные уровни: валидация
    # обязана остаться прежней, иначе точки кривой «данные -> качество»
    # окажутся посчитаны на разных наборах и сравнивать их будет нельзя.
    if args.hard_fraction < 1.0:
        hard = [g for g in train_groups if g["difficulty"] in ("orta", "zor")]
        random.Random(args.seed).shuffle(hard)
        keep = {id(g) for g in hard[: int(len(hard) * args.hard_fraction)]}
        train_groups = [g for g in train_groups
                        if g["difficulty"] not in ("orta", "zor") or id(g) in keep]
        print(f"сложных групп оставлено {len(keep)} из {len(hard)}", flush=True)
    if set(args.train_levels) != set(LEVELS):
        before = len(train_groups)
        train_groups = [g for g in train_groups if g["difficulty"] in set(args.train_levels)]
        print(f"уровни в обучении {args.train_levels}: {len(train_groups)} из {before}", flush=True)
    sizes = [len(g["passages"]) for g in groups]
    print(f"групп всего {len(groups)} | обучение {len(train_groups)} | валидация {len(val_groups)}",
          flush=True)
    print(f"размер группы: медиана {sorted(sizes)[len(sizes) // 2]} | мин {min(sizes)} "
          f"| макс {max(sizes)}", flush=True)
    print(f"страниц в валидации: {len({g['page'] for g in val_groups})}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        task_type="SEQ_CLS", bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)

    steps_per_epoch = math.ceil(len(train_groups) / args.groups_per_step)
    total_steps = int(steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    # Разогрев, затем косинусное затухание до конца горизонта. Постоянный lr
    # пробовали — модель топчется вокруг оптимума, и хвост кривой рваный.
    # Ранняя остановка остаётся страховкой, но горизонт теперь достижимый.
    def lr_at(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    # Volta без bfloat16 — только fp16, а значит нужен масштаб градиента.
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    def evaluate():
        """Метрики на валидации целиком и отдельно по каждому уровню сложности."""
        model.eval()
        hits = reciprocal = 0
        losses = []
        by_level = {level: [0, 0.0] for level in LEVELS}
        with torch.no_grad():
            for start in range(0, len(val_groups), args.groups_per_step):
                chunk = val_groups[start: start + args.groups_per_step]
                if not chunk:
                    continue
                sizes = [len(g["passages"]) for g in chunk]
                batch = encode(tokenizer, chunk, device, args.max_length)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    logits = model(**batch).logits.float()
                losses.append(group_loss(logits, sizes, torch).item())
                for group, rank in zip(chunk, positive_ranks(logits, sizes)):
                    hits += 1 if rank == 1 else 0
                    reciprocal += 1.0 / rank
                    bucket = by_level.setdefault(group["difficulty"], [0, 0.0])
                    bucket[0] += 1 if rank == 1 else 0
                    bucket[1] += 1
        model.train()
        n = max(len(val_groups), 1)
        levels = {level: (correct / total if total else None)
                  for level, (correct, total) in by_level.items()}
        return hits / n, reciprocal / n, (sum(losses) / len(losses) if losses else 0.0), levels

    history_path = args.output / "history.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")

    def record(step, train_loss, val, grad_norm=None):
        hit_, mrr_, loss_, levels_ = val
        with history_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({
                "step": step,
                "epoch": round(step / max(steps_per_epoch, 1), 3),
                "minutes": round((time.time() - launched) / 60, 2),
                "train_loss": train_loss,
                "val_loss": loss_,
                "val_hit": hit_,
                "val_mrr": mrr_,
                "val_hit_by_level": levels_,
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
            }, ensure_ascii=False) + "\n")

    launched = time.time()
    hit, mrr, val_loss, levels = evaluate()
    print(f"до обучения: позитив первым {hit * 100:.1f}% | MRR {mrr * 100:.1f} "
          f"| val loss {val_loss:.4f}", flush=True)
    print("  по сложности: " + " | ".join(
        f"{k} {v * 100:.1f}%" for k, v in levels.items() if v is not None), flush=True)

    record(0, float("nan"), (hit, mrr, val_loss, levels))
    best = {"hit": hit, "mrr": mrr, "step": 0}
    rng = random.Random(args.seed)
    order = list(range(len(train_groups)))
    started, step, running = time.time(), 0, 0.0
    since_eval = []
    stale, stop, last_grad_norm = 0, False, None
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps and not stop:
        rng.shuffle(order)
        for start in range(0, len(order), args.groups_per_step):
            if step >= total_steps:
                break
            chunk = [train_groups[i] for i in order[start: start + args.groups_per_step]]
            if not chunk:
                continue
            sizes = [len(g["passages"]) for g in chunk]
            batch = encode(tokenizer, chunk, device, args.max_length)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(**batch).logits.float()
            loss = group_loss(logits, sizes, torch) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            since_eval.append(loss.item() * args.accum)

            if (step + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                # Норму сохраняем: по её всплескам видно переполнение fp16 на Volta.
                last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            step += 1

            if step % args.log_every == 0:
                minutes = (time.time() - started) / 60
                print(f"  шаг {step}/{total_steps} | loss {running / args.log_every:.4f} "
                      f"| {minutes:.1f} мин", flush=True)
                running = 0.0

            if step % args.eval_every == 0:
                train_loss = sum(since_eval) / len(since_eval) if since_eval else float("nan")
                since_eval = []
                val = evaluate()
                record(step, train_loss, val, last_grad_norm)
                # Лучшая точка сохраняется отдельно: если валидация развернётся
                # вверх, откатываться будет куда, а переучивать заново не придётся.
                marker = ""
                if val[0] > best["hit"]:
                    best.update(hit=val[0], mrr=val[1], step=step)
                    model.save_pretrained(args.output / "best")
                    marker = "  <- лучшая"
                    stale = 0
                else:
                    stale += 1
                    marker = f"  (без улучшения {stale}/{args.patience})"
                print(f"  [валидация] шаг {step} | train {train_loss:.4f} | val {val[2]:.4f} "
                      f"| позитив первым {val[0] * 100:.1f}% | MRR {val[1] * 100:.1f}{marker}",
                      flush=True)
                print("     по сложности: " + " | ".join(
                    f"{k} {v * 100:.1f}%" for k, v in val[3].items() if v is not None), flush=True)
                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(args.output / f"step{step}")
                if stale >= args.patience:
                    print(f"ранняя остановка: {args.patience} замеров без улучшения, "
                          f"лучшее было на шаге {best['step']}", flush=True)
                    stop = True
                    break

    hit_after, mrr_after, val_loss_after, levels_after = evaluate()
    # Именно step, а не total_steps: при ранней остановке график иначе тянул бы
    # плоскую линию до несостоявшегося конца обучения.
    record(step, float("nan"), (hit_after, mrr_after, val_loss_after, levels_after))
    print(f"после обучения: позитив первым {hit_after * 100:.1f}% "
          f"(было {hit * 100:.1f}) | MRR {mrr_after * 100:.1f} (было {mrr * 100:.1f})", flush=True)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "training_summary.json").write_text(json.dumps({
        "base_model": args.base_model,
        "groups_train": len(train_groups),
        "groups_val": len(val_groups),
        "negatives": args.negatives,
        "steps": total_steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "val_hit_before": hit, "val_hit_after": hit_after,
        "val_mrr_before": mrr, "val_mrr_after": mrr_after,
        "best_hit": best["hit"], "best_mrr": best["mrr"], "best_step": best["step"],
        "minutes": round((time.time() - started) / 60, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"адаптер сохранён: {args.output}", flush=True)
    print(f"лучшая точка: шаг {best['step']} | позитив первым {best['hit'] * 100:.1f}% "
          f"-> {args.output / 'best'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
