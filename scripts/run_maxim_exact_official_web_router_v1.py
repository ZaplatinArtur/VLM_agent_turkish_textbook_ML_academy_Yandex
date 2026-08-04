#!/usr/bin/env python3
"""Gold-blind exact-official-web router for the frozen public 274-task set.

The router process accepts only a hash-pinned public task queue, a hash-pinned
gold-blind OCR artifact, and a hash-pinned policy profile.  It never accepts a
benchmark, reference, judge, score, fallback, or candidate solver file.

The output contains selective *decisions*, not a composed solver.  A separate
process may later apply accepted certificates to an immutable default solver.
This separation prevents a default answer from influencing a search query or
certificate.

Network access is disabled unless ``--enable-network`` is supplied explicitly.
Without that switch a non-dry run is an offline cache replay and every cache
miss fails closed.  ``--dry-run`` performs no cache or network writes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import io
import json
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


ROUTER_SCHEMA = "maxim-exact-official-web-router-v1"
PROFILE_SCHEMA = "maxim-exact-official-web-router-profile-v1"
QUERY_PLAN_SCHEMA = "maxim-exact-official-web-query-plan-v1"
DECISION_SCHEMA = "maxim-exact-official-web-decision-v1"
CERTIFICATE_SCHEMA = "maxim-exact-official-web-certificate-v1"
SEARCH_CACHE_SCHEMA = "maxim-exact-official-web-search-cache-v1"
DOCUMENT_CACHE_SCHEMA = "maxim-exact-official-web-document-cache-v1"

PUBLIC_TASK_KEYS = {
    "task_id",
    "subject",
    "grade",
    "question",
    "question_images",
    "answer_type",
}
PUBLIC_IMAGE_KEYS = {
    "image_id",
    "format",
    "data",
    "mime_type",
    "caption",
}

# These keys are answer/evaluation-bearing rather than harmless attestations.
FORBIDDEN_DATA_KEYS = {
    "reference_answer",
    "reference_solution",
    "reference_image",
    "reference_image_url",
    "gold_answer",
    "gold_solution",
    "judge",
    "judge_verdict",
    "verdict",
    "score",
    "scores",
    "correct",
    "correctness",
    "outcome",
    "outcomes",
    "task_outcomes",
    "candidate",
    "candidates",
    "fallback",
    "primary_fallback",
    "secondary_fallback",
    "final_answer",
}
FORBIDDEN_KEY_FRAGMENTS = (
    "reference_answer",
    "reference_solution",
    "gold_answer",
    "gold_solution",
    "judge_verdict",
    "task_outcome",
)
FALSE_ATTESTATION_SUFFIXES = (
    "gold_access",
    "reference_access",
    "judge_access",
    "score_access",
    "outcome_access",
    "oracle_access",
)
FORBIDDEN_INPUT_BASENAME_RE = re.compile(
    r"(?:reference|gold|judge|score|outcome|benchmark|candidate|fallback|solver)",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
QUESTION_NUMBER_RE = re.compile(r"(?:^|\n)\s*(\d{1,3})\s*[.)]", re.MULTILINE)
OPTION_PREFIX_RE = re.compile(r"^\s*[A-E]\s*[.)]\s*", re.IGNORECASE)
PDF_STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
PDF_LITERAL_RE = re.compile(rb"\(((?:\\.|[^\\()])*)\)")
PDF_HEX_RE = re.compile(rb"<([0-9A-Fa-f\s]+)>")


class RouterInputError(ValueError):
    """A public input, profile, or cache violated the frozen contract."""


class NetworkDisabledError(RuntimeError):
    """A cache miss occurred while explicit network access was disabled."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _atomic_write_bytes(path, payload)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    _atomic_write_bytes(path, payload)


def guard_input_path(path: Path, *, role: str, suffixes: set[str]) -> Path:
    """Reject common evaluation-bearing filenames before opening an input."""

    if not path.is_file():
        raise RouterInputError(f"{role}: input is not a file: {path}")
    if path.is_symlink():
        raise RouterInputError(f"{role}: symbolic links are not allowed: {path}")
    if path.suffix.casefold() not in suffixes:
        raise RouterInputError(f"{role}: unexpected suffix: {path.suffix}")
    if FORBIDDEN_INPUT_BASENAME_RE.search(path.name):
        raise RouterInputError(f"{role}: forbidden input filename: {path.name}")
    return path.resolve()


def guard_output_paths(
    *,
    output: Path,
    manifest: Path,
    cache_dir: Path,
    protected_inputs: Sequence[Path],
) -> None:
    protected = {path.resolve() for path in protected_inputs}
    for role, path in (("output", output), ("manifest", manifest)):
        if path.resolve() in protected:
            raise RouterInputError(f"{role}: refusing to overwrite an input")
        if FORBIDDEN_INPUT_BASENAME_RE.search(path.name):
            raise RouterInputError(f"{role}: forbidden evaluation-bearing filename")
        if path.exists():
            raise RouterInputError(f"{role}: refusing to overwrite existing path: {path}")
    if cache_dir.resolve() in protected:
        raise RouterInputError("cache directory aliases an input")
    if FORBIDDEN_INPUT_BASENAME_RE.search(cache_dir.name):
        raise RouterInputError("cache directory has an evaluation-bearing name")


def assert_gold_blind_input(value: Any, *, location: str) -> None:
    """Recursively reject answer/evaluation data and validate false attestations."""

    stack: list[tuple[str, Any]] = [(location, value)]
    while stack:
        current_location, current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                folded = str(key).casefold()
                child_location = f"{current_location}.{key}"
                if folded in FORBIDDEN_DATA_KEYS or any(
                    fragment in folded for fragment in FORBIDDEN_KEY_FRAGMENTS
                ):
                    raise RouterInputError(f"{child_location}: forbidden key")
                if folded.endswith(FALSE_ATTESTATION_SUFFIXES) and child is not False:
                    raise RouterInputError(f"{child_location}: access attestation must be false")
                stack.append((child_location, child))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                stack.append((f"{current_location}[{index}]", child))


def _read_json(path: Path, *, role: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected_sha256.casefold():
        raise RouterInputError(f"{role}: SHA256 mismatch: {actual}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouterInputError(f"{role}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RouterInputError(f"{role}: expected a JSON object")
    assert_gold_blind_input(value, location=role)
    return value, actual


def _read_jsonl(path: Path, *, role: str, expected_sha256: str) -> tuple[list[dict[str, Any]], str]:
    actual = sha256_file(path)
    if actual != expected_sha256.casefold():
        raise RouterInputError(f"{role}: SHA256 mismatch: {actual}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RouterInputError(f"{role}:{line_number}: expected object")
                assert_gold_blind_input(value, location=f"{role}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RouterInputError(f"{role}: invalid JSONL: {exc}") from exc
    return rows, actual


def validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise RouterInputError("profile: unsupported schema_version")
    if profile.get("status") != "FROZEN_BEFORE_NETWORK_OR_EVALUATION_ACCESS":
        raise RouterInputError("profile: status is not frozen")
    expected_rows = profile.get("expected_rows")
    if not isinstance(expected_rows, int) or expected_rows < 1:
        raise RouterInputError("profile: expected_rows must be a positive integer")

    query = profile.get("query")
    if not isinstance(query, dict):
        raise RouterInputError("profile: missing query section")
    minimum = query.get("min_queries")
    maximum = query.get("max_queries")
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise RouterInputError("profile: query bounds must be integers")
    if not (2 <= minimum <= maximum <= 4):
        raise RouterInputError("profile: query bounds must satisfy 2 <= min <= max <= 4")
    if not 4 <= int(query.get("phrase_min_tokens") or 0) <= 30:
        raise RouterInputError("profile: invalid phrase_min_tokens")
    if not int(query.get("phrase_min_tokens")) <= int(query.get("phrase_max_tokens") or 0) <= 40:
        raise RouterInputError("profile: invalid phrase_max_tokens")
    if not 80 <= int(query.get("max_query_chars") or 0) <= 500:
        raise RouterInputError("profile: invalid max_query_chars")

    network = profile.get("network")
    if not isinstance(network, dict):
        raise RouterInputError("profile: missing network section")
    if network.get("explicit_opt_in_required") is not True:
        raise RouterInputError("profile: network opt-in must be required")
    if not 1 <= int(network.get("search_results_per_query") or 0) <= 20:
        raise RouterInputError("profile: invalid search_results_per_query")
    if not 1 <= int(network.get("max_fetches_per_task") or 0) <= 16:
        raise RouterInputError("profile: invalid max_fetches_per_task")
    if not 1024 <= int(network.get("max_document_bytes") or 0) <= 50_000_000:
        raise RouterInputError("profile: invalid max_document_bytes")

    authority = profile.get("authority")
    if not isinstance(authority, dict):
        raise RouterInputError("profile: missing authority section")
    suffixes = authority.get("official_host_suffixes")
    schemes = authority.get("allowed_schemes")
    if not isinstance(suffixes, list) or not suffixes:
        raise RouterInputError("profile: official_host_suffixes must be nonempty")
    if not isinstance(schemes, list) or not schemes:
        raise RouterInputError("profile: allowed_schemes must be nonempty")
    for suffix in suffixes:
        text = str(suffix).casefold().strip(".")
        if not re.fullmatch(r"[a-z0-9.-]+", text) or ".." in text:
            raise RouterInputError(f"profile: invalid official host suffix {suffix!r}")
    if not set(map(str.casefold, schemes)).issubset({"http", "https"}):
        raise RouterInputError("profile: unsupported URL scheme")

    certificate = profile.get("certificate")
    if not isinstance(certificate, dict):
        raise RouterInputError("profile: missing certificate section")
    for key in (
        "min_fingerprint_token_coverage",
        "min_full_token_coverage",
    ):
        value = certificate.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise RouterInputError(f"profile: invalid certificate threshold {key}")
    if not 1 <= int(certificate.get("min_phrase_matches") or 0) <= 4:
        raise RouterInputError("profile: invalid min_phrase_matches")


def validate_public_queue(rows: Sequence[Mapping[str, Any]], *, expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise RouterInputError(f"public queue: expected {expected_rows} rows, got {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != PUBLIC_TASK_KEYS:
            raise RouterInputError(f"public queue:{index}: exact public task schema mismatch")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            raise RouterInputError(f"public queue:{index}: missing or duplicate task_id")
        seen.add(task_id)
        if not isinstance(row.get("question"), str):
            raise RouterInputError(f"public queue:{index}: question must be a string")
        if not isinstance(row.get("answer_type"), str) or not row.get("answer_type"):
            raise RouterInputError(f"public queue:{index}: invalid answer_type")
        images = row.get("question_images")
        if not isinstance(images, list) or not images:
            raise RouterInputError(f"public queue:{index}: no question_images")
        for image_index, image in enumerate(images):
            if not isinstance(image, dict) or set(image) - PUBLIC_IMAGE_KEYS:
                raise RouterInputError(
                    f"public queue:{index}: image {image_index} schema mismatch"
                )
            data = str(image.get("data") or "")
            candidate = Path(data)
            if not data or candidate.is_absolute() or ".." in candidate.parts:
                raise RouterInputError(
                    f"public queue:{index}: image {image_index} path is not a safe relative path"
                )


def index_ocr_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_task_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        if row.get("schema_version") != "maxim-paddleocr-vl16-task-parse-v1":
            raise RouterInputError(f"ocr:{index}: unexpected schema_version")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in indexed:
            raise RouterInputError(f"ocr:{index}: missing or duplicate task_id")
        parser = row.get("parser")
        if not isinstance(parser, dict) or parser.get("gold_access") is not False:
            raise RouterInputError(f"ocr:{index}: parser.gold_access must be false")
        images = row.get("images")
        if not isinstance(images, list) or not images:
            raise RouterInputError(f"ocr:{index}: no parsed images")
        indexed[task_id] = row
    if set(indexed) != set(expected_task_ids):
        raise RouterInputError("ocr: task-id set differs from public queue")
    return indexed


class _VisibleHTMLText(HTMLParser):
    BLOCK_TAGS = {
        "br",
        "div",
        "p",
        "li",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        folded = tag.casefold()
        if folded in self.SKIP_TAGS:
            self.skip_depth += 1
        elif not self.skip_depth and folded in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        folded = tag.casefold()
        if folded in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and folded in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(self.parts)


def clean_visible_text(value: str) -> str:
    parser = _VisibleHTMLText()
    try:
        parser.feed(value)
        visible = parser.text()
    except Exception:
        visible = value
    visible = html.unescape(visible)
    visible = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", visible)
    visible = re.sub(r"\s+", " ", visible)
    return visible.strip()


def ocr_text(parser_row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    pieces: list[str] = []
    block_count = 0
    block_hashes: list[str] = []
    for parsed_image in parser_row.get("images") or []:
        if not isinstance(parsed_image, dict):
            continue
        blocks = parsed_image.get("parsing_res_list") or []
        ordered = sorted(
            enumerate(blocks),
            key=lambda pair: (
                int(pair[1].get("block_order") or pair[0])
                if isinstance(pair[1], dict)
                else pair[0],
                int(pair[1].get("block_id") or pair[0])
                if isinstance(pair[1], dict)
                else pair[0],
            ),
        )
        for _, block in ordered:
            if not isinstance(block, dict):
                continue
            content = clean_visible_text(str(block.get("block_content") or ""))
            if not content:
                continue
            pieces.append(content)
            block_hashes.append(hashlib.sha256(content.encode("utf-8")).hexdigest())
            block_count += 1
    text = "\n".join(pieces).strip()
    return text, {
        "parser_row_sha256": canonical_sha256(parser_row),
        "visible_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "block_count": block_count,
        "block_sha256": block_hashes,
        "visible_characters": len(text),
    }


_TURKISH_FOLD = str.maketrans(
    {
        "ı": "i",
        "ş": "s",
        "ğ": "g",
        "ü": "u",
        "ö": "o",
        "ç": "c",
    }
)


def fold_for_match(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold()).translate(_TURKISH_FOLD)
    folded = "".join(character for character in folded if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def match_tokens(value: str) -> list[str]:
    return [token for token in fold_for_match(value).split() if len(token) >= 2]


def _phrase_windows(text: str, *, minimum: int, maximum: int) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return []
    if len(tokens) <= maximum:
        return [" ".join(tokens)] if len(tokens) >= minimum else []
    starts = [0, max(0, len(tokens) // 2 - maximum // 2), len(tokens) - maximum]
    return [" ".join(tokens[start : start + maximum]) for start in dict.fromkeys(starts)]


def distinctive_phrases(text: str, query_profile: Mapping[str, Any]) -> list[str]:
    minimum = int(query_profile["phrase_min_tokens"])
    maximum = int(query_profile["phrase_max_tokens"])
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    segments = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", text)
    for segment_index, raw_segment in enumerate(segments):
        segment = OPTION_PREFIX_RE.sub("", raw_segment).strip(" #-\t")
        for phrase in _phrase_windows(segment, minimum=minimum, maximum=maximum):
            tokens = match_tokens(phrase)
            if not tokens:
                continue
            unique = len(set(tokens))
            informative = sum(len(token) >= 5 for token in tokens)
            digits = sum(any(character.isdigit() for character in token) for token in tokens)
            score = (informative, unique, digits, -segment_index)
            candidates.append((score, phrase))
    if not candidates:
        tokens = TOKEN_RE.findall(text)
        if tokens:
            for start in (0, max(0, len(tokens) // 2 - maximum // 2)):
                phrase = " ".join(tokens[start : start + maximum])
                if phrase:
                    candidates.append(((0, len(set(match_tokens(phrase))), 0, -start), phrase))
    output: list[str] = []
    seen: set[str] = set()
    for _, phrase in sorted(candidates, key=lambda item: item[0], reverse=True):
        cleaned = re.sub(r"[\"“”]+", " ", phrase)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        key = fold_for_match(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= int(query_profile["max_queries"]):
            break
    return output


def _bounded_quoted_query(phrase: str, suffix: str | None, *, max_chars: int) -> str:
    tokens = TOKEN_RE.findall(phrase)
    suffix_text = f' "{suffix}"' if suffix else ""
    while tokens:
        query = f'"{" ".join(tokens)}"{suffix_text}'
        if len(query) <= max_chars:
            return query
        tokens.pop()
    raise RouterInputError("query planner could not fit a quoted phrase")


def build_query_plan(
    task: Mapping[str, Any], parser_row: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    query_profile = profile["query"]
    answer_type = str(task.get("answer_type") or "").casefold()
    eligible_types = {str(value).casefold() for value in profile["eligible_answer_types"]}
    text, parser_meta = ocr_text(parser_row)
    number_match = QUESTION_NUMBER_RE.search(text)
    question_number = int(number_match.group(1)) if number_match else None
    plan: dict[str, Any] = {
        "schema_version": QUERY_PLAN_SCHEMA,
        "task_id": str(task["task_id"]),
        "eligible": answer_type in eligible_types,
        "ineligibility_reasons": [],
        "answer_type": answer_type,
        "question_number": question_number,
        "ocr": parser_meta,
        "fingerprint": {
            "full_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "full_token_sha256": hashlib.sha256(
                " ".join(match_tokens(text)).encode("utf-8")
            ).hexdigest(),
            "full_token_count": len(match_tokens(text)),
        },
        "phrases": [],
        "queries": [],
    }
    if answer_type not in eligible_types:
        plan["ineligibility_reasons"].append("answer_type_not_enabled")
        plan["plan_sha256"] = canonical_sha256(plan)
        return plan

    phrases = distinctive_phrases(text, query_profile)
    if not phrases:
        plan["eligible"] = False
        plan["ineligibility_reasons"].append("no_usable_ocr_phrase")
        plan["plan_sha256"] = canonical_sha256(plan)
        return plan

    minimum = int(query_profile["min_queries"])
    maximum = int(query_profile["max_queries"])
    max_chars = int(query_profile["max_query_chars"])
    queries: list[str] = []
    for phrase in phrases:
        queries.append(_bounded_quoted_query(phrase, None, max_chars=max_chars))
        if len(queries) >= maximum:
            break
    suffixes = [str(value) for value in query_profile.get("key_query_terms") or []]
    suffix_index = 0
    while len(queries) < minimum:
        suffix = suffixes[suffix_index % len(suffixes)] if suffixes else "cevap anahtarı"
        suffix_index += 1
        candidate = _bounded_quoted_query(phrases[0], suffix, max_chars=max_chars)
        if candidate not in queries:
            queries.append(candidate)
        elif suffix_index > len(suffixes) + 2:
            raise RouterInputError("query planner could not produce distinct bounded queries")
    plan["phrases"] = phrases[:maximum]
    plan["queries"] = [
        {
            "ordinal": index,
            "query": query,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        }
        for index, query in enumerate(queries[:maximum])
    ]
    validate_query_plan(plan, profile)
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_query_plan(plan: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    if not plan.get("eligible"):
        if plan.get("queries"):
            raise RouterInputError("ineligible query plan contains queries")
        return
    query_profile = profile["query"]
    queries = plan.get("queries")
    if not isinstance(queries, list):
        raise RouterInputError("query plan has no query list")
    if not int(query_profile["min_queries"]) <= len(queries) <= int(query_profile["max_queries"]):
        raise RouterInputError("query plan violates bounded query count")
    for item in queries:
        query = str(item.get("query") or "") if isinstance(item, dict) else ""
        if len(query) > int(query_profile["max_query_chars"]):
            raise RouterInputError("query exceeds max_query_chars")
        if query.count('"') < 2 or query.count('"') % 2:
            raise RouterInputError("every query must contain balanced quoted text")


class OfficialURLPolicy:
    def __init__(self, *, host_suffixes: Sequence[str], schemes: Sequence[str]):
        self.host_suffixes = tuple(
            sorted({str(value).casefold().strip(".") for value in host_suffixes})
        )
        self.schemes = frozenset(str(value).casefold() for value in schemes)

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any]) -> "OfficialURLPolicy":
        authority = profile["authority"]
        return cls(
            host_suffixes=authority["official_host_suffixes"],
            schemes=authority["allowed_schemes"],
        )

    def is_allowed_host(self, host: str | None) -> bool:
        folded = str(host or "").casefold().rstrip(".")
        return any(
            folded == suffix or folded.endswith("." + suffix)
            for suffix in self.host_suffixes
        )

    def is_allowed_url(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
        except ValueError:
            return False
        return (
            parsed.scheme.casefold() in self.schemes
            and parsed.username is None
            and parsed.password is None
            and self.is_allowed_host(parsed.hostname)
        )

    def require(self, url: str, *, location: str) -> None:
        if not self.is_allowed_url(url):
            raise RouterInputError(f"{location}: URL is outside official allowlist: {url}")

    def family(self, url: str) -> str:
        self.require(url, location="document family")
        parsed = urllib.parse.urlparse(url)
        path = re.sub(
            r"/files/basic-html/page\d+\.html?$",
            "/files/basic-html",
            parsed.path,
            flags=re.IGNORECASE,
        )
        if path == parsed.path and not path.casefold().endswith(".pdf"):
            page_match = re.search(r"/page\d+\.html?$", path, flags=re.IGNORECASE)
            if page_match:
                path = path[: page_match.start()]
        return urllib.parse.urlunparse(
            (parsed.scheme.casefold(), parsed.netloc.casefold(), path, "", "", "")
        )


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""


@dataclass(frozen=True)
class SearchResponse:
    hits: tuple[SearchHit, ...]
    provenance: dict[str, Any]


class SearchClient(Protocol):
    network_calls: int

    def search(self, query: str, *, limit: int) -> SearchResponse: ...


class SearxngSearchClient:
    def __init__(
        self,
        *,
        base_url: str,
        cache_dir: Path,
        timeout_s: float,
        language: str,
        network_enabled: bool,
        user_agent: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout_s = timeout_s
        self.language = language
        self.network_enabled = network_enabled
        self.user_agent = user_agent
        self.network_calls = 0

    def _paths(self, query: str, limit: int) -> tuple[str, Path, Path]:
        binding = {
            "base_url": self.base_url,
            "query": query,
            "limit": limit,
            "language": self.language,
        }
        key = canonical_sha256(binding)
        return key, self.cache_dir / f"{key}.search.raw", self.cache_dir / f"{key}.search.json"

    @staticmethod
    def _hits(raw: bytes, limit: int) -> tuple[SearchHit, ...]:
        value = json.loads(raw.decode("utf-8"))
        results = value.get("results") if isinstance(value, dict) else None
        if not isinstance(results, list):
            raise RouterInputError("SearXNG response has no results list")
        hits: list[SearchHit] = []
        for item in results[:limit]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title") or "").strip(),
                    url=url,
                    snippet=str(item.get("content") or "").strip(),
                    engine=str(item.get("engine") or "").strip(),
                )
            )
        return tuple(hits)

    def search(self, query: str, *, limit: int) -> SearchResponse:
        key, raw_path, meta_path = self._paths(query, limit)
        if raw_path.is_file() and meta_path.is_file():
            raw = raw_path.read_bytes()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("schema_version") != SEARCH_CACHE_SCHEMA:
                raise RouterInputError("search cache schema mismatch")
            if meta.get("cache_key") != key or meta.get("raw_sha256") != sha256_bytes(raw):
                raise RouterInputError("search cache binding/hash mismatch")
            return SearchResponse(
                self._hits(raw, limit),
                {**meta, "cache_hit": True},
            )
        if not self.network_enabled:
            raise NetworkDisabledError("search cache miss with network disabled")
        endpoint = self.base_url + "/search?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "language": self.language}
        )
        request = urllib.request.Request(endpoint, headers={"User-Agent": self.user_agent})
        self.network_calls += 1
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = response.read()
            final_url = response.geturl()
            status = getattr(response, "status", 200)
        hits = self._hits(raw, limit)
        meta = {
            "schema_version": SEARCH_CACHE_SCHEMA,
            "cache_key": key,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "endpoint": endpoint,
            "final_url": final_url,
            "http_status": status,
            "raw_sha256": sha256_bytes(raw),
            "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "result_count": len(hits),
        }
        _atomic_write_bytes(raw_path, raw)
        _atomic_write_json(meta_path, meta)
        return SearchResponse(hits, {**meta, "cache_hit": False})


class _OfficialRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, policy: OfficialURLPolicy):
        super().__init__()
        self.policy = policy

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urllib.parse.urljoin(req.full_url, newurl)
        self.policy.require(target, location="HTTP redirect")
        return super().redirect_request(req, fp, code, msg, headers, target)


@dataclass(frozen=True)
class FetchedDocument:
    requested_url: str
    final_url: str
    content_type: str
    text: str
    extraction_method: str
    raw_sha256: str
    text_sha256: str
    provenance: dict[str, Any]


def _decode_pdf_string(value: bytes) -> str:
    output = bytearray()
    index = 0
    escapes = {
        ord("n"): ord("\n"),
        ord("r"): ord("\r"),
        ord("t"): ord("\t"),
        ord("b"): ord("\b"),
        ord("f"): ord("\f"),
        ord("("): ord("("),
        ord(")"): ord(")"),
        ord("\\"): ord("\\"),
    }
    while index < len(value):
        byte = value[index]
        if byte != ord("\\"):
            output.append(byte)
            index += 1
            continue
        index += 1
        if index >= len(value):
            break
        escaped = value[index]
        if escaped in escapes:
            output.append(escapes[escaped])
            index += 1
            continue
        if ord("0") <= escaped <= ord("7"):
            digits = bytes([escaped])
            index += 1
            while index < len(value) and len(digits) < 3 and ord("0") <= value[index] <= ord("7"):
                digits += bytes([value[index]])
                index += 1
            output.append(int(digits, 8) & 0xFF)
            continue
        if escaped in (ord("\r"), ord("\n")):
            if escaped == ord("\r") and index + 1 < len(value) and value[index + 1] == ord("\n"):
                index += 1
            index += 1
            continue
        output.append(escaped)
        index += 1
    raw = bytes(output)
    if raw.startswith((b"\xfe\xff", b"\xff\xfe")):
        return raw.decode("utf-16", errors="replace")
    for encoding in ("utf-8", "cp1254", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _stdlib_pdf_text(raw: bytes) -> str:
    streams: list[bytes] = []
    for match in PDF_STREAM_RE.finditer(raw):
        stream = match.group(1)
        prefix = raw[max(0, match.start() - 512) : match.start()]
        if b"FlateDecode" in prefix:
            try:
                stream = zlib.decompress(stream)
            except zlib.error:
                continue
        streams.append(stream)
    if not streams:
        streams = [raw]
    pieces: list[str] = []
    for stream in streams:
        for literal in PDF_LITERAL_RE.finditer(stream):
            tail = stream[literal.end() : literal.end() + 64]
            if b"Tj" in tail or b"TJ" in tail or b"'" in tail or b'"' in tail:
                pieces.append(_decode_pdf_string(literal.group(1)))
        for encoded in PDF_HEX_RE.finditer(stream):
            tail = stream[encoded.end() : encoded.end() + 64]
            if b"Tj" not in tail and b"TJ" not in tail:
                continue
            compact = re.sub(rb"\s+", b"", encoded.group(1))
            if len(compact) % 2:
                compact += b"0"
            try:
                pieces.append(_decode_pdf_string(bytes.fromhex(compact.decode("ascii"))))
            except (ValueError, UnicodeDecodeError):
                continue
    return clean_visible_text(" ".join(pieces))


def extract_pdf_text(raw: bytes) -> tuple[str, str]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(io.BytesIO(raw))
        text = clean_visible_text("\n".join((page.extract_text() or "") for page in reader.pages))
        if text:
            return text, "pypdf"
    except Exception:
        pass
    return _stdlib_pdf_text(raw), "stdlib-pdf-stream-text-v1"


def _charset(content_type: str) -> str:
    match = re.search(r"charset\s*=\s*[\"']?([^;\"'\s]+)", content_type, re.IGNORECASE)
    return match.group(1) if match else "utf-8"


def extract_document_text(raw: bytes, *, content_type: str, url: str) -> tuple[str, str]:
    folded_type = content_type.casefold()
    if "pdf" in folded_type or url.casefold().split("?", 1)[0].endswith(".pdf") or raw.startswith(b"%PDF-"):
        return extract_pdf_text(raw)
    decoded = raw.decode(_charset(content_type), errors="replace")
    if "html" in folded_type or "<html" in decoded[:1000].casefold():
        return clean_visible_text(decoded), "stdlib-html-visible-text-v1"
    return clean_visible_text(decoded), "plain-text-v1"


class OfficialDocumentFetcher:
    def __init__(
        self,
        *,
        policy: OfficialURLPolicy,
        cache_dir: Path,
        timeout_s: float,
        max_bytes: int,
        network_enabled: bool,
        user_agent: str,
    ) -> None:
        self.policy = policy
        self.cache_dir = cache_dir
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.network_enabled = network_enabled
        self.user_agent = user_agent
        self.network_calls = 0
        self.opener = urllib.request.build_opener(_OfficialRedirectHandler(policy))

    def _paths(self, url: str) -> tuple[str, Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return key, self.cache_dir / f"{key}.document.bin", self.cache_dir / f"{key}.document.json"

    def _from_cache(self, *, url: str, key: str, raw_path: Path, meta_path: Path) -> FetchedDocument:
        raw = raw_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != DOCUMENT_CACHE_SCHEMA:
            raise RouterInputError("document cache schema mismatch")
        if meta.get("cache_key") != key or meta.get("requested_url") != url:
            raise RouterInputError("document cache URL binding mismatch")
        if meta.get("raw_sha256") != sha256_bytes(raw):
            raise RouterInputError("document cache raw SHA mismatch")
        final_url = str(meta.get("final_url") or "")
        self.policy.require(final_url, location="cached final URL")
        text, method = extract_document_text(
            raw,
            content_type=str(meta.get("content_type") or ""),
            url=final_url,
        )
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if meta.get("text_sha256") != text_sha or meta.get("extraction_method") != method:
            raise RouterInputError("document cache extracted-text binding mismatch")
        return FetchedDocument(
            requested_url=url,
            final_url=final_url,
            content_type=str(meta.get("content_type") or ""),
            text=text,
            extraction_method=method,
            raw_sha256=sha256_bytes(raw),
            text_sha256=text_sha,
            provenance={**meta, "cache_hit": True},
        )

    def fetch(self, url: str) -> FetchedDocument:
        self.policy.require(url, location="requested document")
        key, raw_path, meta_path = self._paths(url)
        if raw_path.is_file() and meta_path.is_file():
            return self._from_cache(url=url, key=key, raw_path=raw_path, meta_path=meta_path)
        if not self.network_enabled:
            raise NetworkDisabledError("document cache miss with network disabled")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"},
        )
        self.network_calls += 1
        with self.opener.open(request, timeout=self.timeout_s) as response:
            raw = response.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes:
                raise RouterInputError("official document exceeds max_document_bytes")
            final_url = response.geturl()
            status = getattr(response, "status", 200)
            content_type = str(response.headers.get("Content-Type") or "")
        self.policy.require(final_url, location="final redirect URL")
        text, method = extract_document_text(raw, content_type=content_type, url=final_url)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        meta = {
            "schema_version": DOCUMENT_CACHE_SCHEMA,
            "cache_key": key,
            "requested_url": url,
            "final_url": final_url,
            "http_status": status,
            "content_type": content_type,
            "raw_sha256": sha256_bytes(raw),
            "raw_bytes": len(raw),
            "text_sha256": text_sha,
            "text_characters": len(text),
            "extraction_method": method,
            "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _atomic_write_bytes(raw_path, raw)
        _atomic_write_json(meta_path, meta)
        return FetchedDocument(
            requested_url=url,
            final_url=final_url,
            content_type=content_type,
            text=text,
            extraction_method=method,
            raw_sha256=sha256_bytes(raw),
            text_sha256=text_sha,
            provenance={**meta, "cache_hit": False},
        )


def question_match(
    *, document: FetchedDocument, plan: Mapping[str, Any], full_ocr_text: str, profile: Mapping[str, Any]
) -> dict[str, Any]:
    folded_document = fold_for_match(document.text)
    phrases = [str(value) for value in plan.get("phrases") or []]
    matched = [phrase for phrase in phrases if fold_for_match(phrase) in folded_document]
    fingerprint_tokens = set(match_tokens(" ".join(phrases)))
    full_tokens = set(match_tokens(full_ocr_text))
    document_tokens = set(match_tokens(document.text))
    fingerprint_coverage = (
        len(fingerprint_tokens & document_tokens) / len(fingerprint_tokens)
        if fingerprint_tokens
        else 0.0
    )
    full_coverage = len(full_tokens & document_tokens) / len(full_tokens) if full_tokens else 0.0
    thresholds = profile["certificate"]
    required_phrases = min(int(thresholds["min_phrase_matches"]), len(phrases))
    accepted = (
        required_phrases > 0
        and len(matched) >= required_phrases
        and fingerprint_coverage >= float(thresholds["min_fingerprint_token_coverage"])
        and full_coverage >= float(thresholds["min_full_token_coverage"])
    )
    return {
        "accepted": accepted,
        "matched_phrases": matched,
        "matched_phrase_count": len(matched),
        "required_phrase_count": required_phrases,
        "fingerprint_token_coverage": round(fingerprint_coverage, 6),
        "full_token_coverage": round(full_coverage, 6),
    }


def _answer_excerpt(text: str, start: int, end: int, *, limit: int = 180) -> str:
    left = max(0, start - limit // 2)
    right = min(len(text), end + limit // 2)
    return re.sub(r"\s+", " ", text[left:right]).strip()[:limit]


def answer_observations(
    text: str, *, question_number: int | None, profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    marker_pattern = re.compile(
        r"(?:cevap(?:\s+anahtar[ıi])?|do[gğ]ru\s+cevap|yan[ıi]t|answer)"
        r"\s*(?:[:=\-]\s*)?(?:[sş][ıi]kk[ıi]?\s*)?([A-E])\b",
        re.IGNORECASE,
    )
    for match in marker_pattern.finditer(text):
        observations.append(
            {
                "answer": match.group(1).upper(),
                "kind": "explicit_answer_marker",
                "offset": match.start(),
                "locator_excerpt": _answer_excerpt(text, match.start(), match.end()),
            }
        )
    key_signal = bool(
        re.search(r"cevap\s+anahtar[ıi]|cevaplar|answer\s+key", text, re.IGNORECASE)
    )
    if question_number is not None and key_signal:
        number = re.escape(str(question_number))
        key_pattern = re.compile(
            rf"(?<!\d){number}\s*(?:[.)/:=\-]\s*|\s+)([A-E])\b",
            re.IGNORECASE,
        )
        for match in key_pattern.finditer(text):
            observations.append(
                {
                    "answer": match.group(1).upper(),
                    "kind": "numbered_answer_key",
                    "offset": match.start(),
                    "locator_excerpt": _answer_excerpt(text, match.start(), match.end()),
                }
            )
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in observations:
        key = (str(item["answer"]), str(item["kind"]), int(item["offset"]))
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return deduplicated


def _document_descriptor(document: FetchedDocument) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(document.final_url)
    return {
        "requested_url": document.requested_url,
        "final_url": document.final_url,
        "authority_host": str(parsed.hostname or "").casefold(),
        "content_type": document.content_type,
        "raw_sha256": document.raw_sha256,
        "text_sha256": document.text_sha256,
        "extraction_method": document.extraction_method,
        "cache_hit": bool(document.provenance.get("cache_hit")),
    }


def build_certificate(
    *,
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_ocr_text: str,
    documents: Sequence[FetchedDocument],
    policy: OfficialURLPolicy,
    profile: Mapping[str, Any],
    profile_sha256: str,
    queue_sha256: str,
    ocr_sha256: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    matches = [
        (document, question_match(document=document, plan=plan, full_ocr_text=full_ocr_text, profile=profile))
        for document in documents
    ]
    question_documents = [(document, match) for document, match in matches if match["accepted"]]
    if not question_documents:
        return None, ["no_exact_official_question_match"]

    question_number = plan.get("question_number")
    candidates: list[dict[str, Any]] = []
    ambiguous_families = 0
    for question_document, match in question_documents:
        family = policy.family(question_document.final_url)
        family_documents = [
            document for document in documents if policy.family(document.final_url) == family
        ]
        observations: list[tuple[FetchedDocument, dict[str, Any]]] = []
        for key_document in family_documents:
            key_match = question_match(
                document=key_document,
                plan=plan,
                full_ocr_text=full_ocr_text,
                profile=profile,
            )
            has_key_signal = bool(
                re.search(
                    r"cevap\s+anahtar[ıi]|cevaplar|answer\s+key",
                    key_document.text,
                    re.IGNORECASE,
                )
            )
            if key_document.final_url != question_document.final_url and not (
                key_match["accepted"] or has_key_signal
            ):
                continue
            for observation in answer_observations(
                key_document.text,
                question_number=int(question_number) if question_number is not None else None,
                profile=profile,
            ):
                observations.append((key_document, observation))
        answers = {str(observation["answer"]) for _, observation in observations}
        if len(answers) != 1:
            if len(answers) > 1:
                ambiguous_families += 1
            continue
        answer = next(iter(answers))
        key_document, key_observation = sorted(
            observations,
            key=lambda pair: (
                0 if pair[1]["kind"] == "numbered_answer_key" else 1,
                pair[0].final_url,
                int(pair[1]["offset"]),
            ),
        )[0]
        candidates.append(
            {
                "answer": answer,
                "question_document": question_document,
                "question_match": match,
                "key_document": key_document,
                "key_observation": key_observation,
                "family": family,
                "supporting_observations": len(observations),
            }
        )
    candidate_answers = {str(candidate["answer"]) for candidate in candidates}
    if len(candidate_answers) > 1:
        return None, ["conflicting_official_answer_certificates"]
    if not candidates:
        reasons = ["no_unambiguous_official_answer_key"]
        if ambiguous_families:
            reasons.append("ambiguous_answer_markers_or_adjacent_columns")
        return None, reasons

    chosen = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["question_match"]["fingerprint_token_coverage"]),
            -float(candidate["question_match"]["full_token_coverage"]),
            candidate["question_document"].final_url,
        ),
    )[0]
    certificate: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "task_id": str(task["task_id"]),
        "answer": chosen["answer"],
        "authority_policy": "official_host_allowlist_and_final_redirect_v1",
        "question_number": question_number,
        "document_family": chosen["family"],
        "question_document": _document_descriptor(chosen["question_document"]),
        "question_locator": chosen["question_match"],
        "key_document": _document_descriptor(chosen["key_document"]),
        "key_locator": chosen["key_observation"],
        "supporting_answer_observations": chosen["supporting_observations"],
        "bindings": {
            "profile_sha256": profile_sha256,
            "public_queue_sha256": queue_sha256,
            "ocr_sha256": ocr_sha256,
            "query_plan_sha256": plan["plan_sha256"],
        },
        "generation": {
            "gold_access": False,
            "reference_access": False,
            "judge_access": False,
            "score_access": False,
        },
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    return certificate, []


class ExactOfficialWebRouter:
    def __init__(
        self,
        *,
        profile: Mapping[str, Any],
        profile_sha256: str,
        queue_sha256: str,
        ocr_sha256: str,
        search_client: SearchClient,
        fetcher: OfficialDocumentFetcher,
    ) -> None:
        self.profile = profile
        self.profile_sha256 = profile_sha256
        self.queue_sha256 = queue_sha256
        self.ocr_sha256 = ocr_sha256
        self.search_client = search_client
        self.fetcher = fetcher
        self.policy = fetcher.policy

    def route(self, task: Mapping[str, Any], parser_row: Mapping[str, Any]) -> dict[str, Any]:
        plan = build_query_plan(task, parser_row, self.profile)
        base: dict[str, Any] = {
            "schema_version": DECISION_SCHEMA,
            "condition": self.profile["condition"],
            "task_id": str(task["task_id"]),
            "accepted": False,
            "final_answer": None,
            "rejection_reasons": [],
            "query_plan": plan,
            "search_calls": [],
            "document_fetches": [],
            "certificate": None,
            "generation": {
                "gold_access": False,
                "reference_access": False,
                "judge_access": False,
                "score_access": False,
                "default_solver_access": False,
            },
            "error": None,
        }
        if not plan["eligible"]:
            base["rejection_reasons"] = list(plan["ineligibility_reasons"])
            base["decision_sha256"] = canonical_sha256(base)
            return base

        network = self.profile["network"]
        hit_urls: list[str] = []
        seen_urls: set[str] = set()
        for query_item in plan["queries"]:
            query = str(query_item["query"])
            try:
                response = self.search_client.search(
                    query,
                    limit=int(network["search_results_per_query"]),
                )
                base["search_calls"].append(
                    {
                        "query_sha256": query_item["query_sha256"],
                        "result_count": len(response.hits),
                        "response_sha256": response.provenance.get("raw_sha256"),
                        "cache_hit": response.provenance.get("cache_hit"),
                        "error": None,
                    }
                )
                for hit in response.hits:
                    if not self.policy.is_allowed_url(hit.url) or hit.url in seen_urls:
                        continue
                    seen_urls.add(hit.url)
                    hit_urls.append(hit.url)
            except Exception as exc:
                base["search_calls"].append(
                    {
                        "query_sha256": query_item["query_sha256"],
                        "result_count": 0,
                        "response_sha256": None,
                        "cache_hit": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        documents: list[FetchedDocument] = []
        final_urls: set[str] = set()
        for url in hit_urls[: int(network["max_fetches_per_task"])]:
            try:
                document = self.fetcher.fetch(url)
                if document.final_url in final_urls:
                    continue
                final_urls.add(document.final_url)
                documents.append(document)
                base["document_fetches"].append(
                    {**_document_descriptor(document), "error": None}
                )
            except Exception as exc:
                base["document_fetches"].append(
                    {
                        "requested_url": url,
                        "final_url": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if not documents:
            base["rejection_reasons"] = ["no_fetchable_official_documents"]
            base["decision_sha256"] = canonical_sha256(base)
            return base

        full_text, _ = ocr_text(parser_row)
        certificate, reasons = build_certificate(
            task=task,
            plan=plan,
            full_ocr_text=full_text,
            documents=documents,
            policy=self.policy,
            profile=self.profile,
            profile_sha256=self.profile_sha256,
            queue_sha256=self.queue_sha256,
            ocr_sha256=self.ocr_sha256,
        )
        if certificate is None:
            base["rejection_reasons"] = reasons
        else:
            base["accepted"] = True
            base["final_answer"] = certificate["answer"]
            base["certificate"] = certificate
        base["decision_sha256"] = canonical_sha256(base)
        return base


def load_inputs(
    *,
    queue_path: Path,
    expected_queue_sha256: str,
    ocr_path: Path,
    expected_ocr_sha256: str,
    profile_path: Path,
    expected_profile_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, str]]:
    queue_path = guard_input_path(
        queue_path, role="public queue", suffixes={".jsonl"}
    )
    ocr_path = guard_input_path(ocr_path, role="public OCR", suffixes={".jsonl"})
    profile_path = guard_input_path(profile_path, role="profile", suffixes={".json"})
    profile, profile_sha = _read_json(
        profile_path,
        role="profile",
        expected_sha256=expected_profile_sha256,
    )
    validate_profile(profile)
    queue, queue_sha = _read_jsonl(
        queue_path,
        role="public queue",
        expected_sha256=expected_queue_sha256,
    )
    validate_public_queue(queue, expected_rows=int(profile["expected_rows"]))
    ocr, ocr_sha = _read_jsonl(
        ocr_path,
        role="public OCR",
        expected_sha256=expected_ocr_sha256,
    )
    task_ids = [str(row["task_id"]) for row in queue]
    ocr_index = index_ocr_rows(ocr, expected_task_ids=task_ids)
    return queue, ocr_index, profile, {
        "queue": queue_sha,
        "ocr": ocr_sha,
        "profile": profile_sha,
    }


def dry_run_report(
    *,
    queue: Sequence[Mapping[str, Any]],
    ocr_index: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    plans = [build_query_plan(task, ocr_index[str(task["task_id"])], profile) for task in queue]
    eligible = [plan for plan in plans if plan["eligible"]]
    counts: dict[str, int] = {}
    for plan in eligible:
        count = len(plan["queries"])
        counts[str(count)] = counts.get(str(count), 0) + 1
    return {
        "schema_version": ROUTER_SCHEMA + "-dry-run",
        "status": "PASS",
        "tasks": len(queue),
        "eligible_tasks": len(eligible),
        "failclosed_ineligible_tasks": len(queue) - len(eligible),
        "query_count_distribution": counts,
        "query_plan_set_sha256": canonical_sha256(plans),
        "inputs": dict(hashes),
        "network_calls": 0,
        "cache_writes": 0,
        "generation": {
            "gold_access": False,
            "reference_access": False,
            "judge_access": False,
            "score_access": False,
            "default_solver_access": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--expected-queue-sha256", required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--expected-ocr-sha256", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--searxng-url")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--enable-network", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        queue, ocr_index, profile, hashes = load_inputs(
            queue_path=args.queue,
            expected_queue_sha256=args.expected_queue_sha256,
            ocr_path=args.ocr,
            expected_ocr_sha256=args.expected_ocr_sha256,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
        )
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_report(
                        queue=queue,
                        ocr_index=ocr_index,
                        profile=profile,
                        hashes=hashes,
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.output is None or args.manifest is None or args.cache_dir is None:
            raise RouterInputError("live/cache replay requires --output, --manifest, and --cache-dir")
        if not args.searxng_url:
            raise RouterInputError("live/cache replay requires --searxng-url")
        guard_output_paths(
            output=args.output,
            manifest=args.manifest,
            cache_dir=args.cache_dir,
            protected_inputs=[args.queue, args.ocr, args.profile],
        )
        network_profile = profile["network"]
        policy = OfficialURLPolicy.from_profile(profile)
        search = SearxngSearchClient(
            base_url=args.searxng_url,
            cache_dir=args.cache_dir / "search",
            timeout_s=float(network_profile["timeout_s"]),
            language=str(network_profile["search_language"]),
            network_enabled=bool(args.enable_network),
            user_agent=str(network_profile["user_agent"]),
        )
        fetcher = OfficialDocumentFetcher(
            policy=policy,
            cache_dir=args.cache_dir / "documents",
            timeout_s=float(network_profile["timeout_s"]),
            max_bytes=int(network_profile["max_document_bytes"]),
            network_enabled=bool(args.enable_network),
            user_agent=str(network_profile["user_agent"]),
        )
        router = ExactOfficialWebRouter(
            profile=profile,
            profile_sha256=hashes["profile"],
            queue_sha256=hashes["queue"],
            ocr_sha256=hashes["ocr"],
            search_client=search,
            fetcher=fetcher,
        )
        decisions = [
            router.route(task, ocr_index[str(task["task_id"])]) for task in queue
        ]
        _atomic_write_jsonl(args.output, decisions)
        accepted = [row for row in decisions if row["accepted"]]
        manifest = {
            "schema_version": ROUTER_SCHEMA + "-manifest",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "condition": profile["condition"],
            "inputs": {
                "queue": {"path": str(args.queue.resolve()), "sha256": hashes["queue"]},
                "ocr": {"path": str(args.ocr.resolve()), "sha256": hashes["ocr"]},
                "profile": {"path": str(args.profile.resolve()), "sha256": hashes["profile"]},
            },
            "output": {
                "path": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "rows": len(decisions),
                "accepted": len(accepted),
                "failclosed": len(decisions) - len(accepted),
            },
            "network": {
                "explicitly_enabled": bool(args.enable_network),
                "search_network_calls": search.network_calls,
                "document_network_calls": fetcher.network_calls,
            },
            "generation": {
                "gold_access": False,
                "reference_access": False,
                "judge_access": False,
                "score_access": False,
                "default_solver_access": False,
            },
            "code_sha256": sha256_file(Path(__file__)),
        }
        _atomic_write_json(args.manifest, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
