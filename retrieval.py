"""Reference implementation of permission-first hybrid memory retrieval."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass
import math
import re
import unicodedata

TERM = re.compile(r"[^\W_]+")
STOP = frozenset(
    "a an and are at for from how in is of on or the to was what where which with "
    "без был была были где для или как какие какой когда мне мой надо наш о он она "
    "они от под при про так у уже что чтобы это".split()
)
WEIGHTS = {"phrase": 1.0, "lexical": 1.0, "dense": 1.0, "fresh": 0.8}


@dataclass(frozen=True, slots=True)
class Memory:
    memory_id: str
    scope: str
    revision: str
    text: str
    updated_at: float  # Unix seconds; assigned by the canonical store.
    status: str = "active"


@dataclass(frozen=True, slots=True)
class Embedding:
    space: str  # Include the model, version and preprocessing in this identifier.
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class IndexedMemory:
    snapshot: Memory
    embedding: Embedding | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    memory: Memory
    lanes: tuple[str, ...]
    score: float


def _key(value):
    return type(value) is str and 0 < len(value) <= 256 and value.isprintable() and bool(value.strip())


def _valid_text(text, max_chars):
    return (type(text) is str and len(text) <= max_chars
            and not any(0xD800 <= ord(char) <= 0xDFFF for char in text))


def _tokens(text):
    # Ignore invisible formatting for matching. Never return this normalized text.
    visible = "".join(
        " " if char.isspace() else char
        for char in text
        if unicodedata.category(char) not in {"Cf", "Cc"} or char.isspace()
    )
    return TERM.findall(unicodedata.normalize("NFC", visible).casefold())


def _terms(tokens):
    return {token for token in tokens if len(token) >= 2 and token not in STOP}


def _valid_record(record, allowed_scopes):
    return (
        type(record) is Memory and type(record.status) is str and record.status == "active"
        and _key(record.memory_id) and _key(record.scope) and record.scope in allowed_scopes
        and _key(record.revision) and _valid_text(record.text, 64_000)
        and type(record.updated_at) in (float, int) and 0 <= record.updated_at < 1e14
    )


def _unit(embedding):
    if (type(embedding) is not Embedding or not _key(embedding.space)
            or type(embedding.values) is not tuple or not 1 <= len(embedding.values) <= 4096
            or not all(type(x) in (float, int) and -1e300 <= x <= 1e300 for x in embedding.values)):
        raise ValueError("invalid embedding")
    scale = max(map(abs, embedding.values))
    if scale == 0:
        raise ValueError("embedding has zero length")
    scaled = tuple(value / scale for value in embedding.values)
    norm = math.hypot(*scaled)
    return tuple(value / norm for value in scaled)


def _cosine(query, query_unit, embedding):
    if (type(embedding) is not Embedding or embedding.space != query.space
            or type(embedding.values) is not tuple or len(embedding.values) != len(query_unit)):
        return None
    try:
        other = _unit(embedding)
    except ValueError:
        return None
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(query_unit, other))))


def search(
    query: str,
    index: Iterable[IndexedMemory],
    *,
    read_current: Callable[[str], Memory | None],
    allowed_scopes: Set[str],
    fresh: Iterable[Memory] = (),
    query_embedding: Embedding | None = None,
    limit: int = 5,
    depth: int = 20,
    cosine_min: float = 0.58,
    coverage_min: float = 0.67,
) -> list[Hit]:
    """Find eligible memories, or return an empty list if none has enough evidence.

    Supply scopes from trusted policy, not from the query. A candidate must match
    the entire canonical active snapshot before ranking and again before serving.
    Adapter errors propagate; the caller must handle them without using stale hits.
    """
    if not _valid_text(query, 4096):
        raise ValueError("query must be Unicode text of at most 4096 characters")
    if not isinstance(allowed_scopes, Set):
        raise ValueError("allowed_scopes must be a set of scope names")
    allowed_scopes = frozenset(allowed_scopes)
    if not all(_key(scope) for scope in allowed_scopes):
        raise ValueError("invalid scope name")
    if not (type(limit) is int and type(depth) is int and 1 <= limit <= depth <= 1000):
        raise ValueError("require 1 <= limit <= depth <= 1000")
    for value in (cosine_min, coverage_min):
        if type(value) not in (int, float) or not 0 < value <= 1:
            raise ValueError("evidence thresholds must be finite and in (0, 1]")

    query_tokens = _tokens(query)
    terms = _terms(query_tokens)
    qvec = None if query_embedding is None else _unit(query_embedding)
    if not terms or not allowed_scopes:
        return []

    accepted = {}
    is_fresh = set()

    def admit(snapshot):
        if not _valid_record(snapshot, allowed_scopes):
            return None
        current = read_current(snapshot.memory_id)
        if not _valid_record(current, allowed_scopes) or current != snapshot:
            return None
        return current

    for candidate in index:
        if type(candidate) is not IndexedMemory:
            continue
        current = admit(candidate.snapshot)
        if current is not None:
            mid = current.memory_id
            similarity = None if qvec is None else _cosine(query_embedding, qvec, candidate.embedding)
            # A duplicate with different content cannot pass the same snapshot.
            previous = accepted.get(mid)
            if previous is None or previous[0] != current or (
                similarity is not None and (previous[1] is None or similarity > previous[1])
            ):
                accepted[mid] = (current, similarity)

    for snapshot in fresh:
        current = admit(snapshot)
        if current is not None:
            mid = current.memory_id
            if mid not in accepted or accepted[mid][0] != current:
                accepted[mid] = (current, None)
            is_fresh.add(mid)

    ranked = defaultdict(list)
    coverage = {}
    phrase = " " + " ".join(query_tokens) + " "
    for mid, (record, cosine) in accepted.items():
        tokens = _tokens(record.text)
        overlap = len(terms & _terms(tokens))
        coverage[mid] = overlap / len(terms)
        if overlap:
            if len(query_tokens) > 1 and phrase in (" " + " ".join(tokens) + " "):
                ranked["phrase"].append((overlap, mid))
            ranked["lexical"].append((overlap, mid))
            if mid in is_fresh:
                ranked["fresh"].append((record.updated_at, mid))
        if cosine is not None and cosine > 0:
            ranked["dense"].append((cosine, mid))

    scores = defaultdict(float)
    lanes = defaultdict(list)
    for lane, weight in WEIGHTS.items():
        entries = sorted(ranked[lane], key=lambda row: (-row[0], row[1]))
        for rank, (_, mid) in enumerate(entries[:depth], 1):
            scores[mid] += weight / (20 + rank)
            lanes[mid].append(lane)

    result = []
    for mid in sorted(scores, key=lambda key: (-scores[key], key)):
        record, cosine = accepted[mid]
        term_cover = coverage[mid]
        agreement = "dense" in lanes[mid] and any(lane != "dense" for lane in lanes[mid])
        sufficient = (
            term_cover >= coverage_min
            or (cosine is not None and cosine >= cosine_min)
            or (agreement and cosine is not None
                and cosine >= cosine_min - 0.12 and term_cover >= coverage_min / 2)
        )
        if not sufficient:
            continue
        # Serve canonical content, never material straight from the index.
        current = admit(record)
        if current is not None:
            result.append(Hit(current, tuple(lanes[mid]), scores[mid]))
        if len(result) == limit:
            break
    return result
