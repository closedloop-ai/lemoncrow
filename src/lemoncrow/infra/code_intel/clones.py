"""F6 -- near-duplicate detection over the engine's symbol table.

The question this answers is "where else does this code already exist", and the
honest form of that answer is a similarity score with a stated method, not a
binary.

Method
------

Each symbol's source is sliced out of its file using ``start_byte``/``end_byte``
from the engine's ``symbols`` table, stripped of comments and string bodies,
tokenised, and shingled into overlapping runs of :data:`SHINGLE_K` tokens. The
shingle set becomes a :class:`~lemoncrow.core.foundation._minhash.MinHash`
signature, and estimated Jaccard similarity over those signatures is the score.

Shingling is what makes this a clone detector rather than a bag-of-words
comparison: order matters, so two functions built from the same vocabulary in a
different sequence score low.

Identifiers are replaced by a placeholder before shingling, and that choice is
the difference between the detector working and not. Measured on a copied
function whose every local name was changed, verbatim tokens scored **0.039** --
because at k=5 nearly every shingle contains at least one renamed identifier, so
nearly every shingle differs. Renaming is the *first* thing that happens to
copied code, so a detector that misses it misses the population it exists to
find. Keywords and punctuation survive, which is what keeps control-flow shape
in the signature; numeric literals are placeholdered for the same reason names
are. The cost is precision on structurally identical but semantically unrelated
code, which :data:`MIN_TOKENS` and :data:`DEFAULT_THRESHOLD` between them hold
down, and which the reported score lets a reader judge for themselves.

Why LSH is not optional
-----------------------

This repository indexes ~23k symbols, so all-pairs comparison is ~277 million
comparisons -- not a tuning problem, a different program. Signatures are banded
instead: :data:`BANDS` bands of :data:`ROWS_PER_BAND` rows, and two symbols
become *candidates* only when some band matches exactly. A pair with true
similarity ``s`` survives banding with probability
``1 - (1 - s**ROWS_PER_BAND)**BANDS`` -- at the shipped 16x8 split, 99.99% at
s=0.9, 94.7% at s=0.8, 61.3% at s=0.7 and 6.1% at s=0.5. Steep enough to
discard almost everything while keeping almost every real clone.
(``test_banding_recall_curve_matches_the_documented_probabilities`` pins these,
so the split cannot be retuned without the numbers here being updated.)

Candidates are then scored individually, so banding costs recall and never
precision: a reported pair was always measured, never assumed.

Provenance
----------

Results land in the sidecar (:mod:`lemoncrow.infra.code_intel.sidecar`) stamped
with the ``engine_index_version`` they were built from. A reindex bumps that
number and every stored pair immediately reads as stale rather than as current.
Reading a stale clone table is the same silent-wrong-answer failure F11 removed
from the engine cache, so :func:`load_clones` refuses instead of guessing.

This module reads the engine's databases and writes only the sidecar.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from lemoncrow.core.foundation._minhash import MinHash
from lemoncrow.infra.code_intel.freshness import IndexRebuilding, index_state
from lemoncrow.infra.code_intel.sidecar import open_sidecar, stamp, stamp_of
from lemoncrow.infra.code_intel.store import CodeIntelStore, SymbolRow

__all__ = [
    "BANDS",
    "CLONES_TABLE",
    "SIGNATURES_TABLE",
    "CloneView",
    "current_symbol_hashes",
    "open_clone_table",
    "signature_coverage",
    "DEFAULT_THRESHOLD",
    "MIN_TOKENS",
    "NUM_PERM",
    "ROWS_PER_BAND",
    "SHINGLE_K",
    "ClonePair",
    "CloneReport",
    "ClonesStale",
    "build_clones",
    "load_clones",
    "normalise_tokens",
    "shingle",
    "signature",
]

logger = logging.getLogger(__name__)

#: Sidecar table these pairs live in, and the key :func:`stamp` records against.
CLONES_TABLE = "symbol_clones"

#: Cached MinHash signatures, keyed by the content they were taken over.
SIGNATURES_TABLE = "symbol_signatures"

#: Signature width. 128 is the ``MinHash`` default and puts the Jaccard estimate
#: error at ~1/sqrt(128) ~= 0.088 -- fine for ranking candidates, which is all
#: the score is used for.
NUM_PERM = 128

#: ``BANDS * ROWS_PER_BAND`` must equal :data:`NUM_PERM`; asserted at import.
BANDS = 16
ROWS_PER_BAND = 8

#: Consecutive tokens per shingle. Below ~4 the shingles stop encoding order and
#: the score drifts toward a vocabulary comparison; above ~7 a single edit
#: breaks too many shingles and near-clones stop scoring as clones.
SHINGLE_K = 5

#: Symbols with fewer tokens than this are skipped entirely. A three-line getter
#: is identical to every other three-line getter, and reporting those as clones
#: buries the findings that mean something under the ones that never did.
MIN_TOKENS = 40

#: Pairs scoring below this are not recorded.
DEFAULT_THRESHOLD = 0.8

if BANDS * ROWS_PER_BAND != NUM_PERM:  # pragma: no cover - guards a constant edit
    raise AssertionError(f"BANDS*ROWS_PER_BAND ({BANDS * ROWS_PER_BAND}) must equal NUM_PERM ({NUM_PERM})")

# One left-to-right scan, not a stack of substitutions. Alternation order is the
# whole point: a docstring is recognised before a plain string, and a string
# before a comment, so a comment marker *inside* a literal is consumed as part
# of that literal instead of eating the rest of the line.
#
# Regression this shape exists to prevent. Comment patterns previously ran as
# `re.sub` over raw text before tokenising, with no idea where strings were:
#
#     @click.option("--limit", type=int, help="Pairs to print.")
#         -> ['@', ID, '.', ID, '(', '"']
#     url = "https://api.example.com/v1/things"
#         -> [ID, '=', '"', ID, ':']
#
# Every single-line click option and every URL literal in the repository was
# truncated at its first `--` or `//`. That defeated the decision documented
# above -- keeping string literals is what stops every `to_dict` scoring 1.000 --
# because option strings are exactly the content distinguishing one CLI command
# from another.
#
# `--` is deliberately absent as a comment marker. It serves only SQL, Lua and
# Haskell, none of which dominate here, and it is a decrement operator in the
# C-family languages that do -- so as a comment rule it destroys more real code
# than it removes prose. SQL comments surviving as tokens is mild noise; a
# truncated C-family line is a wrong answer.
_SCANNER = re.compile(
    r"(?P<doc>\"\"\".*?\"\"\"|'''.*?''')"
    r"|(?P<string>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\\n])*`)"
    r"|(?P<comment>/\*.*?\*/|//[^\n]*|\#[^\n]*)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<number>\d+(?:\.\d+)?)"
    r"|(?P<punct>[^\sA-Za-z_0-9])",
    re.DOTALL,
)

#: Scanner groups that are prose about the code rather than the code. A
#: docstring shared by two unrelated functions is not evidence they are clones,
#: and a copied block whose comments were rewritten is still a copied block --
#: both directions of that error point the same way.
_DISCARDED_GROUPS = frozenset({"doc", "comment"})

#: Stands in for any identifier, and any numeric literal, respectively. Single
#: control characters so they can never collide with a real token.
IDENTIFIER_PLACEHOLDER = "\x01"
LITERAL_PLACEHOLDER = "\x02"

#: Words that keep their identity through normalisation, because they carry the
#: shape of the code rather than the author's naming. Deliberately the union
#: across the indexed languages rather than a per-language table: a keyword that
#: is merely an identifier in some other language costs one token of precision,
#: whereas maintaining a second lexer beside the engine's costs considerably
#: more. Anything absent here is treated as a name, which is the safe default --
#: it can only make two symbols look *less* alike.
_KEYWORDS = frozenset(
    {
        # control flow
        "if", "elif", "else", "for", "while", "do", "switch", "case", "default",
        "break", "continue", "return", "yield", "goto", "match", "when", "loop",
        # binding and declaration
        "def", "fn", "func", "function", "lambda", "class", "struct", "enum",
        "interface", "trait", "impl", "type", "var", "let", "const", "static",
        "final", "global", "nonlocal", "public", "private", "protected",
        # errors and resources
        "try", "except", "catch", "finally", "raise", "throw", "throws",
        "with", "using", "defer", "assert", "panic", "recover",
        # modules
        "import", "from", "export", "package", "module", "use", "require",
        # operators spelled as words
        "and", "or", "not", "in", "is", "as", "new", "delete", "typeof",
        "instanceof", "await", "async", "go", "select", "chan", "mut", "ref",
        # literals that are structure, not naming
        "true", "false", "none", "null", "nil", "undefined", "self", "this",
        "super", "pass", "void",
    }
)


class ClonesStale(RuntimeError):
    """Nothing has ever built the clone table for this workspace.

    Not a subclass of anything a caller degrades to an empty result on. An empty
    clone list reads as "this code is unique", which is precisely the wrong
    conclusion to hand back when the truth is "nobody has looked yet".

    Deliberately *not* raised merely because the engine has reindexed. That was
    the first design and it made the feature unusable: ``index_version`` is a
    global counter, so one unrelated edit invalidated a table still correct for
    every symbol it described. Readers now check each pair against the two
    content hashes it was built from and return the subset that still holds,
    with explicit coverage -- a verified-current answer beats a refusal, and
    both beat a silently-superseded one.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"{detail}; run `lc code clones` to build it")
        self.detail = detail


@dataclass(frozen=True)
class ClonePair:
    """Two symbols measured as near-duplicates, with the score that says so."""

    symbol_a: str
    symbol_b: str
    content_hash_a: str
    content_hash_b: str
    qualified_name_a: str
    qualified_name_b: str
    file_path_a: str
    file_path_b: str
    token_count_a: int
    token_count_b: int
    jaccard: float

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "qualified_name_a": self.qualified_name_a,
            "qualified_name_b": self.qualified_name_b,
            "file_path_a": self.file_path_a,
            "file_path_b": self.file_path_b,
            "token_count_a": self.token_count_a,
            "token_count_b": self.token_count_b,
            "jaccard": self.jaccard,
        }

    def still_current(self, hashes: dict[str, str]) -> bool:
        """True when both symbols still hold the content this pair was measured on."""
        return (
            hashes.get(self.symbol_a) == self.content_hash_a
            and hashes.get(self.symbol_b) == self.content_hash_b
        )


@dataclass(frozen=True)
class CloneReport:
    """What one build pass did, including everything it declined to look at."""

    pairs: tuple[ClonePair, ...]
    symbols_considered: int
    symbols_skipped_short: int
    symbols_unreadable: int
    candidate_pairs: int
    threshold: float
    engine_index_version: int
    signatures_reused: int = 0
    signatures_computed: int = 0
    #: Symbols whose source was read off disk and tokenised this pass. Larger
    #: than `signatures_computed`, because a symbol under MIN_TOKENS is read and
    #: tokenised before it can be measured as too short to be worth signing.
    symbols_read: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "pairs": [pair.as_dict() for pair in self.pairs],
            "pair_count": len(self.pairs),
            "symbols_considered": self.symbols_considered,
            "symbols_skipped_short": self.symbols_skipped_short,
            "symbols_unreadable": self.symbols_unreadable,
            "candidate_pairs": self.candidate_pairs,
            "threshold": self.threshold,
            "engine_index_version": self.engine_index_version,
            "signatures_reused": self.signatures_reused,
            "signatures_computed": self.signatures_computed,
            "symbols_read": self.symbols_read,
        }


@dataclass(frozen=True)
class CloneView:
    """Stored pairs that are still true, plus how much of the repo they cover.

    Every pair here was re-checked against the live ``symbols.content_hash``, so
    the list is current by construction rather than by assumption. ``coverage``
    is what the caller needs in order to read the *absence* of a pair correctly:
    at 1.0 "no clone reported" means measured-and-none, below it means some
    symbols have changed since the last build and were not examined.
    """

    pairs: tuple[ClonePair, ...]
    recorded: int
    superseded: int
    symbols_total: int
    symbols_unchanged: int
    built_from_index_version: int
    engine_index_version: int

    @property
    def coverage(self) -> float:
        if self.symbols_total == 0:
            return 1.0
        return self.symbols_unchanged / self.symbols_total

    @property
    def stale_symbols(self) -> int:
        return self.symbols_total - self.symbols_unchanged

    def as_dict(self) -> dict[str, object]:
        return {
            "pairs": [pair.as_dict() for pair in self.pairs],
            "pair_count": len(self.pairs),
            "recorded": self.recorded,
            "superseded": self.superseded,
            "coverage": round(self.coverage, 4),
            "stale_symbols": self.stale_symbols,
            "symbols_total": self.symbols_total,
            "built_from_index_version": self.built_from_index_version,
            "engine_index_version": self.engine_index_version,
        }


def normalise_tokens(source: str, *, rename_blind: bool = True) -> list[str]:
    """Strip comments and string bodies from *source*, then tokenise.

    With *rename_blind* (the default) every identifier becomes
    :data:`IDENTIFIER_PLACEHOLDER` and every numeric literal becomes
    :data:`LITERAL_PLACEHOLDER`, so a copy whose names were all changed still
    matches its original. String literals keep their contents either way -- they
    are the strongest signal left once names are gone, and dropping them is what
    made every ``to_dict`` in this repository look identical. Pass
    ``rename_blind=False`` for the verbatim token stream, which finds only
    literal copies.

    Deliberately language-agnostic. The comment and string patterns cover the
    syntaxes the indexed languages actually use, and applying a Python comment
    rule to a Go file removes nothing because Go has no ``#`` comments. Getting
    this wrong in the tolerant direction costs a little precision on one symbol;
    getting it wrong in the strict direction would mean maintaining a per-language
    lexer beside the one the engine already owns.
    """
    tokens: list[str] = []
    for match in _SCANNER.finditer(source):
        group = match.lastgroup or ""
        if group in _DISCARDED_GROUPS:
            continue
        text = match.group()
        if not rename_blind:
            tokens.append(text)
        elif group == "word":
            tokens.append(text if text.lower() in _KEYWORDS else IDENTIFIER_PLACEHOLDER)
        elif group == "number":
            tokens.append(LITERAL_PLACEHOLDER)
        else:
            tokens.append(text)
    return tokens


def shingle(tokens: list[str], k: int = SHINGLE_K) -> set[bytes]:
    """Overlapping runs of *k* tokens, as the set a MinHash is taken over.

    A token list shorter than *k* yields one shingle covering the whole run
    rather than nothing -- returning an empty set would make every short symbol
    identical to every other, which is the opposite of the intent.
    """
    if not tokens:
        return set()
    if len(tokens) <= k:
        return {"\x00".join(tokens).encode("utf-8")}
    return {"\x00".join(tokens[i : i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}


def signature(tokens: list[str], k: int = SHINGLE_K, num_perm: int = NUM_PERM) -> MinHash:
    """MinHash signature over the shingle set of *tokens*."""
    minhash = MinHash(num_perm=num_perm)
    for item in shingle(tokens, k):
        minhash.update(item)
    return minhash


def _band_keys(minhash: MinHash) -> list[tuple[int, tuple[int, ...]]]:
    """The (band index, band contents) pairs two symbols must share to be candidates."""
    values = minhash.hashvalues
    return [(band, tuple(values[band * ROWS_PER_BAND : (band + 1) * ROWS_PER_BAND])) for band in range(BANDS)]


def _read_symbol_sources(
    symbols: list[SymbolRow],
    repo_root: Path,
) -> tuple[dict[str, list[str]], int]:
    """Token lists per ``symbol_id``, reading each file exactly once.

    Byte offsets come from the engine, so the file is read as bytes and sliced
    before decoding -- decoding first and slicing by character index would
    silently misalign on any file with non-ASCII content.
    """
    by_file: dict[str, list[SymbolRow]] = defaultdict(list)
    for row in symbols:
        by_file[row.file_path].append(row)

    tokens_by_symbol: dict[str, list[str]] = {}
    unreadable = 0
    for file_path, rows in by_file.items():
        path = repo_root / file_path
        try:
            blob = path.read_bytes()
        except OSError:
            # Indexed but gone or unreadable now. Counted, not raised: one
            # deleted file must not cost the whole pass.
            unreadable += len(rows)
            continue
        for row in rows:
            start = max(0, row.start_byte)
            end = min(len(blob), row.end_byte)
            if end <= start:
                unreadable += 1
                continue
            source = blob[start:end].decode("utf-8", errors="replace")
            tokens_by_symbol[row.symbol_id] = normalise_tokens(source)
    return tokens_by_symbol, unreadable


@dataclass(frozen=True)
class _CachedSignature:
    content_hash: str
    token_count: int
    packed: bytes

    @property
    def signed(self) -> bool:
        """False for a symbol that was examined and found too short to sign.

        Recorded rather than omitted, so "we looked and it was trivial" stays
        distinguishable from "we never looked". Without that distinction the
        coverage figure read 0.48 on a fully-current index, because half this
        repo's symbols are below MIN_TOKENS.
        """
        return len(self.packed) == NUM_PERM * 4

    def minhash(self) -> MinHash:
        restored = MinHash(num_perm=NUM_PERM)
        restored.hashvalues = list(struct.unpack(f"<{NUM_PERM}I", self.packed))
        return restored


def _pack(minhash: MinHash) -> bytes:
    return struct.pack(f"<{NUM_PERM}I", *minhash.hashvalues)


def _load_signature_cache(repo_root: Path, repo_id: str) -> dict[str, _CachedSignature]:
    conn = open_sidecar(repo_root)
    try:
        rows = conn.execute(
            "SELECT symbol_id, content_hash, token_count, signature FROM symbol_signatures WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        # The cache is an accelerator, never a source of truth. A missing or
        # unreadable one costs time, not correctness.
        return {}
    finally:
        conn.close()
    return {
        str(row["symbol_id"]): _CachedSignature(
            content_hash=str(row["content_hash"]),
            token_count=int(row["token_count"]),
            packed=bytes(row["signature"]),
        )
        for row in rows
    }


def _cache_hit(cached: dict[str, _CachedSignature], row: SymbolRow, min_tokens: int = MIN_TOKENS) -> bool:
    """True when the cache already covers this symbol's current content.

    Keyed on ``content_hash``, not ``symbol_id``: an edited symbol keeps its id,
    and reusing a signature across a content change is the one way this cache
    could produce a wrong answer rather than merely a slow one.

    A symbol recorded as too short counts as covered -- it was read and measured,
    which is what coverage asks. It stops counting the moment *min_tokens* drops
    below its token count, because then it would need a signature it never got.
    A stored signature of the wrong width is a miss, so changing NUM_PERM can
    never mix two geometries into one comparison.
    """
    entry = cached.get(row.symbol_id)
    if entry is None or entry.content_hash != row.content_hash:
        return False
    if entry.token_count < min_tokens:
        return True
    return entry.signed


def _encloses(a: SymbolRow, b: SymbolRow) -> bool:
    """True when one symbol's byte range contains the other's in the same file."""
    if a.file_path != b.file_path:
        return False
    return (a.start_byte <= b.start_byte and b.end_byte <= a.end_byte) or (
        b.start_byte <= a.start_byte and a.end_byte <= b.end_byte
    )


def _candidate_pairs(signatures: dict[str, MinHash]) -> set[tuple[str, str]]:
    """Symbol pairs sharing at least one exact band, ordered so a pair appears once."""
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for symbol_id, minhash in signatures.items():
        for key in _band_keys(minhash):
            buckets[key].append(symbol_id)

    candidates: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                candidates.add((left, right))
    return candidates


def build_clones(
    repo_root: Path | str = ".",
    threshold: float = DEFAULT_THRESHOLD,
    min_tokens: int = MIN_TOKENS,
) -> CloneReport:
    """Detect near-duplicate symbols and replace the sidecar clone table.

    The table is rewritten wholesale rather than merged. A symbol that stopped
    being a clone must stop being reported, and an incremental merge cannot know
    that without tracking deletions the engine does not expose.

    Signatures, unlike pairs, *are* reused: they are cached against the content
    hash they were taken over, so an unchanged symbol is never re-read or
    re-hashed. Tokenising and hashing was nearly all of a 33s full pass; banding
    over cached signatures is the cheap part.

    Raises :class:`IndexRebuilding` when the engine's index is mid-write. F11
    built that probe so nothing derives from a torn index, and a clone table
    stamped authoritative from a half-populated one is exactly that failure with
    an extra step.
    """
    root = Path(repo_root).resolve()
    state = index_state(root)
    if state.rebuilding:
        raise IndexRebuilding(root, state.detail)

    with CodeIntelStore(root) as store:
        engine_index_version = store.engine_state("index_version")
        repo_id = store.repo_id_or_none()
        symbols = store.symbols() if repo_id is not None else []

    by_id = {row.symbol_id: row for row in symbols}
    cached = _load_signature_cache(root, repo_id or "")

    # Only symbols whose content the cache does not already cover are read off
    # disk; everything else skips tokenising entirely.
    unchanged = {row.symbol_id for row in symbols if _cache_hit(cached, row, min_tokens)}
    to_read = [row for row in symbols if row.symbol_id not in unchanged]
    tokens_by_symbol, unreadable = _read_symbol_sources(to_read, root)

    signatures: dict[str, MinHash] = {}
    token_counts: dict[str, int] = {}
    skipped_short = 0
    reused = 0

    for symbol_id in unchanged:
        entry = cached[symbol_id]
        token_counts[symbol_id] = entry.token_count
        if entry.token_count < min_tokens:
            skipped_short += 1
            continue
        reused += 1
        signatures[symbol_id] = entry.minhash()

    for symbol_id, tokens in tokens_by_symbol.items():
        token_counts[symbol_id] = len(tokens)
        if len(tokens) < min_tokens:
            skipped_short += 1
            continue
        signatures[symbol_id] = signature(tokens)

    candidates = _candidate_pairs(signatures)
    pairs: list[ClonePair] = []
    for left, right in candidates:
        row_a, row_b = by_id[left], by_id[right]
        if _encloses(row_a, row_b):
            # A class's byte range spans its methods, so a class whose body is
            # one long method scores against that method and gets written to the
            # table as a clone of itself. Observed on this repo:
            # `HermesImporter` <-> `HermesImporter.import_all` at 0.953 and
            # `LedgerReconstructor` <-> `LedgerReconstructor.reconstruct` at
            # 0.961, both in the top eight results. Markdown sections nest the
            # same way. Containment is the test rather than `parent_symbol`
            # because it also catches grandchildren.
            continue
        score = signatures[left].jaccard(signatures[right])
        if score < threshold:
            continue
        pairs.append(
            ClonePair(
                symbol_a=left,
                symbol_b=right,
                content_hash_a=row_a.content_hash,
                content_hash_b=row_b.content_hash,
                qualified_name_a=row_a.qualified_name,
                qualified_name_b=row_b.qualified_name,
                file_path_a=row_a.file_path,
                file_path_b=row_b.file_path,
                token_count_a=token_counts[left],
                token_count_b=token_counts[right],
                jaccard=score,
            )
        )
    pairs.sort(key=lambda pair: (-pair.jaccard, pair.qualified_name_a, pair.qualified_name_b))

    report = CloneReport(
        pairs=tuple(pairs),
        symbols_considered=len(signatures),
        symbols_skipped_short=skipped_short,
        symbols_unreadable=unreadable,
        candidate_pairs=len(candidates),
        threshold=threshold,
        engine_index_version=engine_index_version,
        signatures_reused=reused,
        signatures_computed=len(signatures) - reused,
        symbols_read=len(tokens_by_symbol),
    )
    _persist(root, repo_id or "", report, by_id, token_counts, signatures)
    logger.info(
        "clones: %d pairs from %d candidates over %d symbols "
        "(index_version %d, %d signatures reused, %d computed, %d symbols read)",
        len(pairs),
        len(candidates),
        len(signatures),
        engine_index_version,
        reused,
        len(signatures) - reused,
        len(tokens_by_symbol),
    )
    return report


def _persist(
    repo_root: Path,
    repo_id: str,
    report: CloneReport,
    by_id: dict[str, SymbolRow],
    token_counts: dict[str, int],
    signatures: dict[str, MinHash],
) -> None:
    conn = open_sidecar(repo_root)
    try:
        conn.execute("DELETE FROM symbol_clones WHERE repo_id = ?", (repo_id,))
        conn.executemany(
            "INSERT INTO symbol_clones (repo_id, symbol_a, symbol_b, content_hash_a, content_hash_b, "
            "qualified_name_a, qualified_name_b, file_path_a, file_path_b, "
            "token_count_a, token_count_b, jaccard) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    repo_id,
                    pair.symbol_a,
                    pair.symbol_b,
                    pair.content_hash_a,
                    pair.content_hash_b,
                    pair.qualified_name_a,
                    pair.qualified_name_b,
                    pair.file_path_a,
                    pair.file_path_b,
                    pair.token_count_a,
                    pair.token_count_b,
                    pair.jaccard,
                )
                for pair in report.pairs
            ],
        )

        # Signatures outlive one build, so the cache is replaced for symbols the
        # engine still knows about and dropped for the rest. Leaving deleted
        # symbols behind would grow the table without bound and let a recycled
        # symbol_id collide with a stale entry.
        try:
            conn.execute("DELETE FROM symbol_signatures WHERE repo_id = ?", (repo_id,))
            conn.executemany(
                "INSERT INTO symbol_signatures (repo_id, symbol_id, content_hash, token_count, signature) "
                "VALUES (?, ?, ?, ?, ?)",
                # Every symbol examined this pass is recorded, signed or not.
                # A short symbol with an empty signature says "looked at, not
                # worth signing", which keeps it out of the next pass's reads
                # *and* keeps the coverage figure honest.
                [
                    (
                        repo_id,
                        symbol_id,
                        by_id[symbol_id].content_hash,
                        token_count,
                        _pack(signatures[symbol_id]) if symbol_id in signatures else b"",
                    )
                    for symbol_id, token_count in token_counts.items()
                    if symbol_id in by_id
                ],
            )
        except sqlite3.OperationalError:
            # Symmetrical with the read: the cache is an accelerator, so failing
            # to write it costs the next build time and costs this one nothing.
            # The pairs above are already committed-in-transaction and correct.
            logger.warning("clones: signature cache not written", exc_info=True)
        conn.commit()
        stamp(conn, CLONES_TABLE, report.engine_index_version)
    finally:
        conn.close()


def open_clone_table(repo_root: Path | str) -> tuple[sqlite3.Connection, int]:
    """Open the sidecar for reading; raise only if nothing ever built the table.

    Returns the connection and the index generation the table was built from.
    The single definition of "never built" and how it is worded -- two copies of
    that check meant two copies of the message, each pinned by its own test.

    A *superseded* table is deliberately not an error here. Validity is decided
    per pair against ``symbols.content_hash`` by :func:`current_pairs`, so a
    reindex costs coverage rather than the whole answer.
    """
    conn = open_sidecar(repo_root)
    try:
        recorded = stamp_of(conn, CLONES_TABLE)
        if recorded is None:
            raise ClonesStale("the clone table has never been built")
    except BaseException:
        conn.close()
        raise
    return conn, recorded.engine_index_version


def current_symbol_hashes(repo_root: Path | str = ".") -> dict[str, str]:
    """``symbol_id`` -> ``content_hash`` as the engine holds it right now."""
    with CodeIntelStore(repo_root) as store:
        if store.repo_id_or_none() is None:
            return {}
        return {row.symbol_id: row.content_hash for row in store.symbols()}


def signature_coverage(repo_root: Path | str = ".") -> tuple[int, int]:
    """``(symbols examined by the last build, symbols the engine holds now)``.

    The signature cache records every symbol a build looked at, keyed by the
    content it looked at. So the count of live symbols whose hash the cache still
    matches is exactly how much of the repository the stored answer covers --
    which is what makes a *missing* clone pair readable: at full coverage it
    means measured-and-none, below it means never examined.
    """
    with CodeIntelStore(repo_root) as store:
        repo_id = store.repo_id_or_none()
        symbols = store.symbols() if repo_id is not None else []
    if not symbols:
        return 0, 0
    cached = _load_signature_cache(Path(repo_root).resolve(), repo_id or "")
    return sum(1 for row in symbols if _cache_hit(cached, row)), len(symbols)


def load_clones(repo_root: Path | str = ".", limit: int = 50) -> CloneView:
    """Stored pairs that still hold, with the coverage needed to read them.

    Raises :class:`ClonesStale` only when nothing has ever built the table --
    the one case where an empty list would be indistinguishable from "this code
    has no duplicates". Once built, every pair is re-checked against the live
    content hashes and the superseded ones are dropped, so a reindex narrows the
    answer instead of destroying it.
    """
    root = Path(repo_root).resolve()
    with CodeIntelStore(root) as store:
        engine_index_version = store.engine_state("index_version")
        repo_id = store.repo_id_or_none()
        symbols = store.symbols() if repo_id is not None else []
    hashes = {row.symbol_id: row.content_hash for row in symbols}

    conn, built_from = open_clone_table(root)
    try:
        # No LIMIT in SQL: superseded rows are dropped after the read, so
        # limiting first would silently return fewer pairs than asked for
        # whenever any of the top-scoring ones had gone stale.
        rows = conn.execute(
            "SELECT symbol_a, symbol_b, content_hash_a, content_hash_b, qualified_name_a, "
            "qualified_name_b, file_path_a, file_path_b, token_count_a, token_count_b, jaccard "
            "FROM symbol_clones WHERE repo_id = ? "
            "ORDER BY jaccard DESC, qualified_name_a, qualified_name_b",
            (repo_id or "",),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise ClonesStale(f"the clone table is unreadable ({exc})") from exc
    finally:
        conn.close()

    recorded = [_row_to_pair(row) for row in rows]
    current = [pair for pair in recorded if pair.still_current(hashes)]

    cached = _load_signature_cache(root, repo_id or "")
    examined = sum(1 for row in symbols if _cache_hit(cached, row))

    return CloneView(
        pairs=tuple(current[: max(0, int(limit))]),
        recorded=len(recorded),
        superseded=len(recorded) - len(current),
        symbols_total=len(hashes),
        symbols_unchanged=examined,
        built_from_index_version=built_from,
        engine_index_version=engine_index_version,
    )


def _row_to_pair(row: sqlite3.Row) -> ClonePair:
    return ClonePair(
        symbol_a=str(row["symbol_a"]),
        symbol_b=str(row["symbol_b"]),
        content_hash_a=str(row["content_hash_a"]),
        content_hash_b=str(row["content_hash_b"]),
        qualified_name_a=str(row["qualified_name_a"]),
        qualified_name_b=str(row["qualified_name_b"]),
        file_path_a=str(row["file_path_a"]),
        file_path_b=str(row["file_path_b"]),
        token_count_a=int(row["token_count_a"]),
        token_count_b=int(row["token_count_b"]),
        jaccard=float(row["jaccard"]),
    )
