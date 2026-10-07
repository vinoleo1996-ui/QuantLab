"""Append-only evidence ledger for forward-captured research signals.

This module deliberately sits beside, rather than inside, the frozen strategy,
router, market dataset, or paper-account code.  It records what was actually
captured, how it was mapped, whether a signal was observed or abstained, and
only later its settled outcome.  Nothing here creates an order or changes a
portfolio.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import uuid

import pandas as pd

from quantlab.data.contracts import timestamp


LEDGER_VERSION = "research-signal-ledger-v1"
_KINDS = ("scope", "receipt", "entity", "observation", "outcome")
_TERMINAL = {
    "scope": ("declared",),
    "entity": ("resolved", "abstained"),
    "observation": ("observed", "abstained"),
    "outcome": ("settled", "unavailable"),
}
_RECEIPT_ROLES = ("source_response", "transport_evidence")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_" + field)
    return value.strip()


def _sha256_hex(value: Any, field: str) -> str:
    """Return one canonical SHA-256 digest, never an opaque label.

    A scope source hash is an evidence identity, not a human-readable source
    name.  The ledger cannot prove which external bytes produced a caller's
    digest, but it can reject values that are not even a SHA-256 digest and
    store the canonical lower-case representation.
    """
    digest = _text(value, field).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("invalid_" + field)
    return digest


def _time(value: Any, field: str) -> pd.Timestamp:
    try:
        result = timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_" + field) from exc
    if pd.isna(result):
        raise ValueError("invalid_" + field)
    return result


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat()


def _normal(value: Any) -> Any:
    """Turn a research record into strict, deterministic JSON data.

    ``json.dumps(default=str)`` is useful for reports but too permissive for a
    ledger: NaN, arbitrary objects and implicit timezone conversion would make
    a content hash misleading.  This normalizer accepts only portable values.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite_ledger_value")
        return value
    if isinstance(value, pd.Timestamp):
        return _iso(_time(value, "timestamp"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normal(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _normal(item())
    raise TypeError("unsupported_ledger_value:" + type(value).__name__)


def _canonical_json(value: Any) -> str:
    return json.dumps(_normal(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: Any) -> str:
    return _sha256(_canonical_json(value).encode("utf-8"))


class ResearchSignalLedger:
    """Content-addressed, append-only records for research-only signal inputs.

    Callers supply source/capture times explicitly, but the ledger owns the
    time at which an immutable event is actually written.  Every externally
    observed value has a raw receipt hash, a source/origin identity, and a
    decision cutoff.  Expected data-quality failures become explicit
    abstentions, never a numeric zero or an implicit fallback score.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ledger_id: str = LEDGER_VERSION,
        cutoff_hour: int = 18,
        now_fn: Callable[[], Any] | None = None,
    ):
        if not isinstance(cutoff_hour, int) or cutoff_hour < 0 or cutoff_hour > 23:
            raise ValueError("invalid_decision_cutoff")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_root = self.root / "blobs"
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ledger.sqlite3"
        self.ledger_id = _text(ledger_id, "ledger_id")
        self.cutoff_hour = cutoff_hour
        self.now_fn = now_fn or (lambda: pd.Timestamp.now(tz="Asia/Shanghai"))
        self._integrity_checked = False
        self._integrity_tip: str | None = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL CHECK(kind IN ('scope','receipt','entity','observation','outcome')),
                    logical_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    available_at TEXT,
                    body_json TEXT NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS events_kind_logical ON events(kind, logical_key, seq);
                CREATE TRIGGER IF NOT EXISTS events_no_update
                BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'append_only_events'); END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete
                BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'append_only_events'); END;
                CREATE TRIGGER IF NOT EXISTS meta_no_update
                BEFORE UPDATE ON meta
                BEGIN SELECT RAISE(ABORT, 'append_only_meta'); END;
                CREATE TRIGGER IF NOT EXISTS meta_no_delete
                BEFORE DELETE ON meta
                BEGIN SELECT RAISE(ABORT, 'append_only_meta'); END;
                """
            )
            identity = dict(ledger_id=self.ledger_id, version=LEDGER_VERSION, cutoff_hour=self.cutoff_hour)
            value = _canonical_json(identity)
            row = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if row is None:
                db.execute("INSERT INTO meta(key,value) VALUES ('identity',?)", (value,))
            elif row["value"] != value:
                raise ValueError("ledger_identity_changed")

    def _now(self, now: Any | None) -> pd.Timestamp:
        return _time(self.now_fn() if now is None else now, "recorded_at")

    def _write_now(self) -> pd.Timestamp:
        """Return the ledger-owned clock for a signal-evidence write.

        ``now`` remains on the public signal APIs for call-site compatibility,
        but it is not evidence of when a write happened: a caller could
        otherwise backdate a receipt, entity mapping, or feature past a frozen
        decision boundary.  Tests that need a deterministic clock must inject
        ``now_fn`` when constructing the ledger.

        Outcomes retain their existing explicit-maturity clock semantics; this
        helper is deliberately limited to the forward signal evidence chain.
        """
        return _time(self.now_fn(), "recorded_at")

    def runtime_now(self) -> pd.Timestamp:
        """Read the ledger-owned runtime clock for a live materialisation step.

        Transport callers must not substitute a page's capture timestamp for
        the time at which evidence actually reached the append-only ledger.
        Tests can still inject ``now_fn`` when constructing the ledger.
        """
        return self._write_now()

    @staticmethod
    def _event_envelope(
        *,
        event_id: str,
        kind: str,
        logical_key: str,
        status: str,
        recorded_at: str,
        available_at: str | None,
        body_sha256: str,
        previous_hash: str | None,
    ) -> dict[str, Any]:
        return dict(
            event_id=event_id,
            kind=kind,
            logical_key=logical_key,
            status=status,
            recorded_at=recorded_at,
            available_at=available_at,
            body_sha256=body_sha256,
            previous_hash=previous_hash,
        )

    def _event_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        body = json.loads(row["body_json"])
        result = dict(
            seq=int(row["seq"]),
            event_id=row["event_id"],
            kind=row["kind"],
            logical_key=row["logical_key"],
            status=row["status"],
            recorded_at=row["recorded_at"],
            available_at=row["available_at"],
            body=body,
            event_hash=row["event_hash"],
        )
        if row["kind"] == "receipt":
            result["canonical_receipt_id"] = body.get("duplicate_of", row["event_id"])
        return result

    def _verify_db(self, db: sqlite3.Connection) -> list[str]:
        reasons: list[str] = []
        previous: str | None = None
        for row in db.execute("SELECT * FROM events ORDER BY seq"):
            try:
                body = json.loads(row["body_json"])
                if _canonical_json(body) != row["body_json"]:
                    raise ValueError("body_not_canonical")
                body_hash = _sha256(row["body_json"].encode("utf-8"))
                if body_hash != row["body_sha256"]:
                    raise ValueError("body_hash_mismatch")
                envelope = self._event_envelope(
                    event_id=row["event_id"],
                    kind=row["kind"],
                    logical_key=row["logical_key"],
                    status=row["status"],
                    recorded_at=row["recorded_at"],
                    available_at=row["available_at"],
                    body_sha256=row["body_sha256"],
                    previous_hash=previous,
                )
                if row["previous_hash"] != previous or _hash_json(envelope) != row["event_hash"]:
                    raise ValueError("hash_chain_mismatch")
                if row["kind"] == "receipt":
                    raw_hash = _text(body.get("raw_sha256"), "receipt_raw_sha256")
                    expected = Path("blobs") / raw_hash
                    if Path(body.get("blob_path", "")) != expected:
                        raise ValueError("blob_path_mismatch")
                    blob = (self.root / expected).resolve()
                    if not blob.is_relative_to(self.blob_root.resolve()):
                        raise ValueError("blob_path_escape")
                    if not blob.is_file():
                        raise ValueError("blob_missing")
                    if _sha256(blob.read_bytes()) != raw_hash:
                        raise ValueError("blob_hash_mismatch")
                if row["kind"] == "scope":
                    members = self._target_keys(body.get("entity_keys", []))
                    if members != body.get("entity_keys"):
                        raise ValueError("scope_members_not_canonical")
                    if _hash_json(members) != body.get("membership_sha256"):
                        raise ValueError("scope_membership_hash_mismatch")
                    source_hash = _sha256_hex(body.get("source_hash"), "scope_source_hash")
                    if source_hash != body.get("source_hash"):
                        raise ValueError("scope_source_hash_not_canonical")
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                reasons.append("event:" + str(row["seq"]) + ":" + str(exc))
            previous = row["event_hash"]
        return reasons

    def verify(self) -> dict[str, Any]:
        with self._connect() as db:
            reasons = self._verify_db(db)
            row = db.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            count = db.execute("SELECT COUNT(*) AS count FROM events").fetchone()["count"]
        return dict(
            verified=not reasons,
            reasons=reasons,
            events=int(count),
            journal_hash=None if row is None else row["event_hash"],
            ledger_id=self.ledger_id,
        )

    def _assert_integrity(self, db: sqlite3.Connection) -> None:
        """Re-read every blob only when the chain tip has changed.

        ``verify`` still walks the blobs every time.  Repeated receipt exports
        during one score reuse the last successful check until a new event is
        appended, because hashing the same raw file once per row does not
        change the answer inside a single process.
        """
        tip = db.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        tip_hash = None if tip is None else tip["event_hash"]
        if self._integrity_checked and tip_hash == self._integrity_tip:
            return
        reasons = self._verify_db(db)
        if reasons:
            self._integrity_checked = False
            raise ValueError("ledger_integrity_failed:" + "|".join(reasons))
        self._integrity_tip = tip_hash
        self._integrity_checked = True

    def _append_in_tx(
        self,
        db: sqlite3.Connection,
        *,
        kind: str,
        logical_key: str,
        status: str,
        body: Mapping[str, Any],
        available_at: pd.Timestamp | None,
        now: pd.Timestamp,
    ) -> dict[str, Any]:
        if kind not in _KINDS:
            raise ValueError("unknown_ledger_event_kind")
        body_json = _canonical_json(body)
        body_sha256 = _sha256(body_json.encode("utf-8"))
        prior = db.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous_hash = None if prior is None else prior["event_hash"]
        event_id = uuid.uuid4().hex
        envelope = self._event_envelope(
            event_id=event_id,
            kind=kind,
            logical_key=logical_key,
            status=status,
            recorded_at=_iso(now),
            available_at=None if available_at is None else _iso(available_at),
            body_sha256=body_sha256,
            previous_hash=previous_hash,
        )
        event_hash = _hash_json(envelope)
        cursor = db.execute(
            """INSERT INTO events(event_id,kind,logical_key,status,recorded_at,available_at,body_json,body_sha256,previous_hash,event_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id,
                kind,
                logical_key,
                status,
                envelope["recorded_at"],
                envelope["available_at"],
                body_json,
                body_sha256,
                previous_hash,
                event_hash,
            ),
        )
        row = db.execute("SELECT * FROM events WHERE seq=?", (cursor.lastrowid,)).fetchone()
        return self._event_dict(row)

    def _terminal_in_tx(
        self,
        db: sqlite3.Connection,
        *,
        kind: str,
        logical_key: str,
        status: str,
        body: Mapping[str, Any],
        available_at: pd.Timestamp | None,
        now: pd.Timestamp,
    ) -> dict[str, Any]:
        terminal = _TERMINAL[kind]
        marks = ",".join("?" for _ in terminal)
        existing = db.execute(
            "SELECT * FROM events WHERE kind=? AND logical_key=? AND status IN (" + marks + ") ORDER BY seq LIMIT 1",
            (kind, logical_key, *terminal),
        ).fetchone()
        body_json = _canonical_json(body)
        expected_available = None if available_at is None else _iso(available_at)
        if existing is not None:
            if (
                existing["status"] == status
                and existing["body_json"] == body_json
                and existing["available_at"] == expected_available
            ):
                return self._event_dict(existing)
            raise ValueError("immutable_event_conflict")
        return self._append_in_tx(
            db,
            kind=kind,
            logical_key=logical_key,
            status=status,
            body=body,
            available_at=available_at,
            now=now,
        )

    def _store_blob(self, raw: bytes) -> tuple[str, Path]:
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise ValueError("receipt_raw_bytes_required")
        value = bytes(raw)
        digest = _sha256(value)
        path = self.blob_root / digest
        if path.exists():
            if not path.is_file() or _sha256(path.read_bytes()) != digest:
                raise ValueError("existing_blob_integrity_failed")
            return digest, path
        fd, temp_name = tempfile.mkstemp(prefix=".blob-", dir=self.blob_root)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(value)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp_name, path)
        finally:
            if Path(temp_name).exists():
                Path(temp_name).unlink()
        if _sha256(path.read_bytes()) != digest:
            raise ValueError("blob_write_integrity_failed")
        return digest, path

    def _receipt_row_in_tx(
        self,
        db: sqlite3.Connection,
        event_id: str,
        *,
        require_source_response: bool = True,
    ) -> sqlite3.Row:
        seen: set[str] = set()
        current = _text(event_id, "receipt_id")
        while True:
            if current in seen:
                raise ValueError("receipt_duplicate_cycle")
            seen.add(current)
            row = db.execute("SELECT * FROM events WHERE event_id=?", (current,)).fetchone()
            if row is None or row["kind"] != "receipt":
                raise ValueError("unknown_receipt")
            body = json.loads(row["body_json"])
            if row["status"] == "duplicate":
                current = _text(body.get("duplicate_of"), "duplicate_of")
                continue
            if row["status"] != "accepted":
                raise ValueError("receipt_not_accepted")
            if require_source_response and body.get("receipt_role", "source_response") != "source_response":
                raise ValueError("receipt_not_source_response")
            return row

    def _receipt_rows_in_tx(self, db: sqlite3.Connection, event_ids: Sequence[str]) -> list[sqlite3.Row]:
        rows: dict[str, sqlite3.Row] = {}
        for event_id in event_ids:
            row = self._receipt_row_in_tx(db, event_id)
            rows[row["event_id"]] = row
        return [rows[key] for key in sorted(rows)]

    @staticmethod
    def _event_body(row: sqlite3.Row) -> dict[str, Any]:
        return json.loads(row["body_json"])

    def _scope_in_tx(self, db: sqlite3.Connection, scope_id: str, decision_at: pd.Timestamp) -> tuple[sqlite3.Row, dict[str, Any]]:
        logical_key = _hash_json(dict(scope_id=scope_id, decision_at=_iso(decision_at)))
        row = db.execute(
            "SELECT * FROM events WHERE kind='scope' AND logical_key=? AND status='declared'", (logical_key,)
        ).fetchone()
        if row is None:
            raise ValueError("scope_not_declared")
        body = self._event_body(row)
        if body.get("scope_id") != scope_id or body.get("decision_at") != _iso(decision_at):
            raise ValueError("scope_identity_mismatch")
        return row, body

    @staticmethod
    def _target_keys(values: Sequence[str]) -> list[str]:
        keys = [_text(value, "target_entity_key") for value in values]
        if not keys:
            raise ValueError("missing_target_entity_keys")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate_target_entity_key")
        return sorted(keys)

    def _check_live_decision(self, decision_at: pd.Timestamp, now: pd.Timestamp) -> None:
        if (
            decision_at.hour != self.cutoff_hour
            or decision_at.minute != 0
            or decision_at.second != 0
            or decision_at.microsecond != 0
        ):
            raise ValueError("decision_cutoff_required")
        # A decision may be materialised before its 18:00 freeze, and an
        # abstention may be appended after the freeze to explain a late or
        # failed input.  We only permit the live decision session itself;
        # neither a prior-day prewrite nor a next-day historical backfill is a
        # valid signal write.
        if now.date() != decision_at.date():
            raise ValueError("historical_observation_forbidden")

    @staticmethod
    def _validate_extractor(extractor: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(extractor, Mapping):
            raise ValueError("extractor_metadata_required")
        result = _normal(extractor)
        kind = _text(result.get("kind"), "extractor_kind")
        _text(result.get("version"), "extractor_version")
        if kind == "llm":
            responses = result.get("response_receipt_ids")
            if (
                not result.get("model_id")
                or not result.get("prompt_sha256")
                or not isinstance(responses, list)
                or not responses
            ):
                raise ValueError("llm_extractor_metadata_missing")
            _text(result.get("model_id"), "extractor_model_id")
            # A prompt identity is evidence, not a display label.  Without a
            # real digest a caller could silently change an LLM instruction
            # while leaving a plausible-looking string in the ledger.
            _sha256_hex(result.get("prompt_sha256"), "extractor_prompt_sha256")
        return result

    def _ensure_observation_targets_unclaimed(
        self,
        db: sqlite3.Connection,
        *,
        logical_key: str,
        scope_id: str,
        decision_at: pd.Timestamp,
        feature_id: str,
        target_entity_keys: Sequence[str],
    ) -> None:
        target = set(target_entity_keys)
        rows = db.execute("SELECT * FROM events WHERE kind='observation' AND status IN ('observed','abstained')").fetchall()
        for row in rows:
            if row["logical_key"] == logical_key:
                continue
            body = self._event_body(row)
            if (
                body.get("scope_id") == scope_id
                and body.get("decision_at") == _iso(decision_at)
                and body.get("feature_id") == feature_id
                and target.intersection(body.get("target_entity_keys", []))
            ):
                raise ValueError("observation_entity_conflict")

    def _append_observation_terminal_in_tx(
        self,
        db: sqlite3.Connection,
        *,
        observation_key: str,
        scope_id: str,
        decision_at: pd.Timestamp,
        feature_id: str,
        target_entity_keys: Sequence[str],
        status: str,
        body: Mapping[str, Any],
        now: pd.Timestamp,
    ) -> dict[str, Any]:
        logical_key = _hash_json(
            dict(observation_key=observation_key, scope_id=scope_id, decision_at=_iso(decision_at))
        )
        existing = db.execute(
            "SELECT * FROM events WHERE kind='observation' AND logical_key=? AND status IN ('observed','abstained')",
            (logical_key,),
        ).fetchone()
        if existing is not None:
            expected = _canonical_json(body)
            if existing["status"] == status and existing["body_json"] == expected:
                return self._event_dict(existing)
            raise ValueError("immutable_event_conflict")
        self._ensure_observation_targets_unclaimed(
            db,
            logical_key=logical_key,
            scope_id=scope_id,
            decision_at=decision_at,
            feature_id=feature_id,
            target_entity_keys=target_entity_keys,
        )
        return self._append_in_tx(
            db,
            kind="observation",
            logical_key=logical_key,
            status=status,
            body=body,
            available_at=decision_at,
            now=now,
        )

    def _abstain_in_tx(
        self,
        db: sqlite3.Connection,
        *,
        observation_key: str,
        scope_id: str,
        decision_at: pd.Timestamp,
        feature_id: str,
        target_entity_keys: Sequence[str],
        stage: str,
        reason: str,
        receipt_ids: Sequence[str],
        entity_ids: Sequence[str],
        extractor: Mapping[str, Any] | None,
        now: pd.Timestamp,
    ) -> dict[str, Any]:
        body = dict(
            observation_key=observation_key,
            scope_id=scope_id,
            decision_at=_iso(decision_at),
            feature_id=feature_id,
            target_entity_keys=list(target_entity_keys),
            receipt_ids=list(receipt_ids),
            entity_ids=list(entity_ids),
            stage=_text(stage, "abstention_stage"),
            reason=_text(reason, "abstention_reason"),
            value=None,
            extractor=None if extractor is None else _normal(extractor),
        )
        return self._append_observation_terminal_in_tx(
            db,
            observation_key=observation_key,
            scope_id=scope_id,
            decision_at=decision_at,
            feature_id=feature_id,
            target_entity_keys=target_entity_keys,
            status="abstained",
            body=body,
            now=now,
        )

    def declare_scope(
        self,
        scope_id: str,
        decision_at: Any,
        entity_keys: Sequence[str],
        *,
        definition_version: str,
        source_hash: str,
        now: Any | None = None,
    ) -> dict[str, Any]:
        scope_id = _text(scope_id, "scope_id")
        decision = _time(decision_at, "decision_at")
        # The scheduler-provided ``now`` is not a trustworthy proof that a
        # scope existed before the cutoff.  Only the ledger-owned write clock
        # may establish that fact.
        recorded = self._write_now()
        if decision.hour != self.cutoff_hour or decision.minute or decision.second or decision.microsecond:
            raise ValueError("decision_cutoff_required")
        if recorded.date() != decision.date() or recorded > decision:
            raise ValueError("scope_must_be_declared_by_cutoff")
        members = self._target_keys(entity_keys)
        membership_sha256 = _hash_json(members)
        body = dict(
            scope_id=scope_id,
            decision_at=_iso(decision),
            entity_keys=members,
            membership_sha256=membership_sha256,
            definition_version=_text(definition_version, "scope_definition_version"),
            source_hash=_sha256_hex(source_hash, "scope_source_hash"),
        )
        logical_key = _hash_json(dict(scope_id=scope_id, decision_at=_iso(decision)))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._assert_integrity(db)
                result = self._terminal_in_tx(
                    db,
                    kind="scope",
                    logical_key=logical_key,
                    status="declared",
                    body=body,
                    available_at=decision,
                    now=recorded,
                )
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise

    def record_receipt(
        self,
        *,
        source_id: str,
        source_family: str,
        origin_id: str,
        canonical_url: str,
        raw: bytes,
        content_type: str,
        published_at: Any,
        first_captured_at: Any,
        available_at: Any,
        availability_basis: str,
        receipt_role: str = "source_response",
        metadata: Mapping[str, Any] | None = None,
        now: Any | None = None,
    ) -> dict[str, Any]:
        # ``recorded_at`` is an immutable evidence fact, not a caller-supplied
        # capture clock.  A late receipt remains recordable for audit, but its
        # actual ledger write time will later make it ineligible for a frozen
        # decision.
        return self._record_receipt_at(
            source_id=source_id,
            source_family=source_family,
            origin_id=origin_id,
            canonical_url=canonical_url,
            raw=raw,
            content_type=content_type,
            published_at=published_at,
            first_captured_at=first_captured_at,
            available_at=available_at,
            availability_basis=availability_basis,
            receipt_role=receipt_role,
            metadata=metadata,
            recorded=self._write_now(),
        )

    def _record_receipt_at(
        self,
        *,
        source_id: str,
        source_family: str,
        origin_id: str,
        canonical_url: str,
        raw: bytes,
        content_type: str,
        published_at: Any,
        first_captured_at: Any,
        available_at: Any,
        availability_basis: str,
        receipt_role: str = "source_response",
        metadata: Mapping[str, Any] | None = None,
        recorded: pd.Timestamp,
    ) -> dict[str, Any]:
        """Append one receipt using a timestamp created by this ledger only.

        This is intentionally private.  It lets a compound ledger operation
        build a payload and its enclosing receipt with exactly the same
        ledger-owned timestamp, without reopening a public caller-controlled
        write-clock path.
        """
        recorded = _time(recorded, "recorded_at")
        source_id = _text(source_id, "source_id")
        source_family = _text(source_family, "source_family")
        origin_id = _text(origin_id, "origin_id")
        canonical_url = _text(canonical_url, "canonical_url")
        content_type = _text(content_type, "content_type")
        basis = _text(availability_basis, "availability_basis")
        role = _text(receipt_role, "receipt_role")
        if role not in _RECEIPT_ROLES:
            raise ValueError("invalid_receipt_role")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("receipt_metadata_mapping_required")
        normalized_metadata = None if metadata is None else _normal(metadata)
        captured = _time(first_captured_at, "first_captured_at")
        available = _time(available_at, "available_at")
        # A source without an original publication timestamp cannot be used as
        # a forward signal.  Callers can still record an explicit observation
        # abstention, but may not manufacture an apparently eligible receipt.
        if published_at is None:
            raise ValueError("missing_published_at")
        published = _time(published_at, "published_at")
        if published > captured:
            raise ValueError("published_after_capture")
        if captured > recorded:
            raise ValueError("capture_after_record")
        if available < captured:
            raise ValueError("availability_before_capture")
        raw_sha256, _ = self._store_blob(raw)
        logical_key = _hash_json(
            dict(source_family=source_family, origin_id=origin_id, raw_sha256=raw_sha256)
        )
        base = dict(
            source_id=source_id,
            source_family=source_family,
            origin_id=origin_id,
            canonical_url=canonical_url,
            content_type=content_type,
            raw_sha256=raw_sha256,
            blob_path=str(Path("blobs") / raw_sha256),
            byte_count=len(raw),
            published_at=_iso(published),
            first_captured_at=_iso(captured),
            availability_basis=basis,
            receipt_role=role,
            metadata=normalized_metadata,
        )
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._assert_integrity(db)
                existing = db.execute(
                    "SELECT * FROM events WHERE kind='receipt' AND logical_key=? AND status='accepted' ORDER BY seq LIMIT 1",
                    (logical_key,),
                ).fetchone()
                if existing is None:
                    result = self._append_in_tx(
                        db,
                        kind="receipt",
                        logical_key=logical_key,
                        status="accepted",
                        body=base,
                        available_at=available,
                        now=recorded,
                    )
                else:
                    body = dict(base, duplicate_of=existing["event_id"])
                    result = self._append_in_tx(
                        db,
                        kind="receipt",
                        logical_key=logical_key,
                        status="duplicate",
                        body=body,
                        available_at=available,
                        now=recorded,
                    )
                    result["canonical_receipt_id"] = existing["event_id"]
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise

    def record_transport_evidence(
        self,
        *,
        collection_id: str,
        source_manifest_sha256: str,
        source_manifest_event_sha256: str,
        session: Any,
        decision_at: Any,
        status: str,
        failure_reason: str | None,
        attempts: Sequence[Mapping[str, Any]],
        first_captured_at: Any,
        available_at: Any,
        reported_at: Any | None = None,
        now: Any | None = None,
    ) -> dict[str, Any]:
        """Persist a raw transport-attempt envelope without making it a signal source.

        The envelope is deliberately stored as a receipt blob so a process
        restart still retains every attempted page, capture clock, URL, error
        label and successful-page hash.  Its ``transport_evidence`` role is
        rejected by entity, feature and outcome APIs: it proves collection
        history, not market content.
        """
        # Resolve the ledger-owned clock before constructing the immutable
        # payload.  This is the actual evidence-write time, deliberately
        # distinct from an HTTP response's received/capture timestamp.
        recorded = self._write_now()
        identifier = _text(collection_id, "transport_collection_id")
        manifest_sha = _sha256_hex(source_manifest_sha256, "transport_manifest_sha256")
        manifest_event = _sha256_hex(source_manifest_event_sha256, "transport_manifest_event_sha256")
        if status not in ("prepared", "failed"):
            raise ValueError("invalid_transport_status")
        if status == "failed":
            reason = _text(failure_reason, "transport_failure_reason")
        elif failure_reason is not None:
            raise ValueError("prepared_transport_must_not_have_failure_reason")
        else:
            reason = None
        if isinstance(attempts, (str, bytes)) or not isinstance(attempts, Sequence):
            raise ValueError("transport_attempts_required")
        has_attempts = bool(attempts)
        failure_reported_at = None if reported_at is None else _time(reported_at, "transport_failure_reported_at")
        if not has_attempts:
            # A transport can fail before a worker ever opens a page (for
            # example, registration or credential setup can fail).  That
            # failure is still material collection evidence, but only a
            # concrete failure clock can anchor a zero-page envelope.
            if status != "failed" or failure_reported_at is None:
                raise ValueError("transport_attempts_required")
            captured = failure_reported_at
            available = failure_reported_at
        else:
            captured = _time(first_captured_at, "transport_first_captured_at")
            available = _time(available_at, "transport_available_at")
        if failure_reported_at is not None and failure_reported_at > recorded:
            raise ValueError("transport_failure_reported_after_record")
        if available < captured:
            raise ValueError("transport_available_before_first_capture")
        normalized_attempts = _normal(list(attempts))
        payload = dict(
            schema="quantlab.forward-market-transport-evidence.v1",
            collection_id=identifier,
            source_manifest_sha256=manifest_sha,
            source_manifest_event_sha256=manifest_event,
            session=_iso(_time(session, "transport_session")),
            decision_at=_iso(_time(decision_at, "transport_decision_at")),
            status=status,
            failure_reason=reason,
            failure_reported_at=None if failure_reported_at is None else _iso(failure_reported_at),
            receipt_written_at=_iso(recorded),
            attempts=normalized_attempts,
        )
        raw = _canonical_json(payload).encode("utf-8")
        return self._record_receipt_at(
            source_id="quantlab.forward-market-transport",
            source_family="quantlab.forward-market-transport",
            origin_id="collection:" + identifier,
            canonical_url="urn:quantlab:forward-market-transport:" + identifier,
            raw=raw,
            content_type="application/vnd.quantlab.forward-market-transport-evidence+json",
            published_at=captured,
            first_captured_at=captured,
            available_at=available,
            availability_basis="transport_attempt_clock",
            receipt_role="transport_evidence",
            metadata={"transport": payload},
            recorded=recorded,
        )

    def record_entity(
        self,
        receipt_id: str,
        *,
        entity_type: str,
        canonical_id: str | None,
        display_name: str | None,
        mapping_method: str,
        mapping_version: str,
        status: str = "resolved",
        reason: str | None = None,
        now: Any | None = None,
    ) -> dict[str, Any]:
        # Entity mappings participate in decision eligibility, so their
        # write time must be ledger-owned for the same reason as receipts.
        recorded = self._write_now()
        entity_type = _text(entity_type, "entity_type")
        display = None if display_name is None else _text(display_name, "entity_display_name")
        if status not in ("resolved", "abstained"):
            raise ValueError("invalid_entity_status")
        if status == "resolved":
            canonical = _text(canonical_id, "canonical_entity_id")
            abstention_reason = None
        else:
            canonical = None
            abstention_reason = _text(reason, "entity_abstention_reason")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._assert_integrity(db)
                receipt = self._receipt_row_in_tx(db, receipt_id)
                receipt_id = receipt["event_id"]
                body = dict(
                    receipt_id=receipt_id,
                    entity_type=entity_type,
                    canonical_id=canonical,
                    display_name=display,
                    mapping_method=_text(mapping_method, "mapping_method"),
                    mapping_version=_text(mapping_version, "mapping_version"),
                    reason=abstention_reason,
                )
                logical_key = _hash_json(
                    dict(
                        receipt_id=receipt_id,
                        entity_type=entity_type,
                        display_name=display,
                        mapping_version=body["mapping_version"],
                    )
                )
                result = self._terminal_in_tx(
                    db,
                    kind="entity",
                    logical_key=logical_key,
                    status=status,
                    body=body,
                    available_at=recorded,
                    now=recorded,
                )
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise

    def abstain(
        self,
        *,
        observation_key: str,
        scope_id: str,
        decision_at: Any,
        feature_id: str,
        target_entity_keys: Sequence[str],
        stage: str,
        reason: str,
        receipt_ids: Sequence[str] = (),
        entity_ids: Sequence[str] = (),
        extractor: Mapping[str, Any] | None = None,
        now: Any | None = None,
    ) -> dict[str, Any]:
        observation_key = _text(observation_key, "observation_key")
        scope_id = _text(scope_id, "scope_id")
        feature_id = _text(feature_id, "feature_id")
        decision = _time(decision_at, "decision_at")
        # An abstention never becomes a usable feature.  It may be written
        # after the decision freeze to retain the audit trail, but the event
        # timestamp must still come from the ledger rather than a caller.
        recorded = self._write_now()
        self._check_live_decision(decision, recorded)
        targets = self._target_keys(target_entity_keys)
        if extractor is not None:
            extractor = self._validate_extractor(extractor)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._assert_integrity(db)
                _, scope = self._scope_in_tx(db, scope_id, decision)
                if not set(targets).issubset(scope["entity_keys"]):
                    raise ValueError("observation_outside_scope")
                receipts = self._receipt_rows_in_tx(db, receipt_ids)
                entities = []
                for event_id in entity_ids:
                    row = db.execute("SELECT * FROM events WHERE event_id=?", (_text(event_id, "entity_id"),)).fetchone()
                    if row is None or row["kind"] != "entity":
                        raise ValueError("unknown_entity")
                    entities.append(row)
                result = self._abstain_in_tx(
                    db,
                    observation_key=observation_key,
                    scope_id=scope_id,
                    decision_at=decision,
                    feature_id=feature_id,
                    target_entity_keys=targets,
                    stage=stage,
                    reason=reason,
                    receipt_ids=[row["event_id"] for row in receipts],
                    entity_ids=sorted({row["event_id"] for row in entities}),
                    extractor=extractor,
                    now=recorded,
                )
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise

    def record_observation(
        self,
        *,
        observation_key: str,
        scope_id: str,
        decision_at: Any,
        feature_id: str,
        target_entity_keys: Sequence[str],
        receipt_ids: Sequence[str],
        entity_ids: Sequence[str],
        value: Any,
        unit: str,
        extractor: Mapping[str, Any],
        min_independent_sources: int = 1,
        now: Any | None = None,
    ) -> dict[str, Any]:
        observation_key = _text(observation_key, "observation_key")
        scope_id = _text(scope_id, "scope_id")
        feature_id = _text(feature_id, "feature_id")
        unit = _text(unit, "observation_unit")
        if not isinstance(min_independent_sources, int) or min_independent_sources < 1:
            raise ValueError("invalid_min_independent_sources")
        decision = _time(decision_at, "decision_at")
        # The feature's actual write time is the decision boundary.  Do not
        # let a scheduler/caller backdate an observed score by supplying a
        # convenient ``now`` argument.
        recorded = self._write_now()
        self._check_live_decision(decision, recorded)
        targets = self._target_keys(target_entity_keys)
        extractor = self._validate_extractor(extractor)
        # Validate values before any append: a malformed feature must not be
        # replaced by a plausible-looking abstention or a coerced zero.
        normalized_value = _normal(value)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._assert_integrity(db)
                _, scope = self._scope_in_tx(db, scope_id, decision)
                if not set(targets).issubset(scope["entity_keys"]):
                    raise ValueError("observation_outside_scope")
                receipts = self._receipt_rows_in_tx(db, receipt_ids)
                if extractor["kind"] == "llm":
                    responses = self._receipt_rows_in_tx(db, extractor["response_receipt_ids"])
                    if not {row["event_id"] for row in responses}.issubset({row["event_id"] for row in receipts}):
                        raise ValueError("llm_response_receipt_not_linked")
                entities: list[sqlite3.Row] = []
                for event_id in entity_ids:
                    row = db.execute("SELECT * FROM events WHERE event_id=?", (_text(event_id, "entity_id"),)).fetchone()
                    if row is None or row["kind"] != "entity":
                        raise ValueError("unknown_entity")
                    entities.append(row)

                reason: str | None = None
                if not receipts:
                    reason = "missing_source_receipt"
                else:
                    for receipt in receipts:
                        body = self._event_body(receipt)
                        if _time(receipt["recorded_at"], "receipt_recorded_at") > decision or _time(
                            receipt["available_at"], "receipt_available_at"
                        ) > decision:
                            reason = "late_source_receipt"
                            break
                if reason is None and not entities:
                    reason = "missing_entity_mapping"
                canonical_receipt_ids = [row["event_id"] for row in receipts]
                mapped_target_keys: set[str] = set()
                if reason is None:
                    for entity in entities:
                        if entity["status"] != "resolved":
                            reason = "ambiguous_entity"
                            break
                        if _time(entity["recorded_at"], "entity_recorded_at") > decision:
                            reason = "late_entity_mapping"
                            break
                        if self._event_body(entity)["receipt_id"] not in canonical_receipt_ids:
                            reason = "entity_mapping_receipt_not_linked"
                            break
                        canonical_id = self._event_body(entity).get("canonical_id")
                        if canonical_id not in targets:
                            reason = "entity_mapping_target_mismatch"
                            break
                        mapped_target_keys.add(canonical_id)
                    # A feature can cover multiple stock/entity keys, but it
                    # cannot claim an observed value for an unmapped member.
                    # Multiple corroborating mappings for one target remain
                    # valid; the set must simply cover exactly the targets.
                    if reason is None and mapped_target_keys != set(targets):
                        reason = "entity_mapping_target_mismatch"
                origins = []
                source_families = []
                raw_content_hashes = []
                if reason is None:
                    origins = sorted({self._event_body(receipt)["origin_id"] for receipt in receipts})
                    source_families = sorted({self._event_body(receipt)["source_family"] for receipt in receipts})
                    raw_content_hashes = sorted(
                        {self._event_body(receipt)["raw_sha256"] for receipt in receipts}
                    )
                    if len(origins) < min_independent_sources or len(source_families) < min_independent_sources:
                        reason = "insufficient_independent_sources"
                    # Source-family and original-source labels are necessary
                    # but not sufficient evidence of independence.  The same
                    # captured bytes relabelled under two identities is a
                    # reprint/duplicate until a future collector can prove
                    # otherwise, so it must not satisfy a multi-source text
                    # consensus through the generic observation API.
                    elif len(raw_content_hashes) < min_independent_sources:
                        reason = "duplicate_content_evidence"
                canonical_entity_ids = sorted({row["event_id"] for row in entities})
                # A complete signal is usable only when the *actual* receipt,
                # entity, and feature writes have all finished no later than
                # ``decision_at``.  Post-cutoff feature attempts are retained
                # as a value-less abstention rather than silently succeeding
                # or dropping their audit trail.  Existing input-provenance
                # failures take precedence because they explain the more
                # specific reason for ineligibility.
                if reason is None and recorded > decision:
                    reason = "observation_write_after_decision_cutoff"
                if reason is not None:
                    result = self._abstain_in_tx(
                        db,
                        observation_key=observation_key,
                        scope_id=scope_id,
                        decision_at=decision,
                        feature_id=feature_id,
                        target_entity_keys=targets,
                        stage="observation",
                        reason=reason,
                        receipt_ids=canonical_receipt_ids,
                        entity_ids=canonical_entity_ids,
                        extractor=extractor,
                        now=recorded,
                    )
                else:
                    body = dict(
                        observation_key=observation_key,
                        scope_id=scope_id,
                        decision_at=_iso(decision),
                        feature_id=feature_id,
                        target_entity_keys=targets,
                        receipt_ids=canonical_receipt_ids,
                        entity_ids=canonical_entity_ids,
                        value=normalized_value,
                        unit=unit,
                        extractor=extractor,
                        extractor_sha256=_hash_json(extractor),
                        min_independent_sources=min_independent_sources,
                        independent_origin_ids=origins,
                        independent_source_families=source_families,
                        independent_raw_content_sha256s=raw_content_hashes,
                    )
                    result = self._append_observation_terminal_in_tx(
                        db,
                        observation_key=observation_key,
                        scope_id=scope_id,
                        decision_at=decision,
                        feature_id=feature_id,
                        target_entity_keys=targets,
                        status="observed",
                        body=body,
                        now=recorded,
                    )
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise

    def record_outcome(
        self,
        *,
        observation_id: str,
        outcome_key: str,
        label_definition: Mapping[str, Any],
        entry_at: Any,
        exit_at: Any,
        available_at: Any,
        value: Any | None = None,
        status: str = "settled",
        reason: str | None = None,
        receipt_ids: Sequence[str] = (),
        now: Any | None = None,
    ) -> dict[str, Any]:
        recorded = self._now(now)
        outcome_key = _text(outcome_key, "outcome_key")
        if status not in ("settled", "unavailable"):
            raise ValueError("invalid_outcome_status")
        if not isinstance(label_definition, Mapping) or not label_definition.get("version"):
            raise ValueError("label_definition_version_required")
        entry = _time(entry_at, "outcome_entry_at")
        exit_ = _time(exit_at, "outcome_exit_at")
        available = _time(available_at, "outcome_available_at")
        if exit_ <= entry or available < exit_ or recorded < available:
            raise ValueError("outcome_not_matured")
        if status == "settled":
            if value is None:
                raise ValueError("settled_outcome_value_required")
            normalized_value = _normal(value)
            outcome_reason = None
        else:
            if value is not None:
                raise ValueError("unavailable_outcome_must_not_have_value")
            normalized_value = None
            outcome_reason = _text(reason, "outcome_unavailable_reason")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._assert_integrity(db)
                observation = db.execute("SELECT * FROM events WHERE event_id=?", (_text(observation_id, "observation_id"),)).fetchone()
                if observation is None or observation["kind"] != "observation" or observation["status"] != "observed":
                    raise ValueError("outcome_requires_observed_observation")
                observation_body = self._event_body(observation)
                decision = _time(observation_body["decision_at"], "observation_decision_at")
                if entry <= decision:
                    raise ValueError("outcome_entry_not_after_decision")
                receipts = self._receipt_rows_in_tx(db, receipt_ids)
                if status == "settled" and not receipts:
                    raise ValueError("outcome_receipt_required")
                for receipt in receipts:
                    if _time(receipt["recorded_at"], "outcome_receipt_recorded_at") > available or _time(
                        receipt["available_at"], "outcome_receipt_available_at"
                    ) > available:
                        raise ValueError("outcome_receipt_not_available")
                body = dict(
                    observation_id=observation["event_id"],
                    outcome_key=outcome_key,
                    label_definition=_normal(label_definition),
                    entry_at=_iso(entry),
                    exit_at=_iso(exit_),
                    value=normalized_value,
                    reason=outcome_reason,
                    receipt_ids=[row["event_id"] for row in receipts],
                )
                logical_key = _hash_json(dict(observation_id=observation["event_id"], outcome_key=outcome_key))
                result = self._terminal_in_tx(
                    db,
                    kind="outcome",
                    logical_key=logical_key,
                    status=status,
                    body=body,
                    available_at=available,
                    now=recorded,
                )
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise

    def coverage(self, scope_id: str, decision_at: Any, feature_id: str) -> dict[str, Any]:
        scope_id = _text(scope_id, "scope_id")
        decision = _time(decision_at, "decision_at")
        feature_id = _text(feature_id, "feature_id")
        with self._connect() as db:
            self._assert_integrity(db)
            _, scope = self._scope_in_tx(db, scope_id, decision)
            states: dict[str, dict[str, Any]] = {}
            for row in db.execute("SELECT * FROM events WHERE kind='observation' AND status IN ('observed','abstained')"):
                body = self._event_body(row)
                if (
                    body.get("scope_id") != scope_id
                    or body.get("decision_at") != _iso(decision)
                    or body.get("feature_id") != feature_id
                ):
                    continue
                for key in body.get("target_entity_keys", []):
                    if key in states:
                        raise ValueError("coverage_ambiguous_member")
                    states[key] = dict(status=row["status"], reason=body.get("reason"), event_id=row["event_id"])
            members = list(scope["entity_keys"])
            observed = [key for key in members if states.get(key, {}).get("status") == "observed"]
            abstained = [key for key in members if states.get(key, {}).get("status") == "abstained"]
            missing = [key for key in members if key not in states]
            reasons = Counter(states[key]["reason"] for key in abstained if states[key].get("reason"))
            return dict(
                scope_id=scope_id,
                decision_at=_iso(decision),
                feature_id=feature_id,
                expected=len(members),
                observed=len(observed),
                abstained=len(abstained),
                missing=missing,
                observed_entity_keys=observed,
                abstained_entity_keys=abstained,
                reason_counts=dict(sorted(reasons.items())),
                membership_sha256=scope["membership_sha256"],
            )

    def export_receipt_evidence(self, receipt_id: str) -> dict[str, Any]:
        """Return an integrity-checked, read-only receipt evidence reference.

        The raw response stays content-addressed in the ledger's blob store;
        callers receive only its relative path, byte count and digest so an
        evidence bundle can be independently rechecked without copying or
        silently rewriting the captured payload.
        """
        with self._connect() as db:
            self._assert_integrity(db)
            requested_id = _text(receipt_id, "receipt_id")
            requested = db.execute("SELECT * FROM events WHERE event_id=?", (requested_id,)).fetchone()
            if requested is None or requested["kind"] != "receipt":
                raise ValueError("unknown_receipt")
            canonical = self._receipt_row_in_tx(
                db,
                requested_id,
                require_source_response=False,
            )
            body = self._event_body(canonical)
            return dict(
                kind="research_signal_receipt_evidence",
                ledger_id=self.ledger_id,
                requested_receipt_id=requested_id,
                canonical_receipt_id=canonical["event_id"],
                receipt=self._event_dict(canonical),
                raw=dict(
                    relative_path=body["blob_path"],
                    sha256=body["raw_sha256"],
                    byte_count=body["byte_count"],
                    content_type=body["content_type"],
                ),
            )

    def export_observation_evidence(self, observation_id: str) -> dict[str, Any]:
        """Return one integrity-checked observed feature without exposing writes.

        A forward labeler needs the exact score vector that was frozen at the
        prior decision, not a caller-maintained copy that might have changed
        while the H=5 window matured.  This is deliberately a read-only view:
        it verifies the full chain first and returns no capability to amend an
        observation or attach an order.
        """
        with self._connect() as db:
            self._assert_integrity(db)
            requested_id = _text(observation_id, "observation_id")
            row = db.execute("SELECT * FROM events WHERE event_id=?", (requested_id,)).fetchone()
            if row is None or row["kind"] != "observation":
                raise ValueError("unknown_observation")
            if row["status"] != "observed":
                raise ValueError("observation_not_observed")
            verification = self._verify_db(db)
            if verification:
                raise ValueError("ledger_integrity_invalid")
            journal = db.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            return dict(
                kind="research_signal_observation_evidence",
                ledger_id=self.ledger_id,
                ledger_journal_hash=None if journal is None else journal["event_hash"],
                observation=self._event_dict(row),
            )

    def outcomes(
        self,
        as_of: Any,
        *,
        outcome_key: str | None = None,
        statuses: Sequence[str] = ("settled", "unavailable"),
    ) -> list[dict[str, Any]]:
        """Read only outcomes already available at ``as_of``.

        Unlike :meth:`settled_outcomes`, this includes explicit unavailable
        labels so forward reports disclose collection failures instead of
        silently counting only the successful H=5 settlements.  It never
        returns a future or merely pending label, so it is safe for the same
        no-feedback boundary as the settled-only view.
        """
        cutoff = _time(as_of, "outcome_query_as_of")
        if outcome_key is not None:
            outcome_key = _text(outcome_key, "outcome_key")
        accepted = tuple(_text(status, "outcome_status") for status in statuses)
        if not accepted or any(status not in _TERMINAL["outcome"] for status in accepted):
            raise ValueError("invalid_outcome_status_filter")
        with self._connect() as db:
            self._assert_integrity(db)
            result = []
            for row in db.execute("SELECT * FROM events WHERE kind='outcome' ORDER BY seq"):
                if row["status"] not in accepted:
                    continue
                body = self._event_body(row)
                if outcome_key is not None and body.get("outcome_key") != outcome_key:
                    continue
                if _time(row["available_at"], "outcome_available_at") <= cutoff and _time(
                    row["recorded_at"], "outcome_recorded_at"
                ) <= cutoff:
                    result.append(self._event_dict(row))
            return result

    def settled_outcomes(self, as_of: Any, *, outcome_key: str | None = None) -> list[dict[str, Any]]:
        return self.outcomes(as_of, outcome_key=outcome_key, statuses=("settled",))
