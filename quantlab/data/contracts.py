"""Versioned research data. Unknown availability or market rules never imply permission."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    pass


def timestamp(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("Asia/Shanghai") if t.tzinfo is None else t.tz_convert("Asia/Shanghai")


def decision_time(date) -> pd.Timestamp:
    return timestamp(pd.Timestamp(date).normalize()) + pd.Timedelta(hours=18)


def fingerprint(value) -> str:
    def normalize(v):
        if isinstance(v, dict): return {str(k): normalize(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [normalize(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def asof_records(history: pd.DataFrame, as_of, effective=None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    required = {"instrument_id", "available_at", "effective_at", "source", "revision", "quality_status"}
    if not required.issubset(history.columns):
        raise DataContractError(f"history missing {sorted(required - set(history.columns))}")
    t = timestamp(as_of)
    e = timestamp(effective) if effective is not None else t
    h = history.drop_duplicates().copy()
    h["available_at"] = h.available_at.map(timestamp)
    h["effective_at"] = h.effective_at.map(timestamp)
    h = h[(h.available_at <= t) & (h.effective_at <= e)]
    if h.duplicated(["instrument_id", "available_at", "effective_at"]).any():
        raise DataContractError("ambiguous_instrument_revision")
    # Latest known revision of the latest effective record. Revisions may arrive later.
    h = h.sort_values(["effective_at", "available_at", "revision"])
    return h.drop_duplicates("instrument_id", keep="last").set_index("instrument_id", drop=False)


@dataclass(frozen=True)
class InstrumentRules:
    instrument_id: str
    asset_type: str
    buy_lot: int
    sell_lot: int
    sell_delay: int
    price_tick: float
    commission: float
    min_commission: float
    limit_ratio: float | None

    @classmethod
    def from_record(cls, row):
        names = list(cls.__dataclass_fields__)
        if row is None or any(k not in row or (pd.isna(row[k]) and k != "limit_ratio") for k in names):
            raise DataContractError("missing instrument execution rules")
        obj = cls(**{k: row[k] for k in names})
        if obj.asset_type not in ("stock", "equity_etf"):
            raise DataContractError(f"unvalidated asset class: {obj.asset_type}")
        if obj.buy_lot < 1 or obj.sell_lot < 1 or obj.sell_delay < 0 or obj.price_tick <= 0:
            raise DataContractError("invalid lot/settlement rules")
        if obj.commission < 0 or obj.min_commission < 0:
            raise DataContractError("invalid fees")
        return obj


@dataclass
class MarketDataset:
    panels: dict[str, pd.DataFrame]
    instruments: pd.DataFrame
    calendar: pd.DatetimeIndex
    actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark: pd.DataFrame = field(default_factory=pd.DataFrame)
    announcements: pd.DataFrame = field(default_factory=pd.DataFrame)
    exposures: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest: dict = field(default_factory=dict)
    evidence_root: Path | None = field(default=None, repr=False)

    @property
    def version(self):
        return fingerprint({"manifest": self.manifest,
                            "panels": {k: frame_hash(v) for k, v in sorted(self.panels.items())},
                            "history": frame_hash(self.instruments), "actions": frame_hash(self.actions),
                            "calendar": list(self.calendar), "benchmark": frame_hash(self.benchmark),
                            "announcements": frame_hash(self.announcements), "exposures": frame_hash(self.exposures)})

    def validate(self) -> list[str]:
        errors = []
        m = self.manifest
        for field in ("source", "revision", "universe_coverage", "actions_coverage", "price_basis", "as_of"):
            if not m.get(field): errors.append(f"manifest_missing:{field}")
        if m.get("price_basis") != "raw": errors.append("raw_execution_prices_required")
        if m.get("currency") != "CNY" or m.get("amount_unit") != "CNY" or m.get("turnover_unit") not in ("fraction", "percent"):
            errors.append("market_units_unverified")
        if m.get("universe_coverage") not in ("historical", "snapshot_research"):
            errors.append("historical_universe_unverified")
        if m.get("actions_coverage") != "complete": errors.append("corporate_actions_unverified")
        if not self.calendar.is_monotonic_increasing or self.calendar.has_duplicates or self.calendar.empty:
            errors.append("invalid_exchange_calendar")
        try:
            asof_records(self.instruments, m.get("as_of", pd.Timestamp.today()))
            for _, row in self.instruments.iterrows(): InstrumentRules.from_record(row)
            for name in ("active", "is_st", "listing_date", "sector"):
                if name not in self.instruments or self.instruments[name].isna().any(): errors.append("security_history_missing:"+name)
        except (ValueError, TypeError) as exc: errors.append(str(exc))
        for code, p in self.panels.items():
            if not {"open", "close", "available_at", "source", "revision", "quality_status", "open_status", "open_status_available_at"}.issubset(p):
                errors.append(f"bar_contract_missing:{code}")
            if not p.index.is_monotonic_increasing or p.index.has_duplicates:
                errors.append(f"invalid_bar_index:{code}")
            if not p.index.isin(self.calendar).all(): errors.append(f"bar_outside_calendar:{code}")
            if "close" in p and (p.close.dropna() <= 0).any(): errors.append(f"invalid_price:{code}")
        if not self.actions.empty:
            required = {"action_id", "instrument_id", "record_date", "ex_date", "pay_date", "share_date",
                        "cash_per_share", "share_ratio", "available_at", "source", "revision", "quality_status"}
            if not required.issubset(self.actions): errors.append("action_contract_missing")
            elif (pd.to_datetime(self.actions.record_date) >= pd.to_datetime(self.actions.ex_date)).any():
                errors.append("invalid_action_record_date")
            elif (self.actions.cash_per_share < 0).any() or (self.actions.share_ratio < 0).any():
                errors.append("unsupported_action")
        if not self.actions.empty and 'cash_per_share' in self.actions:
            cash_actions=self.actions[self.actions.cash_per_share>0]
            if len(cash_actions) and ('tax_policy' not in cash_actions or not cash_actions.tax_policy.isin(['net_fixed','gross_cn_individual','exempt_etf']).all()):
                errors.append('dividend_tax_policy_unverified')
            if 'action_type' in self.actions and not self.actions.action_type.eq('distribution').all():errors.append('unsupported_corporate_action')
        if self.benchmark.empty or self.manifest.get("benchmark_type") not in ("index", "equity_etf"):
            errors.append("benchmark_identity_unverified")
        elif not {"available_at", "source", "revision", "quality_status", "close"}.issubset(self.benchmark):
            errors.append("benchmark_availability_unverified")
        else:
            for date, row in self.benchmark.iterrows():
                if pd.isna(row.available_at) or timestamp(row.available_at)>decision_time(date) or row.quality_status!="ok":
                    errors.append("benchmark_temporal_integrity_unverified")
                    break
        return sorted(set(errors))

    def known_panels(self, date):
        t = decision_time(date)
        return {c: p.loc[(p.index <= pd.Timestamp(date)) & p.available_at.map(timestamp).le(t)].copy()
                for c, p in self.panels.items()}


def load_dataset(path: str | Path) -> MarketDataset:
    """Load immutable parquet/CSV inputs; verify hashes instead of silently accepting edits."""
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    root = path.parent
    def read(entry):
        p = (root / entry["path"]).resolve()
        if not p.is_relative_to(root): raise DataContractError("dataset path escapes its root")
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry.get("sha256"):
            raise DataContractError(f"checksum mismatch: {p.name}")
        if p.suffix == ".parquet": return pd.read_parquet(p)
        return pd.read_csv(p, dtype={"instrument_id": str})
    def dated(frame):
        if "event_time" in frame: frame = frame.set_index("event_time")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        return frame.sort_index()
    panels = {c: dated(read(v)) for c, v in m["panels"].items()}
    cal = read(m["calendar"])
    dataset = MarketDataset(panels, read(m["instruments"]), pd.DatetimeIndex(pd.to_datetime(cal["session"])),
                            read(m["actions"]) if m.get("actions") else pd.DataFrame(),
                            dated(read(m["benchmark"])),
                            read(m["announcements"]) if m.get("announcements") else pd.DataFrame(),
                            read(m["exposures"]) if m.get("exposures") else pd.DataFrame(), m)
    errors = dataset.validate()
    if errors: raise DataContractError("; ".join(errors))
    dataset.evidence_root = root
    return dataset
