"""Dataset v5: the frozen v4 extension snapshot plus cloud (Tushare) daily rows after its cutoff.

History up to the base ``as_of`` is never rewritten: unchanged panel files are
hard-linked and changed panels keep every old row byte-for-byte at the frame
level. New rows reproduce the base panel contract column by column:

- raw prices from ``daily``; ``volume`` = ``vol`` x 100 shares, ``amount`` = ``amount`` x 1000 CNY;
- ``factor`` continues the base factor through the cloud ``adj_factor`` ratio at the seam,
  so ``feature_close`` ratios are continuous across the join;
- limits from ``stk_limit``; ST from ``stock_st``; suspension from ``suspend_d``;
- ``available_at`` = trade date 18:00, the base research assumption.

Before anything is written, cloud rows on the overlap window are compared with
the base rows; a mismatch rate above ``tolerance`` aborts the build.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.exchange_rules import RULE_VERSION

ROOT = Path(__file__).resolve().parents[1]
V5_ROOT = ROOT / 'data/dataset_v5'
VOLUME_PER_LOT = 100.0
AMOUNT_PER_UNIT = 1000.0
REQUIRED = ('daily', 'adj_factor', 'stk_limit', 'suspend_d', 'stock_st', 'daily_basic')


class BuildError(RuntimeError):
    pass


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canon(ts_code):
    number, market = str(ts_code).split('.')
    return f'{market.lower()}.{number}'


def _date(series):
    series = pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series)
        return (out.dt.tz_localize(None) if out.dt.tz is not None else out).dt.normalize()
    return pd.to_datetime(series.astype(str).str[:10].str.replace('-', ''), format='%Y%m%d', errors='coerce').dt.normalize()


def _read(root, entry):
    payload = (root / entry['path']).read_bytes()
    if sha256_bytes(payload) != entry['sha256']:
        raise BuildError('base_hash_mismatch:' + entry['path'])
    return pd.read_parquet(io.BytesIO(payload))


def _dated(frame):
    out = frame.set_index('event_time') if 'event_time' in frame else frame.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None).normalize()
    return out.sort_index()


def prepare_cloud(tables):
    """Normalize cloud tables to ``(code, date)`` keyed frames."""
    missing = [name for name in REQUIRED if name not in tables]
    if missing:
        raise BuildError('missing_cloud_tables:' + ','.join(missing))
    daily = tables['daily'].copy()
    daily['code'], daily['date'] = daily.ts_code.map(canon), _date(daily.trade_date)
    daily = daily.set_index(['code', 'date'])
    if daily.index.duplicated().any():
        raise BuildError('duplicate_daily_rows')
    adj = tables['adj_factor'].assign(code=lambda d: d.ts_code.map(canon), date=lambda d: _date(d.trade_date))
    limits = tables['stk_limit'].assign(code=lambda d: d.ts_code.map(canon), date=lambda d: _date(d.trade_date))
    basic = tables['daily_basic'].assign(code=lambda d: d.ts_code.map(canon), date=lambda d: _date(d.trade_date))
    st = tables['stock_st'].assign(code=lambda d: d.ts_code.map(canon), date=lambda d: _date(d.trade_date))
    suspend = tables['suspend_d'].assign(code=lambda d: d.ts_code.map(canon), date=lambda d: _date(d.trade_date))
    if 'suspend_type' in suspend:
        suspend = suspend[suspend.suspend_type.astype(str).eq('S')]
    return dict(daily=daily, volume_mult=VOLUME_PER_LOT, amount_mult=AMOUNT_PER_UNIT,
                adj=adj.set_index(['code', 'date'])['adj_factor'],
                limits=limits.set_index(['code', 'date'])[['up_limit', 'down_limit']],
                basic=basic.set_index(['code', 'date']),
                st=set(zip(st.code, st.date)),
                suspend=set(zip(suspend.code, suspend.date)))


UNIT_MULT = {'share': 1.0, 'lot': 100.0, 'CNY': 1.0, 'kCNY': 1000.0}


def prepare_combined(daily, basic, *, overlap_close=None, end=None):
    """Normalize the cloud's single combined bar table (``daily_ohlcv_raw_pit``) plus ``daily_basic_full``.

    Units come from the table's own ``volume_unit``/``amount_unit`` columns; a unit outside ``UNIT_MULT``
    or more than one unit in a column aborts. Suspended rows (``tradestatus == 0``) become the suspension
    set. ``overlap_close`` (``daily_basic_full`` closes before the bar table starts) is kept for the
    close-only seam check."""
    frame = daily.copy()
    for column in ('volume_unit', 'amount_unit'):
        units = set(frame[column].dropna().astype(str))
        if len(units) != 1 or next(iter(units)) not in UNIT_MULT:
            raise BuildError(f'unknown_or_mixed_{column}:{sorted(units)}')
    frame['code'] = frame['instrument_id'] if 'instrument_id' in frame else frame.ts_code.map(canon)
    frame['date'] = pd.to_datetime(frame['trade_date']).dt.tz_localize(None).dt.normalize()
    if end is not None:
        frame = frame[frame['date'] <= pd.Timestamp(end)]
    keyed = frame.set_index(['code', 'date'])
    if keyed.index.duplicated().any():
        raise BuildError('duplicate_daily_rows')
    st_by_day = keyed['is_st_at_open'].astype(bool).groupby(level='date').sum().sort_index()
    previous = st_by_day.shift(1)
    collapsed = st_by_day[(previous >= 20) & (st_by_day < 0.5 * previous)]
    if len(collapsed):
        raise BuildError('st_flags_collapsed:' + ','.join(f'{d.date()}={int(n)}/{int(previous[d])}'
                                                          for d, n in collapsed.items()))
    trading = keyed[keyed['tradestatus'].eq(1) & keyed['close'].notna()]
    suspended = keyed[~keyed.index.isin(trading.index)]
    bars = trading.rename(columns={'volume': 'vol'})[['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']]
    basic = basic.copy()
    basic['code'] = basic['instrument_id'] if 'instrument_id' in basic else basic.ts_code.map(canon)
    basic['date'] = pd.to_datetime(basic['trade_date']).dt.tz_localize(None).dt.normalize()
    return dict(daily=bars, volume_mult=UNIT_MULT[str(frame.volume_unit.iloc[0])],
                amount_mult=UNIT_MULT[str(frame.amount_unit.iloc[0])],
                adj=keyed['adj_factor'], limits=keyed[['limit_up', 'limit_down']].rename(
                    columns={'limit_up': 'up_limit', 'limit_down': 'down_limit'}),
                basic=basic.set_index(['code', 'date']),
                st=set(keyed.index[keyed['is_st_at_open'].astype(bool)]),
                suspend=set(suspended.index), overlap_close=overlap_close)


def overlap_close_check(base_frames, closes, start, end, tolerance=0.001):
    """Close-only seam check against ``daily_basic_full`` when the bar table starts after the base cutoff."""
    checked = mismatched = 0
    examples = []
    for code, frame in base_frames.items():
        rows = frame.loc[(frame.index >= start) & (frame.index <= end)]
        if 'tradestatus' in rows:
            rows = rows[rows['tradestatus'].eq(1)]
        for day, value in rows['close'].dropna().items():
            other = closes.get((code, day))
            if other is None or not np.isfinite(other):
                continue
            checked += 1
            if not np.isclose(float(value), float(other), rtol=1e-3, atol=0.011):
                mismatched += 1
                if len(examples) < 20:
                    examples.append(dict(code=code, date=str(day.date()), base=float(value), cloud=float(other)))
    rate = mismatched / checked if checked else 1.0
    report = dict(kind='close_only_daily_basic', start=str(start.date()), end=str(end.date()), rows_checked=checked,
                  rows_mismatched=mismatched, mismatch_rate=rate, examples=examples, tolerance=tolerance)
    if checked == 0 or rate > tolerance:
        raise BuildError('overlap_check_failed:' + json.dumps(report))
    return report


def overlap_check(base_frames, cloud, start, end, tolerance=0.001):
    """Compare raw open/close/volume/amount on ``[start, end]``; return the report or raise."""
    daily = cloud['daily']
    checked = mismatched = 0
    examples = []
    for code, frame in base_frames.items():
        rows = frame.loc[(frame.index >= start) & (frame.index <= end)]
        ok = rows['quality_status'].astype(str).eq('ok') if 'quality_status' in rows else True
        trading = rows['tradestatus'].eq(1) if 'tradestatus' in rows else True
        rows = rows[ok & trading]
        for day, row in rows.iterrows():
            key = (code, day)
            if key not in daily.index:
                continue
            bar = daily.loc[key]
            pairs = dict(open=(row.open, bar.open), close=(row.close, bar.close),
                         volume=(row.volume, bar.vol * cloud['volume_mult']),
                         amount=(row.amount, bar.amount * cloud['amount_mult']))
            checked += 1
            bad = [name for name, (a, b) in pairs.items()
                   if not np.isclose(float(a), float(b), rtol=1e-3, atol=0.011 if name in ('open', 'close') else 1.0)]
            if bad:
                mismatched += 1
                if len(examples) < 20:
                    examples.append(dict(code=code, date=str(day.date()), fields=bad))
    rate = mismatched / checked if checked else 1.0
    report = dict(start=str(start.date()), end=str(end.date()), rows_checked=checked, rows_mismatched=mismatched,
                  mismatch_rate=rate, examples=examples, tolerance=tolerance)
    if checked == 0 or rate > tolerance:
        raise BuildError('overlap_check_failed:' + json.dumps(report))
    return report


def new_rows(code, base, cloud, sessions, revision):
    """Rows for ``sessions`` (all after the base cutoff) in the base column layout."""
    last = base.dropna(subset=['close']).iloc[-1] if base is not None and base['close'].notna().any() else None
    last_day = base.dropna(subset=['close']).index[-1] if last is not None else None
    base_factor = float(last['factor']) if last is not None and pd.notna(last.get('factor')) else None
    seam_adj = cloud['adj'].get((code, last_day)) if last_day is not None else None
    if seam_adj is None and last is not None:
        seam_adj = _implied_seam_adj(code, cloud, sessions, float(last['close']))
    lower_bound = int(last['listing_session_lower_bound']) if last is not None and 'listing_session_lower_bound' in last else 0
    previous_close = float(last['close']) if last is not None else np.nan
    out = []
    for day in sessions:
        key = (code, day)
        has_bar = key in cloud['daily'].index
        suspended = key in cloud['suspend']
        if not has_bar and not suspended:
            continue
        lower_bound += 1
        adj = cloud['adj'].get(key)
        if base_factor is not None and seam_adj and adj:
            factor = base_factor * float(adj) / float(seam_adj)
        elif adj and base_factor is None:
            factor = float(adj)
        else:
            factor = np.nan
        basic = cloud['basic'].loc[key] if key in cloud['basic'].index else None
        limits = cloud['limits'].loc[key] if key in cloud['limits'].index else None
        if has_bar:
            bar = cloud['daily'].loc[key]
            prices = dict(open=float(bar.open), high=float(bar.high), low=float(bar.low), close=float(bar.close),
                          volume=float(bar.vol) * cloud['volume_mult'], amount=float(bar.amount) * cloud['amount_mult'])
            reference = float(bar.pre_close)
            status = 1.0
        else:
            prices = dict(open=previous_close, high=previous_close, low=previous_close, close=previous_close,
                          volume=0.0, amount=0.0)
            reference = previous_close
            status = 0.0
        up = float(limits.up_limit) if limits is not None else np.nan
        down = float(limits.down_limit) if limits is not None else np.nan
        complete = limits is not None and basic is not None and np.isfinite(factor)
        row = dict(event_time=day.strftime('%Y-%m-%d'), **prices, factor=factor, change=np.nan,
                   feature_close=prices['close'] * factor if np.isfinite(factor) else np.nan,
                   reference_price=reference, source='cloud.tushare.daily', revision=revision,
                   symbol=f'{code.split(".")[1]}.{code.split(".")[0].upper()}', upper_limit=None, lower_limit=None,
                   adjusted=False, pb_mrq=float(basic.pb) if basic is not None and pd.notna(basic.get('pb')) else np.nan,
                   turnover=float(basic.turnover_rate) if basic is not None and pd.notna(basic.get('turnover_rate')) else np.nan,
                   is_st_at_open=1.0 if key in cloud['st'] else 0.0, tradestatus=status,
                   state_source='cloud.tushare.stock_st+suspend_d', turnover_source='cloud.tushare.daily_basic',
                   listing_session_lower_bound=lower_bound,
                   limit_ratio_at_open=round(up / reference - 1, 2) if reference and np.isfinite(up) else np.nan,
                   limit_up=up, limit_down=down, open_status='derived', open_status_reason='',
                   available_at=f'{day.strftime("%Y-%m-%d")} 18:00:00',
                   availability_basis='daily_close_plus_buffer_research_assumption',
                   open_status_available_at=f'{day.strftime("%Y-%m-%d")} 09:30:00',
                   open_status_basis='daily_open_assumption', rule_version=RULE_VERSION,
                   quality_status='ok' if complete else 'incomplete_cloud_fields', unexplained_ex_reference=np.nan)
        previous_close = prices['close']
        out.append(row)
    return out


def _implied_seam_adj(code, cloud, sessions, base_close):
    """Cloud adj factor equivalent at the base's last bar when the cloud table starts after it.

    Tushare ``pre_close`` is the ex-rights reference price, so ``adj[t-1] = adj[t] * pre_close[t] / close[t-1]``;
    this stays correct when the first cloud session is itself an ex-date."""
    for day in sessions:
        key = (code, day)
        if key in cloud['daily'].index and key in cloud['adj'].index:
            bar = cloud['daily'].loc[key]
            if base_close and np.isfinite(bar.pre_close) and bar.pre_close > 0:
                return float(cloud['adj'][key]) * float(bar.pre_close) / base_close
            return None
    return None


def _append(base_frame, rows):
    """Append rows in the base layout. ``event_time`` stays where the base keeps it (index or column)."""
    addition = pd.DataFrame(rows)
    if base_frame is None:
        addition['event_time'] = pd.to_datetime(addition['event_time']).astype('datetime64[us]')
        return addition.set_index('event_time')
    indexed = base_frame.index.name == 'event_time'
    if indexed:
        addition = addition.set_index('event_time')
        if pd.api.types.is_datetime64_any_dtype(base_frame.index):
            addition.index = pd.DatetimeIndex(pd.to_datetime(addition.index), name='event_time').astype(base_frame.index.dtype)
    addition = addition.reindex(columns=list(base_frame.columns))
    for column in base_frame.columns:
        if isinstance(base_frame[column].dtype, pd.CategoricalDtype):
            base_frame[column] = base_frame[column].astype(object)
    return pd.concat([base_frame, addition], ignore_index=not indexed)


def cloud_actions(dividend, start, end, sessions):
    """Implemented Tushare distributions with ex-date in ``(start, end]`` in the base action layout."""
    if dividend is None or dividend.empty:
        return pd.DataFrame()
    frame = dividend[dividend.get('div_proc', '实施').astype(str).eq('实施')].copy()
    frame['ex'] = _date(frame.ex_date)
    frame = frame[(frame.ex > start) & (frame.ex <= end)]
    rows = []

    def value(row, *names, default=None):
        for name in names:
            item = row.get(name)
            if item is not None and not (isinstance(item, float) and np.isnan(item)):
                return item
        return default

    for _, row in frame.iterrows():
        code = canon(row.ts_code)
        notice = _date(pd.Series([value(row, 'imp_ann_date', 'ann_date')])).iloc[0]
        position = sessions.searchsorted(notice, side='right') if pd.notna(notice) else None
        available = sessions[position] if position is not None and position < len(sessions) else pd.NaT
        rows.append(dict(action_id=f'{code}:{row.ex.date()}', instrument_id=code,
                         record_date=str(_date(pd.Series([row.record_date])).iloc[0].date()),
                         ex_date=str(row.ex.date()), pay_date=str(_date(pd.Series([row.pay_date])).iloc[0].date()),
                         share_date=str(_date(pd.Series([value(row, 'div_listdate', 'ex_date')])).iloc[0].date()),
                         cash_per_share=float(value(row, 'cash_div_tax', default=0.0)),
                         share_ratio=float(value(row, 'stk_div', default=0.0)),
                         available_at=available, source='cloud.tushare.dividend', revision='',
                         quality_status='ok' if pd.notna(available) else 'missing_notice_date',
                         tax_policy='gross_cn_individual', action_type='distribution',
                         ingested_at=datetime.now(timezone.utc).isoformat(),
                         availability_basis='implementation_notice_date_next_session'))
    return pd.DataFrame(rows)


def build(base_manifest, tables, *, end, overlap=('2026-09-01', '2026-09-22'), version=None, out_root=V5_ROOT,
          tolerance=0.001, cloud_snapshots=None):
    base_manifest = Path(base_manifest)
    base_root = base_manifest.parent
    manifest = json.loads(base_manifest.read_text())
    cutoff, end = pd.Timestamp(manifest['as_of']), pd.Timestamp(end)
    cloud = tables if 'volume_mult' in tables else prepare_cloud(tables)
    revision = sha256_bytes(json.dumps(cloud_snapshots or {}, sort_keys=True).encode())
    base_frames = {code: _dated(_read(base_root, entry)) for code, entry in manifest['panels'].items()}
    if cloud.get('overlap_close') is not None:
        report = overlap_close_check(base_frames, cloud['overlap_close'], pd.Timestamp(overlap[0]),
                                     pd.Timestamp(overlap[1]), tolerance)
    else:
        report = overlap_check(base_frames, cloud, pd.Timestamp(overlap[0]), pd.Timestamp(overlap[1]), tolerance)
    base_calendar = pd.DatetimeIndex(pd.to_datetime(_read(base_root, manifest['calendar']).session)).normalize()
    cloud_days = cloud['daily'].index.get_level_values('date').unique()
    new_sessions = pd.DatetimeIndex(sorted(set(cloud_days[(cloud_days > cutoff) & (cloud_days <= end)])))
    if new_sessions.empty:
        raise BuildError('no_cloud_sessions_after_cutoff')
    version = version or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = Path(out_root) / version
    if target.exists():
        raise FileExistsError(target)
    (target / 'panels').mkdir(parents=True)
    codes = sorted(set(manifest['panels']) | set(cloud['daily'].index.get_level_values('code')))
    panels, appended, gaps = {}, 0, []
    for code in codes:
        base_entry = manifest['panels'].get(code)
        base_raw = _read(base_root, base_entry) if base_entry else None
        rows = new_rows(code, base_frames.get(code), cloud, new_sessions, revision)
        path = target / 'panels' / f'{code}.parquet'
        if not rows and base_entry:
            try:
                os.link(base_root / base_entry['path'], path)
            except OSError:
                shutil.copy2(base_root / base_entry['path'], path)
        else:
            frame = _append(base_raw.copy() if base_raw is not None else None, rows)
            frame.to_parquet(path, index=frame.index.name == 'event_time')
            appended += len(rows)
        panels[code] = dict(path=f'panels/{code}.parquet', sha256=sha256_bytes(path.read_bytes()))
        if base_entry and not rows and base_frames[code]['close'].notna().any() \
                and base_frames[code].index.max() >= cutoff - pd.Timedelta(days=10):
            gaps.append(dict(date=str(new_sessions[0].date()), instrument_id=code, reason='no_cloud_bar_after_cutoff'))
    calendar = pd.DataFrame({'session': sorted(set(base_calendar) | set(new_sessions))})
    _write_support(target, 'calendar', calendar.assign(session=calendar.session.dt.strftime('%Y-%m-%d')))
    actions = _read(base_root, manifest['actions'])
    extra = cloud_actions(tables.get('dividend'), cutoff, end, pd.DatetimeIndex(calendar.session))
    if not extra.empty:
        extra = extra.reindex(columns=actions.columns)
        for column in actions.columns:
            if not pd.api.types.is_numeric_dtype(actions[column]) and pd.api.types.is_datetime64_any_dtype(extra[column]):
                extra[column] = extra[column].dt.strftime('%Y-%m-%d %H:%M:%S')
        for column in actions.columns:
            if isinstance(actions[column].dtype, pd.CategoricalDtype):
                actions[column] = actions[column].astype(object)
        actions = pd.concat([actions, extra], ignore_index=True)
    _write_support(target, 'actions', actions)
    out = dict(manifest)
    instruments = extend_instruments(_read(base_root, manifest['instruments']), tables.get('stock_basic'), codes)
    _write_support(target, 'instruments', instruments)
    benchmark = extend_benchmark(_read(base_root, manifest['benchmark']), tables.get('index_daily'), cutoff, end,
                                 manifest.get('benchmark_id', 'sh.000300'))
    _write_support(target, 'benchmark', benchmark)
    for name in ('instruments', 'benchmark'):
        out[name] = dict(path=f'{name}.parquet', sha256=sha256_bytes((target / f'{name}.parquet').read_bytes()))
    for name in ('financials', 'industry_records', 'coverage'):
        if isinstance(manifest.get(name), dict) and 'path' in manifest[name]:
            source = base_root / manifest[name]['path']
            shutil.copy2(source, target / f'{name}.parquet')
            out[name] = dict(path=f'{name}.parquet', sha256=sha256_bytes((target / f'{name}.parquet').read_bytes()))
    out.update(panels=panels, as_of=str(new_sessions[-1].date()), revision='dataset_v5',
               calendar=dict(path='calendar.parquet', sha256=sha256_bytes((target / 'calendar.parquet').read_bytes())),
               actions=dict(path='actions.parquet', sha256=sha256_bytes((target / 'actions.parquet').read_bytes())),
               data_gaps=list(manifest.get('data_gaps', [])) + gaps,
               v5_provenance=dict(base_manifest=str(base_manifest), base_manifest_sha256=sha256_bytes(base_manifest.read_bytes()),
                                  base_as_of=str(cutoff.date()), cloud_snapshots=cloud_snapshots or {},
                                  overlap_check=report, new_sessions=[str(d.date()) for d in new_sessions],
                                  rows_appended=appended, cloud_actions_added=int(len(extra)),
                                  benchmark_rows_added=int(len(benchmark) - len(_read(base_root, manifest['benchmark']))),
                                  instruments_added=int(len(instruments) - len(_read(base_root, manifest['instruments']))),
                                  promotion_approved=False))
    (target / 'manifest.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return target


def extend_instruments(instruments, stock_basic, codes):
    """New listings get a record cloned from a same-exchange template; without ``stock_basic`` they are
    left out, so the ledger refuses to buy them rather than guessing a listing date."""
    known = set(instruments.instrument_id)
    if stock_basic is None or stock_basic.empty:
        return instruments
    basic = stock_basic.assign(code=stock_basic.ts_code.map(canon)).set_index('code')
    rows = []
    for code in sorted(set(codes) - known):
        if code not in basic.index:
            continue
        template = instruments[instruments.instrument_id.str.startswith(code[:5])]
        template = template.iloc[-1] if len(template) else instruments.iloc[-1]
        listed = _date(pd.Series([basic.loc[code, 'list_date']])).iloc[0]
        rows.append(dict(template.to_dict(), instrument_id=code, effective_at=str(listed.date()),
                         available_at=f'{listed.date()} 18:00:00', listing_date=str(listed.date()),
                         observation_start=str(listed.date()), source='cloud.tushare.stock_basic',
                         lifecycle_basis='tushare_stock_basic', active=True, quality_status='ok'))
    if not rows:
        return instruments
    return pd.concat([instruments, pd.DataFrame(rows).reindex(columns=instruments.columns)], ignore_index=True)


def extend_benchmark(benchmark, index_daily, cutoff, end, benchmark_id):
    if index_daily is None or index_daily.empty:
        return benchmark
    frame = index_daily.assign(code=index_daily.ts_code.map(canon), date=_date(index_daily.trade_date))
    frame = frame[(frame.code == benchmark_id) & (frame.date > cutoff) & (frame.date <= end)].sort_values('date')
    rows = [dict(event_time=row.date.strftime('%Y-%m-%d'), code=benchmark_id, open=row.open, high=row.high, low=row.low,
                 close=row.close, preclose=row.pre_close, volume=float(row.vol) * VOLUME_PER_LOT,
                 amount=float(row.amount) * AMOUNT_PER_UNIT, source='cloud.tushare.index_daily', revision='',
                 quality_status='ok', available_at=f'{row.date.strftime("%Y-%m-%d")} 18:00:00')
            for row in frame.itertuples()]
    if not rows:
        return benchmark
    return pd.concat([benchmark, pd.DataFrame(rows).reindex(columns=benchmark.columns)], ignore_index=True)


def _write_support(target, name, frame):
    frame.to_parquet(target / f'{name}.parquet', index=False)
