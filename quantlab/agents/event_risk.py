"""Event-risk review of held / incoming names (TradingAgents-style roles, own data and own output).

Evidence is built deterministically from local tables with every fact dated on or before ``asof``.
Three LLM turns per name: a bull note, a bear note, then a judge that must return JSON
``{"veto": bool, "severity": 0-3, "reasons": [...], "cited_facts": [...]}`` citing evidence keys only.
The output is a flag for the forward record; it is never a rating, a weight or an order. A judge
reply that does not validate is recorded as an abstention, not as a pass.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from quantlab import llm

BULL = """You are the bull analyst for A-share {code} on {asof}. Using ONLY the evidence JSON below, argue in at most
120 words why holding it for the next 20 trading days carries no unusual event risk. Cite evidence keys in [brackets].
Evidence: {evidence}"""
BEAR = """You are the bear analyst for A-share {code} on {asof}. Using ONLY the evidence JSON below, argue in at most
120 words which specific events (unlocks, insider reductions, pledges, earnings warnings, abnormal trading, limit-downs)
could cause a large loss in the next 20 trading days. Cite evidence keys in [brackets].
Evidence: {evidence}
Bull note: {bull}"""
JUDGE = """You judge event risk for A-share {code} on {asof}. Weigh the two notes against the evidence. Conflict alone
is not a reason to veto; veto only for a concrete, dated event in the evidence that makes a large 20-day loss likely.
Return JSON only: {{"veto": true|false, "severity": 0-3, "reasons": ["..."], "cited_facts": ["evidence keys"]}}.
Evidence: {evidence}
Bull note: {bull}
Bear note: {bear}"""


def _code6(code):
    return code.split('.')[-1] if '.' in code else code


def evidence(code, asof, tables, panel_fields):
    """Dated facts for one name. ``tables``: dict of local frames; ``panel_fields``: date x code frames."""
    asof = pd.Timestamp(asof)
    six = _code6(code)
    facts = {}
    close = panel_fields['adj_close'][code].loc[:asof].dropna() if code in panel_fields['adj_close'] else pd.Series(dtype=float)
    if len(close) > 60:
        window = close.iloc[-60:]
        facts['price.last_session'] = str(close.index[-1].date())
        facts['price.ret_20'] = round(float(close.iloc[-1] / close.iloc[-21] - 1), 4)
        facts['price.ret_60'] = round(float(close.iloc[-1] / close.iloc[-61] - 1), 4)
        facts['price.drawdown_60'] = round(float(window.iloc[-1] / window.max() - 1), 4)
        raw = panel_fields['close'][code].loc[:asof].iloc[-20:]
        down = panel_fields['limit_down'][code].loc[:asof].iloc[-20:]
        facts['price.limit_down_days_20'] = int((raw <= down + 1e-6).sum())
        facts['price.suspended_days_20'] = int(panel_fields['tradestatus'][code].loc[:asof].iloc[-20:].eq(0).sum())
    unlock = tables.get('unlock')
    if unlock is not None:
        rows = unlock[(unlock.SECURITY_CODE.astype(str) == six)]
        dates = pd.to_datetime(rows.FREE_DATE)
        upcoming = rows[(dates > asof) & (dates <= asof + pd.Timedelta(days=60))]
        facts['unlock.next_60d_ratio'] = round(float(pd.to_numeric(upcoming.FREE_RATIO, errors='coerce').sum()), 4)
        if len(upcoming):
            facts['unlock.next_date'] = str(pd.to_datetime(upcoming.FREE_DATE).min().date())
    holders = tables.get('holder_change')
    if holders is not None:
        rows = holders[(holders.SECURITY_CODE.astype(str) == six)]
        notice = pd.to_datetime(rows.NOTICE_DATE)
        recent = rows[(notice <= asof) & (notice > asof - pd.Timedelta(days=90))]
        cut = recent[recent.DIRECTION.astype(str).str.contains('减')]
        facts['insider.reduction_notices_90d'] = int(len(cut))
        facts['insider.reduction_ratio_90d'] = round(float(pd.to_numeric(cut.CHANGE_FREE_RATIO, errors='coerce').sum()), 4)
    pledge = tables.get('pledge')
    if pledge is not None:
        rows = pledge[(pledge.SECURITY_CODE.astype(str) == six) & (pd.to_datetime(pledge.TRADE_DATE) <= asof)]
        if len(rows):
            last = rows.sort_values('TRADE_DATE').iloc[-1]
            facts['pledge.ratio'] = float(last.PLEDGE_RATIO)
            facts['pledge.as_of'] = str(pd.Timestamp(last.TRADE_DATE).date())
    forecast = tables.get('forecast')
    if forecast is not None:
        rows = forecast[(forecast.SECURITY_CODE.astype(str) == six) & (pd.to_datetime(forecast.NOTICE_DATE) <= asof)]
        rows = rows[pd.to_datetime(rows.NOTICE_DATE) > asof - pd.Timedelta(days=120)]
        if len(rows):
            last = rows.sort_values('NOTICE_DATE').iloc[-1]
            facts['forecast.type'] = str(last.PREDICT_TYPE)
            facts['forecast.notice'] = str(pd.Timestamp(last.NOTICE_DATE).date())
            facts['forecast.growth_lower'] = float(last.ADD_AMP_LOWER) if pd.notna(last.ADD_AMP_LOWER) else None
    billboard = tables.get('billboard')
    if billboard is not None:
        rows = billboard[(billboard.SECURITY_CODE.astype(str) == six)]
        day = pd.to_datetime(rows.TRADE_DATE)
        recent = rows[(day <= asof) & (day > asof - pd.Timedelta(days=30))]
        facts['billboard.entries_30d'] = int(len(recent))
        if len(recent):
            facts['billboard.net_amt_30d'] = float(pd.to_numeric(recent.BILLBOARD_NET_AMT, errors='coerce').sum())
    return facts


def parse_verdict(text, evidence_keys):
    match = re.search(r'\{.*\}', text or '', flags=re.S)
    if not match:
        return None, 'no_json'
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 'bad_json'
    if not isinstance(verdict.get('veto'), bool) or verdict.get('severity') not in (0, 1, 2, 3):
        return None, 'bad_fields'
    cited = [str(k).strip('[]') for k in verdict.get('cited_facts', [])]
    if verdict['veto'] and not any(k in evidence_keys for k in cited):
        return None, 'veto_without_evidence'
    return dict(veto=verdict['veto'], severity=verdict['severity'], reasons=[str(r)[:200] for r in verdict.get('reasons', [])][:5],
                cited_facts=[k for k in cited if k in evidence_keys]), None


def review(code, asof, facts, chat=llm.chat):
    payload = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    turns = {}
    model_b, bull, _ = chat(BULL.format(code=code, asof=asof, evidence=payload), caller='event_agent', max_tokens=1200)
    turns['bull'] = dict(model=model_b, text=bull[:1500])
    model_r, bear, _ = chat(BEAR.format(code=code, asof=asof, evidence=payload, bull=bull[:1500]), caller='event_agent',
                            max_tokens=1200)
    turns['bear'] = dict(model=model_r, text=bear[:1500])
    model_j, judged, digest = chat(JUDGE.format(code=code, asof=asof, evidence=payload, bull=bull[:1500], bear=bear[:1500]),
                                   caller='event_agent', max_tokens=1500)
    verdict, problem = parse_verdict(judged, set(facts))
    return dict(code=code, asof=str(asof), evidence=facts, turns=turns, judge_model=model_j, judge_prompt_sha=digest,
                verdict=verdict, abstained=verdict is None, abstain_reason=problem, rating=None, order=None)
