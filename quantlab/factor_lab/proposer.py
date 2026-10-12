"""LLM factor proposer: prompt, parse and validate; transport, rotation and the daily cap live in ``quantlab.llm``."""
from __future__ import annotations

import json
import re

from quantlab import llm
from quantlab.factor_lab.dsl import FIELDS, FormulaError, canonical, formula_id

MODELS, DAILY_CAP = llm.MODELS, llm.DAILY_CAP

PROMPT = """You propose stock-selection factors for China A-share small caps (bottom 40% by turnover),
held 20 trading days, long only. Write formulas in this DSL only:
fields: {fields}
time series (window integer 2..250, ends today): ts_mean ts_std ts_sum ts_max ts_min ts_rank delay(x,d>=1) delta(x,d>=1) ts_corr(x,y,d>=5)
cross section: rank zscore ; elementwise: log abs sign neg + - * / and numeric constants.
No other functions, no attributes, no negative lags.
Existing features you should NOT duplicate: {existing}.
Previously proposed (avoid repeating): {seen}.
Return exactly {n} candidates as a JSON list of objects with keys "name", "formula", "rationale" (one sentence
on the economic mechanism). JSON only."""


def calls_today(usage_path):
    return llm.calls_today(usage_path)


def parse_candidates(text):
    match = re.search(r'\[.*\]', text, flags=re.S)
    if not match:
        return [], ['no_json_list']
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], [f'json:{exc.msg}']
    good, errors = [], []
    for item in items if isinstance(items, list) else []:
        formula = str(item.get('formula', ''))
        try:
            good.append(dict(name=str(item.get('name', ''))[:60], formula=canonical(formula),
                             id=formula_id(formula), rationale=str(item.get('rationale', ''))[:300]))
        except FormulaError as exc:
            errors.append(f'{formula[:80]}:{exc}')
    return good, errors


def propose(existing, seen, usage_path, n=10, post=llm.http_post, key=None, models=MODELS):
    prompt = PROMPT.format(fields=', '.join(FIELDS), existing=', '.join(existing), seen='; '.join(seen[-40:]) or 'none', n=n)
    try:
        model, text, digest = llm.chat(prompt, caller='factor_lab', usage=usage_path, models=models, post=post, key=key)
    except llm.LLMError as exc:
        raise RuntimeError(str(exc)) from exc
    candidates, errors = parse_candidates(text)
    return dict(model=model, prompt_sha=digest, candidates=candidates, invalid=errors)
