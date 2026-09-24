#!/usr/bin/env python3
"""Offline audit of colleague experiments; never changes source, caches or rev/.
Run: .venv2/bin/python scripts/review_compare.py
"""
from pathlib import Path
from unittest.mock import patch
import sys,json,pickle,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'rev'))
from quantlab.config import Config
from quantlab.data.store import DataStore
from quantlab.report.metrics import performance_table
import quantlab.backtest.engine as em
from exp1_reproduce import buyhold_baseline, pool_baseline
from scripts.review_audit import fixture
OUT=ROOT/'output/review_comparison_20260924';OUT.mkdir(parents=True,exist_ok=True)
saved=pickle.load((ROOT/'output/results.pkl').open('rb'))['result']
store=DataStore(); panels={c:store.load('a','daily_'+c) for c in saved['codes']}
regimes={pd.Timestamp(t['signal_date']):t['regime'] for t in saved['trades']}
def state(_self,t): return regimes.get(t,{'trend':1,'vol_high':False,'multiplier':1.})
original_experts=em.experts_score
observations=[]
def inspect_experts(*a,**kw):
    score,w=original_experts(*a,**kw)
    observations.append({'date':str(score.index[-1].date()),'unique_scores':int(score.iloc[-1].nunique())})
    return score,w

def fixed_experts(full,*a,**kw):
    f=next(iter(full.values()))
    return pd.DataFrame(.5,index=f.index,columns=f.columns),{'momentum':1.,'value':0.,'lowvol_quality':0.,'flow':0.,'composite':0.}

def run(label,override):
    cfg=Config();cfg.factors.neutralize_by_sector=False
    if label=='single_expert_original_filter': cfg.factors.use_experts=False
    with patch.object(em.FactorEngine,'compute',return_value=saved['factor_panels']),patch.object(em.Regime,'state',state),patch.object(em,'experts_score',side_effect=override):
        r=em.Engine(cfg).run(panels)
    p=performance_table(r['nav'],r['daily_return'])
    print(label,p,flush=True)
    return r,p

def main():
    base,bp=run('current_experts',inspect_experts)
    const,cp=run('constant_scores_fixed_expert_weights',fixed_experts)
    single,sp=run('single_expert_original_filter',inspect_experts)
    delta=float((base['daily_return']-const['daily_return']).abs().max())
    weight_delta=float((pd.DataFrame(base['weight_history']).fillna(0)-pd.DataFrame(const['weight_history']).fillna(0)).abs().max().max())
    assert delta==0 and weight_delta==0
    _,toy=fixture('drift');toy={c:toy[c] for c in list(toy)[:3]}
    bh=buyhold_baseline(toy,pd.Timestamp('2023-02-01'))
    sample=(1+bh.loc['2023-02-01':'2023-02-03']).prod()-1
    assert sample>0.002
    bl=pool_baseline(panels,None)
    bperf=performance_table((1+bl).cumprod(),bl)
    idx=base['daily_return'].index.intersection(bl.index)
    ratio=(1+base['daily_return'].loc[idx]).prod()/(1+bl.loc[idx]).prod()
    rel_ann=ratio**(252/len(idx))-1
    # Descriptive statistics only; no causal selection-bias quantification.
    wh=pd.DataFrame(saved['weight_history']).fillna(0)
    touches={c:int((wh[c]>0).sum()) if c in wh else 0 for c in ['300308','300502']}
    a={
      'source_data_unchanged_from_first_review':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in json.loads((ROOT/'output/review_20260924/manifest.json').read_text())['hashes'].items()),
      'current_experts':bp,'constant_experts':cp,'single_expert_with_different_filter':sp,
      'constant_vs_original_max_daily_return_difference':delta,
      'constant_vs_original_max_weight_difference':weight_delta,
      'signal_dates_observed':len(observations),'all_score_sections_constant':all(x['unique_scores']==1 for x in observations),
      'buyhold_counterexample':{'colleague_return':float(sample),'true_fixed_units_return':0.,'path':'one of three initial equal holdings 10 -> 11 -> 10, other prices flat'},
      'colleague_pool_baseline_rerun':bperf,
      'relative_wealth_return':float(ratio-1),'relative_wealth_annualized_correct_formula':float(rel_ann),
      'comparison_period':{'start':str(idx[0].date()),'end':str(idx[-1].date()),'rows':len(idx)},
      'saved_holding_occurrences':touches,
      'saved_gross_scales':sorted(set(t['risk']['gross_scale'] for t in saved['trades'])),
      'notes':['Frozen processed factors/regime controlled ablation; source code retained.',
               '29% pool return reruns colleague implementation, which is not a validated execution baseline.',
               'Relative wealth underperformance is not a risk-adjusted alpha estimate.']}
    (OUT/'comparison.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
    print(json.dumps(a,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
