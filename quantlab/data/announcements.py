"""Verifiable announcement facts only. No LLM order generation or enabled announcement alpha."""
from pathlib import Path
import hashlib
import pandas as pd
from quantlab.data.contracts import timestamp


def available_time(published,calendar,date_only=False):
    t=timestamp(published)
    if date_only:
        following=calendar[calendar>t.tz_localize(None).normalize()]
        if not len(following):raise ValueError('next_exchange_session_unknown')
        return timestamp(following[0])+pd.Timedelta(hours=9,minutes=30)
    return t


def verify_fact(fact,root):
    required=('instrument_id','report_period','field','value','unit','published_at','source_document','sha256','quote','start','end')
    if any(k not in fact for k in required):raise ValueError('incomplete_announcement_fact')
    root=Path(root).resolve();path=(root/fact['source_document']).resolve()
    if not path.is_relative_to(root):raise ValueError('document_path_escape')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=fact['sha256']:raise ValueError('document_hash_mismatch')
    text=raw.decode('utf-8')
    if text[fact['start']:fact['end']]!=fact['quote']:raise ValueError('quote_not_in_document')
    if str(fact['value']) not in fact['quote'] or fact['unit'] not in fact['quote']:raise ValueError('numeric_fact_not_grounded')
    if fact['unit'] not in ('元','万元','亿元'):raise ValueError('unvalidated_unit')
    return dict(fact,value_cny=float(str(fact['value']).replace(',',''))*{'元':1.,'万元':1e4,'亿元':1e8}[fact['unit']],
                status='grounded_fact_requires_financial_review')


def reaction_signal_time(published_at,calendar):
    t=timestamp(published_at)
    sessions=calendar[calendar>t.tz_localize(None).normalize()]
    if len(sessions)<3:raise ValueError('three_completed_reaction_sessions_missing')
    return timestamp(sessions[2])+pd.Timedelta(hours=18)
