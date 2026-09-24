#!/usr/bin/env python3
"""Refresh public caches; report freshness without treating cache reads as updates."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.config import OUTPUT_DIR
from quantlab.data.sources import DataFeed
from quantlab.data.store import DataStore


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=40)
    parser.add_argument('--expected-session',default=None,help='Actual latest completed exchange session, YYYY-MM-DD')
    args=parser.parse_args(argv)
    if args.expected_session is None:
        out=OUTPUT_DIR/'pulse_v2.json'
        out.write_text(json.dumps(dict(status='blocked_calendar', stocks_advanced=0,
                      reason='Provide --expected-session from a verified exchange calendar. No update was claimed.'),indent=2))
        print(out);return
    feed=DataFeed();store=DataStore()
    cand=feed.fetch_candidates('a')
    codes=list(cand.index[:args.limit])
    # Include simulated holdings for data maintenance, labelled as simulation.
    from quantlab.portfolio.current import snapshot
    codes=list(dict.fromkeys(codes+[x['code'] for x in snapshot(with_quotes=False)]))
    rows=[]
    for c in codes:
        old=store.last_date('a',f'daily_raw_{c}')
        try:
            frame=feed.fetch_daily('a',str(c),refresh=True,expected_session=args.expected_session)
            new=frame.index[-1] if frame is not None and len(frame) else None
            rows.append(dict(instrument_id=c,previous=str(old),latest=str(new),
                             advanced=bool(new is not None and (old is None or new>old)),
                             current=bool(new is not None and new>=pd.Timestamp(args.expected_session)),
                             error=frame.attrs.get('update_error') if frame is not None else 'no_data'))
        except Exception as e:rows.append(dict(instrument_id=c,advanced=False,current=False,error=str(e)))
    report=dict(recorded_at=pd.Timestamp.now(tz='UTC').isoformat(),expected_session=args.expected_session,
                status='public_cache_only' if rows and all(x['current'] and not x.get('error') for x in rows) else 'public_cache_incomplete',stocks_advanced=sum(x['advanced'] for x in rows),
                stocks_current=sum(x['current'] for x in rows),rows=rows,promotion_approved=False)
    out=OUTPUT_DIR/'pulse_v2.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(out)


if __name__=='__main__':main()
