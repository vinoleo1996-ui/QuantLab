#!/usr/bin/env python3
"""Freeze an all-observed-market research snapshot from existing local files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.data.local_adapter import export_local_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    parser.add_argument('--qlib-root')
    parser.add_argument('--raw-cache',action='append',default=[])
    parser.add_argument('--output',default='data/local_market')
    parser.add_argument('--start',default='2014-01-02');parser.add_argument('--end',default='2026-06-30')
    parser.add_argument('--industry-mode',choices=['verified_pit','effective_date_proxy','disabled'],default='effective_date_proxy')
    args=parser.parse_args()
    result=export_local_dataset(args.database,args.output,qlib_root=args.qlib_root,raw_cache_roots=args.raw_cache,
        start=args.start,end=args.end,industry_mode=args.industry_mode,progress=lambda x:print(json.dumps(x),flush=True))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
