#!/usr/bin/env python3
"""Pack already licensed, normalized tables. Never invent PIT timestamps or trading states."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.data.contracts import load_dataset
from quantlab.data.qualification import audit_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,help='normalized manifest with source-relative files')
    p.add_argument('--destination',required=True);args=p.parse_args()
    src=Path(args.source).resolve();dst=Path(args.destination).resolve()
    if dst.exists():raise ValueError('destination_must_be_new_version_directory')
    # Validate references/hashes before copying. Preserve rows, units, source and dates exactly.
    load_dataset(src)
    shutil.copytree(src.parent,dst)
    report=audit_manifest(dst/src.name)
    (dst/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
