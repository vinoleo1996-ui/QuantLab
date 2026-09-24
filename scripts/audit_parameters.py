#!/usr/bin/env python3
"""Static read-point evidence plus explicit support classification, never a claim of behavioral coverage."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.config import CONFIG, OUTPUT_DIR
from quantlab.config_validation import config_fields, status


def reads(files):
    result = {}
    for path,text in files.items():
        if path.endswith(('config.py','config_validation.py')): continue
        for n in ast.walk(ast.parse(text)):
            name = n.attr if isinstance(n,ast.Attribute) else None
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('getattr','hasattr') and len(n.args)>1 and isinstance(n.args[1],ast.Constant): name=n.args[1].value
            if name: result.setdefault(name,[]).append(f'{path}:{n.lineno}')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    files={str(p):p.read_text() for p in Path('quantlab').rglob('*.py')}
    evidence={}
    for p in Path('tests').glob('test_*.py'):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name.startswith('test_'):
                for item in ast.walk(node):
                    if isinstance(item,ast.Attribute):evidence.setdefault(item.attr,set()).add(str(p)+':'+node.name)
    live=reads(files)
    rows=[dict(parameter=f'{g}.{n}'.lstrip('.'), default=str(v), support=status(g,n), read_points=live.get(n,[]),
               behavioral_test_references=sorted(evidence.get(n,[])), behavioral_coverage='mapped_test_requires_assertion_review' if evidence.get(n) else 'not_mapped') for g,n,v in config_fields(CONFIG)]
    before={}
    archive=OUTPUT_DIR/'implementation_20260924'/'source_before.tar.gz'
    if archive.exists():
        with tarfile.open(archive) as t:
            old={m.name:t.extractfile(m).read().decode() for m in t.getmembers() if m.name.startswith('quantlab/') and m.name.endswith('.py')}
        original=reads(old)
        names=[f'{c.name}.{n.target.id}' for c in ast.parse(old['quantlab/config.py']).body if isinstance(c,ast.ClassDef)
               for n in c.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]
        missing=[n for n in names if n.split('.')[-1] not in original]
        before=dict(fields=len(names), no_static_or_literal_getattr_reference=len(missing), missing=missing)
    report=dict(original=before, current=rows,
                caveat='Reference scan is not alias-sensitive data-flow analysis. Unsupported parameter overrides fail explicitly.')
    digest=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()[:12]
    out=OUTPUT_DIR/'upgrade_20260924'/('parameter_audit_'+digest+'.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(out)
    missing=[r['parameter'] for r in rows if r['support']=='active' and not r['read_points']]
    print('active_without_read_points',missing)
    print('active_behavior_not_mapped',sum(r['support']=='active' and r['behavioral_coverage']=='not_mapped' for r in rows))
    if args.check and missing:raise SystemExit(1)


if __name__=='__main__': main()
