#!/usr/bin/env python3
"""Static read-point evidence plus explicit support classification, never a claim of behavioral coverage."""
import ast
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
    files={str(p):p.read_text() for p in Path('quantlab').rglob('*.py')}
    live=reads(files)
    rows=[dict(parameter=f'{g}.{n}'.lstrip('.'), default=str(v), support=status(g,n), read_points=live.get(n,[]),
               behavioral_coverage='see tests/test_parameters.py; static reads alone do not prove behavior') for g,n,v in config_fields(CONFIG)]
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
    out=OUTPUT_DIR/'implementation_20260924'/'parameter_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(out);print('original',before)


if __name__=='__main__': main()
