#!/usr/bin/env python3
"""Run each unittest module in a clean interpreter, including inherited bootstrap.

This test-only launcher canonicalizes continuity_cadence.FROZEN in memory. It
never edits research files, calls load_frozen at startup, substitutes engines,
or bypasses the guard that rejects previously imported production modules.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

BOOTSTRAP = '''"""Temporary, explicit QuantLab test-process bootstrap."""
import importlib.util
import os
from pathlib import Path
import sys
root = Path(os.environ['QUANTLAB_ISOLATED_TEST_ROOT']).resolve()
path = root / 'scripts' / 'continuity_cadence.py'
if path.is_file():
    spec = importlib.util.spec_from_file_location('continuity_cadence', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Canonicalize the comparison root; leave load_frozen and its guard intact.
    module.FROZEN = module.FROZEN.resolve()
'''

@contextmanager
def bootstrap_environment(root):
    root=Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix='quantlab-test-bootstrap-') as tmp:
        Path(tmp,'sitecustomize.py').write_text(BOOTSTRAP)
        env=os.environ.copy();env['QUANTLAB_ISOLATED_TEST_ROOT']=str(root)
        env['PYTHONPATH']=os.pathsep.join([tmp,str(root),str(root/'tests'),env.get('PYTHONPATH','')])
        env.setdefault('MPLCONFIGDIR',str(Path(tmp,'matplotlib')))
        yield env

def discover_modules(root):
    tests=Path(root)/'tests';modules=[]
    for path in sorted(tests.rglob('test*.py')):
        # unittest discovers subdirectories only when they are packages.
        if any(not (parent/'__init__.py').exists() for parent in path.parents if parent!=tests and tests in parent.parents):continue
        modules.append('.'.join(path.relative_to(root).with_suffix('').parts))
    return modules

def _ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from _ids(item)
        else:yield item.id()

def child(module,result_path):
    # No test imports occur in the supervising process. Normal module ordering
    # within this one module is retained, including genuine import errors.
    suite=unittest.defaultTestLoader.loadTestsFromName(module)
    discovered=list(_ids(suite));began=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    value=dict(module=module,tests_discovered=len(discovered),test_ids=discovered,
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses),passed=result.wasSuccessful() and result.testsRun>0,
        runner_error='empty_test_module' if result.testsRun==0 else None,
        failure_details=[dict(test=t.id(),traceback=tb) for t,tb in result.failures+result.errors],
        seconds=time.monotonic()-began,
        quantlab_modules={name:str(Path(m.__file__).resolve()) for name,m in sys.modules.items()
            if (name=='quantlab' or name.startswith('quantlab.')) and getattr(m,'__file__',None)})
    Path(result_path).write_text(json.dumps(value,indent=2))
    return 0 if value['passed'] else 1

def run_module(root,module,output,env,timeout=900):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result_path=output/(module+'.json');log_path=output/(module+'.log')
    command=[sys.executable,str(Path(__file__).resolve()),'--child',module,'--result',str(result_path)]
    started=time.monotonic()
    with log_path.open('x') as log:
        try:
            completed=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            code=completed.returncode;problem=None
        except subprocess.TimeoutExpired:
            code=None;problem='module_timeout'
    if result_path.exists():
        value=json.loads(result_path.read_text())
        value['passed']=bool(value['passed'] and code==0)
    else:
        value=dict(module=module,tests_run=0,tests_discovered=0,failures=0,errors=0,skipped=0,
            expected_failures=0,unexpected_successes=0,passed=False,runner_error=problem or 'child_result_missing')
    value.update(exit_code=code,wall_seconds=time.monotonic()-started,command=command,log=str(log_path),result=str(result_path))
    return value

def source_hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for base in ('scripts','quantlab','tests') for p in sorted((Path(root)/base).rglob('*.py'))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--modules',nargs='+')
    parser.add_argument('--timeout',type=float,default=900)
    parser.add_argument('--child');parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    if args.child:
        if not args.result:parser.error('--child requires --result')
        return child(args.child,args.result)
    if not args.output:parser.error('--output is required; choose a new evidence directory')
    root=args.root.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    modules=args.modules or discover_modules(root)
    if len(modules)!=len(set(modules)):parser.error('duplicate test module selections')
    before=source_hashes(root)
    protocol=dict(root=str(root),python=sys.executable,modules=modules,
        created_at=datetime.now(timezone.utc).isoformat(),source_sha256=before,
        bootstrap=BOOTSTRAP,scope='one fresh interpreter per module, inherited test-only path normalization; guard retained')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    began=time.monotonic();results=[]
    with bootstrap_environment(root) as env:
        for module in modules:
            result=run_module(root,module,output/'modules',env,args.timeout);results.append(result)
            print(json.dumps({k:result.get(k) for k in ('module','tests_run','passed','failures','errors','runner_error','wall_seconds')}),flush=True)
    after=source_hashes(root);changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
    totals={k:sum(r[k] for r in results) for k in ('tests_run','tests_discovered','failures','errors','skipped','expected_failures','unexpected_successes')}
    summary=dict(status='passed' if results and totals['tests_run']>0 and all(r['passed'] for r in results) and not changed else 'failed',
        module_count=len(results),passed_modules=sum(r['passed'] for r in results),totals=totals,
        source_changes=changed,seconds=time.monotonic()-began,modules=results,
        interpretation='Actual per-module unittest counts; child/import/timeout errors remain failures. This isolates tests, not research accounts.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='modules'},indent=2),flush=True)
    return 0 if summary['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
