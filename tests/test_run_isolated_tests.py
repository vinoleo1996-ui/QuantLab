"""Test-only import isolation: real guards remain active in nested children."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from scripts.run_isolated_tests import bootstrap_environment, run_module, discover_modules
ROOT=Path(__file__).resolve().parents[1]
class IsolatedTests(unittest.TestCase):
 def invoke(self,code):
  with bootstrap_environment(ROOT) as env:
   return subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,capture_output=True)
 def test_repeated_frozen_load_uses_resolved_path_and_inherits_to_child(self):
  code="import continuity_cadence as c; c.load_frozen(); import quantlab; c.load_frozen(); assert c.FROZEN==c.FROZEN.resolve()"
  result=self.invoke('import subprocess,sys; p=subprocess.run([sys.executable,"-c",'+repr(code)+']); sys.exit(p.returncode)')
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)
 def test_production_import_stays_production_and_mixing_still_fails(self):
  result=self.invoke('import pathlib,quantlab,continuity_cadence as c\nassert pathlib.Path(quantlab.__file__).resolve().is_relative_to(pathlib.Path('+repr(str(ROOT))+').resolve()/"quantlab")\ntry: c.load_frozen()\nexcept ValueError as e: assert str(e).startswith("nonfrozen_quantlab_already_imported:")\nelse: raise AssertionError("guard disabled")')
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)
 def test_module_count_comes_from_real_unittest_result_and_failures_survive(self):
  with tempfile.TemporaryDirectory() as d,bootstrap_environment(ROOT) as env:
   out=Path(d)
   result=run_module(ROOT,'tests.this_module_does_not_exist',out,env,timeout=30)
   self.assertFalse(result['passed']);self.assertEqual(result['tests_run'],1);self.assertEqual(result['errors'],1)
 def test_discovery_includes_all_flat_modules_without_importing_them(self):
  actual=discover_modules(ROOT)
  expected=['tests.'+p.stem for p in sorted((ROOT/'tests').glob('test*.py'))]
  self.assertEqual(actual,expected)
 def test_legacy_top_level_fixture_import(self):
  result=self.invoke('import pathlib,fixtures_v2; assert pathlib.Path(fixtures_v2.__file__).resolve()==pathlib.Path('+repr(str(ROOT/'tests/fixtures_v2.py'))+').resolve()')
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)
 def test_empty_root_cannot_pass(self):
  import json
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'empty';root.mkdir();out=Path(d)/'out'
   result=subprocess.run([sys.executable,str(ROOT/'scripts/run_isolated_tests.py'),'--root',str(root),'--output',str(out)],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0)
   self.assertEqual(json.loads((out/'summary.json').read_text())['status'],'failed')
 def test_zero_test_module_cannot_pass(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'tests').mkdir();(root/'tests/__init__.py').write_text('');(root/'tests/test_empty.py').write_text('import unittest')
   with bootstrap_environment(root) as env:
    result=run_module(root,'tests.test_empty',root/'out',env,timeout=30)
   self.assertEqual(result['tests_run'],0);self.assertFalse(result['passed'])
if __name__=='__main__':unittest.main()
