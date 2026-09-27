import unittest
from collections import Counter
import numpy as np
import pandas as pd
from scripts.router_random_diagnostic import sample_basket, forward_label, block_indices
class DiagnosticTests(unittest.TestCase):
 def test_no_replacement_cap(self):
  h={str(i):float(i<25) for i in range(80)};s={str(i):str(i%4) for i in range(80)};r=np.random.default_rng(1)
  for _ in range(100):
   x=sample_basket(h,s,50,r);self.assertEqual(len(set(x)),50);self.assertLessEqual(max(Counter(s[c] for c in x).values()),15)
 def test_tier_precedence(self):
  h={str(i):float(i<10) for i in range(30)}
  self.assertTrue(set(map(str,range(10)))<=set(sample_basket(h,{c:c for c in h},15,np.random.default_rng(2))))
 def test_exact_strata(self):
  h={str(i):float(i%2) for i in range(60)};s={c:str(int(c)%3) for c in h};a=list(h)[:12]
  x=sample_basket(h,s,12,np.random.default_rng(3),a)
  self.assertEqual(Counter((h[c],s[c]) for c in x),Counter((h[c],s[c]) for c in a))
 def test_reproducible_no_future_filter(self):
  h={str(i):1. for i in range(20)};s={c:c for c in h}
  self.assertEqual(sample_basket(h,s,10,np.random.default_rng(7)),sample_basket(h,s,10,np.random.default_rng(7)))
  self.assertIn('17',set(sum([sample_basket(h,s,10,np.random.default_rng(i)) for i in range(10)],[])))
 def test_append_invariance(self):
  c=pd.bdate_range('2020-01-01',periods=100);p=pd.Series(np.arange(100)+10.,index=c)
  a=forward_label(p.iloc[:50],c[:50],c[2],20,c[49],[])
  self.assertEqual(a,forward_label(p,c,c[2],20,c[49],[]));self.assertEqual(a[0],32/12-1)
 def test_endpoint(self):
  c=pd.bdate_range('2020-01-01',periods=100);p=pd.Series(10.,index=c)
  self.assertEqual(forward_label(p,c,c[80],20,c[-1],[])[1],'immature')
  self.assertEqual(forward_label(p,c,c[2],60,c[40],[])[1],'period_end')
 def test_unknown_gap(self):
  c=pd.bdate_range('2020-01-01',periods=30);p=pd.Series(10.,index=c);p.iloc[5]=np.nan
  self.assertEqual(forward_label(p,c,c[0],20,c[-1],[])[1],'missing_path')
  self.assertEqual(forward_label(p.fillna(10),c,c[0],20,c[-1],[str(c[10].date())])[1],'known_gap')
 def test_session_blocks(self):
  x=block_indices(101,20,np.random.default_rng(8));self.assertEqual(len(x),101)
  for i in range(0,100,20):self.assertTrue(np.all(np.diff(x[i:i+20])%101==1))
if __name__=='__main__':unittest.main()
