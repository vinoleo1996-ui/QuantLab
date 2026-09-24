import unittest

import numpy as np
import pandas as pd

from quantlab.strategy.scoring import adaptive_weight, composite_score, weight_row


class AdaptiveWeightContracts(unittest.TestCase):
    def test_increasing_icir_cannot_reduce_the_adaptive_coefficient(self):
        for floor,cap in ((.30,1.60),(.10,.80)):
            values=pd.Series(sorted(set([-1.,floor-1e-8,floor,floor+1e-8,cap,cap+1.,
                *np.linspace(floor,cap,101)])))
            weights=adaptive_weight(values,floor,cap)
            self.assertTrue(weights.diff().dropna().ge(-1e-14).all())
            self.assertTrue(weights.between(.1,1.).all())
            np.testing.assert_allclose(weights,[adaptive_weight(x,floor,cap) for x in values])
            # Preserve the original nonzero minimum and established upper ramp.
            self.assertEqual(adaptive_weight(-.5,floor,cap),.1)
            self.assertAlmostEqual(adaptive_weight((floor+cap)/2,floor,cap),.5)
            self.assertEqual(adaptive_weight(cap,floor,cap),1.)

    def test_stronger_factor_is_not_suppressed_at_the_floor_in_actual_scoring(self):
        day=pd.Timestamp('2024-01-31')
        a=pd.Series({'A':1.,'B':0.,'C':-1.})
        b=pd.Series({'A':-1/np.sqrt(3),'B':2/np.sqrt(3),'C':-1/np.sqrt(3)})
        self.assertAlmostEqual(a.std(),b.std())
        factors={'lower_icir':a.to_frame().T.set_axis([day]),
                 'higher_icir':b.to_frame().T.set_axis([day])}
        weights=adaptive_weight(pd.Series({'lower_icir':.29,'higher_icir':.31}))
        scores=composite_score(factors,weight_row(weights,day),None).loc[day]
        self.assertEqual(scores.idxmax(),'B')

    def test_unknown_icir_is_not_converted_to_a_valid_minimum_weight(self):
        self.assertTrue(np.isnan(adaptive_weight(float('nan'))))
        self.assertTrue(np.isnan(adaptive_weight(pd.Series([np.nan])).iloc[0]))


if __name__=='__main__':unittest.main()
