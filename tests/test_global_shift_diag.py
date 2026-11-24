import unittest
import numpy as np
from dacsc.global_shift_diag import moments,fit_moments,distance,loto_stats,pullback

class GlobalDiagTests(unittest.TestCase):
    def test_moments_reproduce_direct_z_distance(self):
        rng=np.random.default_rng(1);source=rng.normal(size=(9,48,10));target=rng.normal(2,size=(48,10))
        stats=fit_moments([moments(x) for x in source]);d=distance(moments(target),stats)
        direct=np.sqrt(np.mean(((target-stats['mean'])/stats['std'])**2))
        self.assertAlmostEqual(d['z_rms'],direct)
    def test_loto_never_reads_own_task_or_shift(self):
        records=[]
        for i in range(10):records.append({'case_id':str(i),'base_task':str(i),'factor':'clean','moments':{'key__agentview':moments(np.ones((2,3))*i)}})
        records.append({'case_id':'s','base_task':'other','factor':'sensor_noise','moments':{'key__agentview':moments(np.ones((2,3))*1e9)}})
        a,ids=loto_stats(records,'0','key','agentview');records[0]['moments']['key__agentview']=moments(np.ones((2,3))*1e9)
        b,_=loto_stats(records,'0','key','agentview');np.testing.assert_array_equal(a['mean'],b['mean']);self.assertNotIn('0',ids);self.assertNotIn('s',ids)
    def test_normal_components_are_exactly_unchanged(self):
        x=np.array([[-4.,-2.,0.,2.,4.]])
        y,mask=pullback(x,np.zeros(5),np.ones(5),2,.5)
        np.testing.assert_array_equal(y[~mask],x[~mask]);np.testing.assert_array_equal(y,np.array([[-3.,-2.,0.,2.,3.]]))
    def test_wrong_source_count_fails(self):
        with self.assertRaises(ValueError):loto_stats([],'t','key','agentview')

if __name__=='__main__':unittest.main()
