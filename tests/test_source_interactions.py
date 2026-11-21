import unittest
import numpy as np
from tests.test_interaction_records import fixture
from dacsc.source_interactions import role_ids,validate_source_episode
from scripts.audit_failure_timing import opportunities_before

SPLIT={'source_train_state_ids':[1],'source_val_state_ids':[2],'target_dev_state_ids':[3],
       'target_test_state_ids':[4],'evaluation_seeds':[2014]}
class SourceContracts(unittest.TestCase):
    def test_role_separation(self):
        self.assertEqual(role_ids(SPLIT,'source_fit'),[1])
        for role,ids in [('source_fit',[2]),('source_fit',[3]),('source_validation',[4]),('target_dev',[3])]:
            with self.subTest(role=role,ids=ids),self.assertRaises(ValueError):role_ids(SPLIT,role,ids)
    def test_independent_source_without_fake_probe(self):
        a,m=fixture()
        for k in list(a):
            if k.startswith('probe') or k=='x_probe':del a[k]
        m.update(role='source_fit',domain='clean',data_origin='independent_source_policy_rollout',seed=2014)
        a['state8_executed']=np.zeros((2,12,8));a['sim_time_executed']=np.zeros((2,12))
        for i,n in enumerate(a['executed_steps']):a['sim_time_executed'][i,:n]=a['boundary_sim_time'][i]+.05*np.arange(1,n+1)
        self.assertTrue(validate_source_episode(a,m,SPLIT)['pass'])
        m['data_origin']='target_dev_clean_replay'
        with self.assertRaises(ValueError):validate_source_episode(a,m,SPLIT)
    def test_no_anticipatory_observer_boundary(self):
        self.assertEqual(opportunities_before(1),[])
        self.assertEqual(opportunities_before(12),[])
        self.assertEqual(opportunities_before(13),[12])
        self.assertEqual(opportunities_before(24),[12])

if __name__=='__main__':unittest.main()
