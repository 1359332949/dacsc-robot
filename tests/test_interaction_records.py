import copy
import unittest
import numpy as np
from dacsc.interaction_records import array_hash, step_flags, validate_episode, validate_pair


def fixture():
    n=np.array([12,3]); mask=np.arange(12)[None,:]<n[:,None]
    a={'executed_steps':n,'executed_mask':mask,'actions_executed':np.zeros((2,12,7)),
       'actions_planned':np.zeros((2,12,7)),'boundary_step':np.array([0,12,15]),
       'boundary_sim_time':np.array([0.5,1.1,1.25]),'state8':np.zeros((3,8)),
       'probe_anchor_state8':np.zeros((3,8)),'full_chunk':n==12,'fixed_horizon_eligible':n==12,
       'tokens_agentview':np.zeros((3,48,768)),'tokens_eye_in_hand':np.zeros((3,48,768)),
       'probe_hidden':np.zeros((3,768)),'probe_velocity':np.zeros((3,12,7)),'x_probe':np.zeros((1,12,7))}
    for k in ('success','terminated','truncated','done'): a[k]=np.zeros((2,12),bool)
    a['truncated'][1,2]=True
    m=dict(control_dt=.05,x_probe_sha256=array_hash(a['x_probe']),state_id=1,seed=7,probe_tau=.5,
           probe_action_index=0,language='task',initial_state_sha256='state')
    return a,m

class Contracts(unittest.TestCase):
    def test_partial_chunk_kept_but_not_fixed_horizon(self):
        a,m=fixture();r=validate_episode(a,m);self.assertEqual(r['partial_chunks'],1)
    def test_reject_temporal_mask_feature_and_probe_corruption(self):
        for key in ('boundary_step','executed_mask','tokens_agentview','probe_anchor_state8','x_probe'):
            a,m=fixture()
            if key=='boundary_step':a[key][1]+=1
            elif key=='executed_mask':a[key][1,8]=True
            elif key=='tokens_agentview':a[key][0,0,0]=np.nan
            elif key=='probe_anchor_state8':a[key][1,0]=1
            else:a[key][0,0,0]=1
            with self.subTest(key=key),self.assertRaises(ValueError):validate_episode(a,m)
    def test_reject_action_swap_between_domains(self):
        a,m=fixture();b=copy.deepcopy(a);b['actions_executed'][0,0,0]=.5;b['actions_planned'][0,0,0]=.5
        with self.assertRaises(ValueError):validate_pair(a,b,m,m)
    def test_terminal_chunk_cannot_continue(self):
        a,m=fixture();a['success'][0,5]=True
        with self.assertRaises(ValueError):validate_episode(a,m)
    def test_timeout_is_not_success(self):
        class E:
            def check_success(self):return False
        _,f=step_flags(({},0,True,{'TimeLimit.truncated':True}),E())
        self.assertTrue(f['truncated']);self.assertFalse(f['success']);self.assertFalse(f['terminated'])
    def test_success_independent_of_done(self):
        _,f=step_flags(({},1,False,{'success':True}),None)
        self.assertTrue(f['success']);self.assertFalse(f['done']);self.assertFalse(f['terminated'])

if __name__=='__main__':unittest.main()
