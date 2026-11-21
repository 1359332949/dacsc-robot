import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dacsc.source_observer import (Data,fit,predict,state_weights,weighted_pca,design,load_role,sha,gate,comparison,state_metrics)

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'experiments/source_observer_v1.json').read_text())


def toy():
    rng=np.random.default_rng(3);n=30;states=np.repeat([0,1,2],10)
    q=rng.normal(size=(n,8));a=rng.normal(size=(n,21));cur={};nxt={};bounds={}
    for v in CONFIG['views']:
        cur[v]=rng.normal(size=(n,12));nxt[v]=cur[v]+.1*a[:,0,None]*rng.normal(size=(1,12))+.02*rng.normal(size=(n,12))
        bounds[v]=np.r_[cur[v],nxt[v]]
    return Data('source_fit',states,q,a,cur,nxt,bounds,np.r_[states,states],[],[],0)


class ObserverIsolation(unittest.TestCase):
    def test_fit_refuses_validation(self):
        with self.assertRaises(ValueError):fit(replace(toy(),role='source_validation'),CONFIG)

    def test_state_total_weights_equal(self):
        s=np.array([1,1,2,2,2,2]);w=state_weights(s)
        self.assertAlmostEqual(w[s==1].sum(),.5);self.assertAlmostEqual(w[s==2].sum(),.5)

    def test_weighted_pca_matches_equal_weight_reference(self):
        from sklearn.decomposition import PCA
        rng=np.random.default_rng(5);x=rng.normal(size=(30,12))
        mu,p,_=weighted_pca(x,np.ones(30)/30,8);ref=PCA(n_components=8,svd_solver='full').fit(x)
        np.testing.assert_allclose(mu,ref.mean_,atol=1e-12)
        np.testing.assert_allclose(p.T@p,ref.components_.T@ref.components_,atol=1e-12)

    def test_weighted_ridge_matches_declared_objective(self):
        data=toy();arrays,_=fit(data,CONFIG);v=CONFIG['views'][0];k='M1';pre=v+'_'+k
        z=(data.current[v]-arrays[v+'_mu'])@arrays[v+'_pca'].T
        x=design(k,z,data.q,data.actions);x=(x-arrays[pre+'_x_mu'])/arrays[pre+'_x_scale'];d=np.c_[x,np.ones(len(x))]
        y=(data.following[v]-data.current[v])@arrays[v+'_pca'].T/arrays[v+'_target_scale']
        w=state_weights(data.states);penalty=np.eye(d.shape[1]);penalty[-1,-1]=0
        coef=np.linalg.solve(d.T@(w[:,None]*d)+penalty,d.T@(w[:,None]*y))
        np.testing.assert_allclose(coef[:-1],arrays[pre+'_coef'],atol=1e-10)
        np.testing.assert_allclose(coef[-1],arrays[pre+'_intercept'],atol=1e-10)

    def test_predictions_cannot_mutate_frozen_arrays(self):
        data=toy();arrays,_=fit(data,CONFIG);before={k:a.copy() for k,a in arrays.items()}
        for v in CONFIG['views']:
            for k in ('M0','M1','M2'):
                predict(arrays,v,k,data.current[v]+1000,data.q-50,data.actions+10)
        for k in arrays:np.testing.assert_array_equal(arrays[k],before[k])

    def test_time_success_metadata_not_model_inputs(self):
        data=toy();a,_=fit(data,CONFIG)
        poisoned=replace(data,timing=[{'remaining_steps':-999,'episode_success':True}]*len(data.states))
        b,_=fit(poisoned,CONFIG)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_gate_checks_both_references_and_subsets(self):
        base={str(s):{'rmse':1.} for s in range(10)};good={str(s):{'rmse':.5} for s in range(10)}
        refs=['M0','M2','zero_delta','fit_mean_next_minus_current']
        r={'comparisons':{k:comparison(good,base,CONFIG) for k in refs},
           'subsets':{name:{'comparisons':{k:comparison(good,base,CONFIG,False) for k in ('M0','M2')}} for name in ('common_early','nonterminal')}}
        self.assertTrue(gate(r,CONFIG)['pass'])
        r['comparisons']['M2']['wins']=7
        self.assertFalse(gate(r,CONFIG)['pass'])

    def test_fit_loader_does_not_open_validation_or_partial_endpoint(self):
        from tests.test_interaction_records import fixture
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);split={'source_train_state_ids':[1],'source_val_state_ids':[2],
                'target_dev_state_ids':[3],'target_test_state_ids':[4],'evaluation_seeds':[2014],'policy':{'checkpoint_sha256':'checkpoint'}}
            sp=root/'split.json';sp.write_text(json.dumps(split))
            folder=root/'fit'/'source_fit_state1_seed2014';folder.mkdir(parents=True)
            bad=root/'val'/'source_validation_state2_seed2014';bad.mkdir(parents=True);(bad/'metadata.json').write_text('INVALID DO NOT READ')
            a,m=fixture();a['state8_executed']=np.zeros((2,12,8));a['sim_time_executed']=np.zeros((2,12))
            for i,n in enumerate(a['executed_steps']):a['sim_time_executed'][i,:n]=a['boundary_sim_time'][i]+.05*np.arange(1,n+1)
            for v in CONFIG['views']:a['tokens_'+v][2]=99999 # endpoint of partial chunk must be excluded
            np.savez(folder/'episode.npz',**a)
            m.update(role='source_fit',domain='clean',data_origin='independent_source_policy_rollout',seed=2014,
                split_sha256=sha(sp),checkpoint_sha256='checkpoint',npz_sha256=sha(folder/'episode.npz'),steps=15,success=False)
            from dacsc.source_provenance import FIELDS
            for key in FIELDS:m.setdefault(key,'synthetic-test')
            for key in ('target_environment_accessed','weights_updated','chunk_risk_labels_assigned'):m[key]=False
            m['upstream_sha256']={'synthetic.py':'test-hash'}
            (folder/'metadata.json').write_text(json.dumps(m))
            d=load_role(root,sp,'source_fit',CONFIG['views'])
            self.assertEqual(len(d.states),1);self.assertEqual(d.excluded_partial,1)
            self.assertEqual(len(d.boundary_states),2)
            for v in CONFIG['views']:self.assertEqual(np.max(d.boundaries[v]),0)

if __name__=='__main__':unittest.main()
