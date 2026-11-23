import copy,json,random,tempfile,unittest
from pathlib import Path
import numpy as np
from dacsc.static_visual_v2 import (resolve_task,equivalence_key,seed_for,authorize,EnvironmentRNG,isolated_seed,sensor_transform,deterministic_partition,file_hash)

BASE='pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate'
class StaticV2Tests(unittest.TestCase):
    def test_wrapper_parameters_distinguish_same_bddl(self):
        cam=resolve_task(BASE+'_view_0_0_100_2_352_initstate_0','camera')
        noise=resolve_task(BASE+'_view_0_0_100_0_0_initstate_0_noise_3','sensor_noise')
        self.assertEqual(cam['resolved_bddl'],noise['resolved_bddl'])
        self.assertNotEqual(equivalence_key('s',BASE,'same',cam['runtime_params'],'init'),equivalence_key('s',BASE,'same',noise['runtime_params'],'init'))
        self.assertEqual(cam['runtime_params']['view'],[0,0,100,2,352]);self.assertEqual(noise['runtime_params']['noise'],3)
    def test_initial_pose_is_not_episode_index(self):
        with self.assertRaises(ValueError):resolve_task(BASE+'_view_0_0_100_0_0_initstate_3','camera')
    def test_mixed_shift_rejected(self):
        for name,factor in [(BASE+'_view_1_0_100_0_0_initstate_0_noise_3','sensor_noise'),(BASE+'_light_1','camera')]:
            with self.assertRaises(ValueError):resolve_task(name,factor)
    def test_static_asset_resolution(self):
        x=resolve_task(BASE+'_light_50','lighting');self.assertEqual(x['base_task'],BASE);self.assertEqual(x['resolved_bddl'],BASE+'_light_50.bddl')
        self.assertEqual(x['init_file'],BASE+'.pruned_init')
    def test_exact_asset_alias_dedup(self):
        a=resolve_task(BASE+'_light_1','lighting')['runtime_params'];b=resolve_task(BASE+'_light_2','lighting')['runtime_params']
        self.assertEqual(equivalence_key('s',BASE,'identical_asset',a,'init'),equivalence_key('s',BASE,'identical_asset',b,'init'))
    def test_rng_stream_separation(self):
        def simulate(extra):
            env=EnvironmentRNG(7);values=[]
            for step in range(4):
                with env.use():values.append((np.random.rand(),random.random()))
                with isolated_seed(seed_for('augmentation',step)):np.random.rand(extra);[random.random() for _ in range(extra)]
                with isolated_seed(seed_for('sensor',step)):values.append((np.random.rand(),random.random()))
            return values
        self.assertEqual(simulate(1),simulate(1000))
    def test_rng_restores_even_on_failure(self):
        np.random.seed(9);expected=np.random.rand(3);np.random.seed(9)
        with self.assertRaises(RuntimeError):
            with isolated_seed(77):np.random.rand(100);raise RuntimeError()
        np.testing.assert_array_equal(np.random.rand(3),expected)
    def test_sensor_only_agentview_and_same_realization(self):
        class Wrapper:
            @staticmethod
            def motion_blur(image,severity):return np.asarray(image)+np.random.randint(0,10,size=np.asarray(image).shape)
        w=Wrapper();w.gaussian_blur=w.zoom_blur=w.fog=w.glass_blur=w.motion_blur
        obs={'agentview_image':np.zeros((8,8,3),np.uint8),'robot0_eye_in_hand_image':np.ones((8,8,3),np.uint8)}
        a=sensor_transform(obs,1,22,3,w)
        np.random.rand(300)
        b=sensor_transform(obs,1,22,3,w)
        np.testing.assert_array_equal(a['agentview_image'],b['agentview_image']);np.testing.assert_array_equal(a['robot0_eye_in_hand_image'],obs['robot0_eye_in_hand_image'])
        self.assertEqual(obs['agentview_image'].sum(),0)
    def test_blind_denied_even_if_status_changed(self):
        for role in ('target_blind','blind_clean','clean_holdout'):
            c={'role':role};m={'status':'FROZEN','kind':'dev','cases':[c]}
            with self.assertRaises(PermissionError):authorize(m,c,'render')
        c={'role':'source_fit'}
        with self.assertRaises(PermissionError):authorize({'kind':'blind','status':'FROZEN','cases':[c]},c,'rollout')
    def test_p0_only_source(self):
        c={'role':'target_dev'}
        with self.assertRaises(PermissionError):authorize({'kind':'dev','status':'P0_SMOKE_ONLY','cases':[c]},c,'render')
    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';p.write_text('{}');old=file_hash(p);p.write_text('{"edited":true}')
            c={'role':'source_fit'}
            with self.assertRaises(PermissionError):authorize({'kind':'dev','status':'FROZEN','cases':[c]},c,'run',old,p)
    def test_partition_deterministic_no_group_overlap(self):
        rows=[{'base_task':'t','factor':'f','equivalence_group':str(i),'logical_task_name':str(i),'history_status':'candidate'} for i in range(20)]
        a=deterministic_partition(rows,3,10,2026112301);b=deterministic_partition(list(reversed(rows)),3,10,2026112301)
        self.assertEqual(a,b);self.assertFalse({r['equivalence_group'] for r in a['dev']}&{r['equivalence_group'] for r in a['blind']})
        rows[0]['history_status']='quarantine';c=deterministic_partition(rows,3,10,99)
        self.assertNotIn('0',[r['equivalence_group'] for r in c['dev']+c['blind']])
    def test_equivalent_alias_order_does_not_change_manifest(self):
        rows=[{'base_task':'t','factor':'f','equivalence_group':str(i),'logical_task_name':str(i),'history_status':'candidate'} for i in range(13)]
        rows.append(dict(rows[0],logical_task_name='alias'))
        self.assertEqual(deterministic_partition(rows,3,10,5),deterministic_partition(rows[::-1],3,10,5))

    def test_empty_blind_cli_rejects_before_assets(self):
        from argparse import Namespace
        from scripts.static_visual_v2 import run
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'blind.json';path.write_text(json.dumps({'kind':'blind','cases':[]}))
            with self.assertRaises(PermissionError):run(Namespace(manifest=path,case_id='anything'))
    def test_history_incomplete_cannot_freeze(self):
        from argparse import Namespace
        from scripts.static_visual_v2 import freeze
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'p0_audit.json').write_text(json.dumps({'settings':{'history_complete':False},'parse_errors':[]}))
            with self.assertRaises(PermissionError):freeze(Namespace(audit=root))
    def test_summary_keeps_missing_instances(self):
        from argparse import Namespace
        from scripts.static_visual_v2 import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp=root/'m.json';mp.write_text(json.dumps({'kind':'dev','cases':[{'role':'target_dev','case_id':'x','base_task':'t','factor':'camera'}]}))
            results=root/'results';results.mkdir();out=root/'report.json'
            summarize(Namespace(manifest=mp,results=results,out=out))
            d=json.loads(out.read_text());self.assertFalse(d['complete']);self.assertIsNone(d['frozen_shift_minus_clean_macro']);self.assertEqual(d['cells']['t|camera']['missing'],1)

    def test_freeze_keeps_independent_clean_holdout_separate(self):
        from argparse import Namespace
        from scripts.static_visual_v2 import freeze
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit=root/'audit';audit.mkdir();results=root/'runs';results.mkdir()
            source=[{'base_task':'t','case_id':'smoke'+str(i),'role':'source_fit','planned_smoke':True} for i in range(5)]
            source += [{'base_task':'t','case_id':'hold'+str(i),'role':'clean_holdout','planned_smoke':False} for i in range(10)]
            registry=[]
            for f in ('camera','lighting','sensor_noise','background'):
                for i in range(13):registry.append({'base_task':'t','factor':f,'case_id':f+str(i),'logical_task_name':f+str(i),'equivalence_group':f+str(i),'history_status':'candidate','pair_group_id':'same-default-init','clean_bddl':'clean.bddl','clean_bddl_sha256':'clean'})
            a={'settings':{'history_complete':True,'dev_per_cell':3,'blind_per_cell':10,'sampling_seed':7},'parse_errors':[],'assets_present':True,'shortfalls':[],'candidate_tasks':['t'],'registry':registry,'source_plan':source}
            ap=audit/'p0_audit.json';ap.write_text(json.dumps(a));h=file_hash(ap)
            for i in range(5):(results/(str(i)+'.json')).write_text(json.dumps({'base_task':'t','role':'source_fit','case_id':'smoke'+str(i),'status':'completed','success':True,'audit_sha256':h}))
            review=root/'review.json';review.write_text(json.dumps({'audit_sha256':h,'independent_clean_holdout_verified':True}))
            freeze(Namespace(audit=audit,results=results,review=review,out=root/'frozen'))
            d=json.loads((root/'frozen/dev_manifest.json').read_text());b=json.loads((root/'frozen/blind_manifest.json').read_text())
            self.assertEqual(sum(c['role']=='target_dev' for c in d['cases']),12)
            self.assertEqual(sum(c['role']=='dev_clean' for c in d['cases']),1)
            self.assertEqual(sum(c['role']=='target_blind' for c in b['cases']),40)
            self.assertEqual(sum(c['role']=='clean_holdout' for c in b['cases']),10)
            self.assertFalse(any(c['role']=='blind_clean' for c in b['cases']))

    def test_no_seed_based_quota_filling(self):
        rows=[{'base_task':'t','factor':'f','equivalence_group':'same','logical_task_name':str(i),'history_status':'candidate'} for i in range(20)]
        with self.assertRaises(ValueError):deterministic_partition(rows,3,10,7)
if __name__=='__main__':unittest.main()
