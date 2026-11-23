import json,tempfile,unittest
from pathlib import Path
from argparse import Namespace
from dacsc.static_visual_v2 import deterministic_partition,deterministic_dev_sample,file_hash
from scripts.static_visual_v2 import freeze_suite

class SuiteFactorTests(unittest.TestCase):
    def rows(self):
        return [{'suite':'s','base_task':'task'+str(i%10),'factor':'lighting','equivalence_group':str(i),'logical_task_name':'n'+str(i),'history_status':'candidate'} for i in range(140)]
    def test_suite_quota_ignores_sparse_base_task(self):
        rows=self.rows();rows[0]['base_task']='sparse'
        d=deterministic_partition(rows,30,100,7,cell_fields=('suite','factor'))
        self.assertEqual(len(d['dev']),30);self.assertEqual(len(d['blind']),100)
        self.assertEqual(d,deterministic_partition(rows[::-1],30,100,7,cell_fields=('suite','factor')))
    def test_dev_reuse_explicitly_not_a_blind_pool(self):
        rows=self.rows()
        for r in rows:r['history_status']='quarantine'
        self.assertEqual(len(deterministic_dev_sample(rows,30,7,['lighting'])),30)
        self.assertEqual(deterministic_partition(rows,30,100,7,cell_fields=('suite','factor')),{'dev':[],'blind':[]})
    def test_suite_freeze_never_manufactures_fresh_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit=root/'a';audit.mkdir();results=root/'r';results.mkdir()
            rows=self.rows()
            for r in rows:r.update(case_id=r['logical_task_name'],pair_group_id='p'+r['base_task'],clean_bddl='c',clean_bddl_sha256='h')
            a={'settings':{'sampling_unit':'suite_factor','historical_dev_reuse_authorized':True,'dev_per_factor':30,'blind_per_factor':100,'sampling_seed':7,'dev_factors':['lighting'],'confirmatory_factors':['lighting']},'parse_errors':[],'assets_present':True,'source_plan':[{'case_id':str(i),'base_task':'t','planned_smoke':True} for i in range(5)],'candidate_tasks':['t'],'registry':rows}
            p=audit/'p0_audit.json';p.write_text(json.dumps(a))
            for i in range(5):(results/(str(i)+'.json')).write_text(json.dumps({'role':'source_fit','case_id':str(i),'base_task':'t','status':'completed','success':i==0,'audit_sha256':file_hash(p)}))
            freeze_suite(Namespace(audit=audit,results=results,out=root/'out'))
            b=json.loads((root/'out/blind_manifest.json').read_text());d=json.loads((root/'out/dev_manifest.json').read_text())
            self.assertEqual(b['cases'],[]);self.assertEqual(b['status'],'NO_UNACCESSED_OFFICIAL_POOL')
            self.assertEqual(len([r for r in d['cases'] if r['role']=='target_dev']),30)
            self.assertTrue(all(r['freshness']=='historically_accessed_development_reuse' for r in d['cases']))
    def test_zero_of_five_stops_before_target_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit=root/'a';audit.mkdir();results=root/'r';results.mkdir()
            a={'settings':{'sampling_unit':'suite_factor'},'parse_errors':[],'assets_present':True,
               'source_plan':[{'case_id':str(i),'base_task':'t','planned_smoke':True} for i in range(5)],'candidate_tasks':['t']}
            p=audit/'p0_audit.json';p.write_text(json.dumps(a))
            for i in range(5):(results/(str(i)+'.json')).write_text(json.dumps({'role':'source_fit','case_id':str(i),'base_task':'t','status':'completed','success':False,'audit_sha256':file_hash(p)}))
            with self.assertRaisesRegex(PermissionError,'0/5'):freeze_suite(Namespace(audit=audit,results=results,out=root/'out'))
            self.assertFalse((root/'out').exists())

    def test_step_keyed_sensor_discarded_frames_do_not_affect_query(self):
        import numpy as np
        from dacsc.static_visual_v2 import sensor_transform
        class W:
            @staticmethod
            def motion_blur(image,severity):return np.asarray(image)+np.random.randint(0,10,size=np.asarray(image).shape)
        w=W();w.gaussian_blur=w.zoom_blur=w.fog=w.glass_blur=w.motion_blur
        obs={'agentview_image':np.zeros((8,8,3),np.uint8),'robot0_eye_in_hand_image':np.ones((8,8,3),np.uint8)}
        full={i:sensor_transform(obs,1,123,i,w)['agentview_image'] for i in range(25)}
        for i in (0,12,24):np.testing.assert_array_equal(full[i],sensor_transform(obs,1,123,i,w)['agentview_image'])

    def test_no_unresolved_clean_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'p0_audit.json').write_text(json.dumps({'settings':{'sampling_unit':'suite_factor'},'parse_errors':[],'assets_present':True,'source_plan':[{'planned_smoke':True,'case_id':'x'}]}))
            results=root/'results';results.mkdir()
            with self.assertRaises(PermissionError):freeze_suite(Namespace(audit=root,results=results,out=root/'out'))
if __name__=='__main__':unittest.main()
