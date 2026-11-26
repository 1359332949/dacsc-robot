import hashlib,json,math
from pathlib import Path
r=Path('/hy-tmp/patch-diag/runs/corp_main_table_v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
protocol=read(r/'run_protocol.json');cal=read(r/'calibration_receipt.json')
assert read(r/'status.json')['phase']=='complete'
for name,h in protocol['source_sha256'].items():assert sha(r/'source_code'/name)==h
for key in ('loto','global'):assert sha(cal[key+'_path'])==cal[key+'_sha256']==protocol[key+'_sha256']
assert sha(cal['frozen_ours_path'])==protocol['frozen_reference_sha256']
assert sha('/hy-tmp/patch-diag/repos/CoRP/checkpoints/corp_libero.pt')==protocol['checkpoint_sha256']
assert sha('/hy-tmp/LIBERO-plus/libero/libero/envs/env_wrapper.py')==protocol['sensor_wrapper_sha256']
assert cal['object_source_frames']==0 and cal['source_frames']==101
assert all(x['b3_moments_bit_exact'] and x['actions_bit_exact'] for x in cal['source_replay'])
inputs=read(r/'reproduction_inputs.json')['files']
for p,h in inputs.items():assert sha(p)==h,p
arms=['frozen','pixel_align','deep_block12','actmad_inspired_viewwise_v2','ours_v1'];summary={}
for suite in ('spatial','object'):
 manifest=read(r/(suite+'_manifest.json'));assert sha(r/(suite+'_manifest.json'))==protocol['manifest_sha256'][suite]
 expected={(c['case_id'],s) for c in manifest['cases'] for s in manifest['seeds']};files=list((r/suite/'jobs').glob('*.json'));assert len(files)==300
 export=read(r/'export/docs/data/corp_main_table_v2'/(suite+'_5arm.json'));counts={a:0 for a in arms};seen=set();roundings=0;updates=0;zero_steps=0
 for p in files:
  v=read(p);pair=(v['case_id'],v['seed']);assert pair in expected and pair not in seen;seen.add(pair)
  assert v['run_fingerprint']==protocol['fingerprints'][suite]
  assert export['raw_sha256'][p.name]==sha(p)
  assert v['paired_initial_state_pixels_and_flow'] and v['all_weights_restored_and_frozen']
  assert [a['arm'] for a in v['arms']]==arms
  f=v['arms'][0]['trace']
  for a in v['arms']:
   assert a['finite'] and len(a['trace'])==a['chunks'] and 0<a['steps']<=220
   assert all(a['trace'][0][k]==f[0][k] for k in ('state','robot','images','z'))
   assert all(x['z']==y['z'] for x,y in zip(a['trace'],f))
   assert len(a['action_sha256'])==64
   for rounding in a['roundings']:assert rounding[2]<=64 and rounding[3]<=1
   roundings+=len(a['roundings']);counts[a['arm']]+=int(a['success'])
   if a['arm']=='actmad_inspired_viewwise_v2':
    assert len(a['updates'])==a['chunks']
    for i,u in enumerate(a['updates']):
     assert u['frames']==min(i+1,4)
     assert math.isfinite(u['loss']) and math.isfinite(u['parameter_step_l2'])
     updates+=1;zero_steps+=int(u['parameter_step_l2']==0)
 assert seen==expected
 for a,n in counts.items():assert n==export['groups']['total']['frozen'][a]['success']
 assert sum(c['factor']=='sensor_bilateral' and c['bilateral'] for c in manifest['cases'])==30
 if suite=='object':assert set(c['task'] for c in manifest['cases']).isdisjoint(cal['source_tasks'])
 summary[suite]={'jobs':300,'episodes':1500,'success_counts':counts,'paired_raw_hashes_verified':True,'rounding_events_within_limits':roundings,'actmad_updates':updates,'actmad_zero_parameter_steps':zero_steps}
receipt={'status':'PASS','jobs':600,'episodes':3000,'all_source_and_calibration_hashes_unchanged':True,'evaluation_and_dependency_files_verified':len(inputs),'source_episodes':10,'source_chunks':101,'source_block3_and_action_replay_bit_exact':True,'object_calibration_frames':0,'suites':summary,'runtime_note':'Weight restoration and action finiteness were asserted inside each actual episode; this audit additionally verifies saved assertions, pairing hashes, fingerprints and independent count totals.'}
(r/'final_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
