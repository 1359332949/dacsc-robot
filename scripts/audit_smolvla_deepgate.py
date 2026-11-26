import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);r=p.parse_args().run
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=read(r/'manifest.json');summary=read(r/'summary.json');status=read(r/'status.json');gate=read(r/'source_gate.json')['tasks']
assert status['phase']=='complete' and status['main_episodes']==100 and status['supplemental_episodes']==2
assert summary['fingerprint']==m['fingerprint']
for name,h in m['source_sha256'].items():assert sha(r/'source_code'/name)==h
assert sha(r/'source_gate.json')==m['source_gate_sha256']==sha(r.parent/'phase3_smolvla_unified_diag_v1/source_gate.json')
assert sha(r.parent/'phase3_smolvla/spatial_init0_calib.npz')==m['calib_sha256']
for v in read(r/'weights_receipt.json').values():assert sha(v['path'])==v['sha256']
assert len(list((r/'pairs').glob('*.json')))==51
arms=('frozen','deep_gate_L3_a0.75');rows=[];events=[];render_different=0;equal_input_checks=0;all_frames=0;hist=[]
for c in m['cases']:
 f=r/'pairs'/(c['id']+'.json');pair=read(f);assert pair['case']==c and pair['fingerprint']==m['fingerprint']
 assert sha(f)==summary['raw_sha256'][c['id']]
 a,b=[pair['arms'][k] for k in arms];assert a['start']==b['start'] and a['observed_start']==b['observed_start']
 for arm in (a,b):
  assert arm['finite_actions'] and arm['weights_frozen_and_versions_unchanged'] and len(arm['trace'])==arm['steps'] and 0<arm['steps']<=220
  h=hashlib.sha256()
  for i,t in enumerate(arm['trace']):
   action=np.array(t['action'],dtype=np.float32);assert t['step']==i and action.shape==(7,) and np.isfinite(action).all();h.update(action.tobytes());all_frames+=1
   if arm['arm']!=arms[0]:
    g=t['gate'];assert g['decision_rule']=='source' and g['threshold']==gate[str(c['task_id'])]['threshold']
    assert np.isfinite(g['score']) and g['score']==float(np.mean(g['raw_layer12_R']))
    assert g['accepted']==(g['score']>g['threshold']) and g['encoder_calls']==(4 if g['accepted'] else 2) and g['prefix_builds']==1
  assert h.hexdigest()==arm['action_sha256']
  for event in arm['rounding_events']:assert event['pixels']<=64 and event['max_lsb']<=1;events.append(event)
 for x,y in zip(a['trace'],b['trace']):
  assert x['rng_before']==y['rng_before'] and x['rng_after']==y['rng_after']
  eq=all(x['snapshot'][k]==y['snapshot'][k] for k in ('sim','eef','quat','gripper'))
  if eq and x['render_state']!=y['render_state']:render_different+=1
  if eq and x['render_state']==y['render_state']:
   equal_input_checks+=1;assert x['snapshot']==y['snapshot']
   if not y['gate']['accepted']:assert x['action']==y['action']
 rows.append(pair)
 if c['scored']:
  folder=r.parent/('phase3_smolvla_sev3' if c['group']=='Sensor' else 'phase3_smolvla_shift')
  old=read(folder/'pairs'/(c['id']+'_frozen.json'))
  n=min(a['steps'],old['steps']);assert [t['rng_before'] for t in a['trace'][:n]]==old['rng_before_replans'][:n]
  first=next((i for i,(x,y) in enumerate(zip(a['trace'],old['frames'])) if x['action']!=y['action']),None)
  hist.append({'case_id':c['id'],'old_success':old['success'],'current_success':a['success'],'actions_bit_exact':old['actions_sha256']==a['action_sha256'],'first_action_difference_step':first,'initial_pixels_and_state_equal':a['start']==old['start'],'observed_initial_pixels_equal':a['observed_start']==old['observed_start']})
assert sum(c['scored'] for c in m['cases'])==50 and len(rows)==51
for name,g in summary['groups'].items():
 def keep(c):
  if not c['scored']:return False
  if name=='All50':return True
  if name=='Shift40':return c['group']!='Clean'
  return c['group']==name or c['subgroup']==name
 part=[v for v in rows if keep(v['case'])];f=[bool(v['arms'][arms[0]]['success']) for v in part];v=[bool(p['arms'][arms[1]]['success']) for p in part]
 rescued=sum(not x and y for x,y in zip(f,v));harmed=sum(x and not y for x,y in zip(f,v));assert (len(part),sum(f),sum(v),rescued,harmed)==(g['n'],g['frozen'],g['deepgate'],g['rescued'],g['harmed'])
 pv=float(binomtest(rescued,rescued+harmed,.5).pvalue) if rescued+harmed else 1.;assert pv==g['mcnemar_exact_two_sided']
 traces=[p['arms'][arms[1]]['trace'] for p in part];accepts=[sum(t['gate']['accepted'] for t in trace) for trace in traces]
 assert sum(accepts)==g['gate']['accepted_replans'] and sum(map(len,traces))==g['gate']['total_replans']
 assert sum(a>0 for a in accepts)==g['gate']['episodes_ever_accepted']
result={'status':'PASS','main_conditions':50,'main_episodes':100,'supplemental_episodes':2,'atomic_pairs':51,'all_replans_checked':all_frames,'source_and_weights_and_calibration_hashes_unchanged':True,'all_raw_hashes_actions_rng_gate_decisions_and_counts_verified':True,'same_physical_render_input_checks':equal_input_checks,'physical_equal_render_different_frames':render_different,'canonicalization_events':len(events),'max_canonical_pixels':max([v['pixels'] for v in events],default=0),'max_canonical_lsb':max([v['max_lsb'] for v in events],default=0),'historical_frozen_actions_exact':sum(v['actions_bit_exact'] for v in hist),'historical_frozen_success_agreement':sum(v['old_success']==v['current_success'] for v in hist),'historical_frozen_cases':hist,'historical_note':'Cross-run renderer/numerical trajectory sensitivity retained; the current concurrent paired baseline is authoritative. Old excluded attempt and old V1 are not mixed into current statistical comparisons.'}
(r/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='historical_frozen_cases'},indent=2))
