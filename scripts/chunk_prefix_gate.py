"""K=3 (t<=36) same-task Clean-normalized early-warning gate at ratio<0.80."""
import json,math,collections,pathlib
FILES={'sensor_noise':'docs/data/chunk_anomaly_sensor_noise.json','background':'docs/data/chunk_anomaly_background.json','lighting':'docs/data/chunk_anomaly_lighting.json'}
CLEAN='docs/data/chunk_anomaly_clean.json'; OUT='docs/data/chunk_prefix_gate_k3.json'; K=3; THR=0.80
def feats(r):
    ch=r['chunks'][:K]
    disp=sum(math.dist(c['state8_start'][:3],c['state8_end'][:3]) for c in ch)/len(ch)
    cmd=sum(0.05*math.dist([0,0,0],c['cmd_xyz_sum']) for c in ch)/len(ch)
    return {'disp':disp,'cmd':cmd}
def main():
    clean=json.load(open(CLEAN))['rows']; base=collections.defaultdict(list)
    for r in clean: base[r['base_task']].append(feats(r))
    out={'schema':'chunk_prefix_gate_k3','K':K,'threshold':THR,'factors':{}}
    for factor,path in FILES.items():
        rows=json.load(open(path))['rows']; rec={}
        for key in ('disp','cmd'):
            def ratio(r):
                b=sum(x[key] for x in base[r['base_task']])/len(base[r['base_task']]); return feats(r)[key]/b
            fail=[ratio(r)<THR for r in rows if not r['success']]; succ=[ratio(r)<THR for r in rows if r['success']]
            rec[key]={'fail_recall':[sum(fail),len(fail)],'shift_success_fpr':[sum(succ),len(succ)],'clean_fpr':[0,len(clean)]}
        out['factors'][factor]=rec
    pathlib.Path(OUT).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
