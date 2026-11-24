import sys,json
from pathlib import Path

def check(path):
    data={}
    for p in path.glob('pilot_*.json'):
        r=json.loads(p.read_text());data.setdefault(r['case_id'],{})[r['method']]=r
    assert len(data)==4
    counts=[0,0]
    for group in data.values():
        assert set(group)=={'trust','frozen','rgb','joint'}
        for i,(arm,reference) in enumerate([('trust','frozen'),('joint','rgb')]):
            r=group[arm];b=group[reference]
            if not r['gate']['accepted']:
                assert (r['action_sha'],r['steps'],r['success'])==(b['action_sha'],b['steps'],b['success']);counts[i]+=1
    print('pilot rejected exact matches',counts,flush=True)
if __name__=='__main__':check(Path(sys.argv[1]))
