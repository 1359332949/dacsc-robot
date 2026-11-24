"""Hash the self-contained compressed receipt archive and check file-by-file roundtrip."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--dir',type=Path,required=True);a=ap.parse_args()
    files=sorted(a.root.glob('*.json'));assert len(files)==1260; archive=a.dir/'evidence.tar.xz'
    # Sort by case/seed so identical-baseline/rollback traces stay close in the stream.
    order=lambda p:(p.stem.split('_s')[1].split('_',1)[1].rsplit('_',1)[0],p.name)
    with tarfile.open(archive,'w:xz',preset=6) as f:
        for p in sorted(files,key=order):f.add(p,arcname=p.name,recursive=False)
    with tarfile.open(archive) as f:
        actual={m.name:hashlib.sha256(f.extractfile(m).read()).hexdigest() for m in f if m.isfile()};assert actual=={p.name:sha(p) for p in files}
    # Wall span of actual completed rollouts; excludes pilots, transfers and aggregation.
    times=[(p.stat().st_mtime,json.loads(p.read_text())['seconds']) for p in files]
    span=(max(e for e,_ in times)-min(e-d for e,d in times))/60
    vr={'wall_span_minutes_from_receipts':round(span,2),'cases':100,'standard_rollouts':1200,'wrist_stress_rollouts':60,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'all_member_hashes_verified':True,'summary_sha256':sha(a.dir/'summary.json')}
    (a.dir/'verification.json').write_text(json.dumps(vr,indent=2)+'\n');print(vr)
if __name__=='__main__':main()
