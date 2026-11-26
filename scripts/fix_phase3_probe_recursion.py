"""Guard the legacy bridge against python -c probe startup; terminate only its probe chain."""
from pathlib import Path
import hashlib,json,os,signal,time
out=Path('/hy-tmp/patch-diag/runs/smolvla_reproduction_v1');out.mkdir(parents=True,exist_ok=True)
p=Path('/hy-tmp/phase3_patch/sitecustomize.py');old=p.read_text();(out/'legacy_sitecustomize_before.py').write_text(old)
if 'if not (sys.argv and sys.argv[0] == "-c"):' not in old:
    pre,body=old.split('try:\n',1);new=pre+'import sys\n# GLFW probes spawn python -c; reimporting LIBERO there recursively spawns probes.\nif not (sys.argv and sys.argv[0] == "-c"):\n'+''.join('    '+line if line.strip() else line for line in ('try:\n'+body).splitlines(keepends=True));compile(new,str(p),'exec');p.write_text(new)
rows=[]
for turn in range(5):
    selected=[]
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:
            cmd=(item/'cmdline').read_bytes();env=(item/'environ').read_bytes();st=(item/'stat').read_text().split()
            if not (cmd.startswith(b'/hy-tmp/corp_venv/bin/python\0-c\0') and b'Queries and returns the library version tuple' in cmd and b'ctypes.CDLL(filename)' in cmd and any(v.startswith(b'PYTHONPATH=') and b'/hy-tmp/phase3_patch' in v for v in env.split(b'\0'))):continue
            pid=int(item.name);os.kill(pid,signal.SIGSTOP);selected.append(pid);rows.append(dict(pid=pid,ppid=int(st[3]),start_ticks=int(st[21]),command_sha256=hashlib.sha256(cmd).hexdigest()))
        except (ProcessLookupError,FileNotFoundError):pass
    for pid in selected:
        try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    if not selected:break
    time.sleep(.2)
receipt=dict(kind='scoped_legacy_glfw_probe_cleanup',unique_pids=len({r['pid'] for r in rows}),rows=rows,sitecustomize_before_sha256=hashlib.sha256(old.encode()).hexdigest(),sitecustomize_after_sha256=hashlib.sha256(p.read_bytes()).hexdigest());(out/'probe_cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n');print('guard installed; terminated verified probe processes:',receipt['unique_pids'])
