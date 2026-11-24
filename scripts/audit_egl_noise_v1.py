"""Measure legacy EGL constructor noise before reset (both EGL runners use resolved BDDL)."""
import argparse, json, os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--libero',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    from dacsc.static_visual_v2 import file_hash
    m=json.loads(a.manifest.read_text());assert m['kind']=='dev' and m['status']=='FROZEN' and os.environ['MUJOCO_GL']=='egl'
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.env_wrapper as wrapper
    rows=[]
    for c in m['cases']:
        if c['role']!='target_dev' or c['factor']!='sensor_noise':continue
        env=OffScreenRenderEnv(bddl_file_name=str(a.libero/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
        try:rows.append(dict(case_id=c['case_id'],bddl=c['bddl'],native_before_reset=int(env.noise),manual_code=c['runtime_params']['noise']))
        finally:env.close()
    assert len(rows)==30
    a.out.write_text(json.dumps(dict(manifest_sha256=file_hash(a.manifest),wrapper_sha256=file_hash(wrapper.__file__),rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
