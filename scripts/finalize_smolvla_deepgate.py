import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
read=lambda p:json.loads(Path(p).read_text())
s=read(r/'summary.json');a=read(r/'final_audit.json');status=read(r/'status.json');assert a['status']=='PASS' and status['phase']=='complete'
out=r/'export/docs/data/phase3';out.mkdir(parents=True,exist_ok=True)
s['postchecks']=a;s['execution']=status;s['renderer_recovery_review']=read(r/'renderer_recovery_review.json')
(out/'smolvla_deepgate_cl.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
text=(r/'PHASE3_SMOLVLA_DEEPGATE_CL.md').read_text();g=s['groups']['Shift40'];c=s['groups']['Clean'];extra=s['supplemental_task0_clean_init1'];lo,hi=g['task_cluster_bootstrap95_pp']
notes=['','## 完成后的独立核验与裁决','',f"正式重跑用时{status['seconds']/60:.1f}分钟；主表100局与额外2局分开。全部51个原始配对文件、动作SHA、每步门控规则、双侧p值与各组计数已独立复算。",'',f"Shift40：Frozen {g['frozen']}/{g['n']} → DeepGate {g['deepgate']}/{g['n']}；救{g['rescued']}、伤{g['harmed']}，净{g['net']:+d}，任务聚类95%区间[{lo:+.1f},{hi:+.1f}]个百分点。"]
if g['net']<=0:notes.append('本轮不支持该冻结候选在Shift40上的净正收益，不按单个子类报喜，也不回头修改阈值追回开发分数。')
elif lo<=0:notes.append('本次是探索性净正收益，任务聚类区间仍包含0；不能称为稳健显著，也不能替代独立多种子确认。')
else:notes.append('本轮任务聚类区间为正，但仍只有一个策略种子且是反复访问过的开发池，不能代替独立确认。')
notes+=['',f"Clean主表：救{c['rescued']}、伤{c['harmed']}；额外task0/init1：救{extra['rescued']}、伤{extra['harmed']}。"]
if c['harmed'] or extra['harmed']:notes.append('存在Clean误伤，不得把首帧0触发写成整局零误伤或通用安全保证。')
else:notes.append('仅本轮已测Clean配对无新增误伤，不推出其他初始状态/任务/种子的普适安全。')
notes+=['',f"本次记录到{a['canonicalization_events']}次同物理/记录渲染状态的原始RGB微小差异，最大{a['max_canonical_pixels']}像素、{a['max_canonical_lsb']}灰度；均在既有CoRP数值界限内，替换前后图像哈希逐项保留。另有{a['physical_equal_render_different_frames']}个共同步仅SimState相同而记录渲染状态不同，未强行复用图片。",f"与之前独立运行的Frozen对照相比，动作SHA完全一致{a['historical_frozen_actions_exact']}/50，成功结局一致{a['historical_frozen_success_agreement']}/50。跨运行数值/渲染轨迹敏感性如实保留；本轮效果只以当前同期配对Frozen为准，不把旧V1当作同期第三臂。",'', '配对控制读取模拟器/渲染状态仅用于验证同输入和缓存，不进入门控或动作模型。部署门控模块只接收原策略图像、固定源参考与固定阈值。', '源域拟合不使用这些目标观测或成功标签；不过方法假设是在已访问开发数据上形成，不能把源域标定等同于完全未使用目标域开发信息。', 'SmolVLA加门控的结果不证明CoRP加相同门控有效；原CoRP五臂表仍是未加该总闸的V1。','']
(r/'export/docs/PHASE3_SMOLVLA_DEEPGATE_CL.md').write_text(text+'\n'.join(notes))
print('Final exported report and JSON generated with independent audit')
