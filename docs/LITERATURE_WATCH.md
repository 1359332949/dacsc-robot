# Literature Watch — VLA Test-Time Adaptation

## 2026-11-24 — Weekly update

### Executive read

- No independently verified reproduction was found for SALT, Verify-Act-Distill (VAD), or COAST. In the sources checked, SALT and VAD remain arXiv v1 preprints; COAST is arXiv v1 with a NeurIPS 2026 submission noted by a secondary index, but no acceptance was verified. Do not count revisions as new papers.
- The clearest new item is HDH (10 Nov): a lightweight verifier trained on the frozen policy's own success/failure rollouts selects between two sampled chunks. Its project page reports 82.3% mean success on four LIBERO-10 tasks with 5.4% median query-latency overhead. This is outcome-supervised candidate selection, not target-unlabeled TTA.
- PAV (15 Oct) uses terminal-outcome supervision to decide how much of an action chunk to execute. It reinforces that action-prefix validity is different from generation stability, but it does not adapt policy weights.
- **No material evidence this week conflicts with our current conclusions**: interface novelty/statistics are not yet validated as harmful-shift detectors; Clean retention remains unresolved; and no low-rank interface adapter has shown closed-loop SR gains over matched candidate aggregation or SALT-like baselines.

### Paper ledger

#### 1. SALT — Self-supervised Adaptation from Leftover Trajectories

- **Title/authors:** Adapting Vision-Language-Action Models to Unknown Visual Disruptions During Execution — Ahin Lee, Jinwoo Seo, Youngsoo Jang, Taesik Gong.
- **Date/status/ID:** first posted 2026-10-06; arXiv:2610.07946v1, cs.RO; no later version or peer-reviewed venue verified.
- **Supervision:** target-unlabeled self-supervision from the unexecuted remainder of the previous action chunk; no target demos, expert actions, disruption labels, or reward. Gate calibrated on nominal trajectories.
- **Signal/location:** current predicted chunk is aligned against the previous chunk's leftover over the shared future interval; policy parameters are updated toward that leftover, then the chunk is regenerated.
- **Regime/timing/chunk:** online within an episode, with sequential correction propagation. The key transition condition is a shift after a nominal chunk has been generated, so its leftover may retain the pre-shift plan. A shift already present at the first replan removes this anchor and is reported as a weak/no-benefit condition. Exact action horizon and execution stride were not independently extracted from the appendix in this audit; do not assume they match our H/E.
- **Benchmark/results:** LIBERO-10, five persistent visual corruptions. Authors report mean SR 43.9→53.2% (SmolVLA), 58.7→66.0% (GR00T N1.7), and real-robot task progress 0.49→0.61 over digital/physical disruptions. Abstract gives no episode counts, CI, or significance tests.
- **Compute:** update plus policy regeneration at replanning; no verified wall-clock overhead.
- **Code/reproduction:** no official code link or independent reproduction confirmed.
- **Limitations/impact:** closest to our target-unlabeled constraint, but depends on a pre-shift leftover. Our E=H setting may leave no target; a faithful baseline needs an explicit overlap/execution stride and shift timestamp. Retain SALT as a required comparator, but do not infer general robustness from the reported aggregate results. Its nominal gate does not validate our gate.

#### 2. Verify, Act, Distill (VAD)

- **Title/authors:** Verify, Act, Distill: Turning Test-Time Compute into Test-Time Adaptation for Vision-Language-Action Models — Kaneyoshi Hiratsuka, Satoshi Yamamori, Jun Morimoto.
- **Date/status/ID:** first posted 2026-10-10; updated 2026-10-14 per arXiv metadata; arXiv:2610.14885v1, cs.RO. Listed as under review at ICLR 2027 by a secondary index; acceptance not verified.
- **Supervision:** fixed verifier scores candidates; weighted aggregate is executed; observations and aggregated chunks from successful adaptation episodes are distilled. Success/failure episode filtering means this is not strictly target-unlabeled. Verifier's training provenance should be checked before calling it source-only.
- **Signal/location:** aggregated chunk becomes a teacher; standard flow-matching loss updates LoRA parameters in the flow VLA.
- **Regime/timing/chunk:** adaptation from successful episodes, then updated policy is used at inference. Secondary technical summary reports planned chunk H1=50, verifier scores leading H2=10 and executes 10; verify against primary appendix before implementation. Abstract does not define shift timing.
- **Benchmark/results:** all 40 LIBERO tasks, SmolVLA 5k checkpoint. Authors report verifier-free single-candidate SR 54.1→70.6% after 10 episodes/task; applying eight-candidate aggregation to distilled policy gives 76.4→82.3%; LIBERO-Long 44.0→57.8%. Abstract says single-candidate also improves on LIBERO-Plus shifts but gives no Plus-specific numbers. No CI/tests exposed in abstract.
- **Compute:** LoRA training and verifier/candidate costs not quantified in accessible summary; single-candidate inference avoids repeated aggregation but adaptation is not free.
- **Code/reproduction:** no official repository or independent reproduction confirmed.
- **Limitations/impact:** evidence that verifier-generated candidate aggregates can be useful distillation targets, but relies on successful episodes and a verifier. It does not show an unlabeled DINO/interface adapter beats candidate aggregation. Use only as a privileged/success-supervised reference unless exact supervision assumptions can be matched.

#### 3. COAST — Contrastive Conceptor Activation Steering

- **Title/authors:** Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States — Miranda Muqing Miao, Subin Kim, Brandon Yang, Lyle Ungar.
- **Date/status/ID:** first posted 2026-05-16; arXiv:2605.17144v1 (cs.RO/cs.AI/cs.LG). Secondary index notes submission to NeurIPS 2026; acceptance not verified.
- **Supervision:** few target-task success and failure rollouts; target outcome-labelled, not target-unlabeled.
- **Signal/location:** contrastive conceptors identify success-critical activation subspaces; inference-time steering modifies the action expert's residual/hidden activations. No persistent policy parameter update.
- **Regime/timing/chunk:** per-inference activation steering; not continual mid-episode visual-shift adaptation. Action-chunk horizon/shift timing not specified in abstract.
- **Benchmark/results:** authors claim absolute mean success gains >20 percentage points in simulation and >40 points on real robots across flow VLA, autoregressive VLA, and Diffusion Policy. Abstract does not give sample sizes, task split, uncertainty, or tests. Secondary commentary reports 3 real tasks ×15 trials and early checkpoints on two simulation benchmarks; not verified against primary appendix here.
- **Compute:** conceptor fitting plus activation steering; no verified latency figure.
- **Code/reproduction:** no official code link or independent reproduction confirmed in this audit.
- **Limitations/impact:** supports the hypothesis that action-expert activations contain task-relevant directions, but target success/failure labels violate our core protocol. Treat as privileged diagnostic/oracle reference, not deployable TTA evidence.

#### 4. HDH — Learning Low-Latency Action Verification for Frozen VLAs

- **Title/authors:** Learning Low-Latency Action Verification for Frozen Vision-Language-Action Models — Wen-Han Hsieh, Ken Jiankun Zheng, Chia-Chien Chen, David M. Chan, Trevor Darrell.
- **Date/status/ID:** first posted 2026-11-10; arXiv:2611.14834v1, cs.RO; preprint, no venue verified.
- **Supervision:** offline compact verifier trained from the frozen policy's autonomous rollouts labelled by episode success/failure. Target-task outcome supervision, not unlabeled TTA.
- **Signal/location:** verifier scores two sampled action chunks; frozen VLA remains unchanged.
- **Regime/timing/chunk:** verifier trained offline, applied at each query; project page says execute 8 actions from the higher-scoring chunk. No mid-episode visual-shift experiment.
- **Benchmark/results:** four LIBERO-10 tasks; project page says 50 held-out states × 3 runs/task. Mean SR 78.8% baseline vs 82.3% HDH. Per-task HDH: 76.7±2.5, 97.3±2.5, 72.0±5.9, 83.3±5.0; baseline: 73.3±2.5, 96.0±0.0, 66.0±2.8, 80.0±4.3. ± is SD over three runs, not CI; no significance test shown.
- **Compute:** median/p95 query latency 718/796 ms vs 681/760 ms baseline (+5.4% median).
- **Code/reproduction:** official project page links simulation and real-robot code; author release, not independent reproduction.
- **Limitations/impact:** four tasks and three runs/task in the displayed table; verifier requires labelled rollouts and is not a shift-adaptation method. Add as candidate-verification reference; compare parameter updates against frozen verifier/aggregation controls.

#### 5. PAV — Policy-Aligned Predictive Verification

- **Title/authors:** Policy-Aligned Predictive Verification for Adaptive VLA Action-Chunk Execution — Minkyu Kim, Junhoo Lee, Seungyeon Kim, Suhyun Jeon, Baekseung Kim, Donghoon Han, Nojun Kwak.
- **Date/status/ID:** first posted 2026-10-15; arXiv:2610.21434v1, cs.RO/cs.LG; preprint, no venue acceptance verified.
- **Supervision:** trajectory-level survival labels from terminal task outcomes plus localization prior preserving useful prefixes in failed rollouts. Outcome-supervised verifier, not target-unlabeled.
- **Signal/location:** predicts whether a candidate prefix remains compatible with successful continuation in the VLA's task-conditioned representation; selects execution horizon. VLA weights unchanged.
- **Regime/timing/chunk:** offline verifier training, per-query adaptive execution. Secondary appendix extraction reports LIBERO chunk H=50, prediction stride 5, candidate execution horizons {5,10,...,50}; verify primary PDF before implementing. Not a shift-onset adaptation method.
- **Benchmark/results:** pi_0/pi_0.5 on LIBERO, RoboTwin 2.0 and real manipulation; abstract reports improved success-query trade-off but no numeric table, sample size, CI, or tests. A secondary extraction reports 6,000 LIBERO rollouts (4,200/840/960 split, one seed), not independently checked against primary PDF.
- **Compute:** designed to reduce policy queries via adaptive execution; no verified latency figure.
- **Code/reproduction:** no confirmed official code or independent reproduction found.
- **Limitations/impact:** action-prefix validity is not generation stability, but PAV needs terminal outcome supervision and does not adapt visual features. It makes H/E and execution horizon explicit requirements for any leftover-target experiment.

#### 6. TT-VLA — On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning

- **Title/authors:** On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning — Changyu Liu, Yiyang Liu, Taowen Wang, Qiao Zhuang, James Chenhao Liang, Wenhao Yang, Renjing Xu, Qifan Wang, Dongfang Liu, Cheng Han.
- **Date/status/ID:** first posted 2026-01-11; published ACL 2026, 2026.acl-long.1863.
- **Supervision:** dense step-by-step task-progress reward/weak labels at test time; not target-unlabeled.
- **Signal/location:** progress between adjacent observations drives a test-time RL objective that updates the VLA action policy while retaining pretrained SFT/RL priors.
- **Regime/timing/chunk:** online during deployment; chunk horizon and exact shift timing not stated in ACL abstract.
- **Benchmark/results:** simulated and real-world dynamic unseen scenarios claimed; abstract provides no numerical SR, sample size, or uncertainty.
- **Compute:** online policy optimization adds cost; abstract has no latency/compute accounting.
- **Code/reproduction:** GitHub search surfaced author repository https://github.com/ChangyuLiu2022/TT-VLA; current reproducibility was not audited; no independent reproduction confirmed.
- **Limitations/impact:** reward/progress availability separates it from our no-target-reward protocol. Keep as reward-assisted reference only; disclose whether progress is simulator-derived.

### Cross-paper assessment against project claims

| Question | Evidence this week | Not established |
|---|---|---|
| Do interface statistics identify harmful shifts? | No new paper validates a target-unlabeled interface-statistics gate with Clean false-update/skip accounting. SALT uses leftover/action discrepancy plus nominal calibration, a different signal. | Our residual novelty/channel statistics separate style from task content. |
| Can action-sensitive changes encode useful compensation? | VAD candidate distillation, COAST outcome-labelled activation subspaces, and HDH/PAV action verification show action-dependent signals can help. | An unlabeled shallow DINO/interface adapter infers useful compensation or improves closed-loop SR. |
| Do leftovers require pre-shift nominal chunks? | SALT explicitly relies on a leftover that can preserve a pre-shift plan; no independent reproduction found. | SALT works when shift exists at first observation/replan, or when E=H leaves no target. |
| Do Clean gates really skip updates? | SALT claims nominal-calibrated gating; no independent false-update audit located. | Our gate preserves Clean success. |
| Does low-rank adaptation beat candidate aggregation/SALT? | VAD reports LoRA distillation gains over single-candidate behavior and additional gains from re-aggregation. | Fair superiority over candidate aggregation/SALT under our CoRP location, target-unlabeled constraint, paired episodes and sealed LIBERO-Plus test. |

### Decision rule

Do not change the main experimental plan based on these papers alone. Before promoting a method to the main SR table require: (1) protocol-matched multi-domain LIBERO-Plus shifts and sealed target-test; (2) explicit H, E, overlap, and shift timestamp relative to chunk generation; (3) paired Clean episodes with update/skip rates and Clean SR non-inferiority; (4) frozen, candidate-only aggregation, SALT-like, and adapter comparisons on identical states/seeds; (5) initial-state-clustered uncertainty and update/inference overhead; and (6) an independent rerun or reproducible public implementation.

### Sources

- [SALT — arXiv:2610.07946](https://arxiv.org/abs/2610.07946)
- [Verify, Act, Distill — arXiv:2610.14885](https://arxiv.org/abs/2610.14885)
- [COAST — arXiv:2605.17144](https://arxiv.org/abs/2605.17144)
- [HDH — arXiv:2611.14834](https://arxiv.org/abs/2611.14834) · [project page / code links](https://wen-hanshieh.github.io/hdh/)
- [PAV — arXiv:2610.21434](https://arxiv.org/abs/2610.21434)
- [TT-VLA — ACL Anthology 2026.acl-long.1863](https://aclanthology.org/2026.acl-long.1863/) · [author repository](https://github.com/ChangyuLiu2022/TT-VLA)
- [VAD secondary chunk/setup summary](https://takahiromiki.com/arxiv-digest/paper.html?id=2610.14885)
- [PAV secondary appendix extraction; verify against primary PDF](https://www.paperpeel.com/paper/2610.21434)
