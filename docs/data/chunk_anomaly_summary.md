# Frozen chunk-boundary anomaly, 30 cases/factor

Frozen replay success reproduces recorded dev labels exactly:
Sensor 17/30, Background 17/30, Lighting 24/30.

Features per episode (chunk state8 EEF displacement, command XYZ, efficiency,
direction reversals). Failure-vs-success AUROC at t>=12:

| feature | Sensor | Background | Lighting |
|---|---:|---:|---:|
| final chunk displacement | .086 | .109 | .014 |
| mean chunk displacement | .018 | .000 | .000 |
| min chunk displacement | .199 | .131 | .056 |
| late-3 min displacement | .104 | .072 | .000 |
| mean actual/command efficiency | .394 | .511 | .424 |
| min efficiency | .294 | .312 | .167 |
| command direction reversals | .774 | .943 | 1.000 |
| mean command magnitude | .041 | .000 | .000 |

Interpretation: simple "stuck / low displacement / low efficiency" features do
NOT identify failures (AUC near or below chance). Direction-reversal count is
the only feature with failure>success ordering, but it is a post-hoc aggregate
and not yet a deployment signal. No target-test.
