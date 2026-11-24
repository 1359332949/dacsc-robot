# Flow consensus closed-loop, t=0-conditional (dev only)

Clean 10: Frozen 10/10, Medoid 10/10, Trimmed-Mean 10/10, z0=0 10/10.

| Factor | Frozen | Medoid | Trimmed | z0=0 |
|---|---:|---:|---:|---:|
| Sensor Noise (30) | 17 | 18 | 21 | 20 |
| Background (30) | 17 | 14 | 18 | 20 |
| Lighting (30) | 24 | 25 | 24 | 24 |

Paired transitions vs Frozen (rescued / harmed / net):
- Sensor: Medoid 1/0/+1; Trimmed 5/1/+4; z0=0 5/2/+3.
- Background: Medoid 2/5/-3; Trimmed 4/3/+1; z0=0 3/0/+3.
- Lighting: Medoid 2/1/+1; Trimmed 2/2/0; z0=0 1/1/0.

Interpretation: z0=0 is the only operator with positive net on both hard factors
and zero Clean loss; Trimmed is strong on Sensor but mixed on Background;
Medoid hurts Background. No target-test; no Adapter.
