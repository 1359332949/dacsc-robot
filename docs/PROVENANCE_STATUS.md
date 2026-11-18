# Provenance status (2026-11-18)

## Compliant run

`docs/data/perturb_dev_v2.json` (Frozen 11/30, First 7/30):

- git commit: `913290c1e47afbb602ed712533d53efb757a39e5`
- split SHA256: `0141c2e8...`
- capture manifest SHA256: `d66b26a4...`
- seed pool: screening_seeds (7777/8888/9999)
- sampler: sample_z; precision: fp32
- per-record z0 schedule hash and hit_max_steps are present
- source capture manifest has `source_role_fingerprint`

## Legacy / pre-fix results

`perturb_test_clean.json` and `perturb_test_shift.json` (Frozen admission on
Lighting 2312) were produced before the provenance fixes:

- no `git_commit` or `run_config`;
- test split was a temporary eval-seed variant (`a568c776...`);
- capture manifest had no source-role fingerprint.

They remain useful as exploratory admission probes, but are not confirmatory
and must not be mixed with protocol-compliant results.

## Immutability

`tta_split_libero_spatial_perturbation.json` is restored to the original
`0141c2e8...` content used by the compliant run. Target-test access state is
kept in `tta_access_ledger.json`, not in the split file, so split hashes do not
change when access changes.
