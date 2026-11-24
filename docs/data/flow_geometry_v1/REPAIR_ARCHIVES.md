# Repair probe raw archives

Raw results are compressed to avoid committing roughly 175 MB of pretty-printed duplicate arrays. No model/checkpoint or credentials are included.

- `repair_token_raw.tar.gz`: original token-placement 10-case grid (`repair_raw/`), fixed token-scaling controls (`view_control_raw/`), and mean-L2 preflight evidence/source (`repair_preflight/`).
- `repair_value_raw.tar.gz`: corrected post-LayerNorm Value-placement grid (`repair_kv_raw/`) and fixed Value controls (`value_control_raw/`).
- `repair_archive_hashes.json`: archive SHA-256 and bytes.

From repository root, unpack with:

```sh
tar -xzf docs/data/flow_geometry_v1/repair_token_raw.tar.gz -C docs/data/flow_geometry_v1
tar -xzf docs/data/flow_geometry_v1/repair_value_raw.tar.gz -C docs/data/flow_geometry_v1
python scripts/verify_flow_causal_repair.py --results docs/data/flow_geometry_v1/repair_kv_raw --selection docs/data/flow_geometry_v1/repair_selection.json --out /tmp/repair_verification.json
python scripts/analyze_flow_causal_repair.py --results docs/data/flow_geometry_v1/repair_kv_raw --selection docs/data/flow_geometry_v1/repair_selection.json --out /tmp/causal_repair_probe.json
```

The verification checks selected-case coverage, all 18 fixed configurations, noise disjointness, per-case end receipts, native/zero-gate parity, parameter bounds and matched-null norms. The original token-placement directory can be checked with the same command. Additional scripts reproduce the fixed controls, claim audit and figure. `repair_code_hashes.json` binds the saved sources. See FLOW_CAUSAL_REPAIR_RESULTS.md for interpretation; data integrity is not task-repair validation.
