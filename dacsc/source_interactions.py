"""Source-only role and transition validation. No target data can enter here."""
from __future__ import annotations
import numpy as np
from .interaction_records import validate_episode

ROLE_KEYS={'source_fit':'source_train_state_ids','source_validation':'source_val_state_ids'}


def role_ids(split, role, requested=None):
    if role not in ROLE_KEYS:raise ValueError('source-only role required')
    fit=set(split['source_train_state_ids']);val=set(split['source_val_state_ids'])
    target=set(split['target_dev_state_ids'])|set(split['target_test_state_ids'])
    if fit&val or (fit|val)&target:raise ValueError('source/target role overlap')
    allowed=split[ROLE_KEYS[role]]
    ids=list(allowed if requested is None else requested)
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(allowed):raise ValueError('states outside declared source role')
    return ids


def validate_source_episode(arrays,metadata,split):
    role_ids(split,metadata['role'],[metadata['state_id']])
    if metadata['domain']!='clean' or metadata['data_origin']!='independent_source_policy_rollout':raise ValueError('not independent Source Clean')
    if metadata['seed'] not in split['evaluation_seeds']:raise ValueError('source seed outside frozen evaluation pool')
    report=validate_episode(arrays,metadata,require_probe=False)
    n=arrays['executed_steps'];mask=arrays['executed_mask']
    if arrays['state8_executed'].shape!=(len(n),12,8):raise ValueError('executed state shape')
    if arrays['sim_time_executed'].shape!=(len(n),12):raise ValueError('executed time shape')
    if np.any(arrays['state8_executed'][~mask]!=0) or np.any(arrays['sim_time_executed'][~mask]!=0):raise ValueError('nonzero execution padding')
    for i,count in enumerate(n):
        if not np.array_equal(arrays['state8'][i+1],arrays['state8_executed'][i,count-1]):raise ValueError('next state mismatch')
        expected=arrays['boundary_sim_time'][i]+metadata['control_dt']*np.arange(1,count+1)
        if not np.allclose(arrays['sim_time_executed'][i,:count],expected,atol=1e-8,rtol=0):raise ValueError('step timestamps mismatch')
    return report
