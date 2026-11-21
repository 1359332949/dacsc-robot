"""Strict, model-independent contracts for paired chunk transition records."""
from __future__ import annotations
import hashlib
import numpy as np

VIEWS = ('agentview', 'eye_in_hand')
HORIZON = 12
ACTION_DIM = 7


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def step_flags(result, env):
    """Preserve legacy done separately; never infer task success from timeout."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        source = 'five_value_api'
    elif len(result) == 4:
        obs, reward, done, info = result
        truncated = bool(info.get('TimeLimit.truncated', False))
        terminated = bool(done and not truncated)
        source = 'legacy_done_minus_explicit_time_limit'
    else:
        raise ValueError('unsupported step API')
    if 'success' in info:
        success = bool(info['success'])
    elif callable(getattr(env, 'check_success', None)):
        success = bool(env.check_success())
    else:
        raise ValueError('success cannot be inferred from done/reward')
    return obs, dict(reward=float(reward), success=success, terminated=bool(terminated),
                     truncated=bool(truncated), done=bool(done), flags_source=source)


def validate_episode(a, metadata):
    """Raise on misaligned times/actions/features, including truncated tail chunks."""
    count = len(a['executed_steps'])
    if count == 0: raise ValueError('no transitions')
    n = np.asarray(a['executed_steps'])
    if not np.issubdtype(n.dtype, np.integer) or np.any((n < 1) | (n > HORIZON)):
        raise ValueError('invalid executed_steps')
    expected = np.arange(HORIZON)[None, :] < n[:, None]
    if a['executed_mask'].shape != (count, HORIZON) or not np.array_equal(a['executed_mask'], expected):
        raise ValueError('execution mask is not contiguous prefix')
    if a['actions_executed'].shape != (count,HORIZON,ACTION_DIM): raise ValueError('action shape')
    if np.any(a['actions_executed'][~expected] != 0): raise ValueError('nonzero padded actions')
    if np.any(np.abs(a['actions_executed']) > 1): raise ValueError('actions are not clipped')
    if not np.array_equal(a['actions_executed'][expected], np.clip(a['actions_planned'], -1, 1)[expected]):
        raise ValueError('executed commands differ from clipped plan')
    if not np.array_equal(a['boundary_step'], np.r_[0, np.cumsum(n)]): raise ValueError('boundary time mismatch')
    if a['boundary_sim_time'].shape != (count+1,): raise ValueError('simulation time length')
    if not np.allclose(np.diff(a['boundary_sim_time']), n * metadata['control_dt'], atol=1e-8, rtol=0):
        raise ValueError('simulation elapsed time mismatch')
    for key in ('state8','probe_anchor_state8'):
        if a[key].shape != (count+1,8): raise ValueError(key+' shape')
    if not np.array_equal(a['probe_anchor_state8'], np.repeat(a['probe_anchor_state8'][:1],count+1,axis=0)):
        raise ValueError('probe state anchor changed')
    if not np.array_equal(a['full_chunk'], n == HORIZON): raise ValueError('full_chunk flag')
    if not np.array_equal(a['fixed_horizon_eligible'], n == HORIZON): raise ValueError('partial chunk eligibility')
    for view in VIEWS:
        if a['tokens_'+view].shape != (count+1,48,768): raise ValueError(view+' token shape')
    if a['probe_hidden'].shape != (count+1,768): raise ValueError('hidden shape')
    if a['probe_velocity'].shape != (count+1,HORIZON,ACTION_DIM): raise ValueError('velocity shape')
    if a['x_probe'].shape != (1,HORIZON,ACTION_DIM): raise ValueError('x_probe shape')
    if metadata['x_probe_sha256'] != array_hash(a['x_probe']): raise ValueError('probe hash mismatch')
    for key in ('success','terminated','truncated','done'):
        if a[key].shape != (count,HORIZON): raise ValueError(key+' shape')
        if a[key][~expected].any(): raise ValueError(key+' on unexecuted step')
    for i in range(count):
        stop = a['success'][i] | a['terminated'][i] | a['truncated'][i] | a['done'][i]
        hits = np.flatnonzero(stop & expected[i])
        if hits.size and (hits[0] != n[i]-1 or i != count-1):
            raise ValueError('continued execution after terminal event')
    for key, value in a.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all(): raise ValueError('nonfinite '+key)
    return {'pass': True, 'transitions': count, 'full_chunks': int((n == HORIZON).sum()),
            'partial_chunks': int((n < HORIZON).sum()), 'executed_steps': int(n.sum())}


def validate_pair(clean, shift, cm, sm, atol=1e-6):
    cr, sr = validate_episode(clean,cm), validate_episode(shift,sm)
    for key in ('state_id','seed','x_probe_sha256','probe_tau','probe_action_index','language','initial_state_sha256'):
        if cm[key] != sm[key]: raise ValueError('pair metadata mismatch '+key)
    for key in ('actions_executed','executed_steps','executed_mask','boundary_step','x_probe','probe_anchor_state8'):
        if not np.array_equal(clean[key],shift[key]): raise ValueError('pair mismatch '+key)
    if not np.allclose(clean['state8'], shift['state8'], atol=atol, rtol=0): raise ValueError('paired state mismatch')
    if not np.allclose(clean['boundary_sim_time'],shift['boundary_sim_time'],atol=1e-9,rtol=0): raise ValueError('paired time mismatch')
    return {'pass': True, 'clean':cr, 'shift':sr, 'state8_max_abs':float(np.max(np.abs(clean['state8']-shift['state8'])))}
