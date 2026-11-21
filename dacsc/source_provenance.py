"""Fail closed on mixed source collection provenance; ignore shard display names."""
import hashlib
import json

FIELDS=('schema','domain','data_origin','seed','H','E','max_steps','split_sha256',
        'checkpoint_sha256','base_bddl_sha256','initial_state_file_sha256','matrix_sha256',
        'script_sha256','source_contract_sha256','transition_contract_sha256','shared_inference_sha256',
        'renderer','precision','torch_version','numpy_version','target_environment_accessed',
        'weights_updated','chunk_risk_labels_assigned','observation_semantics','upstream_sha256',
        'control_dt','language')


def collection_signature(metadata):
    missing=[key for key in FIELDS if key not in metadata or metadata[key] is None]
    if missing:raise ValueError('missing collection provenance: '+', '.join(missing))
    for key in ('target_environment_accessed','weights_updated','chunk_risk_labels_assigned'):
        if metadata[key] is not False:raise ValueError('source protocol violation: '+key)
    if not metadata['upstream_sha256']:raise ValueError('empty upstream fingerprints')
    payload={key:metadata[key] for key in FIELDS}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def require_same_collection(metadata,expected=None):
    signature=collection_signature(metadata)
    if expected is not None and signature!=expected:raise ValueError('mixed collection provenance')
    return signature
