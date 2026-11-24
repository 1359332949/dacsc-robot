import unittest
from dacsc.source_provenance import FIELDS,require_same_collection


def fixture():
    m={key:'fixed' for key in FIELDS}
    for key in ('target_environment_accessed','weights_updated','chunk_risk_labels_assigned'):m[key]=False
    m['upstream_sha256']={'policy.py':'same'}
    return m

class ProvenanceTests(unittest.TestCase):
    def test_shard_names_and_state_ids_are_not_protocol(self):
        a=fixture();signature=require_same_collection(a)
        a.update(state_ids=[0],state_id=0,role='source_fit',directory='fit_smoke')
        self.assertEqual(require_same_collection(a,signature),signature)
    def test_each_protocol_field_mismatch_rejected(self):
        a=fixture();signature=require_same_collection(a)
        for key in FIELDS:
            b=dict(a);b[key]={'different':'value'} if key=='upstream_sha256' else 'changed'
            with self.subTest(key=key),self.assertRaises(ValueError):require_same_collection(b,signature)
    def test_missing_is_not_consistent(self):
        a=fixture();del a['script_sha256']
        with self.assertRaises(ValueError):require_same_collection(a)
if __name__=='__main__':unittest.main()
