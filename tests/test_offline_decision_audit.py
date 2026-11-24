import random
import unittest
from scripts.offline_decision_audit import compute_wrist_late, validate_alignment


def make_rows():
    rows=[]
    for si,state in enumerate((18,20)):
        for c in range(3):
            rows.append({'role':'source_validation','state_id':state,'chunk_index':c,'window':'late',
                         'pairing_pass':True,
                         'attention_metrics':{'eye_in_hand':{'js_mean':0.1*si+0.03*c}},
                         'action_exec':{'norm':0.2*si+0.01*c,'xyz':0.3*si+0.02*c,'rot':0.1*si+0.005*c,'grip':0.01*si},
                         '_clean_action_norm':0.5+0.1*si})
    return rows


class TestOfflineDecisionAudit(unittest.TestCase):
    def test_input_order_does_not_change_statistics(self):
        rows=make_rows(); a=compute_wrist_late(rows)
        shuffled=list(rows); random.Random(7).shuffle(shuffled); b=compute_wrist_late(shuffled)
        self.assertEqual(a, b)

    def test_negative_feature_row_rejected(self):
        rows=make_rows(); records=[{'feature_row':0,'role':'source_validation','state_id':18,'chunk_index':0,'pairing_pass':True}]
        rows[0]['feature_row']=-1
        with self.assertRaises(RuntimeError):
            validate_alignment(rows, records)

    def test_duplicate_attention_row_rejected(self):
        rows=make_rows(); records=[{'feature_row':i,'role':'source_validation','state_id':18,'chunk_index':i,'pairing_pass':True} for i in range(len(rows))]
        for i,r in enumerate(rows): r['feature_row']=i
        rows[1]['feature_row']=rows[0]['feature_row']
        with self.assertRaises(RuntimeError):
            validate_alignment(rows, records)

    def test_failed_pair_rejected(self):
        rows=make_rows(); rows[0]['pairing_pass']=False
        with self.assertRaises(RuntimeError):
            compute_wrist_late(rows)


if __name__=='__main__':
    unittest.main()
