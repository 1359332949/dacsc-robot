import numpy as np
from scripts.consensus_eval import pair_dist

def test_pair_dist_equal_weight():
    a=np.zeros((12,7)); b=np.zeros((12,7)); b[0,0]=0.05; b[0,3]=0.05; b[0,6]=0.05
    assert abs(pair_dist(a,b)-np.sqrt((3*0.05**2)/84))<1e-12
