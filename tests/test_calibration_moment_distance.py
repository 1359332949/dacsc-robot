"""An analytic counterexample for the shared-reference threshold builder."""
import ast
from pathlib import Path
import unittest
import numpy as np
import torch

class SharedCalibrationDistance(unittest.TestCase):
    def test_mean_shift_is_squared_once(self):
        path=Path(__file__).resolve().parents[1]/"scripts/aggregate_calib.py"
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="loto_tau")
        examples=[]
        for mean in (0.,2.):
            m=np.full((1,2,3),mean,dtype=np.float32)
            examples.append({"b3_m":m,"b3_q":m*m+1.})
        ns={"torch":torch,"tasks":examples}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),ns)
        actual=ns["loto_tau"]("b3")
        torch.testing.assert_close(actual,torch.full((2,3),4.))

if __name__=="__main__":unittest.main()
