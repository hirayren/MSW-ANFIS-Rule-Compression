import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from model import SugenoANFIS

def test_finite_predictions():
    rng=np.random.default_rng(1);X=rng.random((24,4));y=X[:,0]*3-X[:,1]
    for k in (2,3):
        m=SugenoANFIS(k,max_iter=1).fit(X,y)
        assert np.isfinite(m.predict(X)).all()
        assert m.rule_count==k**4
