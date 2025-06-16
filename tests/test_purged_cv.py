import numpy as np
from src.utils.cv import PurgedWalkForwardCV


def test_purged_cv_no_leak():
    n = 10000
    cv = PurgedWalkForwardCV(n_splits=5, test_size=1000, gap=200)
    data = np.arange(n)
    prev_test = set()
    for train_idx, test_idx in cv.split(data):
        assert len(set(train_idx) & set(test_idx)) == 0
        if len(train_idx):
            assert train_idx[-1] <= test_idx[0] - 200 - 1
        assert not prev_test.intersection(test_idx)
        prev_test.update(test_idx)
