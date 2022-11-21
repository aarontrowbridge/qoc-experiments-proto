import numpy as np

def test(us):
    return np.array([np.sum(u) for u in us])

