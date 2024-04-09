import numpy as np


def select(p):
    C = np.cumsum(p)  # Cumulative sum of probabilities
    r = np.random.rand()  # Random number in the range [0, 1)
    i = np.where(r <= C)[0][0]  # Index of the first element in C that is >= r
    return i
