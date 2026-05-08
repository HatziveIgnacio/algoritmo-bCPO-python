import numpy as np
from math import gamma, sin, pi

def Levy(dim):
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.randn(1, dim) * sigma
    v = np.random.randn(1, dim)
    step = u / (np.abs(v)**(1 / beta))
    return step.flatten()
