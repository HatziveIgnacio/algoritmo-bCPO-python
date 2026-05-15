import numpy as np
from math import pi, exp, sin, log

def Aroma_concentration(Max_iter):
    Q = 100
    M = np.zeros(Max_iter)
    sigma_y = np.zeros(Max_iter)
    sigma_z = np.zeros(Max_iter)
    
    for t in range(Max_iter):
        r1 = np.random.rand()
        H = 0.5 * r1
        r2 = np.random.rand()
        u = 2 + r2
        # Usar t+1 porque las fórmulas matemáticas usualmente asumen pasos desde 1
        t_step = t + 1
        
        sigma_y[t] = 50 - ((10 * t_step) / Max_iter)
        sigma_z[t] = sin((pi * t_step) / Max_iter) + 40 * exp(-t_step / Max_iter) - 10 * log((pi * t_step) / Max_iter)
        M[t] = (Q / (pi * u * sigma_y[t] * sigma_z[t])) * exp(-(H**2) / (2 * (sigma_z[t])**2))
        
    # Reescalar M entre 0 y 1
    m_min = np.min(M)
    m_max = np.max(M)
    if m_max > m_min:
        Cm = (M - m_min) / (m_max - m_min)
    else:
        Cm = np.zeros(Max_iter)
        
    return Cm
