import numpy as np

def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = 1
    if isinstance(ub, (list, np.ndarray)) and isinstance(lb, (list, np.ndarray)):
        if len(ub) > 1:
            Boundary_no = len(ub)
            
    # Si los límites de todas las variables son iguales y el usuario ingresa un solo
    # número para ub y lb
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    # Si cada variable tiene un lb y ub diferente
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
            
    return Positions
