import numpy as np

def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = 1
    if isinstance(ub, (list, np.ndarray)) and isinstance(lb, (list, np.ndarray)):
        if len(ub) > 1:
            Boundary_no = len(ub)
            
    # If the boundaries of all variables are equal and user enter a single
    # number for both ub and lb
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    # If each variable has a different lb and ub
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
            
    return Positions
