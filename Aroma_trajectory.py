import numpy as np

def Aroma_trajectory(N, Dc):
    dt = 1 / N 
    
    # Posición inicial
    x0 = 0
    y0 = 0
    z0 = 0
    
    # Generando tamaños de paso aleatorios
    dWx = np.sqrt(2 * Dc * dt) * np.random.randn(N)
    dWy = np.sqrt(2 * Dc * dt) * np.random.randn(N)
    dWz = np.sqrt(2 * Dc * dt) * np.random.randn(N)
    
    # Calculando la trayectoria del aroma
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    x[0] = x0
    y[0] = y0
    z[0] = z0
    for k in range(1, N):
        x[k] = x[k-1] + dWx[k]
        y[k] = y[k-1] + dWy[k]
        z[k] = z[k-1] + dWz[k]
        
    random_index = np.random.randint(N)
    random_point_x = x[random_index]
    random_point_y = y[random_index]
    random_point_z = z[random_index]
    
    Bs = np.linalg.norm([random_point_x, random_point_y, random_point_z])
    return Bs
