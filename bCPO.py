import numpy as np
from initialization import initialization
from Aroma_concentration import Aroma_concentration
from Aroma_trajectory import Aroma_trajectory
from Levy import Levy

def bCPO(SearchAgents_no, Max_iter, dim, fobj):
    # En problemas binarios, los límites siempre son 0 y 1
    lb = 0
    ub = 1
    
    Manis_pos = np.zeros(dim)
    Manis_score = float('inf')
    
    Ant_pos = np.zeros(dim)
    Ant_score = float('inf')
    
    # Inicializar las posiciones de los agentes de búsqueda (usaremos ceros y unos aleatorios)
    # Reutilizamos la función de inicialización y luego forzamos a binario
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = np.round(Positions) # Aseguramos que la población inicial sea puramente binaria
    
    Convergence_curve = np.zeros(Max_iter)
    Div_curve = np.zeros(Max_iter)

    t = 0
    fitness = np.zeros(SearchAgents_no)
    
    while t < Max_iter:
        for i in range(SearchAgents_no):
            # Calcular la función objetivo para cada pangolín
            fitness[i] = fobj(Positions[i, :])
            
            # Actualizar la ubicación del pangolín (Manis pentadactyla)
            if fitness[i] <= Manis_score:
                Manis_score = fitness[i]
                Manis_pos = Positions[i, :].copy()
                
            if fitness[i] > Manis_score and fitness[i] < Ant_score:
                Ant_score = fitness[i]
                Ant_pos = Positions[i, :].copy()
                
        r1 = (np.random.rand() + np.random.rand()) / 2
        r2 = np.random.rand()
        
        # Factor de concentración de aroma
        Cm = Aroma_concentration(Max_iter)
        
        # Factor de disminución rápida
        C1 = (2 - ((t * 2) / Max_iter))
        
        # Factor de trayectoria del aroma
        a = Aroma_trajectory(SearchAgents_no, 0.6)
        
        # Longitud del paso de Lévy
        Levy_Step_length = Levy(SearchAgents_no)
        
        for i in range(SearchAgents_no):
            # Factor de corrección de energía
            lamda = 0.1 * np.random.rand()
            VO2 = 0.2 * np.random.rand()
            
            # Factor del índice de fatiga
            Fatigue = np.log(((t * np.pi) / Max_iter) + 1)
            
            # Factor de consumo de energía
            E = np.exp(-lamda * VO2 * t * (1 + Fatigue))
            l = np.random.randint(0, Max_iter)
            r3 = np.random.rand()
            
            # Factor de fluctuación de energía
            A1 = lamda * (2 * E * np.random.rand() - E)
            
            # --- Variables temporales para la posición continua ---
            new_position_continuous = np.zeros(dim)
            
            ## Comportamiento de Atracción (Luring)
            if Cm[l] >= 0.2 and r3 <= 0.5:
                ## Etapa de Atracción y Captura
                D_ant = np.abs(a * Ant_pos - Manis_pos)
                New_Ant_pos = Positions[i, :] + Ant_pos - A1 * D_ant
                
                ## Etapa de Movimiento y Alimentación
                D_manis = np.abs(C1 * New_Ant_pos - Positions[i, :]) - Levy_Step_length[i] * (1 - t / Max_iter)
                New_Manis_pos = Positions[i, :] + Manis_pos - A1 * D_manis
                
                ## Se actualizan las posiciones (Posición Continua)
                den = (4 * np.pi) * np.tan(New_Manis_pos * np.exp((t * 4 * np.pi**2) / Max_iter))
                den[den == 0] = np.finfo(float).eps
                
                new_position_continuous = (New_Manis_pos + New_Ant_pos) / 2 + \
                    (np.sin(New_Ant_pos * np.exp(t / Max_iter)) / den) * \
                    r1 * r2 * np.random.rand()
                
            ## Comportamiento de Depredación (Predation)
            elif Cm[l] <= 0.7 or r3 > 0.5:
                ## Etapa de Búsqueda y Localización
                if Cm[l] >= 0 and Cm[l] < 0.3:
                    D_manis = np.abs(Levy_Step_length[i] * Manis_pos - Positions[i, :])
                    New_Manis_pos = np.sin(C1 * Positions[i, :] + A1 * np.abs(Manis_pos - Levy_Step_length[i] * D_manis))
                    new_position_continuous = New_Manis_pos * C1
                ## Etapa de Acercamiento Rápido
                elif Cm[l] >= 0.3 and Cm[l] < 0.6:
                    D_manis = np.abs(a * Manis_pos - Positions[i, :])
                    New_Manis_pos = Positions[i, :] - A1 * np.abs(Manis_pos - np.exp(-a) * (np.random.rand() * np.pi) * D_manis)
                    new_position_continuous = New_Manis_pos * C1
                ## Etapa de Excavación y Alimentación
                elif Cm[l] >= 0.6:
                    D_manis = np.abs(C1 * Manis_pos - Positions[i, :])
                    New_Manis_pos = Positions[i, :] + A1 * np.abs(Manis_pos - D_manis)
                    new_position_continuous = New_Manis_pos * C1
            
            # =========================================================
            # PASO 1: Función de Transferencia (Sigmoide / S-shaped)
            # T(x) = 1 / (1 + exp(-x))
            # =========================================================
            # Usamos clip para evitar overflow en exp
            clipped_pos = np.clip(-new_position_continuous, -709, 709)
            T_x = 1 / (1 + np.exp(clipped_pos))
            
            # =========================================================
            # PASO 2: Regla de Binarización
            # X = 1 si rand < T(x), sino 0
            # =========================================================
            rand_values = np.random.rand(dim)
            Positions[i, :] = (rand_values < T_x).astype(int)
                    
        # -------------------------------------------------------------------------------------
        if t % 100 == 0:
            print(f'At iteration {t} the best solution fitness is {Manis_score}')
            
        Convergence_curve[t] = Manis_score
        
        # Calcular diversidad para Exploración vs Explotación
        median_pos = np.median(Positions, axis=0)
        div = np.mean(np.abs(Positions - median_pos))
        Div_curve[t] = div
        
        t += 1
        
    # Calcular porcentajes de Exploración y Explotación
    Div_max = np.max(Div_curve)
    if Div_max == 0:
        Exploration = np.zeros(Max_iter)
        Exploitation = np.ones(Max_iter) * 100
    else:
        Exploration = (Div_curve / Div_max) * 100
        Exploitation = 100 - Exploration
        
    return Manis_score, Manis_pos, Convergence_curve, Exploration, Exploitation
