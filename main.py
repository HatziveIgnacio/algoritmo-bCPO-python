import time
import numpy as np
import matplotlib.pyplot as plt
from bCPO import bCPO
from uscp_problem import USCP

if __name__ == '__main__':
    # 1. Instanciamos el problema USCP
    problem_file = 'uscpcyc08.txt'
    uscp = USCP(problem_file)
    
    # 2. Configuramos los parámetros del CPO
    SearchAgents_no = 30      # Número de pangolines
    Max_iteration = 500       # Número máximo de iteraciones
    dim = uscp.n              # La dimensión es igual a la cantidad de columnas (1024)
    
    print(f"Instancia USCP cargada: {uscp.m} restricciones (filas) y {uscp.n} dimensiones (columnas).")
    print("Iniciando Binary Chinese Pangolin Optimizer (bCPO) con mecanismo de reparación...")
    
    start_time = time.time()
    
    # Ejecutamos bCPO pasando la función de reparación del USCP como función objetivo
    Best_score, Best_pos, Convergence_curve, Exploration, Exploitation = bCPO(SearchAgents_no, Max_iteration, dim, uscp.repair_and_evaluate)
    
    tend = time.time() - start_time
    
    # Comprobación de sanidad: Verificamos que la mejor solución cubra el 100% de las filas
    coberturas = uscp.coverage_matrix.dot(Best_pos)
    filas_no_cubiertas = len(np.where(coberturas == 0)[0])
    
    print("-" * 50)
    print(f'Tiempo de ejecución: {tend:.4f} segundos')
    print(f'Mejor Costo (Fitness): {Best_score} columnas seleccionadas')
    print(f'Estado de Factibilidad: {"¡Factible!" if filas_no_cubiertas == 0 else f"Inválido. Faltan {filas_no_cubiertas} filas por cubrir."}')
    print("-" * 50)
    
    # Graficamos los resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Gráfico de Convergencia
    ax1.plot(Convergence_curve, color='b', linewidth=2.0, label='bCPO (USCP)')
    ax1.set_title('Curva de Convergencia')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Mejor costo obtenido (Menor es mejor)')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Gráfico de Exploración vs Explotación
    iterations = np.arange(Max_iteration)
    ax2.plot(iterations, Exploration, color='g', linewidth=2.0, label='Exploración (%)')
    ax2.plot(iterations, Exploitation, color='r', linewidth=2.0, label='Explotación (%)')
    ax2.fill_between(iterations, 0, Exploration, color='g', alpha=0.3)
    ax2.fill_between(iterations, Exploration, 100, color='r', alpha=0.3)
    ax2.set_title('Exploración vs Explotación')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Porcentaje (%)')
    ax2.set_ylim([0, 100])
    ax2.grid(True)
    ax2.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig('bCPO_USCP_results.png')
    plt.show()
