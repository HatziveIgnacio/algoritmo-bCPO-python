import numpy as np
import random

class USCP:
    def __init__(self, file_path):
        # Leer todos los enteros del archivo
        with open(file_path, 'r') as f:
            data = f.read().split()
            
        self.m = int(data[0]) # Cantidad de filas
        self.n = int(data[1]) # Cantidad de columnas
        
        # Ignoramos costos individuales porque es Unicost (todos valen 1)
        idx = 2 + self.n
        
        self.rows = []
        for _ in range(self.m):
            num_cols = int(data[idx])
            idx += 1
            # Convertimos a 0-indexed (python usa índices desde 0)
            cols_for_row = [int(data[i]) - 1 for i in range(idx, idx + num_cols)]
            self.rows.append(cols_for_row)
            idx += num_cols
            
        # Matriz de cobertura para cálculos matriciales rápidos
        self.coverage_matrix = np.zeros((self.m, self.n), dtype=bool)
        for i, row_cols in enumerate(self.rows):
            self.coverage_matrix[i, row_cols] = True

    def repair_and_evaluate(self, x):
        """
        Calcula el fitness aplicando un mecanismo de reparación (Repair Mechanism).
        Como la matriz x es de numpy, modificarla aquí se reflejará en la población
        del algoritmo principal (Lamarckian learning).
        """
        # Contamos cuántas columnas cubren cada fila
        coverage_counts = self.coverage_matrix.dot(x)
        
        # Buscamos qué filas quedaron con 0 coberturas
        uncovered_indices = np.where(coverage_counts == 0)[0]
        
        # Reparación: Por cada fila descubierta, forzamos a 1 una columna aleatoria
        for row_idx in uncovered_indices:
            # Columnas que pueden cubrir esta fila específica
            possible_cols = self.rows[row_idx]
            chosen_col = random.choice(possible_cols)
            
            # Encendemos la columna
            x[chosen_col] = 1
            
            # Nota: Esto podría cubrir otras filas, pero es un mecanismo
            # simple y rápido para asegurar factibilidad estricta.
            
        # El costo (fitness) en un Unicost es la suma de los bits encendidos
        return np.sum(x)
