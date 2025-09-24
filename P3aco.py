"""
@author: 
    Perez Lucio Kevyn Alejandro
    Rosales Benhumea Aldo
    Villaseñor Arvizu Anael Ulaila
"""

import numpy as np
import matplotlib.pyplot as plt

class ACO:
    def __init__(self, distancias, n_hormigas, n_iteraciones, evaporacion, alpha=1, beta=5, q=1):
        self.distancias = distancias
        self.n_hormigas = n_hormigas
        self.n_iteraciones = n_iteraciones
        self.evaporacion = evaporacion
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.n_ciudades = len(distancias[0])
        self.feromonas = [np.ones((self.n_ciudades, self.n_ciudades)) / self.n_ciudades for _ in distancias]

    def selecciona_siguiente_ciudad(self, actual, no_visitadas, feromona_actual):
        if not no_visitadas:
            # Si no quedan ciudades no visitadas, retorna la ciudad actual (o una al azar si prefieres)
            return actual
        
        feromona = feromona_actual[actual][no_visitadas] ** self.alpha
        visibilidad = [1.0 / (self.distancias[actual][i] ** self.beta) if self.distancias[actual][i] > 0 else 0 for i in no_visitadas]
    
        if sum(visibilidad) == 0:
            # Si todas las visibilidades son cero, selecciona una ciudad al azar
            siguiente = np.random.choice(no_visitadas)
        else:
            probabilidades = [fer * vis for fer, vis in zip(feromona, visibilidad)]
            probabilidades /= sum(probabilidades)
            siguiente = np.random.choice(len(no_visitadas), 1, p=probabilidades)[0]
    
        return no_visitadas[siguiente]

    def construye_camino(self, inicio):
        camino = [inicio]
        no_visitadas = list(range(self.n_ciudades))
        no_visitadas.remove(inicio)
    
        actual = inicio
        while no_visitadas:
            siguiente = self.selecciona_siguiente_ciudad(actual, no_visitadas, self.feromonas[0])
            camino.append(siguiente)
            no_visitadas.remove(siguiente)
            actual = siguiente
    
        return camino

    def computa_distancia(self, camino):
        distancia = 0
        for i in range(len(camino) - 1):
            distancia += self.distancias[camino[i]][camino[i+1]]
        distancia += self.distancias[camino[-1]][camino[0]]
        return distancia

    def actualiza_feromona(self, caminos, distancias):
        for feromona, camino, distancia in zip(self.feromonas, caminos, distancias):
            for i in range(self.n_ciudades):
                for j in range(self.n_ciudades):
                    feromona[i][j] *= (1.0 - self.evaporacion)
            for i in range(self.n_ciudades - 1):
                feromona[camino[i]][camino[i+1]] += self.q / distancia
            feromona[camino[-1]][camino[0]] += self.q / distancia

    def resuelve(self):
        mejor_camino = None
        mejor_distancia = float('inf')

        for _ in range(self.n_iteraciones):
            caminos = [self.construye_camino(np.random.choice(self.n_ciudades)) for _ in range(self.n_hormigas)]
            distancias = [self.computa_distancia(camino) for camino in caminos]
            self.actualiza_feromona(caminos, distancias)
            min_dist = min(distancias)
            if min_dist < mejor_distancia:
                mejor_distancia = min_dist
                mejor_camino = caminos[distancias.index(min_dist)]

        return mejor_camino, mejor_distancia

def cargar_datos_desde_archivo(archivo):
    with open(archivo, 'r') as file:
        lines = file.read().strip().split('\n')

    # Extraer el tamaño de la matriz
    tamaño_matriz = int(lines[0])
    inicio_distancias = 2  # La matriz de distancias comienza después de dos saltos de línea

    # Extraer la matriz de distancias
    matriz_distancias = [list(map(int, linea.split())) for linea in lines[inicio_distancias:inicio_distancias + tamaño_matriz]]

    # Extraer la matriz de flujos
    matriz_flujos = [list(map(int, linea.split())) for linea in lines[inicio_distancias + tamaño_matriz + 1:]]

    return matriz_distancias, matriz_flujos

matriz_distancias, matriz_flujos = cargar_datos_desde_archivo('tai12.dat')
distancias = matriz_distancias
flujos = matriz_flujos

matriz_distancias = np.array(matriz_distancias)
matriz_flujos = np.array(matriz_flujos)

# Agregar impresiones para verificar los datos cargados
print("Matriz de distancias:")
print(distancias)
print("Matriz de flujos:")
print(flujos)


def calcular_amplitud(matriz_distancias, matriz_flujos, permutacion):
    n = len(permutacion)
    amplitud = 0
    for i in range(n):
        for j in range(n):
            amplitud += matriz_distancias[i][j] * matriz_flujos[permutacion[i]][permutacion[j]]
    return amplitud

class ACO_QAP(ACO):
    def __init__(self, distancias, flujos, n_hormigas, n_iteraciones, evaporacion, alpha=1, beta=5, q=1):
        super().__init__(distancias, n_hormigas, n_iteraciones, evaporacion, alpha, beta, q)
        self.flujos = flujos
        self.best_amplitudes = []  # Lista para almacenar la mejor amplitud en cada iteración

    def computa_amplitud(self, camino):
        return calcular_amplitud(self.distancias, self.flujos, camino)
    
    def resuelve(self):
        mejores_distancias = []  # Lista para almacenar las mejores distancias en cada iteración
        mejor_camino = None
        mejor_distancia = float('inf')
    
        for iteracion in range(self.n_iteraciones):
            caminos = [self.construye_camino(np.random.choice(self.n_ciudades)) for _ in range(self.n_hormigas)]
            distancias = [self.computa_distancia(camino) for camino in caminos]
            self.actualiza_feromona(caminos, distancias)
            min_dist = min(distancias)
            if min_dist < mejor_distancia:
                mejor_distancia = min_dist
                mejor_camino = caminos[distancias.index(min_dist)]
            
            # Agregar la distancia mínima actual a la lista de mejores distancias
            mejores_distancias.append(mejor_distancia)
    
        # Trazar la gráfica de convergencia
        plt.figure()
        plt.plot(mejores_distancias)
        plt.title('Convergencia del ACO')
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Distancia')
        plt.grid(True)
        plt.show()
    
        return mejor_camino, mejor_distancia

if __name__ == "__main__":
    aco_qap = ACO_QAP(distancias, flujos, n_hormigas=10, n_iteraciones=100, evaporacion=0.5)
    
    mejor_camino, mejor_amplitud = aco_qap.resuelve()
    print("Mejor camino:", mejor_camino)  # Mejor camino de la Ant
    print("Costo óptimo:", mejor_amplitud)  # Costo optimo
