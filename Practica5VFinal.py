import random
from itertools import permutations
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time

def calculaAptitudCuadrado(individuo, tamCuad):
    # Función de aptitud para cuadrados mágicos
    sumas = []
    numMag = (tamCuad * (tamCuad**2 + 1)) // 2
    for j in range(tamCuad):
        sumaFila = 0
        sumaColumna = 0
        for i in range(tamCuad):
            sumaFila += individuo[i + j * tamCuad]
            sumaColumna += individuo[j + i * tamCuad]
        sumas.append(sumaFila)
        sumas.append(sumaColumna)
    sumaDiagonal1 = sum(individuo[j * (tamCuad + 1)] for j in range(tamCuad))
    sumaDiagonal2 = sum(individuo[((j + 1) * tamCuad) - (j + 1)] for j in range(tamCuad))
    sumas.append(sumaDiagonal1)
    sumas.append(sumaDiagonal2)
    Error = [abs(elemento - numMag) for elemento in sumas]
    aptitud = sum(Error)
    return aptitud

def mutacion_thanos(poblacion):
    """Cuando termine, la mitad de la humanidad seguirá existiendo. 
    Perfectamente equilibrado, como todas las cosas deberían ser."""
    nueva_poblacion = poblacion.copy()
    mitad_longitud = len(nueva_poblacion) // 2
    indices_a_mutar = random.sample(range(mitad_longitud), mitad_longitud)
    for indice in indices_a_mutar:
        cadena_original = nueva_poblacion[indice]
        nueva_cadena = mutacionPermutacionCompleta(cadena_original)
    nueva_poblacion[indice] = nueva_cadena
    return nueva_poblacion

def mutacionPermutacionCompleta(individuo):
    import random
    from copy import deepcopy
    tam = len(individuo)
    permutacion = random.sample(individuo, tam)
    nuevoIndividuo = deepcopy(permutacion)

    return nuevoIndividuo

def mutacion_thanos_endgame(poblacion):
    """Pensé que eliminando la mitad de la vida, la otra mitad prosperaría, 
    pero me has demostrado… que es imposible."""
    nueva_poblacion = poblacion.copy()
    cantidad_a_no_mutar = 3
    indices_a_mutar = random.sample(range(len(nueva_poblacion)), len(nueva_poblacion) - cantidad_a_no_mutar)

    for indice in indices_a_mutar:
        cadena_original = nueva_poblacion[indice]
        nueva_cadena = mutacionPermutacionCompleta(cadena_original)
        nueva_poblacion[indice] = nueva_cadena

    return nueva_poblacion

def mutacionHeuristica(individuo, porcMuta):
    # Operador de mutación heurística
    tam = len(individuo)
    valorCambio = round(porcMuta*tam)
    tamCuad = int(tam ** 0.5)
    nuevoCromosoma1 = deepcopy(individuo)
    Elementos = random.sample(range(tam), valorCambio)
    perm1 = list(permutations(Elementos))
    mejores_cromosomas = []
    for p in perm1:
        nuevoCromosoma1[p[0]], nuevoCromosoma1[p[1]] = nuevoCromosoma1[p[1]], nuevoCromosoma1[p[0]]
        
        apt1 = calculaAptitudCuadrado(nuevoCromosoma1, tamCuad)
        mejores_cromosomas.append((nuevoCromosoma1.copy(), apt1))
        
    mejor_cromosoma, mejor_aptitud = min(mejores_cromosomas, key=lambda x: x[1])
    return mejor_cromosoma

# -----------------------------------------------------------
# Esta es una funcion nueva, para búsqueda tabú
def mutacionHeuristicaTabu(individuo, porcMuta, lista_tabu):
    # Operador de mutación heurística con búsqueda tabú
    tam = len(individuo)
    valorCambio = round(porcMuta * tam)
    tamCuad = int(tam ** 0.5)
    nuevoCromosoma1 = deepcopy(individuo)
  
    Elementos = [e for e in random.sample(range(tam), valorCambio) if e not in lista_tabu]

    if len(Elementos) >= 2:
        perm1 = list(permutations(Elementos))
        mejores_cromosomas = []
        
        for p in perm1:
            nuevoCromosoma1[p[0]], nuevoCromosoma1[p[1]] = nuevoCromosoma1[p[1]], nuevoCromosoma1[p[0]]
            
            apt1 = calculaAptitudCuadrado(nuevoCromosoma1, tamCuad)
            mejores_cromosomas.append((nuevoCromosoma1.copy(), apt1))
            
        mejor_cromosoma, mejor_aptitud = min(mejores_cromosomas, key=lambda x: x[1])
        return mejor_cromosoma
    else:
        return individuo

# ------------------------------------------------------------------------------------

def cruzaPMX(padre1, padre2):
    # Operador de cruzamiento PMX
    tam = len(padre1)
    punto1 = random.randint(0, tam - 2)
    punto2 = random.randint(punto1, tam - 1)
    
    hijo1 = padre1.copy()
    hijo2 = padre2.copy()
    
    for i in range(punto1, punto2 + 1):
        temp = hijo1[i]
        hijo1[i] = hijo2[i]
        hijo2[i] = temp
        
    for i in range(tam):
        if i < punto1 or i > punto2:
            while hijo1[i] in hijo1[punto1:punto2 + 1]:
                index = padre2.index(hijo1[i])
                hijo1[i] = padre1[index]
                
            while hijo2[i] in hijo2[punto1:punto2 + 1]:
                index = padre1.index(hijo2[i])
                hijo2[i] = padre2[index]
    
    return hijo1, hijo2

def algoritmoGeneticoCuadradoMagico(tamCuad, tamPoblacion, porcCruza, porcMuta, generaciones):
    poblacion = []
    mejor_caso = []
    peor_caso = []
    mediana_caso = []
    
    for _ in range(tamPoblacion):
        cuadrado = list(range(1, tamCuad**2 + 1))
        random.shuffle(cuadrado)
        poblacion.append(cuadrado)
    gen = 0
    mejor_aptitud = 1
    mejor_aptitud_pasado = 1
    estancamiento = 0
    problema = 0
    while True:
        gen = gen + 1
        aptitudes = [calculaAptitudCuadrado(individuo, tamCuad) for individuo in poblacion]
        indices_padres = defineParejas(tamPoblacion, aptitudes)
        padres = [poblacion[i] for i in indices_padres]
        hijos_cruzados = []
        for i in range(0, len(indices_padres), 2):
            padre1 = padres[i]
            padre2 = padres[i + 1]

            if random.uniform(0, 1) <= porcCruza:
                hijo1, hijo2 = cruzaPMX(padre1, padre2)
                hijos_cruzados.extend([hijo1, hijo2])
            else:
                hijos_cruzados.extend([padre1, padre2])

        hijos_mutados = [mutacionHeuristica(individuo, porcMuta) for individuo in hijos_cruzados]

        poblacion = hijos_mutados

        aptitudes_generacion = [calculaAptitudCuadrado(individuo, tamCuad) for individuo in poblacion]
        mejor_individuo = poblacion[np.argmin(aptitudes_generacion)]
        mejor_aptitud = min(aptitudes_generacion)
        print(f"Generación {gen + 1}: Mejor aptitud = {mejor_aptitud}")

        if mejor_aptitud == mejor_aptitud_pasado:
            estancamiento = estancamiento + 1
        mejor_aptitud_pasado = mejor_aptitud
        if estancamiento == 10:
            poblacion = mutacion_thanos(poblacion)
            print("No me quiero ir señor stark")
            problema = problema + 1
            estancamiento = 0
        if problema == 10:
            poblacion = mutacion_thanos_endgame(poblacion)
            print('Thanos gana')
            problema = 0
        aptitudes_generacion = [calculaAptitudCuadrado(individuo, tamCuad) for individuo in poblacion]
        mejor_aptitud = min(aptitudes_generacion)
        promedio_aptitud = np.mean(aptitudes_generacion)
        mejor_caso.append(min(aptitudes))
        peor_caso.append(max(aptitudes))
        mediana_caso.append(sorted(aptitudes)[len(aptitudes) // 2])

        print(f"Generación {gen + 1}: Mejor aptitud = {mejor_aptitud}, Promedio aptitud = {promedio_aptitud}")
        if mejor_aptitud == 0:
            break

    plt.show()
    plt.plot(range(1, gen + 1), mejor_caso, label='Mejor Caso')
    plt.plot(range(1, gen + 1), peor_caso, label='Peor Caso')
    plt.plot(range(1, gen + 1), mediana_caso, label='Caso Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Valor del Mejor Individuo')
    plt.legend()
    plt.title('Cuadrado Magico')
    plt.show()
        
    return mejor_individuo, mejor_aptitud

# ----------------------------------------------------------------------------------
# En esta función trabajaremos con Busqueda Tabú
def algoritmoGeneticoCuadradoMagicoTabu(tamCuad, tamPoblacion, porcCruza, porcMuta, generaciones, tamListaTabu):
    poblacion = []
    mejor_caso = []
    peor_caso = []
    mediana_caso = []
    
    for _ in range(tamPoblacion):
        cuadrado = list(range(1, tamCuad**2 + 1))
        random.shuffle(cuadrado)
        poblacion.append(cuadrado)
    
    lista_tabu = []
    gen = 0
    mejor_aptitud = float('inf')
    mejor_aptitud_pasado = 1
    estancamiento = 0
    problema = 0
    avance = 0
    
    while True:
        gen += 1
        aptitudes = [calculaAptitudCuadrado(individuo, tamCuad) for individuo in poblacion]
        indices_padres = defineParejas(tamPoblacion, aptitudes)
        padres = [poblacion[i] for i in indices_padres]
        hijos_cruzados = []
        for i in range(0, len(indices_padres), 2):
            padre1 = padres[i]
            padre2 = padres[i + 1]

            if random.uniform(0, 1) <= porcCruza:
                hijo1, hijo2 = cruzaPMX(padre1, padre2)
                hijos_cruzados.extend([hijo1, hijo2])
            else:
                hijos_cruzados.extend([padre1, padre2])
        hijos_mutados = [mutacionHeuristicaTabu(individuo, porcMuta, lista_tabu) for individuo in hijos_cruzados]
        poblacion = hijos_mutados

        lista_tabu = actualizarListaTabu(lista_tabu, hijos_mutados, tamListaTabu)

        aptitudes_generacion = [calculaAptitudCuadrado(individuo, tamCuad) for individuo in poblacion]
        mejor_individuo = poblacion[np.argmin(aptitudes_generacion)]
        mejor_aptitud = min(aptitudes_generacion)
        #print(f"Generación {gen}: Mejor aptitud = {mejor_aptitud}")

        #Quitar estancamiento
        if mejor_aptitud == mejor_aptitud_pasado:
            estancamiento = estancamiento + 1
            avance = 0
        else:
            avance = avance + 1
            estancamiento = 0
        mejor_aptitud_pasado = mejor_aptitud
        if estancamiento == 10:
            if round(porcMuta * tamCuad**2) > 5:
                poblacion = mutacion_thanos(poblacion)
                print('Thanos chasquea')
            else:
                porcMuta = porcMuta + 0.01
                print('Mas lento') 
            problema = problema + 1
            estancamiento = 0
        if avance == 7:
            if round(porcMuta * tamCuad**2) > 2:
                porcMuta = porcMuta - 0.01
            else:
                porcMuta = porcMuta
            print('mas rapido')
            avance = 0

        if problema == 6:
            poblacion = mutacion_thanos_endgame(poblacion)
            print('Thanos gana')
            porcMuta = porcMuta - 0.01
            problema = 0

        # Agrega el mejor y peor caso de aptitud en cada generación
        mejor_caso.append(min(aptitudes))
        peor_caso.append(max(aptitudes))
        mediana_caso.append(sorted(aptitudes)[len(aptitudes) // 2])
        promedio_aptitud = np.mean(aptitudes_generacion)
        #print(f"Generación {gen + 1}: Mejor aptitud = {mejor_aptitud}, Promedio aptitud = {promedio_aptitud}")
        if mejor_aptitud == 0:
            break
    print(f'Generación donde se encontro:{gen + 1}')

    plt.plot(range(1, gen + 1), mejor_caso, label='Mejor Caso')
    plt.plot(range(1, gen + 1), peor_caso, label='Peor Caso')
    plt.plot(range(1, gen + 1), mediana_caso, label='Caso Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Valor del Mejor Individuo')
    plt.legend()
    plt.title('Cuadrado Mágico con Búsqueda Tabú')
    plt.show()
    
    return mejor_individuo, mejor_aptitud

def actualizarListaTabu(lista_tabu, poblacion, tamListaTabu):
    # Actualiza la lista tabú con los individuos generados en la población
    lista_tabu.extend(poblacion)
    if len(lista_tabu) > tamListaTabu:
        lista_tabu = lista_tabu[-tamListaTabu:]
    return lista_tabu

def plotCuadradoMagico(cuadrado):
    # Función para visualizar cuadrado mágico
    tam = int(len(cuadrado) ** 0.5)
    matriz = np.array(cuadrado).reshape((tam, tam))
    print(matriz)


def defineParejas(tamPop, aptitudes):
    # Función para seleccionar parejas de padres
    padres_seleccionados = set()
    while len(padres_seleccionados) < tamPop // 2:  
        indice_padre = seleccion_torneo(aptitudes)
        indice_otro_padre = seleccion_torneo(aptitudes)
        
        if indice_padre != indice_otro_padre and (indice_padre, indice_otro_padre) not in padres_seleccionados and (indice_otro_padre, indice_padre) not in padres_seleccionados:
            padres_seleccionados.add((indice_padre, indice_otro_padre))
    
    return [indice for pareja in padres_seleccionados for indice in pareja]

def seleccion_torneo(aptitudes):
    tam_aptitud = len(aptitudes)
    candidato1 = random.randint(0, tam_aptitud - 1)
    candidato2 = random.randint(0, tam_aptitud - 1)
    if candidato1 == candidato2:
        candidato2 = random.randint(0, tam_aptitud - 1)

    if aptitudes[candidato1] < aptitudes[candidato2]:
        return candidato1
    elif aptitudes[candidato1] > aptitudes[candidato2]:
        return candidato2
    else:
        return candidato1 

tamCuad = input('Ingrese el tamaño el cuadrado magico que desea ver:')
tamCuad = int(tamCuad)
tamPoblacion = 70
porcCruza = 0.8
porcMuta = 1
generaciones = 100


tamListaTabu = 5 
primos=[3,17,29,31,43,59,67,71,89,97]

tiempo_total=0

for i,seed_value in enumerate(primos,start=1):
    random.seed(seed_value)
    print(f"\nIteración {i} -- Semilla = {seed_value}")


    tiempo_inicio = time.time()
    while round(porcMuta*tamCuad**2) > 6:
        porcMuta = porcMuta - 0.001

    mejor_cuadrado, mejor_aptitud = algoritmoGeneticoCuadradoMagicoTabu(tamCuad, tamPoblacion, porcCruza, porcMuta, generaciones, tamListaTabu)

    # Visualizar el mejor cuadrado mágico y su aptitud
    print("\nMejor cuadrado mágico:")
    plotCuadradoMagico(mejor_cuadrado)
    print(f"\nMejor aptitud: {mejor_aptitud}")

    tiempo_fin = time.time()

    # Calcula la diferencia de tiempo
    diferencia_tiempo = tiempo_fin - tiempo_inicio

    # Agrega el tiempo de la iteración al tiempo total
    tiempo_total += diferencia_tiempo

    # Convierte la diferencia de tiempo a días, horas, minutos y segundos
    dias = int(diferencia_tiempo // (24 * 3600))
    horas = int((diferencia_tiempo % (24 * 3600)) // 3600)
    minutos = int((diferencia_tiempo % 3600) // 60)
    segundos = int(diferencia_tiempo % 60)

    print(f'Tiempo transcurrido de ejecución: {dias} días, {horas} horas, {minutos} minutos, {segundos} segundos')

# Calcula el tiempo promedio en segundos
promedio_tiempo = tiempo_total / len(primos)

# Calcula el tiempo promedio en días, horas, minutos y segundos
promedio_dias = int(promedio_tiempo // (24 * 3600))
promedio_horas = int((promedio_tiempo % (24 * 3600)) // 3600)
promedio_minutos = int((promedio_tiempo % 3600) // 60)
promedio_segundos = int(promedio_tiempo % 60)

# Muestra el tiempo promedio en días, horas, minutos y segundos
print(f'\nPromedio del tiempo de ejecución para las 10 iteraciones: {promedio_dias} días, {promedio_horas} horas, {promedio_minutos} minutos, {promedio_segundos} segundos')