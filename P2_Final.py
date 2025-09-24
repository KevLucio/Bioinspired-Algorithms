# Importar bibliotecas necesarias
import numpy as np
import math
import random
import itertools
import time
import matplotlib.pyplot as plt

# Definir una lista de números primos
primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

X = []
Y = []

# Definiciones de funciones de aptitud
def rastrigin_fitness(*args):
    # Función de aptitud de Rastrigin
    A = 10
    sum_term = sum(x**2 - A * np.cos(2 * np.pi * x) for x in args)
    return A * len(args) + sum_term

def ackley_fitness(*args):
    # Función de aptitud de Ackley
    A = 20
    B = 0.2
    C = 2 * math.pi
    
    sum_term1 = -A * np.exp(-B * np.sqrt(0.5 * sum(x**2 for x in args)))
    sum_term2 = -np.exp(0.5 * sum(math.cos(C * x) for x in args))
    
    result = sum_term1 + sum_term2 + A + math.exp(1)
    return result

def rosenbrock_fitness(*args):
    # Función de aptitud de Rosenbrock
    n = len(args)
    sum_term = sum((1 - args[i])**2 + 100 * (args[i + 1] - args[i]**2)**2 for i in range(n - 1))
    return sum_term

def griewank_fitness(*args):
    # Función de aptitud de Griewank
    n = len(args)
    sum_term1 = sum(x**2 / 4000 for x in args)
    sum_term2 = np.prod([math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(args)])
    return sum_term1 - sum_term2 + 1

# Menú para seleccionar la función de aptitud
print("Menú de funciones:\n")
print("1. Rastrigin")
print("2. Ackley")
print("3. Rosenbrock")
print("4. Griewank")

opcion = int(input("\nIngrese el número de la función deseada: "))

# Selección de la función de aptitud basada en la opción del usuario
if opcion == 1:
    fitness_function = rastrigin_fitness
    rango = [-5.12, 5.12]  # Rango para la función Rastrigin
    print("\nSe ha seleccionado la función Rastrigin.")
elif opcion == 2:
    fitness_function = ackley_fitness
    rango = [-32.768, 32.768]  # Rango para la función Ackley
    print("\nSe ha seleccionado la función Ackley.")
elif opcion == 3:
    fitness_function = rosenbrock_fitness
    rango = [-2.048, 2.048]  # Rango para la función Rosenbrock
    print("\nSe ha seleccionado la función Rosenbrock.")
elif opcion == 4:
    fitness_function = griewank_fitness
    rango = [-600, 600]  # Rango para la función Griewank
    print("\nSe ha seleccionado la función Griewank.")
else:
    print("Opción no válida. Se utilizará la función de aptitud por defecto (Griewank).")
    fitness_function = griewank_fitness
    rango = [-600, 600]  # Rango por defecto

# Registro de tiempo de inicio
start_time = time.time()

# Función para actualizar la velocidad de las partículas
def update_velocity(particle, velocity, pbest, gbest, w, c1, c2, max=1.0):
    # Inicializar un nuevo arreglo de velocidad
    num_particle = len(particle)
    new_velocity = np.array([0.0 for i in range(num_particle)])
    # Generar aleatoriamente r1, r2 y el peso de inercia desde una distribución normal
    r1 = random.uniform(0, max)
    r2 = random.uniform(0, max)
    # Calcular la nueva velocidad para todas las dimensiones
    for i in range(num_particle):
        new_velocity[i] = (
            w * velocity[i]
            + c1 * r1 * (pbest[i] - particle[i])
            + c2 * r2 * (gbest[i] - particle[i])
        )
    return new_velocity

# Función para actualizar la posición de las partículas
def update_position(particle, velocity, rango):
    # Mover las partículas sumando la velocidad
    new_particle = particle + velocity

    for i in range(len(new_particle)):
        if new_particle[i] < rango[0]:
            new_particle[i] = rango[0]
        if new_particle[i] > rango[1]:
            new_particle[i] = rango[1]
    return new_particle

# Función para ejecutar el algoritmo PSO en 2D
def pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion, c1, c2, w):
    resultados = {
        "Global Best Position": None,
        "Best Fitness Value": None,
        "Average Particle Best Fitness Value": None,
        "Number of Generation": None,
    }

    # Inicialización de partículas y velocidades
    particles = [
        [random.uniform(position_min, position_max) for j in range(dimension)]
        for i in range(population)
    ]
    
    velocity = [[0.0 for j in range(dimension)] for i in range(population)]
    
    pbest_position = particles
    pbest_fitness = [fitness_function(*p) for p in particles]
    
    gbest_index = np.argmin(pbest_fitness)
    gbest_position = pbest_position[gbest_index]

    for t in range(generation):
        if np.average(pbest_fitness) <= fitness_criterion:
            break
        else:
            for n in range(population):
                velocity[n] = update_velocity(
                    particles[n], velocity[n], pbest_position[n], gbest_position, w, c1, c2
                )
                Y.append(grafica(pbest_position[n]))
                particles[n] = update_position(particles[n], velocity[n], rango)

        pbest_fitness = [fitness_function(*p) for p in particles]
        gbest_index = np.argmin(pbest_fitness)
        gbest_position = pbest_position[gbest_index]

    resultados["Global Best Position"] = gbest_position
    resultados["Best Fitness Value"] = min(filter(lambda x: x >= 0, pbest_fitness))
    resultados["Average Particle Best Fitness Value"] = np.average(pbest_fitness)
    resultados["Number of Generation"] = t
    
    return resultados

def grafica(gbest):
    y = fitness_function(*gbest)  # Desempaqueta gbest
    return y

# Configuración de parámetros para PSO
population = 100
dimension = 10
position_min = rango[0]
position_max = rango[1]
generation = 5000
fitness_criterion = 10e-2

w = [0.25, 0.5]
c1 = [0, 1, 2]
c2 = [0, 1, 2]

combinaciones = list(itertools.product(w, c1, c2))

# Función para crear y escribir en archivos de texto
def crear_y_escribir_archivo(nombre_archivo, contenido):
    with open(nombre_archivo, "w") as archivo:
        archivo.write(contenido)

# Bucle para probar diferentes combinaciones de parámetros
for i in combinaciones:
    W = i[0]
    C1 = i[1]
    C2 = i[2]
    # Crear una cadena para almacenar todos los resultados de una combinación
    resultados_combinacion = ""
    for j in range(0, len(primos)):
        random.seed(primos[j])
        resultados = pso_2d(
            population,
            dimension,
            position_min,
            position_max,
            generation,
            fitness_criterion,
            C1,
            C2,
            W
        )
        # Convierte los resultados en una cadena legible (ajusta esto según tus resultados)
        resultados_str = "\n".join([f"{clave}: {valor}" for clave, valor in resultados.items()])
        # Agrega los resultados de esta semilla a la cadena de resultados de la combinación
        resultados_combinacion += f"Resultados para Primo {primos[j]}:\n{resultados_str}\n\n"
    # Crea un nombre de archivo único basado en la combinación
    nombre_archivo = f"{W}_{C1}_{C2}.txt"
    # Crea y escribe en el archivo TXT
    crear_y_escribir_archivo(nombre_archivo, resultados_combinacion)

# Registro de tiempo de finalización
end_time = time.time()

# Cálculo del tiempo total de ejecución
total_time = end_time - start_time

# Imprimir el tiempo total en días, horas, minutos y segundos
days, seconds = divmod(total_time, 86400)
hours, seconds = divmod(seconds, 3600)
minutes, seconds = divmod(seconds, 60)
print(f"\n\nTiempo total de ejecución: {int(days)} días, {int(hours)} horas, {int(minutes)} minutos, {int(seconds)} segundos")

for i in range(2100):
    X.append(i)

datos = list(zip(X, Y))

# Ordenar la lista de tuplas en función de los valores de Y
datos_ordenados = sorted(datos, key=lambda x: x[1])

X_invertidos = list(reversed(X))

# Desempaquetar los valores ordenados de vuelta en X e Y
X_ordenados, Y_ordenados = zip(*datos_ordenados)

print(len(Y_ordenados))

plt.plot(X_invertidos, Y_ordenados)
plt.xlabel('Valores de X')
plt.ylabel('Valores de Y')
if opcion == 1:
    plt.title('RASTRIGIN')
elif opcion == 2:
    plt.title('ACKLEY')
elif opcion == 3:
    plt.title('ROSENBROCK')
elif opcion == 3:
    plt.title('GRIEWANK')
plt.grid(True)
plt.show()

# Mensaje al final de la ejecución
print("\nEl programa ha terminado de ejecutarse.")
