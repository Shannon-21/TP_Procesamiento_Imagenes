import random
from src.Patentes import Patentes
import numpy as np
# [threshold, min_area, max_area, min_ra, max_ra, dist]

class AlgoGenetico:
    def __init__(self, tamanio_poblacion = 100, tasa_mutacion = 0.1):
        self.tamanio_poblacion = tamanio_poblacion
        self.tasa_mutacion = tasa_mutacion
        self.poblacion = self.generar_poblacion()
        self.patente = Patentes()
    
    def generar_individuo(self):
        threshold = self.threshold()
        min_area = self.min_area()
        max_area = self.max_area(min_area)
        min_ra = self.min_ra()
        max_ra = self.max_ra(min_ra)
        dist = self.dist()
        return [threshold, min_area, max_area, min_ra, max_ra, dist]
        
    def generar_poblacion(self):
        return [self.generar_individuo() for _ in range(self.tamanio_poblacion)]

    def calcular_fitness(self, individuo):
        params = [{
            "threshold": individuo[0],
            "area": [individuo[2], individuo[1]],
            "rel_aspect": [individuo[3], individuo[4]],
            "dist": individuo[5]
        }]
        result = self.patente.encontrar_caracteres(params, False)
        sumres = sum(result)
        if sumres == 0: return 0
        return 1 / (abs(6 - len(result)) + 1)

    def seleccionar_padres(self):
        padres = []
        for _ in range(2):
            padres.append(max(self.poblacion, key=self.calcular_fitness))
        return padres

    def reproducir(self, p1, p2):
        corte = random.randint(1, 5)
        return p1[:corte] + p2[corte:]
    
    def mutar(self, individuo):
        for i in range(6):
            if random.random() < self.tasa_mutacion:
                if i == 0:
                    individuo[i] = self.threshold()
                elif i == 1:
                    individuo[i] = self.min_area()
                elif i == 2:
                    individuo[i] = self.max_area(individuo[i - 1])
                elif i == 3:
                    individuo[i] = self.min_ra()
                elif i == 4:
                    individuo[i] = self.max_ra(individuo[i - 1])
                elif i == 5:
                    individuo[i] = self.dist()
        return individuo

    def ejecutar(self, num_generaciones):
        mejor_fitness = 0
        mejor_individuo = None
        for generacion in range(num_generaciones):
            print("Generación: ", generacion + 1)
            nueva_gen = []
            for _ in range(self.tamanio_poblacion // 2):
                print("Trabajando con la población...")
                p1, p2 = self.seleccionar_padres()
                hijo1 = self.reproducir(p1, p2)
                hijo2 = self.reproducir(p2, p1)
                hijo1 = self.mutar(hijo1)
                hijo2 = self.mutar(hijo2)
                nueva_gen.extend([hijo1, hijo2])
            self.poblacion = nueva_gen
            mejor_individuo = max(self.poblacion, key=self.calcular_fitness)
            mejor_fitness = self.calcular_fitness(mejor_individuo)
            # if mejor_fitness == 6:
            #     print("Se encontró la respuesta", mejor_fitness, mejor_individuo)
            #     return mejor_individuo
        print("Se encontró la respuesta", mejor_fitness, mejor_individuo)
        return mejor_individuo
    
    def threshold(self):
        return random.randint(100, 200)
    
    def min_area(self):
        return random.randint(10, 99)
    
    def max_area(self, min_area):
        return random.randint(min_area + 1, 100)
    
    def min_ra(self):
        return random.uniform(1, 2.9)
    
    def max_ra(self, min_ra):
        return random.uniform(min_ra + 0.1, 3)
    
    def dist(self):
        return random.randint(20, 40)