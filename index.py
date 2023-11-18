import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from src.Patentes import Patentes
from src.AlgoGenetico import AlgoGenetico

algo = AlgoGenetico(20, 0.1)
res = algo.ejecutar(10)

patente = Patentes()
patente.encontrar_caracteres([{
            #img 1
            "threshold": res[0],
            "area": [res[2], res[1]],
            "rel_aspect": [res[3], res[4]],
            "dist": res[5]
        }], True)
# print(patente.encontrar_caracteres([{
#             #img 1
#             "threshold": 160,
#             "area": [80, 30],
#             "rel_aspect": [1.2, 2.5],
#             "dist": 43
#         }], True)

# )