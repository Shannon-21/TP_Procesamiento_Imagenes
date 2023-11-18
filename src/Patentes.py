import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from os import path
from scipy.optimize import minimize

class Patentes:
    params = [
        {
            #img 1
            "threshold": 160,
            "area": [80, 30],
            "rel_aspect": [1.2, 2.5],
            "dist": 43
        },
        {
            #img 2
            "threshold": 100,
            "area": [80, 30],
            "rel_aspect": [1.2, 2.5],
            "dist": 43
        },
        {
            #img 3 ######################################## REVISAR!!!!!! #########################
            "threshold": 120,
            "area": [180, 5],
            "rel_aspect": [1, 5],
            "dist": 43
        },
        {
            #img 4
            "threshold": 160,
            "area": [80, 30],
            "rel_aspect": [1.2, 2.5],
            "dist": 43
        },
        {
            #img 5
            "threshold": 160,
            "area": [80, 30],
            "rel_aspect": [1.2, 2.8],
            "dist": 30
        },
        {
            #img 6
            "threshold": 110,
            "area": [100, 30],
            "rel_aspect": [1.2, 2.8],
            "dist": 30
        },
        {
            #img 7
            "threshold": 110,
            "area": [100, 10],
            "rel_aspect": [1.2, 2.8],
            "dist": 30
        },
        {
            #img 8
            "threshold": 140,
            "area": [100, 10],
            "rel_aspect": [1.2, 2.8],
            "dist": 30
        },
        {
            #img 9
            "threshold": 100,
            "area": [100, 15],
            "rel_aspect": [1.2, 2.8],
            "dist": 30
        },
        {
            #img 10
            "threshold": 130,
            "area": [100, 30],
            "rel_aspect": [1.2, 2.5],
            "dist": 30
        },
        {
            #img 11
            "threshold": 133,
            "area": [100, 30],
            "rel_aspect": [1.2, 2.5],
            "dist": 30
        },
        {
            #img 12
            "threshold": 100,
            "area": [100, 15],
            "rel_aspect": [1.8, 2.5],
            "dist": 30
        },
    ]

    def encontrar_caracteres(self, params, show_img = True):
        for i, param in enumerate(params):
            img_num = i + 1
            pthreshold = param["threshold"]
            parea = param["area"]
            prel_aspect = param["rel_aspect"]
            pdist = param["dist"]
            img_gray = cv2.imread(path.join("imagenes", "ejercicio2", f"img{str(img_num).rjust(2, '0')}.png"), cv2.IMREAD_GRAYSCALE)
            _, img_umbral = cv2.threshold(img_gray, pthreshold, 255, cv2.THRESH_BINARY)

            if show_img:
                plt.figure()
                plt.imshow(img_umbral, cmap="gray")
                plt.show(block=False)

            num_labels, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbral, connectivity=8)
            img_zeros = np.zeros_like(img_labels)

            centroides_validos = []

            for i in range(1, len(stats)):
                izq, arriba, ancho, alto, area = stats[i]
                rel_aspecto = alto / ancho
                if area < parea[0] and area > parea[1] and rel_aspecto > prel_aspect[0] and rel_aspecto < prel_aspect[1]:
                    centroides_validos.append((centroids[i], i))
                    img_zeros[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

            if show_img:
                plt.figure()
                plt.imshow(img_zeros, cmap="gray")
                plt.show(block=True)
            componentes = []
            img_zeros2 = np.zeros_like(img_labels)
            for i in range(0, len(centroides_validos)):
                ci, ii = centroides_validos[i]
                centroides_validos_copia = [tupla for j, tupla in enumerate(centroides_validos) if tupla[1] != ii]
                for j in range(0, len(centroides_validos_copia)):
                    cj, ij = centroides_validos_copia[j]
                    dist = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
                    izq, arriba, ancho, alto, area = stats[ii]
                    if dist < pdist:
                        componentes.append(dist)
                        img_zeros2[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

            if show_img:
                plt.figure()
                plt.imshow(img_zeros2, cmap="gray")
                plt.show(block=True)
        return componentes
    
    threshold = 100
    area_min = 10
    area_max = 100
    rel_aspect_min = 1.2
    rel_aspect_max = 2
    
    # def encontrar_img1(self):
    
    #     img_num = i + 1
    #     pthreshold = param["threshold"]
    #     parea = param["area"]
    #     prel_aspect = param["rel_aspect"]
    #     pdist = param["dist"]
    #     img_gray = cv2.imread(path.join("imagenes", "ejercicio2", f"img{str(img_num).rjust(2, '0')}.png"), cv2.IMREAD_GRAYSCALE)
    #     _, img_umbral = cv2.threshold(img_gray, pthreshold, 255, cv2.THRESH_BINARY)

    #     plt.figure()
    #     plt.imshow(img_umbral, cmap="gray")
    #     plt.show(block=False)

    #     num_labels, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbral, connectivity=8)
    #     img_zeros = np.zeros_like(img_labels)

    #     centroides_validos = []

    #     for i in range(1, len(stats)):
    #         izq, arriba, ancho, alto, area = stats[i]
    #         rel_aspecto = alto / ancho
    #         if area < parea[0] and area > parea[1] and rel_aspecto > prel_aspect[0] and rel_aspecto < prel_aspect[1]:
    #             centroides_validos.append((centroids[i], i))
    #             img_zeros[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

    #     plt.figure()
    #     plt.imshow(img_zeros, cmap="gray")
    #     plt.show(block=False)

    #     img_zeros2 = np.zeros_like(img_labels)
    #     for i in range(0, len(centroides_validos)):
    #         ci, ii = centroides_validos[i]
    #         centroides_validos_copia = [tupla for j, tupla in enumerate(centroides_validos) if tupla[1] != ii]
    #         for j in range(0, len(centroides_validos_copia)):
    #             cj, ij = centroides_validos_copia[j]
    #             dist = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
    #             izq, arriba, ancho, alto, area = stats[ii]
    #             if dist < pdist:
    #                 img_zeros2[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

    #     plt.figure()
    #     plt.imshow(img_zeros2, cmap="gray")
    #     plt.show(block=True)
