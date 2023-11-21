import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from os import path


class Patente:

    def __init__(self) -> None:
        self.imagenes = [
            {
                # img 1
                "threshold": 160,
                "area": [80, 30],
                "rel_aspect": [1.2, 2.5],
                "dist": 43
            },
            {
                # img 2
                "threshold": 100,
                "area": [80, 30],
                "rel_aspect": [1.2, 2.5],
                "dist": 43
            },
            {
                # img 3
                "threshold": 120,
                "area": [180, 5],
                "rel_aspect": [1, 5],
                "dist": 43
            },
            {
                # img 4
                "threshold": 160,
                "area": [80, 30],
                "rel_aspect": [1.2, 2.5],
                "dist": 43
            },
            {
                # img 5
                "threshold": 160,
                "area": [80, 30],
                "rel_aspect": [1.2, 2.8],
                "dist": 30
            },
            {
                # img 6
                "threshold": 110,
                "area": [100, 30],
                "rel_aspect": [1.2, 2.8],
                "dist": 30
            },
            {
                # img 7
                "threshold": 110,
                "area": [100, 10],
                "rel_aspect": [1.2, 2.8],
                "dist": 30
            },
            {
                # img 8
                "threshold": 140,
                "area": [100, 10],
                "rel_aspect": [1.2, 2.8],
                "dist": 30
            },
            {
                # img 9
                "threshold": 100,
                "area": [100, 15],
                "rel_aspect": [1.2, 2.8],
                "dist": 30
            },
            {
                # img 10
                "threshold": 130,
                "area": [100, 30],
                "rel_aspect": [1.2, 2.5],
                "dist": 30
            },
            {
                # img 11
                "threshold": 133,
                "area": [100, 30],
                "rel_aspect": [1.2, 2.5],
                "dist": 30
            },
            {
                # img 12
                "threshold": 100,
                "area": [100, 15],
                "rel_aspect": [1.8, 2.5],
                "dist": 30
            },
        ]

        self.formas = [
            {
                #img 1
                "threshold": 150,
                "area": [800, 200],
                "rel_aspect": [0.1, 0.5],
            },
            {
                #img 2
                "threshold": 100,
                "area": [1000, 500],
                "rel_aspect": [0.2, 2.0],
            },
            {
                #img 3 ############## anda mal
                "threshold": 100,
                "area": [10000, 100],
                "rel_aspect": [1, 5],
            },
            {
                #img 4
                "threshold": 150,
                "area": [900, 800],
                "rel_aspect": [0.5, 1.5],
            },
            {
                #img 5
                "threshold": 160,
                "area": [10000, 2000],
                "rel_aspect": [0.1, 1.5],
            },
            {
                #img 6
                "threshold": 100,
                "area": [2000, 1000],
                "rel_aspect": [0.1, 2.0],
            },
            {
                #img 7
                "threshold": 100,
                "area": [1500, 500],
                "rel_aspect": [0.2, 0.5],
            },
            {
                #img 8
                "threshold": 140,
                "area": [1500, 500],
                "rel_aspect": [0.2, 1.0],
            },
            {
                #img 9
                "threshold": 100,
                "area": [1200, 500],
                "rel_aspect": [0.2, 0.5],
            },
            {
                #img 10
                "threshold": 110,
                "area": [2000, 1000],
                "rel_aspect": [0.3, 1.5],
            },
            {
                #img 11
                "threshold": 133,
                "area": [2000, 500],
                "rel_aspect": [0.3, 0.5],
            },
            {
                #img 12 ############# anda mal
                "threshold": 100,
                "area": [100, 15],
                "rel_aspect": [1.8, 2.5],
                "dist": 30
            },
        ]

    def obtener_letras(self):
        for i, param in enumerate(self.imagenes):
            img_num = i + 1
            pthreshold = param["threshold"]
            parea = param["area"]
            prel_aspect = param["rel_aspect"]
            pdist = param["dist"]

            img_gray = cv2.imread(path.join("imagenes", "ejercicio2", f"img{str(img_num).rjust(2, '0')}.png"),
                                  cv2.IMREAD_GRAYSCALE)
            _, img_umbral = cv2.threshold(img_gray, pthreshold, 255, cv2.THRESH_BINARY)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
            axes[0].imshow(img_umbral, cmap="gray")
            axes[0].set_title("Imagen umbralizada")

            num_labels, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbral, connectivity=8)
            img_zeros = np.zeros_like(img_labels)

            centroides_validos = []

            for i in range(1, len(stats)):
                izq, arriba, ancho, alto, area = stats[i]
                rel_aspecto = alto / ancho
                if area < parea[0] and area > parea[1] and rel_aspecto > prel_aspect[0] and rel_aspecto < prel_aspect[1]:
                    centroides_validos.append((centroids[i], i))
                    img_zeros[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

            axes[1].imshow(img_zeros, cmap="gray")
            axes[1].set_title("Imagen con componentes válidos")

            img_zeros2 = np.zeros_like(img_labels)
            for i in range(0, len(centroides_validos)):
                ci, ii = centroides_validos[i]
                centroides_validos_copia = [tupla for j, tupla in enumerate(centroides_validos) if tupla[1] != ii]
                for j in range(0, len(centroides_validos_copia)):
                    cj, ij = centroides_validos_copia[j]
                    dist = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
                    izq, arriba, ancho, alto, area = stats[ii]
                    if dist < pdist:
                        img_zeros2[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto,
                                                                            izq:izq + ancho]

            axes[2].imshow(img_zeros2, cmap="gray")
            axes[2].set_title("Carácteres detectados")

            fig.suptitle(f"Imagen {img_num}")

            # Mostrar imagen
            plt.show(block=True)

    def obtener_formas(self):
        for i, param in enumerate(self.formas):
            img_num = i + 1
            pthreshold = param["threshold"]
            parea = param["area"]
            prel_aspect = param["rel_aspect"]

            img_gray = cv2.imread(path.join("imagenes", "ejercicio2", f"img{str(img_num).rjust(2, '0')}.png"), cv2.IMREAD_GRAYSCALE)
            _, img_umbral = cv2.threshold(img_gray, pthreshold, 255, cv2.THRESH_BINARY)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
            axes[0].imshow(img_umbral, cmap="gray")
            axes[0].set_title('Imagen umbralada')

            # obtencion de componentes conenctados
            num_labels, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbral, connectivity=4)

            # filtro de componentes
            img_zeros = np.zeros_like(img_labels)

            for i in range(1, len(stats)):
                izq, arriba, ancho, alto, area = stats[i]
                rel_aspecto = alto / ancho
                # los componentes deben pasar filtro de area y relacion de aspecto
                if (area < parea[0]) and (area > parea[1]) and (rel_aspecto > prel_aspect[0]) and (rel_aspecto < prel_aspect[1]):
                    img_zeros[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

            axes[1].imshow(img_zeros, cmap="gray")
            axes[1].set_title('Filtro de area')

            fig.suptitle(f"Imagen {img_num}")

            plt.show(block=True)
            
