import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


class Dado:
    def __init__(self, img: np.ndarray, x: float, y: float, width: float, height: float) -> None:
        # inicializa las caracteristicas de un componente conectado
        self.img = img
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contar_nro(self):
        """
        Cuenta los números de la cara de un dado
        """
        cont = 0
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY) # transformar la imagen escala de grises
        _, d_bin = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY) # binarizar la imagen
        d_nro, _, d_stats, _ = cv2.connectedComponentsWithStats(d_bin, connectivity=4) # componentes conectados
        
        # asumiendo que la entrada es un recorte de imagen del dado, filtrar por los puntos
        for j in range(1, d_nro):
            darea = d_stats[j, cv2.CC_STAT_AREA]
            dwidth = d_stats[j, cv2.CC_STAT_WIDTH]
            dheight = d_stats[j, cv2.CC_STAT_HEIGHT]

            # filtro de area y relacion de aspecto
            aspect_ratio = dheight / dwidth
            if darea > 60 and darea < 160 and aspect_ratio > 0.6 and aspect_ratio < 1.2:
                # si cumple, es un punto en la cara del dado
                cont += 1

        return cont


    def es_misma_posicion(self, dado: "Dado"):
        """
        Determina si el dado está en la misma posición calculando la distancia euclidea
        """
        distancia = math.sqrt(
            math.pow(self.x - dado.x, 2)
            +
            math.pow(self.y - dado.y, 2)
        )
        return distancia < 80


    def dibujar(self, img: np.ndarray):
        """
        Dibuja un rectángulo con el nro de la cara en la imagen dada
        """
        # dibujar el rectngulo sobre el dado
        img2 = cv2.rectangle(img, 
                            (self.x, self.y),
                            (self.x + self.width, self.y + self.height),
                            (255, 255, 0),
                            thickness=10)
        
        # escribe el numero de puntos en la cara
        img_dibujada = cv2.putText(img2,
                                   str(self.contar_nro()),
                                   (self.x, self.y - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                   (255, 255, 0), 2)
        
        return img_dibujada
