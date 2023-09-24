import numpy as np
import cv2
class Ecualizador:
    def ecualizar(self, imagen: np.ndarray, kernel: tuple[int, int]):
        """
        Genera una imagen ecualizada.
        Parametros:
            imagen: Arreglo numpy en grayscale
            kernel: Tupla de enteros impares para indicar alto y ancho
        """
        img = imagen.copy()
        nueva_img = np.zeros_like(img)
        #Agregar filas o columnas aplicando Replicación de Bordes
        #Si el kernel es de 11 x 11, entonces se debe agregar 5 filas y 5 columnas extras a la imágen para completar las celdas faltantes del kernel
        mitad_w = round(kernel[0] / 2)
        mitad_h = round(kernel[1] / 2)
        for _ in range(mitad_w):
            img = np.insert(img, 0, img[0, :], axis=0)
            img = np.insert(img, img.shape[0], img[img.shape[0] - 1, :], axis=0)
        for _ in range(mitad_h):
            img = np.insert(img, 0, img[:, 0], axis=1)
            img = np.insert(img, img.shape[1], img[:, img.shape[1] - 1], axis=1)
        #Recorrer imágen
        for i in range(mitad_w - 1, imagen.shape[0] + mitad_w - 2):
            for j in range(mitad_h - 1, imagen.shape[1] + mitad_h - 2):
                seccion = img[i - mitad_h + 1: i + mitad_h, j - mitad_w + 1: j + mitad_w]
                histograma = cv2.calcHist([seccion], [0], None, [256], [0, 256])
                cdf = histograma.cumsum() #Calcular PDF (Función de Densidad de Probabilidad)
                cdfn = cdf * 255 / float(cdf.max()) #Normalizar CDF
                nueva_img[i - mitad_w + 1, j - mitad_h + 1] = cdfn[img[i, j]]
        return nueva_img