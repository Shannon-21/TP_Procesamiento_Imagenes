import numpy as np
import cv2
from matplotlib import pyplot as plt


class Ecualizador:

    def ecualizar(self, imagen: np.ndarray, kernel: tuple[int, int]):
        img = imagen.copy()
        nueva_img = np.zeros_like(img)

        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Aumentar el tamaño de la ventana para reducir los grises en los bordes
        mitad_w = round(kernel[0] / 2) - 1
        mitad_h = round(kernel[1] / 2) - 1

        for _ in range(mitad_w):
            img = np.insert(img, 0, img[0, :], axis=0)
            img = np.insert(img, img.shape[0], img[img.shape[0] - 1, :], axis=0)

        for _ in range(mitad_h):
            img = np.insert(img, 0, img[:, 0], axis=1)
            img = np.insert(img, img.shape[1], img[:, img.shape[1] - 1], axis=1)

        # Recorrer imágen
        for i in range(mitad_w, imagen.shape[0] + mitad_w - 2):
            for j in range(mitad_h, imagen.shape[1] + mitad_h - 2):
                # Obtener el historigrama
                seccion = img[i - mitad_w : mitad_w + i, j - mitad_h : mitad_h + j]
                histograma = cv2.calcHist([seccion], [0], None, [256], [0, 256])
                
                # Obtener la distribicion de frecuencias
                histograma_norm = histograma / img.shape[0] * img.shape[1]
                cdf = histograma_norm.cumsum()
                
                # Mapear la nueva intensidad al pixel original
                nueva_img[i - mitad_w : mitad_w + i, j - mitad_h : mitad_h + j] = cdf[img[i, j]]

        return nueva_img


    def show_equalized(self, imagen: np.ndarray, kernel: tuple[int, int]):
        nueva_img = self.ecualizar(imagen, kernel)

        # Calcular histograma de la imagen original
        hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256])

        # Calcular histograma de la imagen ecualizada
        hist_ecualizada = cv2.calcHist([nueva_img], [0], None, [256], [0, 256])

        fig = plt.figure(figsize=(10, 5))
        ax0 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((2, 3), (0, 2))
        ax2 = plt.subplot2grid((2, 3), (1, 2))

        # Mostrar la imagen ecualizada en ax0
        ax0.imshow(nueva_img, cmap="gray")
        ax0.set_title(f"Imagen Ecualizada con ventana de {kernel[0]} x {kernel[1]}")

        # Mostrar el histograma de la imagen original en ax1
        ax1.plot(hist_original)
        ax1.set_title("Histograma de la Imagen Original")

        # Mostrar el histograma de la imagen ecualizada en ax2
        ax2.plot(hist_ecualizada)
        ax2.set_title("Histograma de la Imagen Ecualizada")

        plt.tight_layout()
        plt.show()
