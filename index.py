import cv2
import matplotlib.pyplot as plt
from src.Ecualizador import Ecualizador
from src.Formulario import Formulario

img = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)


e = Ecualizador()
nueva_img = e.ecualizar(img, (11, 11))
plt.figure()
plt.imshow(nueva_img, cmap="gray")
plt.show()

imagen = cv2.imread("formulario_01.png", cv2.IMREAD_GRAYSCALE)
f = Formulario(imagen)
data = f.escanear()
plt.figure()
plt.imshow(data["comentario"], cmap="gray")
plt.show(block=False)
plt.show()
