import cv2
from src.Ecualizador import Ecualizador
from src.Formulario import Formulario


## Ejercicio 1 ##

details = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)
e = Ecualizador()

kernels = [
    (5, 5),
    (11, 11),
    (11, 21),
    (21, 21),
    (111, 111),
]

for kernel in kernels:
    e.show_equalized(details, (kernel[0], kernel[1]))


##-- Ejercicio 2 --##

form = cv2.imread("formulario_02.png", cv2.IMREAD_GRAYSCALE)
f = Formulario(form)

cells  = f._escanear()
f.validate_form()
