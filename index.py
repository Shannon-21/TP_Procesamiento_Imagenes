import cv2, os, matplotlib.pyplot as plt
from src.Ecualizador import Ecualizador
from src.Formulario import Formulario



## Ejercicio 1 ##

details = cv2.imread("images/Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)
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

archivos = os.listdir("images")
formularios = [archivo for archivo in archivos if archivo.startswith('formulario')]

for formulario in formularios:
    ruta_formulario = os.path.join("images", formulario)
    form = cv2.imread(ruta_formulario, cv2.IMREAD_GRAYSCALE)

    print(f"{formulario}")

    f = Formulario(form)
    cells  = f._escanear()
    f.validate_form()

    plt.imshow(form, cmap="gray"), plt.show()
    print("\n")

    # para explicar funcionamiento
    # f.show_segmentation(cells['nombre'], 10)
    
