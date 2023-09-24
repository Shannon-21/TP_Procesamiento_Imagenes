import cv2
from matplotlib import pyplot as plt
import tinosort

img = cv2.imread('formulario_04.png', cv2.IMREAD_GRAYSCALE)

name_surname = img[63:100 , 330:930]
age = img[105:135 , 330:930]
email = img[145:180 , 330:930]
legajo = img[185:215 , 330:930]
ques1_y, ques1_n = img[265:295 , 330:625], img[265:295 , 635:930]
ques2_y, ques2_n = img[305:335 , 330:625], img[305:335 , 635:930]
ques3_y, ques3_n = img[345:375 , 330:625], img[345:375 , 635:930]
comments = img[385:490 , 330:930]

fields = [name_surname, age, email, legajo, ques1_y, ques1_n, ques2_y, ques2_n, ques3_y, ques3_n, comments]

plt.imshow(img==0, cmap='gray'), plt.show(block=False)
plt.imshow(tinosort.binarize(ques1_n), cmap='gray'), plt.show(block=False)

for field in fields:
    ren_norm = tinosort.normalize(field)
    tinosort.show_segmentation(ren_norm, 3, char_threshold=1) # para ver separacion de caracteres









import cv2
import numpy as np

# Lee la imagen de la celda
celda_img = cv2.imread('ruta_de_la_imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Binariza la imagen (si es necesario)
_, binary_img = cv2.threshold(celda_img, 128, 255, cv2.THRESH_BINARY)

# Aplica la función connectedComponentsWithStats
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, 8, cv2.CV_32S)

# Itera sobre cada componente (ignora el primer componente, ya que es el fondo)
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    
    # Dibuja un rectángulo alrededor del componente en la imagen original
    cv2.rectangle(celda_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Muestra las coordenadas del rectángulo
    print(f"Rectángulo {i}: Coordenadas (x, y): ({x}, {y}), Ancho: {w}, Alto: {h}")

# Muestra la imagen con los rectángulos dibujados
cv2.imshow('Rectángulos', celda_img)
cv2.waitKey(0)
cv2.destroyAllWindows()