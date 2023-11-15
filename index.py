import cv2
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import math

img_gray = cv2.imread("/home/simon/Documentos/FCEIA/PDI/TP1_Procesamiento_Imagenes/imagenes/ejercicio2/img01.png", cv2.IMREAD_GRAYSCALE)
_, img_umbral = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
plt.imshow(img_umbral, cmap="gray")
plt.show(block=True)

num_labels, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbral)

plt.imshow(img_labels, cmap="gray")
plt.show(block=True)

img_zeros = np.zeros_like(img_labels)

centroides_validos = []

for i in range(1, len(stats)):
    izq, arriba, ancho, alto, area = stats[i]
    rel_aspecto = alto / ancho
    if area < 80 and area > 30 and rel_aspecto > 1.2 and rel_aspecto < 2.5:
        centroides_validos.append((centroids[i], i))
        img_zeros[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

# for i in range(1, len(stats)):
#     izq, arriba, ancho, alto, area = stats[i]
#     rel_aspecto = alto / ancho
#     if area < 80 and area > 30 and rel_aspecto > 1.2 and rel_aspecto < 2.5:
#         ci = centroids[i]
#         for j in range(1 + i, len(centroids)):
#             cj = centroids[j]
#             dist = math.sqrt((ci[0] - cj[0]) ** 2 + (cj[0] + cj[1]) ** 2)
#             print("DIST", dist)
#         # centroides_validos.append(centroids[i])
#         img_zeros[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

img_zeros2 = np.zeros_like(img_labels)
for i in range(0, len(centroides_validos)):
    ci, ii = centroides_validos[i]
    for j in range(0, len(centroides_validos)):
        cj, ij = centroides_validos[j]
        dist = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
        print(dist)
        if dist < 46:
            izq, arriba, ancho, alto, area = stats[ii]
            img_zeros2[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

plt.figure()
plt.imshow(img_zeros2, cmap="gray")
plt.show()