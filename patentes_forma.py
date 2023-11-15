import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


# escala de grises
img = cv2.imread('imagenes\ejercicio2\img01.png', cv2.IMREAD_GRAYSCALE)
imshow(img, title="Imagen Original")


# umbralado
ret,thresh1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
imshow(thresh1, title="Imagen Original")


# componentes conectados
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity, cv2.CV_32S)
imshow(img=labels)


# flitro por area
min_area = 700
max_area = 800
new_img = np.zeros_like(img)

for i in range(1, num_labels):    
    if stats[i, cv2.CC_STAT_AREA] > min_area and stats[i, cv2.CC_STAT_AREA] < max_area:
        new_img[labels == i] = 255

imshow(new_img, title="Componentes Conectados Filtrados")


# encierro la patente
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

x = stats[124, cv2.CC_STAT_LEFT]
y = stats[124, cv2.CC_STAT_TOP]
w = stats[124, cv2.CC_STAT_WIDTH]
h = stats[124, cv2.CC_STAT_HEIGHT]

cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

imshow(img_color, color_img=True)

