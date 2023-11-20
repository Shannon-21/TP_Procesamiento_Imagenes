import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

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


imagenes = [
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

path = 'imagenes\ejercicio2'



def obtener_letras(params):
    for i, param in enumerate(params):
    # for i, param in enumerate(params[11:], start=11):    
        # obtener los parametros de la imagen
        img_num = i + 1
        pthreshold = param["threshold"]
        parea = param["area"]
        prel_aspect = param["rel_aspect"]
        # pdist = param["dist"]

        # carga y umbrala la imagen
        img_gray = cv2.imread(f"{path}\img{str(img_num).rjust(2, '0')}.png", cv2.IMREAD_GRAYSCALE)        
        _, img_umbral = cv2.threshold(img_gray, pthreshold, 255, cv2.THRESH_BINARY)
        # _, img_umbral = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        axes[0].imshow(img_umbral, cmap="gray")

        # fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        # axes[1].imshow(img_umbral, cmap="gray", title='Imagen umbralada')


        # obtencion de componentes conenctados
        num_labels, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbral, connectivity=4)

        # filtro de componentes
        img_zeros = np.zeros_like(img_labels)

        for i in range(1, len(stats)):
            izq, arriba, ancho, alto, area = stats[i]
            rel_aspecto = alto / ancho
            # los componentes deben pasar filtro de area y relacion de aspecto
            if (area < parea[0]) and (area > parea[1]) and (rel_aspecto > prel_aspect[0]) and (rel_aspecto < prel_aspect[1]):
            # if (area < 800) and (area > 200):# and (rel_aspecto > prel_aspect[0]) and (rel_aspecto < prel_aspect[1]):
            #     if (rel_aspecto > 0.1) and (rel_aspecto < 0.5):
                    img_zeros[arriba:arriba + alto, izq:izq + ancho] = img_labels[arriba:arriba + alto, izq:izq + ancho]

        axes[1].imshow(img_zeros, cmap="gray")
        fig.show()
        
    
imshow(obtener_letras(imagenes))



# la imagen 3 y 12 no van bien
