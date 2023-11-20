import cv2
from matplotlib import pyplot as plt
import numpy as np

# funcion para visualizar una imagen
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


########## carga y muestra de imagen original
img_path="tp2\TP1_Procesamiento_Imagenes-tp2\imagenes\ejercicio1\monedas.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img, colorbar=False, title='Imagen Original')

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imshow(img_gray, colorbar=False, title='Imagen escala de grises')


########## reduccion de ruido
# f_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0, sigmaY=0)
f_blur = cv2.medianBlur(img_gray, 9)
imshow(f_blur, 'Blur y Escala de grises')


########## probamos deteccion de bordes con sobel
sobel_x = cv2.Sobel(f_blur, -1, 1, 0, ksize=3)
sobel_y = cv2.Sobel(f_blur, -1, 0, 1, ksize=3)
sobel_xy = cv2.Sobel(f_blur, -1, 1, 1, ksize=3)
sobel_combined = cv2.addWeighted((sobel_x), 0.5, (sobel_y), 0.5, 5)

# Crear una figura de 2x2 para mostrar los resultados de Sobel
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

axes[0, 0].imshow((sobel_x), cmap='gray')
axes[0, 0].set_title('Sobel en X')

axes[0, 1].imshow((sobel_y), cmap='gray')
axes[0, 1].set_title('Sobel en Y')

axes[1, 0].imshow((sobel_xy), cmap='gray')
axes[1, 0].set_title('Sobel en XY')

axes[1, 1].imshow(sobel_combined, cmap='gray')
axes[1, 1].set_title('Sobel combinado')

plt.tight_layout()
plt.show()


###### gradiente con umbralado
_, sobel_xy_thresh = cv2.threshold(cv2.convertScaleAbs(sobel_xy), 5, 255, cv2.THRESH_BINARY)
_, sobel_combined_thresh = cv2.threshold(sobel_combined, 10, 255, cv2.THRESH_BINARY)

# Crear una figura de 2x1 para mostrar los resultados de Sobel y umbralado
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

# Mostrar Sobel en XY con umbralado
axes[0].imshow(sobel_xy_thresh, cmap='gray')
axes[0].set_title('Sobel en XY con umbralado')

# Mostrar Sobel combinado con umbralado
axes[1].imshow(sobel_combined_thresh, cmap='gray')
axes[1].set_title('Sobel combinado con umbralado')

plt.tight_layout()
plt.show()



####### Deteccion de bordes con Canny
canny_img = cv2.Canny(f_blur, 0.3*255, 0.1*255, apertureSize=3, L2gradient=True)
imshow(canny_img, title='Deteccion Canny', colorbar=False)

####### Aplicar operaciones morfologicas sobre canny
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

morph = {}
morph["canny"] = canny_img
morph["apertura"] = cv2.morphologyEx(canny_img, cv2.MORPH_OPEN, kernel)
morph["clausura"] = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
morph["gradiente"] = cv2.morphologyEx(canny_img, cv2.MORPH_GRADIENT, kernel)
morph["apertura clausura"] = cv2.morphologyEx(morph["apertura"], cv2.MORPH_CLOSE, kernel)
morph["gradiente clausura"] = cv2.morphologyEx(morph["clausura"], cv2.MORPH_GRADIENT, kernel)

# Crear una figura de 2x1 para mostrar los resultados de Sobel y umbralado
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

# Definir el título de cada subfigura
titles = ['canny', "apertura", "clausura", "gradiente", 'apertura clausura' ,"gradiente clausura"]

for i, (key, value) in enumerate(morph.items()):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(value, cmap='gray')
    axes[row, col].set_title(titles[i])

plt.show()


######## Diltacion
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
canny_dilate = cv2.dilate( morph["gradiente"],dilate_kernel,iterations = 3)

imshow(canny_dilate, title='Dilatacion', colorbar=False)

######## Deteccion de circulos
coins =  cv2.HoughCircles(
            canny_dilate,
            cv2.HOUGH_GRADIENT,
            1.7,  
            300,  
            param1=255,  
            param2=100, 
            minRadius=55,
            maxRadius=200
        )
circles = np.uint16(np.around(coins))

img_circles = img.copy()
for i in circles[0,:]:
    cv2.circle(img_circles,(i[0],i[1]),i[2],(255,0,0),20)
    cv2.circle(img_circles,(i[0],i[1]),2,(0,0,255),10)

imshow(img_circles, title='detected circles', colorbar=False)



###### Clasificacion de monedas
output = img.copy()
result = np.zeros_like(output, np.uint8)

mask_circles = np.zeros_like(output, np.uint8)
mask_circles = mask_circles[:,:,0]

coins_map = {
    "10c" : {
        "radio" : (120,155),
        "color" : (255,0,0),
        "valor" : 0.10,
        "name" : "10 centavos",
        "mask" : []
    },
    "50c" : {
        "radio" : (170,200),
        "color" : (0,255,0),
        "valor" : 0.50,
        "name" : "50 centavos",
        "mask" : []
    },
    "1p" : {
        "radio" : (155,172),
        "color" : (0,0,255),
        "valor" : 1.0,
        "name" : "1 peso",
        "mask" : []
    }
}

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (255, 255, 255)
thickness = 10

for coin in circles[0]:
    x_coor, y_coor, detected_radius = coin
    x_coor, y_coor, detected_radius = int(x_coor), int(y_coor), int(detected_radius)

    key = None
    if detected_radius > coins_map["10c"]["radio"][0] and detected_radius <= coins_map["10c"]["radio"][1]:
        key = "10c"
    elif detected_radius > coins_map["1p"]["radio"][0] and detected_radius <= coins_map["1p"]["radio"][1]:
        key = "1p"
    elif detected_radius > coins_map["50c"]["radio"][0] and detected_radius <= coins_map["50c"]["radio"][1]:
        key = "50c"
    else:
        continue

    coins_map[key]["mask"].append([x_coor, y_coor, detected_radius])

    output = cv2.circle(output, (x_coor, y_coor), detected_radius, coins_map[key]["color"], thickness)
    mask = np.zeros_like(output, np.uint8)
    mask = cv2.circle(mask, (x_coor, y_coor), detected_radius, [255]*3, thickness=cv2.FILLED)
    mask = mask[:,:,0]
    mask_circles += mask
    color_src = np.full_like(output, fill_value=coins_map[key]["color"])
    result = cv2.bitwise_and( color_src, output , dst=result, mask=mask)
    cv2.putText(result, coins_map[key]["name"], (x_coor, y_coor), font,  fontScale, fontColor, thickness, cv2.LINE_AA)

imshow(output, title='Clasificacion', colorbar=False)
imshow(result, title='Conteo', colorbar=False)

##### Contabilizar valores de monedas
def contar_y_sumar_valores(coins_map):
    suma_total = 0

    for clase, detalles in coins_map.items():
        cantidad_recortes = len(detalles["mask"])
        valor_individual = cantidad_recortes * detalles["valor"]
        print(f"{clase}: {cantidad_recortes} recortes, Valor individual: {valor_individual}")
        suma_total += valor_individual

    return suma_total

suma_total = contar_y_sumar_valores(coins_map)
print(f"Suma total de valores: {suma_total}")




###### Reconocieminto de dados
img_copy=img.copy()

for c in coins_map.values():
    for x_coor, y_coor, detected_radius in c["mask"]:

        cv2.circle(canny_img, (x_coor, y_coor), detected_radius+50, [0]*3, thickness=-1)

dilate_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
mask_dices = cv2.dilate(canny_img, kernel, iterations=1)
imshow(mask_dices)


###### clasificaion de dados
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dices, connectivity=8)

# Crear una copia de la máscara para no modificar la original
mask_dices_copy = mask_dices.copy()

# Iterar sobre cada componente conectado y dibujar un rectángulo rojo
for i in range(1, n_labels):  # Empezar desde 1 para evitar el fondo (label 0)
    x, y, w, h, area = stats[i]

    # Dibujar un rectángulo rojo alrededor del componente conectado
    cv2.rectangle(mask_dices_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)

# Mostrar la imagen resultante con los rectángulos rojos
cv2.imshow('Componentes Conectados', mask_dices_copy)










np.unique(mask_dices)
dices = []
final_mask_dices =  np.zeros_like(mask_dices, np.uint8) 
final_mask_dice_values = np.zeros_like(mask_dices, np.uint8)  

for i in range(1, n_labels):
    area = stats[i, cv2.CC_STAT_AREA]  

    if area > 3000 :

        x1 = stats[i, cv2.CC_STAT_LEFT] 
        y1 = stats[i, cv2.CC_STAT_TOP] 
        w = stats[i, cv2.CC_STAT_WIDTH] 
        h = stats[i, cv2.CC_STAT_HEIGHT]        
        p1 = (x1,y1)
        p2 = (x1+w, y1+h)
        
        if area> 10000:
           #Only append dices
            dices.append({"coord": (p1,p2) , "img": mask_dices[y1:y1+h, x1:x1+w]})
            img_rgb_copy= cv2.rectangle( img_rgb_copy, p1, p2, (0,255,0), thickness=10)
            final_mask_dices = cv2.rectangle( final_mask_dices, p1, p2, [255]*3, thickness=-1)
        elif area>3700 and area<4200:
            img_rgb_copy= cv2.rectangle( img_rgb_copy, p1, p2, (0,255,0), thickness=10)
            final_mask_dice_values = cv2.rectangle( final_mask_dice_values, p1, p2, [255]*3, thickness=-1)

plt.imshow(final_mask_dices)
plt.show()
















color_mark_mask = np.full_like(img_rgb_copy, fill_value=(0,255,255))
final_result_dices = cv2.bitwise_and(color_mark_mask,img_rgb_copy,  mask=final_mask_dices)

sub=231
kernel_dice = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
for dice in dices:

    #dice["img"] = cv2.morphologyEx(dice["img"], cv2.MORPH_OPEN, kernel_dice, iterations=1)
    #dice["img"] = cv2.morphologyEx(dice["img"], cv2.MORPH_CLOSE, kernel_dice,iterations=1)

    vals =  cv2.HoughCircles(
            dice["img"],  # source image (blurred and grayscaled)
            cv2.HOUGH_GRADIENT,  # type of detection
            1.5,  # inverse ratio of accumulator res. to image res.
            12,  # minimum distance between the centers of circles
            param1=255,  # Gradient value passed to edge detection
            param2=30, # accumulator threshold for the circle centers
            minRadius=30,
            maxRadius=40
        )

    for val in vals[0] :
        x_coor, y_coor, detected_radius = val
        x_coor, y_coor, detected_radius = int(x_coor), int(y_coor), int(detected_radius)
        dice["img"] = cv2.circle(dice["img"], (x_coor, y_coor), detected_radius, (100,0,0), thickness=cv2.FILLED)

    dice_value = len(vals[0])
    cv2.putText( final_result_dices, f"Value {dice_value}", dice["coord"][0], font,  fontScale, fontColor, thickness, cv2.LINE_AA)
    cv2.putText( img_rgb_copy, f"Value {dice_value}", dice["coord"][0], font,  fontScale, fontColor, thickness, cv2.LINE_AA)

    sub+=1
    plt.subplot(sub)
    plt.imshow(dice["img"], cmap="gray")

plt.figure(figsize=(30,30))

plt.subplot(242)
plt.imshow(final_result_dices)