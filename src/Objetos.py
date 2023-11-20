import cv2
import numpy as np
import matplotlib.pyplot as plt


class Objetos:
    def __init__(self) -> None:
        self.IMG = "imagenes/ejercicio1/monedas.jpg"

    def detectar(self):
        img = cv2.imread(self.IMG, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img_rgb)
        plt.title('Imagen Original')
        plt.show()
        
        
        ### Escala de grises
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        f_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0, sigmaY=0)
        plt.imshow(f_blur, cmap="gray")
        plt.title('Esclada de grises')
        plt.show()
        
        
        ### deteccion con Sobel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        sobel_map = {}
        sobel_map["edge_sobel_x"] = cv2.Sobel(f_blur, -1, 1, 0, ksize=3)
        axes[0].imshow(sobel_map["edge_sobel_x"], cmap="gray")
        axes[0].set_title('edge_sobel_x')
        sobel_map["edge_sobel_y"] = cv2.Sobel(f_blur, -1, 0, 1, ksize=3)
        axes[1].imshow(sobel_map["edge_sobel_y"], cmap="gray")
        axes[1].set_title('edge_sobel_y')
        sobel_map["edge_sobel_combined"] = cv2.addWeighted(sobel_map["edge_sobel_x"], 0.5, sobel_map["edge_sobel_y"], 0.5, 0)
        axes[2].imshow(sobel_map["edge_sobel_combined"], cmap="gray")
        axes[2].set_title('edge_sobel_combined')
        
        fig.show()
        
        ### Sobel con umbralado
        _, sobel_combined_thresh = cv2.threshold(sobel_map["edge_sobel_combined"], 30, 255, cv2.THRESH_BINARY)
        plt.imshow(sobel_combined_thresh, cmap='gray')
        plt.title('Sobel umbralado')
        plt.show()
        
        
        ### Canny
        canny_img = cv2.Canny(f_blur, 35, 90, apertureSize=3, L2gradient=True)
        plt.imshow(canny_img, cmap='gray')
        plt.title('Canny')
        plt.show()
        
        
        ### transformaciones morfologicas
        fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
        morph = {}
        morph["apertura"] = cv2.morphologyEx(canny_img, cv2.MORPH_OPEN, kernel)
        axes[0, 0].imshow(morph["apertura"], cmap="gray")
        axes[0, 0].set_title('apertura')
        
        morph["clausura"] = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
        axes[0, 1].imshow(morph["clausura"], cmap="gray")
        axes[0, 1].set_title('clausura')
        
        morph["gradiente"] = cv2.morphologyEx(canny_img, cv2.MORPH_GRADIENT, kernel)
        axes[0, 2].imshow(morph["gradiente"], cmap="gray")
        axes[0, 2].set_title('gradiente')
        
        morph["gradiente apertura"] = cv2.morphologyEx(morph["apertura"], cv2.MORPH_GRADIENT, kernel)
        axes[1, 0].imshow(morph["gradiente apertura"], cmap="gray")
        axes[1, 0].set_title('gradiente apertura')
        
        morph["gradiente clausura"] = cv2.morphologyEx(morph["clausura"], cv2.MORPH_GRADIENT, kernel)
        axes[1, 1].imshow(morph["gradiente clausura"], cmap="gray")
        axes[1, 1].set_title('gradiente clausura')
        
        fig.delaxes(axes[1, 2])
        
        fig.tight_layout()
        plt.show()
        
        
        ### Diltacion
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        canny_dilate_circles = cv2.dilate(morph["clausura"],dilate_kernel,iterations = 3)
        
        plt.imshow(canny_dilate_circles, cmap='gray')
        plt.title('Clausura dilatado')
        plt.show()
        
        
        ### deteccion de circulos
        coins =  cv2.HoughCircles(
                    canny_dilate_circles, 
                    cv2.HOUGH_GRADIENT, 
                    1.7, 
                    300, 
                    param1=255, 
                    param2=100,
                    minRadius=50,
                    maxRadius=200
                )
        
        circles = np.uint16(np.around(coins))
        
        img_circles = img.copy()
        for i in circles[0,:]:
            cv2.circle(img_circles,(i[0],i[1]),i[2],(255,0,0),20)
            cv2.circle(img_circles,(i[0],i[1]),2,(0,0,255),10)
        
        plt.imshow(img_circles)
        plt.title('Deteccion de monedas')
        plt.show()
        
        
        ### clasificacion de monedas
        coins_map = {
            "10c" : {
                "radio" : (120,145),
                "color" : (255,0,0),
                "valor" : 0.10,
                "name" : "10 centavos",
                "mask" : []
            },
            "1p" : {
                "radio" : (145,170),
                "color" : (0,255,0),
                "valor" : 1.0,
                "name" : "1 peso",
                "mask" : []
            },
            "50c" : {
                "radio" : (170,500),
                "color" : (0,0,255),
                "valor" : 0.50,
                "name" : "50 centavos",
                "mask" : []
            }
        }
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        fontColor = (255, 255, 255)
        thickness = 10
        
        # copia para almacenar
        output = img_rgb.copy()
        result = np.zeros_like(output, np.uint8)
        
        mask_circles = np.zeros_like(output, np.uint8)
        mask_circles = mask_circles[:,:,0]
        
        # clasificar cada circulo
        for coin in coins[0]:
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
        
            output = cv2.circle(output, (x_coor, y_coor), detected_radius, coins_map[key]["color"], thickness=20)
            
            mask = np.zeros_like(output, np.uint8)
            mask = cv2.circle(mask, (x_coor, y_coor), detected_radius, [255]*3, thickness=cv2.FILLED)
            mask = mask[:,:,0]
            mask_circles += mask
            
            color_src = np.full_like(output, fill_value=coins_map[key]["color"])
            result = cv2.bitwise_and( color_src, output , dst=result, mask=mask)
            
            cv2.putText(result, coins_map[key]["name"], (x_coor, y_coor), font,  fontScale, fontColor, thickness, cv2.LINE_AA)
        
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        
        axes[0].imshow(output)
        axes[0].set_title('Clasificacion')
        
        axes[1].imshow(result)
        axes[1].set_title('Conteo')
        
        plt.tight_layout()
        plt.show()
        
        ### Contabilizar valores de monedas
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
        
        
        ### segmentacion dados
        mask_dices = morph["clausura"]
        img_rgb_copy=img_rgb.copy()
        
        for c in coins_map.values():
           for x_coor, y_coor, detected_radius in c["mask"]:
            cv2.circle(mask_dices, (x_coor, y_coor), detected_radius+50, [0]*3, thickness=-1)
        
        dilate_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
        mask_dices = cv2.dilate(mask_dices,kernel,iterations = 10)
        plt.imshow(mask_dices, cmap='gray')
        plt.title('Segmentacion Dados')
        plt.show()
        
        
        ### clasificacion
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dices, connectivity=8)
        
        dices = []
        final_mask_dices =  np.zeros_like(mask_dices, np.uint8)
        final_mask_dice_values = np.zeros_like(mask_dices, np.uint8)
        
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
        
            if area > 3000 : # se deshace de los circulo laterales y otros elementos
        
                x1 = stats[i, cv2.CC_STAT_LEFT]
                y1 = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                p1 = (x1,y1)
                p2 = (x1+w, y1+h)
        
                if area > 10000: # es el cuadrado del dado
                    dices.append({"coord": (p1,p2) , "img": mask_dices[y1:y1+h, x1:x1+w]})
                    img_rgb_copy= cv2.rectangle( img_rgb_copy, p1, p2, (0,255,0), thickness=10)
                    final_mask_dices = cv2.rectangle( final_mask_dices, p1, p2, [255]*3, thickness=-1)
                
                elif area > 3700 and area < 4200: # es un circulo
                    img_rgb_copy= cv2.rectangle( img_rgb_copy, p1, p2, (255,0,0), thickness=10)
                    final_mask_dice_values = cv2.rectangle( final_mask_dice_values, p1, p2, [255]*3, thickness=-1)
        
        plt.imshow(img_rgb_copy)
        plt.title('Deteccion dados')
        plt.show()
        
        
        ### deteccion de puntos
        color_mark_mask = np.full_like(img_rgb_copy, fill_value=(0,255,255))
        final_result_dices = cv2.bitwise_and(color_mark_mask,img_rgb_copy,  mask=final_mask_dices)
        
        kernel_dice = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        for dice in dices:
        
            vals =  cv2.HoughCircles(
                    dice["img"],
                    cv2.HOUGH_GRADIENT,
                    1.5,
                    12, 
                    param1=255,
                    param2=30,
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
        
        plt.imshow(final_result_dices)
        plt.title('segmentacion Dados')
        plt.show()
        
        
        
        
        # Crear una nueva imagen para mostrar monedas y dados
        combined_image = np.zeros_like(img_rgb, np.uint8)
        
        # Mostrar monedas en la imagen combinada
        for coin_key, coin_info in coins_map.items():
            for x_coor, y_coor, detected_radius in coin_info["mask"]:
                cv2.circle(combined_image, (x_coor, y_coor), detected_radius, coin_info["color"], thickness=20)
                cv2.putText(combined_image, coin_info["name"], (x_coor, y_coor), font, fontScale, fontColor, thickness, cv2.LINE_AA)
        
        # Mostrar dados en la imagen combinada
        for dice in dices:
            p1, p2 = dice["coord"]
            cv2.rectangle(combined_image, p1, p2, (0, 255, 0), thickness=10)
        
            vals = cv2.HoughCircles(
                dice["img"],
                cv2.HOUGH_GRADIENT,
                1.5,
                12,
                param1=255,
                param2=30,
                minRadius=30,
                maxRadius=40
            )
        
            for val in vals[0]:
                x_coor, y_coor, detected_radius = val
                x_coor, y_coor, detected_radius = int(x_coor + p1[0]), int(y_coor + p1[1]), int(detected_radius)
                combined_image = cv2.circle(combined_image, (x_coor, y_coor), detected_radius, (255, 255, 0), thickness=cv2.FILLED)
        
            dice_value = len(vals[0])
            cv2.putText(combined_image, f"Value {dice_value}", p1, font, fontScale, fontColor, thickness, cv2.LINE_AA)
        
        
        # Crear una nueva imagen para mostrar monedas, dados y la imagen original
        combined_image_with_original = np.zeros_like(img_rgb, np.uint8)
        
        # Superponer la imagen original de manera clara
        cv2.addWeighted(img_rgb, 0.4, combined_image, 0.6, 0, combined_image_with_original)
        
        # Mostrar la imagen combinada con la imagen original
        plt.imshow(combined_image_with_original)
        plt.title('Monedas y Dados')
        plt.show()
