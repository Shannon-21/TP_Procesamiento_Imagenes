from src.Detector import Detector

Detector().detectar()


# import cv2
# import numpy as np
# from time import sleep

# cap = cv2.VideoCapture('src/videos/tirada_1.mp4')
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# HEIGHT = int(height / 3)
# WIDTH = int(width / 3)

# prevframe = None
# while cap.isOpened():
#     ret, frame = cap.read()

#     if ret==True:
#         # Capturamos el frame y lo convertimos en gris
#         # frame = cv2.resize(frame, dsize=(WIDTH, HEIGHT))
#         framegray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

#         #Cortar parte que no es interesante
#         framegray = framegray[:HEIGHT - 600, :]

#         #Poner en binario
#         fabiosort = cv2.GaussianBlur(framegray, (3, 3), 1)
#         _, framebin = cv2.threshold(fabiosort, 100, 255, cv2.THRESH_BINARY)

#         framebin = ~framebin

#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#         else:
#             if type(prevframe) != "NoneType" and type(framebin) != "NoneType":
#                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(framebin, 4, cv2.CV_32S)
#                 #Detectar cubos
#                 # print([stat[4] for stat in stats])
#                 areas = [(i, stat) for i, stat in enumerate(stats) if stat[4] <= 15000 and stat[4] >= 1000]
#                 print(len(areas))
#                 for stat in areas: #Por cada cubo detectar los círculos internos
#                     arriba, izquierda, alto, ancho, area = stat[1]
#                     i = stat[0]
#                     cuadrado = ~framebin[izquierda : izquierda + ancho, arriba: arriba + alto]
#                     # kernel_ero = np.ones((2, 2), np.uint8)
#                     # cuadrado_ero = cv2.erode(cuadrado, kernel_ero)
#                     num_labels_c, labels_c, stats_c, centroids_c = cv2.connectedComponentsWithStats(cuadrado, 4, cv2.CV_32S)
#                     for stat_c in stats_c:
#                         a, i, al, an, ar = stat_c
#                         if ar <= 10 and ar >= 5:
#                             a = cv2.rectangle(cuadrado, (a, i), (a + al, i + an), (255, 0, 0))
#                             # cv2.imshow("Cara", cuadrado)
#                             # sleep(0.5)
#                     a = cv2.rectangle(frame, (arriba, izquierda), (arriba + alto, izquierda + ancho), (255, 0, 0)) #Dibujar rectánculo en la posición del cubo
#                 prevframe = framegray
#                 cv2.imshow('Frame', frame)
#                 #out.write(frame)
#                 continue
#     else:
#         break

# cv2.destroyAllWindows()