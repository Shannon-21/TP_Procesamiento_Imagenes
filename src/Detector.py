import cv2
from .Dado import Dado
from typing import List
import os
import tqdm
from matplotlib import pyplot as plt


class Detector:
    VIDEOS = [
        "src/videos/tirada_1.mp4",
        "src/videos/tirada_2.mp4",
        "src/videos/tirada_3.mp4",
        "src/videos/tirada_4.mp4",
    ]
    """
    Arreglo de videos
    """

    FPD = 19
    """
    (Fotogramas Por Distancia) Cantidad de fotogramas que deben pasar para medir la distancia entre los dados
    """

    ESCALADO_VISUAL = 3
    """
    Por cuánto escalar la imágen para achicar/aumentar su tamaño
    """

    def __init__(self) -> None:
        self.cont_frame = 0

    def detectar(self):
        """
        Itera sobre cada imágen y detecta los dados con sus números
        """
        flag_visualizacion = 0 # para determinar un frame que muestre un recorte de los dados
        
        # iteracion de cada video
        for i, v in tqdm.tqdm(enumerate(self.VIDEOS), total=len(self.VIDEOS), desc="Procesando videos", unit="video"):
            # para almacenar los recortes de los dados
            prev_dados: List[Dado] = []
            
            # inicializacion del video
            cap = cv2.VideoCapture(v)
            if not cap.isOpened():
                raise Exception("Error al abrir video")
            
            # obtencion de atributos del video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # para almacenar los videos resultantes
            if not os.path.exists("out"):
                os.mkdir("out")

            # se formatea los atributos del video de salida
            video_out = cv2.VideoWriter(f"out/Video_{i + 1}.mp4", cv2.VideoWriter.fourcc(*"mp4v"), fps,
                                        (width // self.ESCALADO_VISUAL, height // self.ESCALADO_VISUAL))

            # reproducir el video para obtener cada frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                org_frame = frame.copy()

                # Para visualización de paso intermedio en el frame 80
                flag_visualizacion += 1
        
                # se saturan los canales verde y azul
                frame[:, :, 0] = 0
                frame[:, :, 1] = 0

                # se binariza la imagen mediante umbral, y se obtienen sus componentes conectados
                t, frame_binario = cv2.threshold(frame[:, :, 2], 80, 255, cv2.THRESH_BINARY)
                n_labels, _, stats, _ = cv2.connectedComponentsWithStats(frame_binario, connectivity=8)
                
                # Para visualización de paso intermedio en el frame 80
                if flag_visualizacion == 80:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
                    axs[0].imshow(frame)
                    axs[0].set_title('Filtro de canal rojo')
                    axs[1].imshow(frame_binario, cmap='gray')
                    axs[1].set_title('Imagen binarizada')
                    plt.show()

                # para almacenar los dados en cada frame
                dados: List[Dado] = []

                # de cada componente obtenner sus caracteristicas
                for j in range(1, n_labels):
                    x1 = stats[j, cv2.CC_STAT_LEFT]
                    y1 = stats[j, cv2.CC_STAT_TOP]
                    w = stats[j, cv2.CC_STAT_WIDTH]
                    h = stats[j, cv2.CC_STAT_HEIGHT]
                    area = stats[j, cv2.CC_STAT_AREA]

                    # se verifica que es un dado, dadas su caracteristicas
                    if self.__es_dado(area, w, h):
                        # se guarda el recorte de imagen del dado
                        dado = Dado(org_frame[y1:y1 + h, x1:x1 + w], x1, y1, w, h)
                        dados.append(dado)

                # itera sobre las imagenes y las previas a la actual
                for d in dados:
                    for pd in prev_dados:
                        if d.es_misma_posicion(pd): # comparar el movimiento
                            org_frame = d.dibujar(org_frame) # dibujar sobre el dado el recuadro

                # reinicia las variables cuando se han iterado todos los frames
                if self.cont_frame == self.FPD:
                    prev_dados = dados
                    self.cont_frame = 0
                self.cont_frame += 1

                # escala el tamanio de la imagen
                resized = cv2.resize(org_frame, (width // self.ESCALADO_VISUAL, height // self.ESCALADO_VISUAL))

                # Para visualización de paso intermedio en el frame 80
                if flag_visualizacion == 80:
                    plt.imshow(resized)
                    plt.title('Dados detectados')
                    plt.show()

                # unir el frame transformado al video final
                video_out.write(resized)
                
                # salir del buclue si se preciona 'q' por mas de 81 milisegundos
                if cv2.waitKey(81) & 0xFF == ord('q'):
                    break

        # cerrar la reporduccion del video
        video_out.release()

    def __es_dado(self, area: float, w: float, h: float):
        ''' Metodo para reconocer un dado por su area y relacion de aspecto '''
        aspect_ratio = h / w # definido como altura sobre ancho
        # si cumple los criterios, retorna que es un dado
        return area > 3700 and area < 5500 and aspect_ratio > 0.7 and aspect_ratio < 1.2
