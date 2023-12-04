import cv2
from .Dado import Dado
from typing import List
import os
import tqdm


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

    FPD = 18
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
        for i, v in tqdm.tqdm(enumerate(self.VIDEOS), total=len(self.VIDEOS), desc="Procesando videos", unit="video"):
            prev_dados: List[Dado] = []
            cap = cv2.VideoCapture(v)
            if not cap.isOpened():
                raise Exception("Error al abrir video")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not os.path.exists("out"):
                os.mkdir("out")

            video_out = cv2.VideoWriter(f"out/Video_{i + 1}.mp4", cv2.VideoWriter.fourcc(*"mp4v"), fps,
                                        (width // self.ESCALADO_VISUAL, height // self.ESCALADO_VISUAL))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                org_frame = frame.copy()

                frame[:, :, 0] = 0
                frame[:, :, 1] = 0
                t, frame_binario = cv2.threshold(frame[:, :, 2], 80, 255, cv2.THRESH_BINARY)
                n_labels, _, stats, _ = cv2.connectedComponentsWithStats(frame_binario, connectivity=8)

                dados: List[Dado] = []
                for j in range(1, n_labels):
                    x1 = stats[j, cv2.CC_STAT_LEFT]
                    y1 = stats[j, cv2.CC_STAT_TOP]
                    w = stats[j, cv2.CC_STAT_WIDTH]
                    h = stats[j, cv2.CC_STAT_HEIGHT]
                    area = stats[j, cv2.CC_STAT_AREA]
                    if self.__es_dado(area, w, h):
                        dado = Dado(org_frame[y1:y1 + h, x1:x1 + w], x1, y1, w, h)
                        dados.append(dado)

                for d in dados:
                    for pd in prev_dados:
                        if d.es_misma_posicion(pd):
                            org_frame = d.dibujar(org_frame)

                if self.cont_frame == self.FPD:
                    prev_dados = dados
                    self.cont_frame = 0
                self.cont_frame += 1

                resized = cv2.resize(org_frame, (width // self.ESCALADO_VISUAL, height // self.ESCALADO_VISUAL))
                video_out.write(resized)

                if cv2.waitKey(81) & 0xFF == ord('q'):
                    break
        video_out.release()

    def __es_dado(self, area: float, w: float, h: float):
        aspect_ratio = h / w
        return area > 3700 and area < 5500 and aspect_ratio > 0.7 and aspect_ratio < 1.2
