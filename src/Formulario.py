import numpy as np
class Formulario:
    
    img: np.ndarray
    def __init__(self, img: np.ndarray):
        """
        Parametros:
            img: Arreglo de numpy en uint8
        """
        self.img = img
        
    def escanear(self):
        """
        Trabaja con la imágen y determina la estructura del formulario.
        Retorna un json con la siguiente estructura:
        {
            nombre: np.ndarray,
            edad: np.ndarray,
            mail: np.ndarray,
            legajo: np.ndarray,
            preg1: {
                si: np.ndarray,
                no: np.ndarray
            },
            preg2: {
                si: np.ndarray,
                no: np.ndarray
            },
            preg3: {
                si: np.ndarray,
                no: np.ndarray
            },
            comentario: np.ndarray
        }
        """
        filas = []
        formulario = {}
        #Obtener posiciones de las filas
        imgumbral = self.img <= 150 #Todos los píxeles menores a 150 son considerados negros.
        imgfilas = np.diff(imgumbral, axis=0)
        imgfilassum = np.sum(imgfilas, 1)
        imgfilas_bool = imgfilassum > 900 #Las filas que tengan más de 900 píxeles negros son consideradas divisoras del formulario
        imgfilas_idxs = np.argwhere(imgfilas_bool) #Obtener índices de las filas
        imgfilas_diff_idxs = np.diff(imgfilas_idxs, axis=0) #Si la diferencia entre índices es menor o igual a diez nos quedamos sólo con una. Ejemplo: [[22, 24, 50, 100]] -> [24, 50, 100]
        indices_a_eliminar = np.where(imgfilas_diff_idxs <= 10)[0]
        imgfilas_idxs_validos = np.delete(imgfilas_idxs, indices_a_eliminar, axis=0)                                                                                                                    #Fila 1   #Fila 2
        filas_y = np.vstack((imgfilas_idxs_validos.ravel()[:-1], imgfilas_idxs_validos.ravel()[1:])).T #Agrupamos los índices de a dos para determinar el rango de la fila. Ejemplo: [24, 50, 100] -> [[24, 50], [50, 100]]
        for i, item in enumerate(filas_y):
            filas.append({
                "img": self.img[item[0]:item[1]],
                "columnas": {}
            })
        #Obtener posiciones de las columnas
        for i, item in enumerate(filas):
            fila = item["img"]
            fila_umbral = fila <= 150
            imgcols = np.diff(fila_umbral, axis=1)
            imgcolssum = np.sum(imgcols, 0)
            imgcols_bool = imgcolssum >= 30 #Columnas que tengan más de 30 píxeles negros son consideradas divisoras del formulario
            imgcols_idxs = np.argwhere(imgcols_bool) #Obtener índices de las columnas
            imgcols_diff_idxs = np.diff(imgcols_idxs, axis=0) #Si la dif entre índices es menor a igual a 2 nos quedamos cólo con una
            indices_a_eliminar = np.where(imgcols_diff_idxs <= 2)[0]
            imgcols_idxs_validos = np.delete(imgcols_idxs, indices_a_eliminar, axis=0)
            columnas_x = np.vstack((imgcols_idxs_validos.ravel()[:-1], imgcols_idxs_validos.ravel()[1:])).T #Agrupamos los índices de a dos para determinar el rango de la columna
            #Iteramos sobre las columnas y en base al índice determinamos a qué parte del formulario pertenece
            """
            ---------------------------------------
            |           FORMULARIO A i = 0        |
            ---------------------------------------
            | Nombre i=1,j=0 | Juan Perez i=1,j=1 |
            ---------------------------------------
            | Edad i=2,j=0   | 45 i=2,j=0         |
            ---------------------------------------
            """
            for j, columna in enumerate(columnas_x):
                img_col = fila_umbral[:, columna[0]:columna[1]] * 255
                if i == 0: continue
                if i == 1:
                    if j == 1:
                        formulario["nombre"] = img_col
                elif i == 2:
                    if j == 1:
                        formulario["edad"] = img_col
                elif i == 3:
                    if j == 1:
                        formulario["mail"] = img_col
                elif i == 4:
                    if j == 1:
                        formulario["legajo"] = img_col
                elif i == 6:
                    if j == 1:
                        formulario["preg1"] = {"si": img_col}
                    elif j == 2:
                        formulario["preg1"]["no"] = img_col
                elif i == 7:
                    if j == 1:
                        formulario["preg2"] = {"si": img_col}
                    elif j == 2:
                        formulario["preg2"]["no"] = img_col
                elif i == 8:
                    if j == 1:
                        formulario["preg3"] = {"si": img_col}
                    elif j == 2:
                        formulario["preg3"]["no"] = img_col
                elif i == 9:
                    if j == 1:
                        formulario["comentario"] = img_col
        return formulario