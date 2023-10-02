import numpy as np
import matplotlib.pyplot as plt

class Formulario:
    
    img: np.ndarray
    mapeo = {
        "nombre": "Nombre y apellido",
        "edad": "Edad",
        "mail": "Mail",
        "legajo": "Legajo",
        "preg1": "Pregunta 1",
        "preg2": "Pregunta 2",
        "preg3": "Pregunta 3",
        "comentario": "Comentario"
    }
    
    def __init__(self, img: np.ndarray):
        """
        Parametros:
            img: Arreglo de numpy en uint8
        """
        self.img = img
        
    def _escanear(self):
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
        Todos los arreglos de numpy son binarios, esto quiere decir que sólo poseen dos valores, 0 para negroy 255 para blanco
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
            | Edad i=2,j=0   | 45 i=2,j=1         |
            ---------------------------------------
            """
            for j, columna in enumerate(columnas_x):
                ARRIBA = 2 #Se agrega padding para evitar tener los bordes de las celdas
                ABAJO = fila_umbral.shape[0] - 1
                IZQUIERDA = columna[0] + 1
                DERECHA = columna[1]
                img_col = fila_umbral[ARRIBA:ABAJO, IZQUIERDA:DERECHA] * 255
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
    
    def validate_form(self):
        """
        Esta función toma un diccionario con los recortes de cada campo de un formulario.
        Usa las funciones existentes para evaluar los criterios especificados para cada campo.
        Argumentos:
            data: Un diccionario con los recortes de cada campo del formulario.
        Salida:
            Un diccionario con los resultados de la validación.
        """
        
        data = self._escanear()
        
        criteria = {
            'nombre': {'min_words': 2, 'max_words': float('inf'), 'min_chars': 1, 'max_chars': 25},
            'edad': {'min_words': 1, 'max_words': 1, 'min_chars': 2, 'max_chars': 3},
            'mail': {'min_words': 1, 'max_words': 1, 'min_chars': 1, 'max_chars': 25},
            'legajo': {'min_words': 1, 'max_words': 1, 'min_chars': 8, 'max_chars': 8},
            'preg1': {'min_words': 1, 'max_words': 1, 'min_chars': 1, 'max_chars': 1},
            'preg2': {'min_words': 1, 'max_words': 1, 'min_chars': 1, 'max_chars': 1},
            'preg3': {'min_words': 1, 'max_words': 1, 'min_chars': 1, 'max_chars': 1},
            'comentario': {'min_words': 1, 'max_words': float('inf'), 'min_chars': 1, 'max_chars': 25},
        }

        for key in data.keys():
            if key.startswith('preg'):
                si = self.validate_text(data[key]['si'] == 0, **criteria[key])
                no = self.validate_text(data[key]['no'] == 0, **criteria[key])
                result = 'OK' if (si and not no) or (no and not si) else 'MAL'
            else:
                result = 'OK' if self.validate_text(data[key] == 0, **criteria[key]) else 'MAL'
            print(self.mapeo[key], result)

    def count_consecutive_values(self, arr):
        """
        Esta función toma una lista de valores booleanos y devuelve una lista de tuplas.
        Cada tupla contiene un valor booleano y la cantidad de veces consecutivas que aparece en la lista.
        Argumentos:
            arr: Una lista de valores booleanos.
        Salida:
            Una lista de tuplas. Cada tupla contiene un valor booleano y la cantidad de veces consecutivas que aparece en la lista.
        """

        result = []
        current_value = arr[0]
        current_count = 1

        for i in range(1, len(arr)):
            if arr[i] == current_value:
                current_count += 1
            else:
                result.append((current_value, current_count))
                current_value = arr[i]
                current_count = 1

        result.append((current_value, current_count))
        return result


    def replace_consecutive_false(self, arr, threshold):
        """
        Esta función toma una lista de valores booleanos y un umbral.
        Reemplaza los valores False consecutivos por debajo del umbral con True.
        Argumentos:
            arr: Una lista de valores booleanos.
            threshold: Un umbral para determinar cuántos valores False consecutivos se deben reemplazar con True.
        Salida:
            Una lista de valores booleanos con los valores False consecutivos por debajo del umbral reemplazados por True.
        """

        arrm = self.count_consecutive_values(arr)
        new_counts = []

        first = arrm[0]
        last = arrm[-1]

        for i in range(1, len(arrm) - 1):
            value, count = arrm[i]

            if not value and count < threshold:
                new_counts.append((True, count))
            else:
                new_counts.append(arrm[i])

        result = [first] + new_counts + [last]
        result_array = np.array([value for value, count in result for _ in range(count)])
        return result_array

    def count_paragraphs(self, arr):
        """
        Esta función toma una lista de valores booleanos y cuenta el número de 'párrafos'.
        Un 'párrafo' se define como una secuencia de valores True consecutivos.
        Argumentos:
            arr: Una lista de valores booleanos.
        Salida:
            El número de 'párrafos' en la lista.
        """

        arrm = self.count_consecutive_values(arr)
        paragraph_counts = [count for value, count in arrm if value]

        return len(paragraph_counts)


    def count_elements(self, img, axis, threshold):
        """
        Esta función toma una imagen, un eje y un umbral.
        Cuenta el número de 'elementos' en la imagen a lo largo del eje especificado.
        Un 'elemento' se define como una secuencia de píxeles blancos (True) consecutivos que son más largos que el umbral.
        Argumentos:
            img: Una imagen en escala de grises.
            axis: El eje a lo largo del cual contar los 'elementos'.
            threshold: Un umbral para determinar qué secuencias de píxeles blancos se consideran 'elementos'.
        Salida:
            El número de 'elementos' en la imagen a lo largo del eje especificado.
        """

        img_zeros = img == 0
        img_sum = img_zeros.any(axis=axis)

        modified = self.replace_consecutive_false(img_sum, threshold)
        num_elements = self.count_paragraphs(modified)

        return num_elements


    def validate_text(self, img, min_chars=0, max_chars=float('inf'), min_words=0, max_words=float('inf'), axis=0, char_threshold=1, word_threshold=10):
        """
        Esta función toma una imagen unit8 y varios parámetros.
        Verifica si el texto en la imagen cumple con los criterios especificados en términos de número de caracteres y palabras.
        Argumentos:
            img: Una imagen en escala de grises.
            min_chars: El número mínimo de caracteres que debe tener el texto.
            max_chars: El número máximo de caracteres que puede tener el texto.
            min_words: El número mínimo de palabras que debe tener el texto.
            max_words: El número máximo de palabras que puede tener el texto.
            axis: El eje a lo largo del cual contar los caracteres y las palabras.
            char_threshold: Un umbral para determinar qué secuencias de píxeles blancos se consideran 'caracteres'.
            word_threshold: Un umbral para determinar qué secuencias de píxeles blancos se consideran 'palabras'.
        Salida:
            Verdadero si el texto en la imagen cumple con los criterios especificados, Falso en caso contrario.
        """    

        num_words = self.count_elements(img, axis, word_threshold)
        num_chars = self.count_elements(img, axis, char_threshold) + num_words - 1 # Espacioes en blanco = num_words - 1

        if min_chars <= num_chars <= max_chars and min_words <= num_words <= max_words:
            return True
        else:
            return False
        
    def show_segmentation(self, arr, threshold, char_threshold=1, word_threshold=10):
        """
        Esta función toma una imagen y un umbral.
        Dibuja líneas rojas para indicar la separación entre elementos (palabras, caracteres, dependiendo del umbral)
        Argumentos:
            arr: Una imagen en escala de grises.
            threshold: Un umbral para determinar la separación entre elementos de interes.
            char_threshold: Un umbral para determinar qué secuencias de píxeles blancos se consideran 'caracteres'.
            word_threshold: Un umbral para determinar qué secuencias de píxeles blancos se consideran 'palabras'.
        Salida:
            Una visualización de la imagen con líneas rojas indicando la separación entre elementos.
        """

        img_pz = arr.any(axis=0)

        modified = self.replace_consecutive_false(img_pz, threshold)

        x = np.diff(modified)
        indxs = np.argwhere(x)

        ii = np.arange(0, len(indxs), 2)
        indxs[ii] += 1

        xx = np.arange(arr.shape[1])
        yy = np.zeros(arr.shape[1])
        yy[indxs] = (arr.shape[0] - 1)

        plt.imshow(arr, cmap='gray')
        plt.plot(xx, yy, c='r')
        plt.show()
