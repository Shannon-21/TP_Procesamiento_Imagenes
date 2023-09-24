import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2


def normalize(img):
    """
    Esta función toma una imagen en escala de grises y devuelve una versión normalizada y binarizada de la imagen.
    La imagen se invierte, se suaviza con un desenfoque gaussiano, se normaliza para tener un rango de 0 a 255, y luego se aplica un umbral para convertirla en una imagen binaria.
    Argumentos:
        img: Una imagen en escala de grises.
    Salida:
        Una imagen binaria con caracteres blancos sobre un fondo negro.
    """

    img_inv = cv2.bitwise_not(img)
    img_smooth = cv2.GaussianBlur(img_inv, (5, 5), 0)
    img_norm = cv2.normalize(img_smooth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, img_bin_inv = cv2.threshold(img_norm, 127, 255, cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_not(img_bin_inv)

    return img_bin


def count_consecutive_values(arr):
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


def replace_consecutive_false(arr, threshold):
    """
    Esta función toma una lista de valores booleanos y un umbral.
    Reemplaza los valores False consecutivos por debajo del umbral con True.
    Argumentos:
        arr: Una lista de valores booleanos.
        threshold: Un umbral para determinar cuántos valores False consecutivos se deben reemplazar con True.
    Salida:
        Una lista de valores booleanos con los valores False consecutivos por debajo del umbral reemplazados por True.
    """

    arrm = count_consecutive_values(arr)
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


def show_segmentation(arr, threshold, char_threshold=3, word_threshold=10):
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
    img_pn = arr == 0
    img_pz = img_pn.any(axis=0)

    modified = replace_consecutive_false(img_pz, threshold)

    x = np.diff(modified)
    indxs = np.argwhere(x)

    ii = np.arange(0, len(indxs), 2)
    indxs[ii] += 1

    xx = np.arange(arr.shape[1])
    yy = np.zeros(arr.shape[1])
    yy[indxs] = (arr.shape[0] - 1)

    num_chars = count_elements(arr, 0, char_threshold)
    num_words = count_elements(arr, 0, word_threshold)

    plt.imshow(arr, cmap='gray')
    plt.plot(xx, yy, c='r')
    plt.title(f'Número de caracteres: {num_chars}, Número de palabras: {num_words}')
    plt.show()


def count_paragraphs(arr):
    """
    Esta función toma una lista de valores booleanos y cuenta el número de 'párrafos'.
    Un 'párrafo' se define como una secuencia de valores True consecutivos.
    Argumentos:
        arr: Una lista de valores booleanos.
    Salida:
        El número de 'párrafos' en la lista.
    """

    arrm = count_consecutive_values(arr)
    paragraph_counts = [count for value, count in arrm if value]

    return len(paragraph_counts)


def count_elements(img, axis, threshold):
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

    modified = replace_consecutive_false(img_sum, threshold)
    num_elements = count_paragraphs(modified)

    return num_elements


def validate_text(img, min_chars=1, max_chars=25, min_words=2, max_words=float('inf'), axis=0, char_threshold=3, word_threshold=10):
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

    num_chars = count_elements(img, axis, char_threshold)
    num_words = count_elements(img, axis, word_threshold)

    if min_chars <= num_chars <= max_chars and min_words <= num_words <= max_words:
        return True
    else:
        return False


#### HAY QUE CORREGUIR ESTA
def validate_form(img, fields):
    """
    Esta función toma una imagen y una lista de recortes de esa imagen.
    Normaliza la imagen y luego usa las funciones existentes para evaluar los criterios especificados.
    Argumentos:
        img: Una imagen en escala de grises.
        fields: Una lista de recortes de la imagen.
    Salida:
        Un DataFrame de pandas con los resultados de la validación.
    """

    # Normalizar la imagen
    img = normalize(img)

    # Definir los criterios de validación
    criteria = [
        {"min_words": 2, "max_words": 2, "min_chars": 2, "max_chars": 25},  # Nombre y apellido
        {"min_words": 1, "max_words": 1, "min_chars": 2, "max_chars": 3},  # Edad
        {"min_words": 1, "max_words": 1, "min_chars": 1, "max_chars": 25},  # Mail
        {"min_words": 1, "max_words": 1, "min_chars": 8, "max_chars": 8},  # Legajo
        {"min_words": 1, "max_words": 1, "min_chars": 1, "max_chars": 1},  # Pregunta 1
        {"min_words": 1, "max_words": 1, "min_chars": 1, "max_chars": 1},  # Pregunta 2
        {"min_words": 1, "max_words": 1, "min_chars": 1, "max_chars": 1},  # Pregunta 3
        {"min_words": 0, "max_words": float('inf'), "min_chars": 0, "max_chars": 25},  # Comentarios
    ]

    # Definir los nombres de los campos
    field_names = ["Nombre y apellido", "Edad", "Mail", "Legajo", "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"]

    # Inicializar el DataFrame de resultados
    results = pd.DataFrame(columns=["Campo", "Resultado"])

    # Validar cada campo
    for i in range(len(fields)):
        field = fields[i]
        criterion = criteria[i]
        result = validate_text(field, **criterion)
        results = results.append({"Campo": field_names[i], "Resultado": "OK" if result else "MAL"}, ignore_index=True)

    return results
