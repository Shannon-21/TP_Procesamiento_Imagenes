# Tecnicatura en Inteligencia Artificial

## Universidad Nacional de Rosario

### Procesamiento de Imagenes 1

### **Morfologia, Color y Restauracion**

---

**Equipo 9:**
- Arevalo Ezequiel
- Ferrucci Constantino
- Giampaoli Fabio
- Revello Simon

---

El presente repositorio tiene el proposito de contener los archivos de codigo e imagenes necesarias para el desarrollo del segundo trabajo practico de la asignatura "Procesamiento de Imagenes 1".

El mismo consta de dos ejercicios: 

* El primero trata de encotrar en una imagem los diferentes objetos: monedas de diferentes tamaños y valor, dados en diferentes posiciones. Se debem segmentar los objetos y clasificarlos. Se deben contabilizar los mismos.

* Para el segundo ejercicio se cuentan con distintas imagenes de patentes de autos. El objetivo es crear un programa que automaticamente segmente las patentes de las imagenes, y luego detectar de cada patente los caracteres de las mismas. 

Para leer mas sobre el proyecto puede acceder a la documentacion: https://docs.google.com/document/d/15uaVN_WLnk7Bqamr9LyvqLQu-mleGBzEVmrUKeUynJk/edit?usp=sharing

---

Para ejecutar estos programas, debe clonar el repositorio, y asegurarse de instalar en su entorno la libreria `opencv-python` mediante pip.

El archivo `index.py` es quien reune todo el codigo del proyecto. Se han organizado la resolucion de cada ejercicio en clases de python. Por lo que sera necesario importar estas clases estando ubicado en la ruta adecuada como:

```
from scr.Objetos import Objetos
from scr.Patentes import Patentes
```

La clase Objetos tiene una unica funcion ejecutada en `index.py` que mostrara progresivamente en distintos graficos de matplotlib el avanze del analisis.
La clase Patente tiene dos metrodos, el primero de reconocer caracteres, que al ser llamada muestra una a una las images de los autos en tres parte: Umbralado, Filtro por area y relacion de aspecto, y filtro por cercania.
La segunda funcion cumple un rol similar, pero solo mostrando la imagen umbralada y el filtro por area y relacion de aspecto.
