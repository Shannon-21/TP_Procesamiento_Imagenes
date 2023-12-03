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

El presente repositorio tiene el proposito de contener los archivos de codigo y vídeos necesarios para el desarrollo del tercer trabajo practico de la asignatura "Procesamiento de Imagenes 1".

El mismo consta de dos ejercicios: 

* Encontrar los dados en el vídeo una vez que estén quietos

* Identificar los números de las caras

Para leer mas sobre el proyecto puede acceder a la documentacion: https://docs.google.com/document/d/15uaVN_WLnk7Bqamr9LyvqLQu-mleGBzEVmrUKeUynJk/edit?usp=sharing

---

Para ejecutar estos programas, debe clonar el repositorio, y asegurarse de instalar en su entorno la libreria `opencv-python` mediante pip.

El archivo `index.py` es quien reune todo el codigo del proyecto. Se han organizado la resolucion de cada ejercicio en clases de python. Por lo que sera necesario importar estas clases estando ubicado en la ruta adecuada como:

El proyecto consta de dos clases:
* La clase Detector que se encarga de detectar los dados
* La clase Dado que representa un dado en el video y detecta los números de su cara
