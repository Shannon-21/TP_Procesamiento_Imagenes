# Tecnicatura en Inteligencia Artificial

## Universidad Nacional de Rosario

### Procesamiento de Imagenes 1

### **Trabajo Práctico 3**

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

Para ejecutar estos programas, se debe clonar el repositorio dentro de su local y posteriormente instalar las librerías necesarias para su ejecución. 

Para ello, se recomienda crear un entorno virtual con `python` de la siguiente forma:

```bash
python -m venv venv
```

Luego activar el entorno virtual con:

```bash
source venv/bin/activate
```
_(En caso de utilizar Windows, el comando anterior cambia a `venv\Scripts\activate.bat`)_

Luego instalar las librerías mediante `pip`:

```bash
pip install -r requirements.txt
```

El archivo `index.py` es quien reune todo el codigo del proyecto.

La solución consta de dos clases:
* La clase Detector que se encarga de detectar los dados
* La clase Dado que representa un dado en el video y detecta los números de su cara
