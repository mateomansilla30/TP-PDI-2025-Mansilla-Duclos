# TP-PDI-2025-Mansilla-Duclos

Este proyecto consta de dos problemas principales relacionados con el procesamiento de imágenes. A continuación, se describen las dependencias necesarias, la organización de carpetas y cómo ejecutar el código.

## Dependencias necesarias

Para ejecutar este proyecto, asegurarse de tener Python instalado en la computadora. Luego, instalar las siguientes dependencias:

- **opencv-python**: Para el procesamiento de imágenes.
- **numpy**: Para operaciones matemáticas y manipulación de datos.
- **matplotlib**: Para visualización de imágenes y gráficos.
- **os**: Para interactuar con el sistema operativo y manejar archivos y directorios.

Se puede instalar estas dependencias ejecutando el siguiente comando:

```bash
pip install opencv-python numpy matplotlib
```

El proyecto debe tener la siguiente estructura de directorios:

TP-PDI-2025-Mansilla-Duclos/
├── problema_1/
│   ├── imagen_con_detalles.jpg     # Imagen con los detalles escondidos (para Problema 1)
│   └── problema_1.py              # Código de solución del Problema 1
│
├── problema_2/
   ├── imagenes_examenes/         # Directorio con los 5 exámenes corregidos
   │   ├── examen_1.png
   │   ├── examen_2.png
   │   ├── examen_3.png
   │   ├── examen_4.png
   │   └── examen_5.png
   └problema_2.py  


### Problema 1
Para el Problema 1, la imagen con los detalles escondidos debe estar ubicada en la carpeta problema_1 junto con el archivo problema_1.py. Asegurarse de que la imagen esté correctamente nombrada y sea la correcta para su procesamiento.

### Problema 2
Para el Problema 2, debes tener un directorio llamado imagenes_examenes dentro de la carpeta problema_2, que contenga los 5 exámenes corregidos. Asegúrarse de que los nombres de las imágenes coincidan con los del código y estén en el formato adecuado (por ejemplo, .png).

## Ejecución del Código
### Problema 1
Colocar la imagen con los detalles escondidos dentro de la carpeta problema_1, junto con el archivo problema_1.py.

Navega al directorio problema_1 en tu terminal.

Ejecutar el siguiente comando:

```bash
python problema_1.py
Este script procesará la imagen, revelará los detalles escondidos y generará las imágenes de salida en la carpeta output_images.
```

### Problema 2
Asegurarse de que el directorio imagenes_examenes dentro de la carpeta problema_2 contenga las imágenes de los exámenes corregidos.

Navegar al directorio problema_2 en la terminal.

Ejecuta el siguiente comando:

```bash
python problema_2.py
```
