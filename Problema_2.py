import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_paths = [
    "multiple_choice_1.png",
    "multiple_choice_2.png",
    "multiple_choice_3.png",
    "multiple_choice_4.png",
    "multiple_choice_5.png"
]

respuestas_correctas = ['A', 'A', 'B', 'A', 'D', 'B', 'B', 'C', 'B', 'A',
                        'D', 'A', 'C', 'C', 'D', 'B', 'A', 'C', 'C', 'D',
                        'B', 'A', 'C', 'C', 'C']

opciones = ['A', 'B', 'C', 'D', 'E']
umbral_respuesta = 290


def corregir_examen(image_paths):
    respuestas_correctas = ['A', 'A', 'B', 'A', 'D', 'B', 'B', 'C', 'B', 'A',
                            'D', 'A', 'C', 'C', 'D', 'B', 'A', 'C', 'C', 'D',
                            'B', 'A', 'C', 'C', 'C']
    opciones = ['A', 'B', 'C', 'D', 'E']
    umbral_respuesta = 290

    resultados = {}

    for image_path in image_paths:
        print(f"\nCorrigiendo: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f" No se pudo cargar la imagen {image_path}.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = w / float(h)
            area = cv2.contourArea(c)
            if 20 < w < 40 and 20 < h < 40 and 0.8 < ratio < 1.5 and area > 250:
                bubbles.append(c)

        bounding_boxes = [cv2.boundingRect(c) for c in bubbles]
        contornos_y_boxes = list(zip(bubbles, bounding_boxes))
        contornos_ordenados = sorted(contornos_y_boxes, key=lambda b: (b[1][1], b[1][0]))
        contornos_ordenados = [c for c, _ in contornos_ordenados]
        respuestas_por_fila = [contornos_ordenados[i:i+5] for i in range(0, len(contornos_ordenados), 5)]

        respuestas_finales = []
        detalles_preguntas = []

        for i, fila in enumerate(respuestas_por_fila):
            negros_por_opcion = []
            for j, c in enumerate(fila):
                (x, y), radius = cv2.minEnclosingCircle(c)
                centro = (int(x), int(y))
                radius = int(radius)

                mascara = np.zeros(thresh.shape, dtype=np.uint8)
                cv2.circle(mascara, centro, radius, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mascara)

                x1, y1 = int(x - radius), int(y - radius)
                x2, y2 = int(x + radius), int(y + radius)
                roi = masked[y1:y2, x1:x2]
                negros = cv2.countNonZero(roi)
                negros_por_opcion.append((opciones[j], negros))

            negros_por_opcion.sort(key=lambda x: x[1], reverse=True)
            max_valor = negros_por_opcion[0][1]
            segundo_valor = negros_por_opcion[1][1]

            if all(negros < umbral_respuesta for _, negros in negros_por_opcion):
                seleccionada = []
            elif max_valor - segundo_valor < max_valor * 0.115:
                seleccionada = [negros_por_opcion[0][0], negros_por_opcion[1][0]]
            else:
                seleccionada = negros_por_opcion[0][0]

            respuestas_finales.append(seleccionada)

            correcta = respuestas_correctas[i]
            estado = (
                "OK" if isinstance(seleccionada, str) and seleccionada == correcta else "MAL"
            )
            detalles_preguntas.append({
                "pregunta": i + 1,
                "respuesta_detectada": seleccionada,
                "respuesta_correcta": correcta,
                "estado": estado
            })

        correctas = sum(1 for d in detalles_preguntas if d["estado"] == "OK")

        # Imprimir los resultados de cada examen
        print("\nRESPUESTAS DETECTADAS:\n")
        for d in detalles_preguntas:
            print(f"Pregunta {d['pregunta']}:{d['respuesta_detectada']}: {d['estado']}")   
            
        resultados[image_path] = {
            "respuestas": respuestas_finales,
            "correctas": correctas,
            "detalles": detalles_preguntas
        }

    return resultados



def validar_campos(image_paths):
    results = {}  # Diccionario para almacenar los resultados de cada imagen

    for image_path in image_paths:
        # Cargar imagen y convertir a escala de grises
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Definir coordenadas de la zona del encabezado y recortar la región de interés (ROI)
        x1, y1, x2, y2 = 28, 105, 770, 130
        roi = gray[y1:y2, x1:x2]

        # Binarización inversa de la imagen
        _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)

        # Procesar líneas verticales
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Procesar líneas horizontales
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (740, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Encontrar y procesar contornos verticales
        v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_lines = [cv2.boundingRect(c)[0] for c in v_contours if cv2.boundingRect(c)[2] < 10]
        v_lines = sorted(set(v_lines + [0, x2 - x1]))  # Añadir el borde derecho

        # Encontrar y procesar contornos horizontales
        h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_lines = [cv2.boundingRect(c)[1] for c in h_contours if cv2.boundingRect(c)[3] < 10]
        h_lines = sorted(set(h_lines + [0, y2 - y1]))  # Añadir el borde inferior

        # Dibujar celdas y guardar coordenadas
        output = img.copy()
        boxes_coordinates = []
        box_id = 0

        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                x_start = v_lines[j] + x1
                x_end = v_lines[j + 1] + x1
                y_start = h_lines[i] + y1
                y_end = h_lines[i + 1] + y1

                boxes_coordinates.append({
                    "id": box_id,
                    "x_start": x_start,
                    "x_end": x_end,
                    "y_start": y_start,
                    "y_end": y_end
                })

                # Borde de 1 píxel
                cv2.rectangle(output, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
                cv2.putText(output, f"Box{box_id}", (x_start + 2, y_start - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                box_id += 1

        # Procesar celdas específicas (boxes 2, 4, 6, 8)
        boxes_to_process = [2, 4, 6, 8]
        boxes_data = {}

        for box_index in boxes_to_process:
            box = boxes_coordinates[box_index]
            x1, x2 = box["x_start"], box["x_end"]
            y1, y2 = box["y_start"], box["y_end"]

            box_roi = gray[y1:y2, x1:x2]
            _, box_bin = cv2.threshold(box_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(box_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Ordenar los contornos por la coordenada X (esto asegura que los caracteres se procesen de izquierda a derecha)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

            character_count = 0
            previous_x = None  # Inicializar variable para las coordenadas X
            previous_x_right = None  # Inicializar variable para la X derecha del carácter anterior
            word_count = 0  # Comenzamos con 1 palabra

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if (w <= 30 and h <= 40 and w > 2 and h > 5) or (w > 2 and h < 5) or (w <= 2.5 and h > 5):
                    character_count += 1
                    if character_count == 1:
                        word_count += 1
                    cv2.rectangle(output, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (0, 0, 255), )

                    if previous_x_right is not None and (x - previous_x_right) < 12:
                        word_count += 1

                    previous_x_right = x + w  # Actualizar la coordenada X derecha del carácter actual

            boxes_data[box['id']] = {
                "character_count": character_count,
                "word_count": word_count
            }

            # Guardar imagen del nombre (box 2)
            if box_index == 2:
                nombre_img = gray[y1:y2, x1:x2]
                os.makedirs('output_images', exist_ok=True)
                filename = f"box_2_{os.path.basename(image_path)}"
                cv2.imwrite(os.path.join('output_images', filename), nombre_img)

        # a. Name: Debe contener al menos dos palabras y no más de 25 caracteres en total.
        name_valid = "OK" if boxes_data.get(2, {}).get('word_count', 0) >= 2 and boxes_data.get(2, {}).get('character_count', 0) <= 25 else "MAL"
        # b. ID: Debe contener sólo 8 caracteres en total.
        id_valid = "OK" if boxes_data.get(4, {}).get('character_count', 0) == 8  else "MAL"
        # c. Code: Debe contener un único carácter.
        code_valid = "OK" if boxes_data.get(6, {}).get('character_count', 0) == 1 and boxes_data.get(6, {}).get('word_count', 0) == 1 else "MAL"
        # d. Date: Debe contener sólo 8 caracteres en total.
        date_valid = "OK" if boxes_data.get(8, {}).get('character_count', 0) == 8  else "MAL"

        results[image_path] = {
            "Name": name_valid,
            "ID": id_valid,
            "Code": code_valid,
            "Date": date_valid,
            "Words": boxes_data.get(2, {}).get('word_count', 0),  
            "Characters": boxes_data.get(2, {}).get('character_count', 0)  
        }

        # Mostrar la imagen final con los rectángulos
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Desactivar los ejes
        plt.show()

    return results



def generar_resumen_aprobados(corrections, carpeta='output_images', umbral_aprobado=20):
    resumen_filas = []

    for i, (image_path, data) in enumerate(corrections.items(), start=1):

        # Ruta a la imagen del nombre recortado
        filename = f'box_2_multiple_choice_{i}.png'
        path = f'{carpeta}/{filename}'

        nombre_img = cv2.imread(path)
        if nombre_img is None:
            print(f"No se encontró la imagen: {path}")
            continue

        nombre_resized = cv2.resize(nombre_img, (400, 60))

        correctas = data.get("correctas", 0)
        estado = "APROBADO" if correctas >= umbral_aprobado else "DESAPROBADO"
        color = (0, 200, 0) if estado == "APROBADO" else (0, 0, 255)

        fila = np.ones((70, 700, 3), dtype=np.uint8) * 255
        fila[5:65, 10:410] = nombre_resized
        cv2.putText(fila, estado, (430, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        resumen_filas.append(fila)


    if resumen_filas:
        resumen_final = cv2.vconcat(resumen_filas)
        cv2.imwrite("resumen_aprobados.png", resumen_final)
        cv2.imshow("Resumen Final", resumen_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # 1. Corrección de exámenes
    correcciones = corregir_examen(image_paths)

    # 2. Validación de encabezados (coordenadas del campo Name)
    coords_name = validar_campos(image_paths)

    # 3. Combinar info de corrección + coordenadas de Name
    for path in image_paths:
        if path in correcciones and path in coords_name:
            correcciones[path].update(coords_name[path])

    # Mostrar los resultados
    for image_path, result in correcciones.items():
        print(f"Resultados para {image_path}:")
        print(f"Name: {result['Name']}")
        print(f"ID: {result['ID']}")
        print(f"Code: {result['Code']}")
        print(f"Date: {result['Date']}")
        print()

    # 4. Generar imagen de salida
    generar_resumen_aprobados(correcciones)

main()
