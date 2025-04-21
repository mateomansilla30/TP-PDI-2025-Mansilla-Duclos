import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local(img, M, N):
    """
    Realiza ecualización local del histograma con ventana de tamaño MxN.
    """
    # Padding de bordes para que se pueda centrar la ventana en todos los píxeles
    pad_M = M // 2
    pad_N = N // 2
    padded = cv2.copyMakeBorder(img, pad_M, pad_M, pad_N, pad_N, cv2.BORDER_REPLICATE)

    # Imagen de salida
    output = np.zeros_like(img)

    # Recorrer cada píxel de la imagen original
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extraer la ventana
            window = padded[i:i+M, j:j+N]

            # Calcular histograma de la ventana
            hist = cv2.calcHist([window], [0], None, [256], [0, 256])
            hist = hist / hist.sum()  # Normalizar

            # Calcular CDF
            cdf = hist.cumsum()
            cdf = (cdf * 255).astype(np.uint8)

            # Aplicar transformación al píxel central
            output[i, j] = cdf[img[i, j]]

    return output

img = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

#Aplicar ecualización local
img_eq = ecualizacion_local(img, M=40, N=40)

#Mostrar imagen procesada
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_eq, cmap='gray')
plt.title('Ecualización local')
plt.axis('off')
plt.show()