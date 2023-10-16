import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Crear un kernel
kernel = np.ones((3, 3), np.uint8)

# Gradiente morfológico manual
erosion = cv2.erode(imagen, kernel, iterations=1)
dilation = cv2.dilate(imagen, kernel, iterations=1)
gradiente_manual = dilation - erosion

# Transformación Top-Hat manual
opening = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
tophat_manual = imagen - opening

# Transformación Bottom-Hat manual
closing = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
bottomhat_manual = closing - imagen

# Guardar los resultados en disco
cv2.imwrite('gradiente_manual.png', gradiente_manual)
cv2.imwrite('tophat_manual.png', tophat_manual)
cv2.imwrite('bottomhat_manual.png', bottomhat_manual)