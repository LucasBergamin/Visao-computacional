import cv2

image = cv2.imread('images/people1.jpg')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector_facil = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
detections = detector_facil.detectMultiScale(image_gray, scaleFactor=1.7)  # ele aumenta a escala, então precisa ser ajustado para obter menos falsos positivos
for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
