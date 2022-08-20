import cv2

image = cv2.imread('images/people1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector_facil = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
detector_eye = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

detections = detector_facil.detectMultiScale(image_gray, scaleFactor=1.3, minSize=(30, 30))
for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

detections_eye = detector_eye.detectMultiScale(image_gray, scaleFactor=1.09, minNeighbors=10, maxSize=(70, 70))
for (x, y, w, h) in detections_eye:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
