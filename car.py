import cv2

image_car = cv2.imread('images/car.jpg')
image_car_gray = cv2.cvtColor(image_car, cv2.COLOR_BGR2GRAY)

detector_car = cv2.CascadeClassifier('Cascades/cars.xml')
detections_car = detector_car.detectMultiScale(image_car_gray, scaleFactor=1.04, minNeighbors=3)

for (x, y, w, h) in detections_car:
    cv2.rectangle(image_car, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', image_car)
cv2.waitKey(0)
cv2.destroyAllWindows()
