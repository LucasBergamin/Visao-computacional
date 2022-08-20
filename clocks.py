import cv2

image_clock = cv2.imread('images/clock.jpg')
image_clock_gray = cv2.cvtColor(image_clock, cv2.COLOR_BGR2GRAY)

detector_clock = cv2.CascadeClassifier('Cascades/clocks.xml')
detections_clock = detector_clock.detectMultiScale(image_clock_gray, scaleFactor=1.001, minNeighbors=5,
                                                   maxSize=(110, 100), minSize=(50, 50))

for (x, y, w, h) in detections_clock:
    cv2.rectangle(image_clock, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('img', image_clock)
cv2.waitKey(0)
cv2.destroyAllWindows()
