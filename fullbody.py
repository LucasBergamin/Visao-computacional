import cv2

image_fullbody = cv2.imread('images/people3.jpg')

detector_fullbody = cv2.CascadeClassifier("Cascades/fullbody.xml")
detections_fullbody = detector_fullbody.detectMultiScale(cv2.cvtColor(image_fullbody, cv2.COLOR_BGR2GRAY),
                                                         scaleFactor=1.03)

for (x, y, w, h) in detections_fullbody:
    cv2.rectangle(image_fullbody, (x, y), (x + w, y + h), (123, 222, 0), 2)

cv2.imshow('img', image_fullbody)
cv2.waitKey(0)
cv2.destroyAllWindows()
