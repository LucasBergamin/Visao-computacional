import dlib
import cv2

image = cv2.imread('images/people2.jpg')

detector_face_hog = dlib.get_frontal_face_detector()
detections = detector_face_hog(image, 1)  # Escala da imagem

for face in detections:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)

cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
