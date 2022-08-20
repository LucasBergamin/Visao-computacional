import cv2
detector_face = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
    #Capturando o frame
    ok, frame = video_capture.read()
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    deteccoes = detector_face.detectMultiScale(imagem_cinza, minSize=(100, 100))

    #desenhando o retângulo
    for (x, y, w, h) in deteccoes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
