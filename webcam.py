import cv2

detector_face = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
reconhecedor_face = cv2.face.LBPHFaceRecognizer_create()
reconhecedor_face.read('Haarcascade/lbph_classifier.yml')
altura, largura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    deteccoes = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30,30))
    for (x, y, w, h) in deteccoes:
        imagem_face = cv2.resize(imagem_cinza[y:y + w, x:x + h], (largura, altura))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        id, confience = reconhecedor_face.predict(imagem_face)
        nome = ""
        if id == 1:
            nome = 'Jones'
        elif id == 2:
            nome = 'Gabriel'
        cv2.putText(frame, nome, (x,y+(w+30)), font, 2, (255,0,0))
        cv2.putText(frame, str(confience), (x,y + (h+50)), font, 1, (255,0,0))
    cv2.imshow('face', frame)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()