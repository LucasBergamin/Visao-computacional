from PIL import Image
import cv2
import numpy as np
import zipfile
import os

path = 'Datasets/yalefaces.zip'
zip_object = zipfile.ZipFile(file=path, mode = 'r')
zip_object.extractall('./')
zip_object.close()
# Esse código extrai os dados do dataset yalefaces que está compactado


def get_image_data():
    #Extraio apenas as imagens de treinamento do datasets yalefaces
    paths = [os.path.join('/content/yalefaces/train', f) for f in os.listdir('/content/yalefaces/train')]
    faces = []
    ids = []
    for path in paths:
        #Converto a imagem para cinza
        image = Image.open(path).convert('L')
        #converto a imagem para o tipo numpy
        image_np = np.array(image, 'uint8')
        #Extraio apenas o número de cada imagem exemplo subject01 então irá sobrar apenas o 01
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(image_np)
    return np.array(ids), faces


ids, faces = get_image_data()

#treinamento do classificador LBPH
lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius=5, neighbors=16, grid_x=12,grid_y=12)
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('/content/lbph_classifier.yml')

#Reconhecimento de faces
imagem_teste = 'yalefaces/test/subject10.sad.gif'

imagem = Image.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')

previsao = lbph_face_classifier.predict(imagem_np)
saida_esperada = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))

cv2.putText(imagem_np, 'Pred: ' + str(previsao[0]), (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.putText(imagem_np, 'Exp: ' + str(saida_esperada), (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.imshow(imagem_np)