# -*- coding: utf-8 -*-
import cv2
'''
    Detecção de faces via webcan
'''
video = cv2.VideoCapture(0)

classificadorFace = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')

while True:
    # Faz a leitura da webcam
    conectado, frame = video.read()
    #print(conectado)

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, scaleFactor=1.1, minNeighbors=9, minSize=(30,30))
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    # Para fechar o video
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


