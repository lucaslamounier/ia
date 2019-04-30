import cv2 

classificador = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')

imagem = cv2.imread('../pessoas/pessoas3.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# faces_detectadas = classificador.detectMultiScale(imagemCinza)

# Adicionado parametro de escala para melhorar a precisão
faces_detectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1)

print(len(faces_detectadas))
print(faces_detectadas)

for (x, y, l, a) in faces_detectadas:
    print(x, y, l, a)
    # Parametros Imagem, Posição original, Quanto quer desenhar da borda, Valores RGB da borda, Espessura da borda
    cv2.rectangle(imagem, (x,y), (x + l, y + a), (0, 253, 255), 2)

cv2.imshow("Faces encontradas", imagem)
cv2.waitKey()
