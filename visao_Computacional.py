import cv2 # Importa a biblioteca opencv-python, para trabalhar com câmeras
import mediapipe
from cvzone.HandTrackingModule import HandDetector # importa o módulo de detecção de mãos

webcam = cv2.VideoCapture(0) # No parenteses vai qual a câmera será capturada... 0 é a camera do note
detectorMaos = HandDetector(detectionCon=0.8, maxHands=2) # 80% de certeza da IA e o total de mãos que serão detectadas 

# Enquanto for verdadeiro ele vai caputara as imagens da camera
while True:
    sucesso, imagem = webcam.read() # Esse cara checa se conseguiu capturar uma imagem da camera do note salva essa resposta na primeira variável e no caso de sim a imagem capturada é salva na segunda variável
    posicao, imagem_Maos = detectorMaos.findHands(imagem) # O detector de maos procura por maos na imagem da camera e salva a posicao delas na 1ª variável e a foto das mãos na segunda variável.
    
    cv2.imshow("Visão Computaciona",imagem) # Aqui ela mostra uma imagem com um título no 1º paramentro e a origem dela no segundo


    # Testa se alguma tecla do teclado foi apertada - escape para sair do loop infinito do While
    if cv2.waitKey(1) != -1: # Esse cara fica checa a cada intervalo colocado no () se alguma tecla foi pressionada. 1 seria a cada 1 milissegundo e caso não tenha sido pressionada o retorno é -1
        break # Se alguma tecla foi pressionada ele sai do loop infinito do While

# Uma vez fora do While eu libero a câmera para outros possíveis dispositivos
webcam.release()
cv2.destroyAllWindows() # E fecho toda e qualquer janela que esteja aberta

