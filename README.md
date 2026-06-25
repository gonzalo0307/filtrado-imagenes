Filtrado Digital de Imagenes

Aplicacion de escritorio en Python con interfaz grafica con tkinter para el procesamiento digital de imagenes en escala de grises. Permite cargar una imagen JPG, seleccionar un recorte de 15x15, una imagen completa o capturada en tiempo real y aplicar filtros de Media, Mediana, Laplaciano y Sobel, mostrando la imagen digitalizada y su representacion matricial antes y despues del filtrado.




Universidad Peruana de Ciencias Aplicadas (UPC)
Curso: Matematica Computacional — Ciclo 4


Requisitos


Python 3.8 o superior
Pillow
NumPy


Instalar dependencias:

bashpip install pillow numpy



Como usar el programa


Cargar imagen — clic en "Abrir JPG / JPEG" y selecciona tu imagen
Aplicar recorte — ingresa las coordenadas X e Y y clic en "Aplicar recorte 15x15"
Seleccionar filtro — elige entre Media, Mediana, Laplaciano o Sobel
Aplicar filtro — elige el modo (recorte, imagen completa o ambos) y clic en "Aplicar filtro"



Filtros implementados

Media: Suaviza la imagen promediando los pixeles vecinos (kernel 3x3)
Mediana: Elimina ruido usando el valor central de la vecindad 3x3
Laplaciano: Detecta bordes mediante la segunda derivada de la imagen
Sobel: Detecta bordes combinando gradiente horizontal y vertical


Caracteristicas

Acepta imagenes JPG/JPEG en color o escala de grises
Conversion automatica a escala de grises para los filtros
Visualizacion de la imagen digitalizada (cada pixel como cuadrado con su valor)
Representacion matricial completa del recorte 15x15
Procesamiento sobre imagen completa con version vectorizada (numpy)
Interfaz con pestanas separadas por tipo de filtro y modo de procesamiento