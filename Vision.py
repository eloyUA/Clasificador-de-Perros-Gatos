import numpy as np
import os
import cv2
import random

from typing import Tuple, List
from abc import ABC, abstractmethod

# ======== SISTEMA DE LECTURA DE DATOS ========
class Lector:
    def __init__(self, ruta_train_perros: str, ruta_train_gatos: str,
                ruta_test_perros: str, ruta_test_gatos: str):
        self.__ruta_train_perros = ruta_train_perros
        self.__ruta_train_gatos = ruta_train_gatos
        self.__ruta_test_perros = ruta_test_perros
        self.__ruta_test_gatos = ruta_test_gatos

        self.__CLASE_PERRO = 0
        self.__CLASE_GATO = 1
        self.__SEMILLA = 50

    def leer_dataset(self) -> Tuple[List[np.ndarray], List, List[np.ndarray], List]:
        ''' Devuelve X_train, y_train, X_test, y_test '''
        X_train, y_train = self.__cargar_conjunto(
            self.__ruta_train_perros,
            self.__ruta_train_gatos
        )
        
        X_test, y_test = self.__cargar_conjunto(
            self.__ruta_test_perros,
            self.__ruta_test_gatos
        )
        return X_train, y_train, X_test, y_test

    def __cargar_conjunto(self, ruta_perros: str, ruta_gatos: str) -> Tuple[List, List]:
        imgs_perros = self.__cargar_imagenes(ruta_perros)
        imgs_gatos = self.__cargar_imagenes(ruta_gatos)
        clases_perros =  [self.__CLASE_PERRO for i in range(len(imgs_perros))]
        clases_gatos = [self.__CLASE_GATO for i in range(len(imgs_gatos))]

        imagenes = imgs_perros + imgs_gatos
        clases = clases_perros + clases_gatos

        random.seed(self.__SEMILLA)
        random.shuffle(imagenes)
        random.seed(self.__SEMILLA)
        random.shuffle(clases)

        return imagenes, clases

    def __cargar_imagenes(self, ruta: str) -> List:
        imagenes = []
        archs_imgs = os.listdir(ruta)
        for arch in archs_imgs:
            img = cv2.imread(ruta + '/' + arch)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagenes.append(img)
        return imagenes

# ======== SISTEMA DE EXTRACCION DE CARACTERISTICAS ========
class AlgoritmoCaracteristicas(ABC):
    @abstractmethod
    def calc_caracteristicas(self, imagen: np.ndarray) -> np.ndarray:
        pass

class AlgoritmoHistograma(AlgoritmoCaracteristicas): 
    def __init__(self):
        super().__init__()
    
    def calc_caracteristicas(self, imagen: np.ndarray) -> np.ndarray:
        hist_r = self.__calc_histograma_canal(imagen[:, :, 0])
        hist_g = self.__calc_histograma_canal(imagen[:, :, 1])
        hist_b = self.__calc_histograma_canal(imagen[:, :, 2])

        vector_caracteristicas = np.concatenate((hist_r, hist_g, hist_b))
        return vector_caracteristicas

    def __calc_histograma_canal(self, canal: np.ndarray) -> np.ndarray:
        return cv2.calcHist([canal], [0], None, [256], [0, 256]).flatten()

class AlgoritmoTexturas(AlgoritmoCaracteristicas):
    def __init__(self):
        super().__init__()

    def calc_caracteristicas(self, imagen: np.ndarray) -> np.ndarray:
        m = 3
        mdiv2 = m // 2
        kernel = np.array([
            [2**7, 2**6, 2**5],
            [2**0, 0000, 2**4],
            [2**1, 2**2, 2**3]
        ], dtype=np.uint8)

        resultado = np.zeros(shape=imagen.shape, dtype=np.uint8)
        img_gris = np.mean(imagen, axis=2).astype(np.uint8)
        img_gris_amp = cv2.copyMakeBorder(img_gris, mdiv2, mdiv2, mdiv2, mdiv2, cv2.BORDER_REPLICATE)
        for i in range(img_gris_amp.shape[0] - m):
            for j in range(img_gris_amp.shape[1] - m):
                zona_comun = img_gris_amp[i:i+m, j:j+m]
                mascara = (zona_comun >= zona_comun[1, 1]).astype(np.uint8)
                resultado[i, j] = np.sum(kernel * mascara)

        return cv2.calcHist([resultado], [0], None, [256], [0, 256]).flatten()

class AlgoritmoOrientaciones(AlgoritmoCaracteristicas):
    def __init__(self):
        super().__init__()

    def calc_caracteristicas(self, img: np.ndarray) -> np.ndarray:
        """ Nota: Para imagenes de distintos tamaños el vector de características también lo es """
        img_gris = np.mean(img, axis=2).astype(np.uint8)
        magnitudes, orientaciones = self.__calcular_derivadas_Sobel(img_gris)
        

    def __calcular_derivadas_Sobel(self, img_gris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        derv_x = cv2.Sobel(img_gris, cv2.CV_64F, 1, 0, ksize=3)
        derv_y = cv2.Sobel(img_gris, cv2.CV_64F, 0, 1, ksize=3)
        magnitudes = np.sqrt(derv_x**2 + derv_y**2)
        orientaciones = np.abs(180*np.arctan2(derv_y, derv_x) / np.pi)
        orientaciones[orientaciones < 0] = 180 + orientaciones

        return magnitudes, orientaciones
    
    def transformar_imagenes_tamaño_optimo(self, imagenes: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def obtener_orientacion_magnitud(derivada_x: np.ndarray, derivada_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass
    def calcular_histograma_orientaciones(magnitud: np.ndarray, orientacion: np.ndarray, num_bins: int ) -> np.ndarray: #EL num_bins es 9 siempre
        pass

class AlgoritmoNuestro(AlgoritmoCaracteristicas):
    def __init__(self):
        super().__init__()

    def calc_caracteristicas():
        pass

class TransformadorCaracteristicas:
    def __init__(self, algoritmo: AlgoritmoCaracteristicas):
        self.__algoritmo = algoritmo # El __ es para poner los miembros privados

    def __preprocesar(self, img: np.ndarray) -> np.ndarray:
        pass

    def transformar(self, img: np.ndarray) -> np.ndarray:
        pass

# ======== SISTEMA DE IA ========
class PredictorPerroGato:
    def __init__(self, modelo):
        ''' modelo: Un modelo de IA para entrenarlo y predecir '''
        pass

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predecir_perro_gato(self, X: np.ndarray) -> np.ndarray:
        pass

class PredictorAdaboost(PredictorPerroGato):
    def __init__(self):
        # modelo = (Creamos el Adaboost)
        super().__init__(modelo)

class PredictorRandomForest(PredictorPerroGato):
    def __init__(self):
        # modelo = (Creamos el RandomForest)
        super().__init__(modelo)

if __name__ == '__main__':
    # Probar la clase Lector (FUNCIONA)
    '''lector = Lector(
        ruta_train_perros='dataset/cat_dog_100/train/dog',
        ruta_train_gatos='dataset/cat_dog_100/train/cat',
        ruta_test_perros='dataset/cat_dog_100/test/dog',
        ruta_test_gatos='dataset/cat_dog_100/test/cat'
    )

    X_train, y_train, X_test, y_test = lector.leer_dataset()
    print(len(X_train), len(y_train), len(X_test), len(y_test))'''

    a = np.array([
        [1, 7, 2],
        [8, 5, 8],
        [1, 3, 3],
    ])

    kernel = np.array([
        [2**7, 2**6, 2**5],
        [2**0, 0000, 2**4],
        [2**1, 2**2, 2**3]
    ], dtype=np.uint8)

    mascara = (a >= a[1, 1]).astype(np.uint8)
    print(mascara)