import numpy as np
import os
import cv2
import random

from typing import Tuple, List
from abc import ABC, abstractmethod

# ======== SISTEMA DE LECTURA DE DATOS ========
class ClasePerroGato:
    CLASE_PERRO = 0
    CLASE_GATO = 1

class Lector:
    def __init__(self, ruta_train_perros: str, ruta_train_gatos: str,
                ruta_test_perros: str, ruta_test_gatos: str):
        self.__ruta_train_perros = ruta_train_perros # El __ es para poner los miembros privados
        self.__ruta_train_gatos = ruta_train_gatos
        self.__ruta_test_perros = ruta_test_perros
        self.__ruta_test_gatos = ruta_test_gatos

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
        clases_perros =  [ClasePerroGato.CLASE_PERRO for i in range(len(imgs_perros))]
        clases_gatos = [ClasePerroGato.CLASE_GATO for i in range(len(imgs_gatos))]

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

# ======== SISTEMA DE PREPROCESAMIENTO ========
class Preprocesador(ABC):
    def __init__(self):
        pass

    def __eliminar_ruido_media(self, img: np.ndarray, m: int) -> np.ndarray:
        pass

    def __eliminar_ruido_mediana(self, img: np.ndarray, m: int) -> np.ndarray:
        pass

    def __equalizar(self, img: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        pass

class PrepMedia(Preprocesador):
    def __init__(self):
        super().__init__()

        self.__TAM_KERNEL = 3

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        return self.__eliminar_ruido_media(img, self.__TAM_KERNEL)

class PrepMediana(Preprocesador):
    def __init__(self):
        super().__init__()

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        pass

class PrepMediaEqu(Preprocesador):
    def __init__(self):
        super().__init__()

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        pass

class PrepMedianaEqu(Preprocesador):
    def __init__(self):
        super().__init__()

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        pass

# ======== SISTEMA DE EXTRACCION DE CARACTERISTICAS ========
class AlgoritmoCaracteristicas(ABC):
    @abstractmethod
    def calc_vector_caracteristicas(self, imagen: np.ndarray) -> np.ndarray:
        pass

class AlgoritmoHistograma(AlgoritmoCaracteristicas): 
    def __init__(self):
        super().__init__()
    
    def calc_vector_caracteristicas(self, imagen: np.ndarray) -> np.ndarray:
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

    def calc_vector_caracteristicas(self, imagen: np.ndarray) -> np.ndarray:
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

        self.__REESCALADO = (256, 256) # Potencias de 2 y ambos numeros iguales

    def calc_vector_caracteristicas(self, img: np.ndarray) -> np.ndarray:
        tam_celda = 8
        img_gris = np.mean(img, axis=2).astype(np.uint8)
        img_reescalada = self.__reescalar_imagen(img_gris)

        magnitudes, orientaciones = self.__calcular_derivadas_Sobel(img_reescalada)
        imagen_histogramas = self.__calc_imagen_histograma(magnitudes, orientaciones, tam_celda)
        vec_carac = self.__normalizacion_por_bloques(imagen_histogramas)
        return vec_carac

    def __reescalar_imagen(self, img: np.ndarray) -> np.ndarray:
        h, w = self.__REESCALADO
        if (np.abs(np.log2(h) - round(np.log2(h))) > 10e-2 or
            np.abs(np.log2(w) - round(np.log2(w))) > 10e-2):
            raise Exception('El reescalado tiene que ser potencia de 2')
        if h != w:
            raise Exception('El reescalado debe ser cuadrado')
        
        tam_obj = self.__REESCALADO[0] # O self.__REESCALADO[1] (Es cuadrada)
        escala = tam_obj / max(h, w)
        nuevo_w = int(w * escala)
        nuevo_h = int(h * escala)
        reescalada = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)

        pad_w = tam_obj - nuevo_w
        pad_h = tam_obj - nuevo_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        resultado = cv2.copyMakeBorder(
            reescalada,
            pad_top, pad_bottom,
            pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        return resultado

    def __calcular_derivadas_Sobel(self, img_gris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        derv_x = cv2.Sobel(img_gris, cv2.CV_64F, 1, 0, ksize=3)
        derv_y = cv2.Sobel(img_gris, cv2.CV_64F, 0, 1, ksize=3)
        magnitudes = np.sqrt(derv_x**2 + derv_y**2)
        orientaciones = np.abs(180*np.arctan2(derv_y, derv_x) / np.pi)
        orientaciones[orientaciones < 0] = 180 + orientaciones
        orientaciones[orientaciones == 180] = 0 # Para que angulo perteneca a [0º, 180º)

        return magnitudes, orientaciones

    def __calc_imagen_histograma(self, img_ori: np.ndarray, img_mag: np.ndarray, m: int=8) -> np.ndarray:
        if (np.abs(np.log2(m) - round(np.log2(m))) > 10e-2):
            raise Exception('El tamaño de celda tiene que ser potencia de 2')
        
        forma_histograma_bloques = (img_ori.shape[0] // m, img_ori.shape[1] // m, 9)
        imagen_histograma = np.zeros(shape=forma_histograma_bloques)
        for i in range(0, img_ori.shape[0], m):
            for j in range(0, img_ori.shape[1], m):
                bloque_ori = img_ori[i:i+m, j:j+m]
                bloque_mag = img_mag[i:i+m, j:j+m]
                hist_bloque = self.__calc_histograma_bloque(bloque_ori, bloque_mag)
                imagen_histograma[i // m, j // m] = hist_bloque
        return imagen_histograma

    def __calc_histograma_bloque(self, bloque_ori: np.ndarray, bloque_mag: np.ndarray) -> np.ndarray:
        pos_l = bloque_ori // 20
        pos_r = (pos_l + 1) % 9
        angulo_l = 20*pos_l
        angulo_r = 20*(pos_l + 1)
        valores_hist_l = (bloque_mag * ((angulo_r - bloque_ori) / 20))
        valores_hist_r = (bloque_mag * ((bloque_ori - angulo_l) / 20))

        hist = np.zeros(shape=(9, ))
        np.add.at(hist, pos_l.ravel(), valores_hist_l.ravel())
        np.add.at(hist, pos_r.ravel(), valores_hist_r.ravel())
        return hist
    
    def __normalizacion_por_bloques(self, imagen_hist: np.ndarray) -> np.ndarray:
        m = 2 # (2x2) celdas = 1 bloque
        mdiv2 = m // 2
        img_hist_norm = np.zeros(shape=(imagen_hist.shape[0] - mdiv2, imagen_hist.shape[1] - mdiv2, 36))
        for i in range(0, imagen_hist.shape[0] - mdiv2):
            for j in range(0, imagen_hist.shape[1] - mdiv2):
                vector = imagen_hist[i:i+m, j:j+m, :].flatten()
                img_hist_norm[i, j] = np.linalg.norm(vector)
        return img_hist_norm.flatten()

class AlgoritmoNuestro(AlgoritmoOrientaciones):
    def __init__(self):
        super().__init__()

    def calc_vector_caracteristicas(self):
        pass

class TransformadorCaracteristicas:
    def __init__(self, algoritmo: AlgoritmoCaracteristicas):
        self.__algoritmo = algoritmo

    def transformar(self, img: np.ndarray) -> np.ndarray:
        return self.__algoritmo.calc_vector_caracteristicas(img)

# ======== SISTEMA DE IA ========
class PredictorPerroGato:
    def __init__(self, modelo):
        ''' modelo: Un modelo de IA para entrenarlo y predecir '''
        pass

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Hacer un entrenamiento buscando los mejores hiperparametros con conjunto de validacion """
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

def experimento(
        preprocesador: Preprocesador,
        algoritmo: AlgoritmoCaracteristicas,
        predictor: PredictorPerroGato
    ) -> float:
    """ Devuelve el accuracy """

    # =========== PROCESO DE ENTRENAMIENTO ===========
    # Leemos los datos
    lector = Lector(
        ruta_train_perros='dataset/cat_dog_100/train/dog',
        ruta_train_gatos='dataset/cat_dog_100/train/cat',
        ruta_test_perros='dataset/cat_dog_100/test/dog',
        ruta_test_gatos='dataset/cat_dog_100/test/cat'
    )
    X_train, y_train, X_test, y_test = lector.leer_dataset()

    # Los preprocesamos
    for i, img in enumerate(X_train):
        X_train[i] = preprocesador.preprocesar(img)
    for i, img in enumerate(X_train):
        X_test[i] = preprocesador.preprocesar(img)

    # Transformamos las imagenes a vectores de caracterisitcas
    transformador = TransformadorCaracteristicas(algoritmo)
    for i, img in enumerate(X_train):
        X_train[i] = transformador.transformar(img)
    for i, img in enumerate(X_train):
        X_test[i] = transformador.transformar(img)

    # Entrenar la IA
    predictor.entrenar(X_train, y_train)

    # =========== PROCESO DE PREDICCION ===========
    y_test_pred = predictor.predecir_perro_gato(X_test)
    accuracy = np.where(y_test == y_test_pred)[0].size / y_test.size
    return accuracy

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

    experimento(PrepMedia(), AlgoritmoHistograma(), PredictorAdaboost())
    experimento(PrepMedia(), AlgoritmoHistograma(), PredictorAdaboost())
    experimento(PrepMedia(), AlgoritmoHistograma(), PredictorAdaboost())
    experimento(PrepMedia(), AlgoritmoHistograma(), PredictorAdaboost())