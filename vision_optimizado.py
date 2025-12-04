import numpy as np
import os
import cv2
import random
import time

from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
from numba import njit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ======== SISTEMA DE LECTURA DE DATOS ========
class ClasePerroGato:
    CLASE_PERRO = -1
    CLASE_GATO = 1

class Lector:
    def __init__(self, ruta_train_perros: str, ruta_train_gatos: str,
                ruta_test_perros: str, ruta_test_gatos: str, semilla: int):
        self.__ruta_train_perros = ruta_train_perros
        self.__ruta_train_gatos = ruta_train_gatos
        self.__ruta_test_perros = ruta_test_perros
        self.__ruta_test_gatos = ruta_test_gatos
        self.__semilla = semilla

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

        random.seed(self.__semilla)
        random.shuffle(imagenes)
        random.seed(self.__semilla)
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
@njit
def conv_media_preprocesador(img_amp: np.ndarray, m: int) -> np.ndarray:
    resultado = np.zeros(shape=(img_amp.shape[0] - m + 1, img_amp.shape[1] - m + 1), dtype=np.float32)
    for i in range(img_amp.shape[0] - m):
        for j in range(img_amp.shape[1] - m):
            zona_comun = img_amp[i:i+m, j:j+m]
            resultado[i, j] = np.mean(zona_comun)
    return resultado.astype(np.uint8)

@njit
def conv_mediana_preprocesador(img_amp: np.ndarray, m: int) -> np.ndarray:
    resultado = np.zeros(shape=(img_amp.shape[0] - m + 1, img_amp.shape[1] - m + 1), dtype=np.float32)
    for i in range(img_amp.shape[0] - m):
        for j in range(img_amp.shape[1] - m):
            zona_comun = img_amp[i:i+m, j:j+m]
            resultado[i, j] = np.median(zona_comun)
    return resultado.astype(np.uint8)

@njit
def conv_kernel_preprocesador(img_amp: np.ndarray, m: int, kernel: np.ndarray) -> np.ndarray:
    resultado = np.zeros(shape=(img_amp.shape[0] - m + 1, img_amp.shape[1] - m + 1), dtype=np.float32)
    for i in range(img_amp.shape[0] - m):
        for j in range(img_amp.shape[1] - m):
            zona_comun = img_amp[i:i+m, j:j+m]
            resultado[i, j] = np.sum(kernel * zona_comun)
    return resultado.astype(np.uint8)

class Preprocesador(ABC):
    def __init__(self, m: int):
        if m % 2 == 0 or m <= 0:
            raise Exception('El tamaño del kernel tiene que ser impar y positivo')
        
        self.__m = m

    @abstractmethod
    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        pass

    def filtro_media(self, img: np.ndarray) -> np.ndarray:
        return self.__aplicar_filtro(img, 'media')

    def filtro_mediana(self, img: np.ndarray) -> np.ndarray:
        return self.__aplicar_filtro(img, 'mediana')

    def filtro_gaussiano(self, img: np.ndarray, sigma: float) -> np.ndarray:
        kernel = cv2.getGaussianKernel(self.__m, sigma) * cv2.getGaussianKernel(self.__m, sigma).T
        return self.__aplicar_filtro(img, 'gaussiano', kernel)

    def __aplicar_filtro(
            self, img: np.ndarray,
            tipo: str,
            kernel: Optional[np.ndarray] = None
        ) -> np.ndarray:
        """ tipo: "media", "mediana", "gaussiano" """

        if not(tipo in ['media', 'mediana', 'gaussiano']):
            raise Exception('Tipo de filtro no valido.')
        if tipo == 'gaussiano' and kernel is None:
            raise Exception('Es necesario especificar el kernel para el filtro gaussiano.')

        mdiv2 = self.__m // 2
        resultado = np.zeros(img.shape, dtype=np.uint8)
        img_amp = cv2.copyMakeBorder(img, mdiv2, mdiv2, mdiv2, mdiv2, cv2.BORDER_REPLICATE)
        for i in range(3):
            if tipo == 'media':
                resultado[:, :, i] = conv_media_preprocesador(img_amp[:, :, i], self.__m)
            elif tipo == 'mediana':
                resultado[:, :, i] = conv_mediana_preprocesador(img_amp[:, :, i], self.__m)
            elif tipo == 'gaussiano':
                resultado[:, :, i] = conv_kernel_preprocesador(img_amp[:, :, i], self.__m, kernel)

        return resultado

    def equalizar(self, img: np.ndarray) -> np.ndarray:
        img_equ = np.zeros(shape=img.shape, dtype=np.uint8)
        for i in range(3):
            img_equ[:, :, i] = self.__equalizar_canal(img[:, :, i])
        return img_equ

    def __equalizar_canal(self, canal: np.ndarray) -> np.ndarray:
        hist = np.bincount(canal.ravel(), minlength=256)
        cumsum = hist.cumsum()
        f_Ixy = (255 * (cumsum / cumsum[-1])).astype(np.uint8)
        return f_Ixy[canal]
        
class PrepMedia(Preprocesador):
    def __init__(self, m: int):
        super().__init__(m)

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        return self.filtro_media(img)

class PrepMediana(Preprocesador):
    def __init__(self, m: int):
        super().__init__(m)

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        return self.filtro_mediana(img)
    
class PrepGaussiano(Preprocesador):
    def __init__(self, m: int, sigma: float):
        super().__init__(m)
        self.__sigma = sigma

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        return self.filtro_gaussiano(img, self.__sigma)

class PrepMediaEqu(Preprocesador):
    def __init__(self, m):
        super().__init__(m)

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        filtrada = self.filtro_media(img)
        return self.equalizar(filtrada)
class PrepMedianaEqu(Preprocesador):
    def __init__(self, m: int):
        super().__init__(m)

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        filtrada = self.filtro_mediana(img)
        return self.equalizar(filtrada)
    
class PrepGaussianoEqu(Preprocesador):
    def __init__(self, m: int, sigma: float):
        super().__init__(m)
        self.__sigma = sigma

    def preprocesar(self, img: np.ndarray) -> np.ndarray:
        filtrada = self.filtro_gaussiano(img, self.__sigma)
        return self.equalizar(filtrada)

# ======== SISTEMA DE EXTRACCION DE CARACTERISTICAS ========
@njit
def conv_algoritmo_texturas(img_amp: np.ndarray, m: int) -> np.ndarray:
    kernel = np.array([
        [2**7, 2**6, 2**5],
        [2**0, 0000, 2**4],
        [2**1, 2**2, 2**3]
    ], dtype=np.uint8)

    resultado = np.zeros(shape=(img_amp.shape[0] - m + 1, img_amp.shape[1] - m + 1), dtype=np.uint8)
    for i in range(img_amp.shape[0] - m):
        for j in range(img_amp.shape[1] - m):
            zona_comun = img_amp[i:i+m, j:j+m]
            mascara = (zona_comun >= zona_comun[1, 1]).astype(np.uint8)
            resultado[i, j] = np.sum(kernel * mascara)
    return resultado

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

        img_gris = np.mean(imagen, axis=2).astype(np.uint8)
        img_gris_amp = cv2.copyMakeBorder(img_gris, mdiv2, mdiv2, mdiv2, mdiv2, cv2.BORDER_REPLICATE)
        resultado = conv_algoritmo_texturas(img_gris_amp, m)

        return cv2.calcHist([resultado], [0], None, [256], [0, 256]).flatten()

class AlgoritmoOrientaciones(AlgoritmoCaracteristicas):
    def __init__(self):
        super().__init__()

        self.__REESCALADO = (256, 256) # Potencias de 2 y ambos numeros iguales

    def calc_vector_caracteristicas(self, img: np.ndarray) -> np.ndarray:
        tam_celda = 8
        img_gris = np.mean(img, axis=2).astype(np.uint8)
        img_reescalada = self._reescalar_imagen(img_gris)

        magnitudes, orientaciones = self._calcular_derivadas_Sobel(img_reescalada)
        imagen_histogramas = self._calc_imagen_histograma(orientaciones, magnitudes, tam_celda)
        vec_carac = self._normalizacion_por_bloques(imagen_histogramas)
        return vec_carac

    def _reescalar_imagen(self, img: np.ndarray) -> np.ndarray:
        h, w = self.__REESCALADO
        if (np.abs(np.log2(h) - round(np.log2(h))) > 10e-2 or
            np.abs(np.log2(w) - round(np.log2(w))) > 10e-2):
            raise Exception('El reescalado tiene que ser potencia de 2')
        if h != w:
            raise Exception('El reescalado debe ser cuadrado')
        
        h, w = img.shape[0], img.shape[1]
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

    def _calcular_derivadas_Sobel(self, img_gris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        derv_x = cv2.Sobel(img_gris, cv2.CV_64F, 1, 0, ksize=3)
        derv_y = cv2.Sobel(img_gris, cv2.CV_64F, 0, 1, ksize=3)
        magnitudes = np.sqrt(derv_x**2 + derv_y**2)
        orientaciones = np.abs(180*np.arctan2(derv_y, derv_x) / np.pi)
        orientaciones = np.where(orientaciones < 0, 180 + orientaciones, orientaciones)
        orientaciones[orientaciones == 180] = 0 # Para que angulo perteneca a [0º, 180º)

        return magnitudes, orientaciones

    def _calc_imagen_histograma(self, img_ori: np.ndarray, img_mag: np.ndarray, m: int=8) -> np.ndarray:
        if (np.abs(np.log2(m) - round(np.log2(m))) > 10e-2):
            raise Exception('El tamaño de celda tiene que ser potencia de 2')
        
        forma_histograma_bloques = (img_ori.shape[0] // m, img_ori.shape[1] // m, 9)
        imagen_histograma = np.zeros(shape=forma_histograma_bloques)
        for i in range(0, img_ori.shape[0], m):
            for j in range(0, img_ori.shape[1], m):
                bloque_ori = img_ori[i:i+m, j:j+m]
                bloque_mag = img_mag[i:i+m, j:j+m]
                hist_bloque = self._calc_histograma_bloque(bloque_ori, bloque_mag)
                imagen_histograma[i // m, j // m] = hist_bloque
        return imagen_histograma

    def _calc_histograma_bloque(self, bloque_ori: np.ndarray, bloque_mag: np.ndarray) -> np.ndarray:
        pos_l = (bloque_ori // 20).astype(np.uint8)
        pos_r = (pos_l + 1) % 9
        angulo_l = 20*pos_l
        angulo_r = 20*(pos_l + 1)
        valores_hist_l = (bloque_mag * ((angulo_r - bloque_ori) / 20))
        valores_hist_r = (bloque_mag * ((bloque_ori - angulo_l) / 20))

        hist = np.zeros(shape=(9, ))
        np.add.at(hist, pos_l.ravel(), valores_hist_l.ravel())
        np.add.at(hist, pos_r.ravel(), valores_hist_r.ravel())
        return hist
    
    def _normalizacion_por_bloques(self, imagen_hist: np.ndarray) -> np.ndarray:
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

    def calc_vector_caracteristicas(self, img: np.ndarray) -> np.ndarray:
        img_gris = np.mean(img, axis=2).astype(np.uint8)
        reescalada = self._reescalar_imagen(img_gris)

        tam_celda_ext = 16
        tam_celda_int = 8

        vec_carac_int = self.__calc_vector_carac_imagen_interior(reescalada, tam_celda_ext, tam_celda_int)
        vec_carac_ext = self.__calc_vector_carac_imagen_exterior(reescalada, tam_celda_ext)
        vec_carac = np.concatenate((vec_carac_int, vec_carac_ext))
        return vec_carac
        
    def __calc_vector_carac_imagen_interior(self, img: np.ndarray, tam_celda_ext: int, tam_celda_int: int) -> np.ndarray:
        img_interior = img[4*tam_celda_ext:-4*tam_celda_ext, 4*tam_celda_ext:-4*tam_celda_ext]
        vec_carac_interior = self.__calc_vector_carac_zona(img_interior, tam_celda_int)
        return vec_carac_interior
    
    def __calc_vector_carac_imagen_exterior(self, img: np.ndarray, tam_celda_ext: int) -> np.ndarray:
        borde_superior = img[:4*tam_celda_ext, :]
        borde_inferior = img[-4*tam_celda_ext:, :]
        borde_izquierdo = img[:, :4*tam_celda_ext]
        borde_derecho = img[:, -4*tam_celda_ext:]
        vec_carac_borde_sup = self.__calc_vector_carac_zona(borde_superior, tam_celda_ext)
        vec_carac_borde_inf = self.__calc_vector_carac_zona(borde_inferior, tam_celda_ext)
        vec_carac_borde_izq = self.__calc_vector_carac_zona(borde_izquierdo, tam_celda_ext)
        vec_carac_borde_der = self.__calc_vector_carac_zona(borde_derecho, tam_celda_ext)
        vec_carac_exterior = np.concatenate((
            vec_carac_borde_sup,
            vec_carac_borde_inf,
            vec_carac_borde_izq,
            vec_carac_borde_der
        ))
        return vec_carac_exterior

    def __calc_vector_carac_zona(self, zona: np.ndarray, tam_celda: int) -> np.ndarray:
        magnitudes, orientaciones = self._calcular_derivadas_Sobel(zona)
        img_hist = self._calc_imagen_histograma(orientaciones, magnitudes, tam_celda)
        vec_carac_zona = self._normalizacion_por_bloques(img_hist)
        return vec_carac_zona

class TransformadorCaracteristicas:
    def __init__(self, algoritmo: AlgoritmoCaracteristicas):
        self.__algoritmo = algoritmo

    def transformar(self, img: np.ndarray) -> np.ndarray:
        return self.__algoritmo.calc_vector_caracteristicas(img)

# ======== SISTEMA DE IA ========
class PredictorPerroGato:
    def __init__(self, tipo_modelo: str, max_est: int):
        ''' tipo_modelo: "adaboost", "random_forest '''
        if not(tipo_modelo in ['adaboost', 'random_forest']):
            raise Exception('Tipo de modelo incorrecto')
        if max_est < 2:
            raise Exception('Requisito: Numero de estimadores >= 2')
        
        self.__max_est = max_est
        self.__tipo_modelo = tipo_modelo
        self.__modelo = None

    def entrenar(
            self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray
        ) -> List[float]:
        
        accuracies = []
        best_acc = -np.inf
        best_model = None
        for n_est in range(2, self.__max_est + 1):
            modelo = self.__crear_modelo(n_est)
            modelo.fit(X_val, y_val)

            y_pred = modelo.predict(X_val)
            acc = np.where(y_val == y_pred)[0].size / y_pred.shape[0]
            if acc > best_acc:
                best_acc = acc
                best_model = modelo
            accuracies.append(acc)
        
        self.__modelo = best_model
        self.__modelo.fit(X_train, y_train)
        return accuracies

    def predecir(self, X: np.ndarray) -> np.ndarray:
        if self.__modelo is None:
            raise Exception("El modelo no esta entrenado.")
        return self.__modelo.predict(X)
    
    def __crear_modelo(self, n_estimadores: int):
        if self.__tipo_modelo == 'adaboost':
            return AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n_estimadores
            )
        elif self.__tipo_modelo == 'random_forest':
            return RandomForestClassifier(n_estimators=n_estimadores)

# ======== SISTEMA DE EXPERIMENTACION ========
class Experimento:
    def __init__(
            self, nombre: str,
            preprocesador: Preprocesador,
            algoritmo: AlgoritmoCaracteristicas,
            predictor: PredictorPerroGato,
            semilla: int
        ) -> None:

        self.__nombre = nombre
        self.__semilla = semilla
        self.__preprocesador = preprocesador
        self.__algoritmo = algoritmo
        self.__predictor = predictor

    def realizar(self):
        print('===============================')
        print('   ' + self.__nombre)
        print('===============================')
        
        t_ini = time.time()
        X_train, y_train, X_test, y_test = self.__leer_imagenes()
        X_train, X_test = self.__preprocesar_imagenes(X_train, X_test)
        X_train, X_test = self.__transformar_imagenes(X_train, X_test)
        accuracy = self.__utilizar_IA(
            np.array(X_train),
            np.array(y_train),
            np.array(X_test),
            np.array(y_test)
        )
        t_fin = time.time()
        
        print('   Resultados:')
        print(f'      Tiempo: {round(t_fin - t_ini, 2)}seg')
        print(f'      Tamaño del vector de caracteristicas: {X_train[0].shape}')
        print(f'      Accuracy en test: ', accuracy)
        print(f'      Sistemas: ', end='')
        print(f'{self.__preprocesador.__class__.__name__}', end=' ')
        print(f'{self.__algoritmo.__class__.__name__}', end=' ')
        print(f'{self.__predictor.__class__.__name__}')
        print(f'----------------------------------------', end='')
        print(f'----------------------------------------\n')

        # Sacamos gráficos

    def __leer_imagenes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print('   Leyendo imágenes...')
        lector = Lector(
            ruta_train_perros='dataset/cat_dog_100/train/dog',
            ruta_train_gatos='dataset/cat_dog_100/train/cat',
            ruta_test_perros='dataset/cat_dog_100/test/dog',
            ruta_test_gatos='dataset/cat_dog_100/test/cat',
            semilla=self.__semilla
        )
        X_train, y_train, X_test, y_test = lector.leer_dataset()
        return X_train, y_train, X_test, y_test
    
    def __preprocesar_imagenes(
            self,
            X_train: np.ndarray,
            X_test: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        print('   Preprocesando imágenes...')
        for i, img in enumerate(X_train):
            X_train[i] = self.__preprocesador.preprocesar(img)
        for i, img in enumerate(X_test):
            X_test[i] = self.__preprocesador.preprocesar(img)
        return X_train, X_test

    def __transformar_imagenes(
            self,
            X_train: np.ndarray,
            X_test: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:

        print('   Calculando vectores de caracteristicas...')
        transformador = TransformadorCaracteristicas(self.__algoritmo)
        for i, img in enumerate(X_train):
            X_train[i] = transformador.transformar(img)
        for i, img in enumerate(X_test):
            X_test[i] = transformador.transformar(img)
        return X_train, X_test
    
    def __utilizar_IA(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray
        ) -> float:
        """ Devuelve el accuracy """

        print('   Entrenando modelo de IA...')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25, random_state=self.__semilla)
        self.__predictor.entrenar(X_train, y_train, X_val, y_val)

        print('   Clasificando imágenes de test...')
        y_test_pred = self.__predictor.predecir(X_test)
        accuracy = np.where(y_test == y_test_pred)[0].size / y_test.size
        return accuracy

if __name__ == '__main__':
    exp1 = Experimento(
        nombre='Experimento 1',
        preprocesador=PrepGaussiano(7, 3),
        algoritmo=AlgoritmoHistograma(),
        predictor=PredictorPerroGato('random_forest', 1000),
        semilla=50
    )

    exp1.realizar()
