import numpy as np
import os
import cv2
import random
import time
import matplotlib.pylab as plt
import pickle
import multiprocessing

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
        self.__REESCALADO = (256, 256) # No tocar

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
            img = self.__reescalar_imagen(img)
            imagenes.append(img)
        return imagenes

    def __reescalar_imagen(self, img: np.ndarray) -> np.ndarray:
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

    def calc_vector_caracteristicas(self, img: np.ndarray) -> np.ndarray:
        """ Requisito: Imagen cuadrada y multiplo de 8 """
        if img.shape[0] != img.shape[1] or img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
            raise Exception('Requisito: Imagen cuadrada y multiplo de 8')

        tam_celda = 8
        img_gris = np.mean(img, axis=2).astype(np.uint8)
        reescalada = cv2.resize(img_gris, (128, 128), interpolation=cv2.INTER_AREA)

        magnitudes, orientaciones = self._calcular_derivadas_Sobel(reescalada)
        imagen_histogramas = self._calc_imagen_histograma(orientaciones, magnitudes, tam_celda)
        vec_carac = self._normalizacion_por_bloques(imagen_histogramas)
        return vec_carac

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

class AlgoritmoNuestro(AlgoritmoCaracteristicas):
    def __init__(self):
        super().__init__()

        self.__alg_tex = AlgoritmoTexturas()

    def calc_vector_caracteristicas(self, img: np.ndarray) -> np.ndarray:
        """ Requisito: Imagen (256x256) """
        if img.shape[0] != 256 and img.shape[1] != 256:
            raise Exception('Requisito: Imagen (256x256)')
        
        vec_carac_int = self.__calc_vector_carac_imagen_interior(img)
        vec_carac_ext = self.__calc_vector_carac_imagen_exterior(img)
        vec_carac = 0.8*vec_carac_int + 0.2*vec_carac_ext
        return vec_carac
        
    def __calc_vector_carac_imagen_interior(self, img: np.ndarray) -> np.ndarray:
        img_interior = img[64:-64, 64:-64, :]
        vec_carac_interior = self.__alg_tex.calc_vector_caracteristicas(img_interior)
        return vec_carac_interior
    
    def __calc_vector_carac_imagen_exterior(self, img: np.ndarray) -> np.ndarray:
        borde_superior = img[:64, :, :]
        borde_inferior = img[-64:, :, :]
        borde_izquierdo = img[64:-64, :64, :]
        borde_derecho = img[64:-64:, -64:, :]
        vec_carac_borde_sup = self.__alg_tex.calc_vector_caracteristicas(borde_superior)
        vec_carac_borde_inf = self.__alg_tex.calc_vector_caracteristicas(borde_inferior)
        vec_carac_borde_izq = self.__alg_tex.calc_vector_caracteristicas(borde_izquierdo)
        vec_carac_borde_der = self.__alg_tex.calc_vector_caracteristicas(borde_derecho)
        vec_carac_exterior = vec_carac_borde_sup + vec_carac_borde_inf
        vec_carac_exterior += vec_carac_borde_izq + vec_carac_borde_der
        return vec_carac_exterior

class TransformadorCaracteristicas:
    def __init__(self, algoritmo: AlgoritmoCaracteristicas):
        self.__algoritmo = algoritmo

    def transformar(self, img: np.ndarray) -> np.ndarray:
        return self.__algoritmo.calc_vector_caracteristicas(img)

# ======== SISTEMA DE IA ========
class PredictorPerroGato(ABC):
    def __init__(self, min_est: int, max_est: int):
        ''' Rango numero estimadores: [min_est, max_est) '''
        if min_est < 1 or max_est < 2:
            raise Exception('Requisito: Numero de estimadores >= 2')
        if max_est <= min_est:
            raise Exception('Requisito: min_est < max_est')

        self.__min_est = min_est
        self.__max_est = max_est
        self.__modelo = None

    def entrenar(
            self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray
        ) -> List[float]:
        
        accs_val = []
        accs_train = []
        best_n_est = None
        best_model = None
        best_acc_val = -np.inf
        for n_est in range(self.__min_est, self.__max_est):
            self.__modelo = self.crear_modelo(n_est)
            self.__modelo.fit(X_train, y_train)

            y_pred_train = self.__modelo.predict(X_train)
            acc_train = np.where(y_train == y_pred_train)[0].size / y_pred_train.shape[0]
            accs_train.append(acc_train)

            y_pred_val = self.__modelo.predict(X_val)
            acc_val = np.where(y_val == y_pred_val)[0].size / y_pred_val.shape[0]
            accs_val.append(acc_val)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_n_est = n_est
                best_model = self.__modelo
        
        self.__modelo = best_model
        self.__modelo.fit(X_train, y_train)
        return accs_val, accs_train, best_n_est

    def predecir(self, X: np.ndarray) -> np.ndarray:
        if self.__modelo is None:
            raise Exception("El modelo no esta entrenado.")
        return self.__modelo.predict(X)
    
    @abstractmethod
    def crear_modelo(self, n_estimadores: int):
        pass

class PredictorAdaboost(PredictorPerroGato):
    def __init__(self, min_est: int, max_est: int, semilla: int):
        super().__init__(min_est, max_est)

        self.__semilla = semilla

    def crear_modelo(self, n_estimadores: int):
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=self.__semilla),
            n_estimators=n_estimadores
        )
    
class PredictorRandomForest(PredictorPerroGato):
    def __init__(self, min_est: int, max_est: int, semilla: int):
        super().__init__(min_est, max_est)

        self.__semilla = semilla

    def crear_modelo(self, n_estimadores: int):
        return RandomForestClassifier(n_estimators=n_estimadores, random_state=self.__semilla)

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

        resultados = self.__ejecutar_procedimiento()

        preprocesador = resultados['preprocesador']
        algoritmo = resultados['algoritmo']
        predictor = resultados['predictor']
        tiempo = resultados['tiempo']
        tam_vector = resultados['tam_vector']
        best_n_est = resultados['best_n_est']
        acc = resultados['acc']
        acc_gatos = resultados['acc_gatos']
        acc_perros = resultados['acc_perros']

        print('   Resultados:')
        print(f'      Tiempo: {tiempo}seg')
        print(f'      Tamaño del vector de caracteristicas: {tam_vector}')
        print(f'      Mejor n_est: ', best_n_est)
        print(f'      Accuracy en test: ', acc)
        print(f'      Accuracy en test (Gatos): ', acc_gatos)
        print(f'      Accuracy en test (Perros): ', acc_perros)
        print(f'      Sistemas: ', end='')
        print(f'{preprocesador}', end=' ')
        print(f'{algoritmo}', end=' ')
        print(f'{predictor}')
        print(f'----------------------------------------', end='')
        print(f'----------------------------------------\n')

        return resultados

    def __ejecutar_procedimiento(self):
        t_ini = time.time()
        X_train, y_train, X_test, y_test = self.__leer_imagenes()
        X_train, X_test = self.__preprocesar_imagenes(X_train, X_test)
        X_train, X_test = self.__transformar_imagenes(X_train, X_test)
        best_n_est, acc, acc_gatos, acc_perros, accs_val, accs_train = self.__utilizar_IA(
            np.array(X_train),
            np.array(y_train),
            np.array(X_test),
            np.array(y_test)
        )
        t_fin = time.time()
        tiempo = round(t_fin - t_ini, 2)

        resultados = {
            'preprocesador': self.__preprocesador.__class__.__name__,
            'algoritmo': self.__algoritmo.__class__.__name__,
            'predictor': self.__predictor.__class__.__name__,
            'tiempo': tiempo,
            'tam_vector': X_train[0].shape,
            'best_n_est': best_n_est,
            'acc': acc,
            'acc_gatos': acc_gatos,
            'acc_perros': acc_perros,
            'accs_val': accs_val,
            'accs_train': accs_train
        }
        return resultados

    def __leer_imagenes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print('   Leyendo imágenes...')
        lector = Lector(
            ruta_train_perros='dataset/cat_dog_500/train/dog',
            ruta_train_gatos='dataset/cat_dog_500/train/cat',
            ruta_test_perros='dataset/cat_dog_500/test/dog',
            ruta_test_gatos='dataset/cat_dog_500/test/cat',
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
        accs_val, accs_train, best_n_est = self.__predictor.entrenar(X_train, y_train, X_val, y_val)
        print('   Clasificando imágenes de test...')

        y_test_pred = self.__predictor.predecir(X_test)
        acc = np.where(y_test == y_test_pred)[0].size / y_test.size
        
        mascara_gatos = y_test == ClasePerroGato.CLASE_GATO
        acc_gatos = np.where(y_test[mascara_gatos] == y_test_pred[mascara_gatos])[0].size / y_test[mascara_gatos].size

        mascara_perros = y_test == ClasePerroGato.CLASE_PERRO
        acc_perros = np.where(y_test[mascara_perros] == y_test_pred[mascara_perros])[0].size / y_test[mascara_perros].size

        return best_n_est, acc, acc_gatos, acc_perros, accs_val, accs_train

if __name__ == '__main__':
    def experimento_proceso(exp: Experimento):
        return exp.realizar()
    
    semilla = 50
    min_est = 1
    max_est = 41
    nucleos_fisicos = 4  # Nucleos fisicios de mi CPU
    resultados = []

    for m in [7]:
        preprocesadores = [
            PrepMedia(m), PrepMediana(m), PrepGaussiano(m, 3),
            PrepMediaEqu(m), PrepMedianaEqu(m), PrepGaussianoEqu(m, 3)
        ]
        algoritmos = [
            AlgoritmoHistograma(), AlgoritmoTexturas(),
            AlgoritmoOrientaciones(), AlgoritmoNuestro()
        ]
        predictores = [
            PredictorRandomForest(min_est, max_est, semilla),
            PredictorAdaboost(min_est, max_est, semilla)
        ]

        # Crear lista de todos los experimentos
        lista_experimentos = [
            Experimento('Experimento', prep, alg, pred, semilla)
            for prep in preprocesadores
            for alg in algoritmos
            for pred in predictores
        ]

        # Ejecutar experimentos en paralelo usando todos los núcleos
        with multiprocessing.Pool(processes=nucleos_fisicos) as pool:
            resultados = pool.map(experimento_proceso, lista_experimentos)

        # Guardar resultados
        ruta = f'experimentos/AllPrep({m}x{m})_AllAlg_AllPred.pkl'
        with open(ruta, 'wb') as f:
            pickle.dump(resultados, f)

        resultados.clear()

    # Leer fichero de resultados
    # with open('prep_alg_pred.pkl', 'rb') as f:
        # resultados = pickle.load(f)
    # print(resultados)