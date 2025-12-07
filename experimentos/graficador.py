import numpy as np
import pickle
from matplotlib import pyplot as plt

def dibujar_grafico_barras(accs: list, labels: list, titulo: str, titulo_leyenda: str):
    # accs(Numero de barras por grupo X Número de grupos)
    # labels(Numero de barras por grupo)

    fig, ax = plt.subplots(figsize=(8,6))

    x = ['Adaboost', 'RandomForest']
    x_indices = np.array([0.25, 0.75]) 

    # Ajustes de tamaño
    num_bars = 6
    total_group_width = 0.35 
    bar_width = total_group_width / num_bars 
    start_position = [-total_group_width / 2 + 0.03, -total_group_width / 2 + 0.03]

    # Dibujar barras (6 datasets)
    barras = []
    colores = ["#66FF7A", "#00BFFF", "#FFB266", "#A066FF", "#FF8C66", "#66FFE8"]
    for i in range(len(accs)):
        bar = ax.bar(x_indices + start_position + i*bar_width, accs[i], bar_width, label=labels[i], color=colores[i % len(colores)])
        barras.append(bar)

    # Títulos y etiquetas
    ax.set_title(titulo, pad=25, fontsize=22)
    ax.set_xlabel('Predictores', labelpad=20, fontsize=14)
    ax.set_ylabel('Accuracy', labelpad=20, fontsize=14)
    ax.set_facecolor('#f1f1f1')

    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_ylim(0, 1.2)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x, fontsize=14)

    # Texto encima de cada barra
    for bars in barras:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height + 0.02,
                f"{height:.2f}", ha='center', va='bottom',
                color='#333333', fontsize=10
            )

    ax.legend(title=titulo_leyenda, loc='upper right', fontsize=10, title_fontsize=11, ncol=3)
    plt.tight_layout()
    plt.show()

def dibujar_grafico_funcion(x_estimadores: list, accs_train: list, accs_val: list):
    fig, ax = plt.subplots()

    ax.plot(x_estimadores, accs_train)
    ax.plot(x_estimadores, accs_val)

    plt.show()

def generar_graficos_barras(ruta: str, titulo_leyenda: str):
    # Leemos los datos
    with open(ruta, 'rb') as f:
        resultados = pickle.load(f)
    
    # Metemos los resultados de cada algoritmo en las listas
    resultados_alg_hist = []
    resultados_alg_texturas = []
    resultados_alg_orientaciones = []
    resultados_alg_nuestro = []
    for res in resultados:
        if res['algoritmo'] == 'AlgoritmoHistograma':
            resultados_alg_hist.append(res)
        elif res['algoritmo'] == 'AlgoritmoTexturas':
            resultados_alg_texturas.append(res)
        elif res['algoritmo'] == 'AlgoritmoOrientaciones':
            resultados_alg_orientaciones.append(res)
        elif res['algoritmo'] == 'AlgoritmoNuestro':
            resultados_alg_nuestro.append(res)

    # Mostramos los graficos de cada algoritmo
    labels = [
        'PrepMedia',
        'PrepMediana',
        'PrepGaussiano',
        'PrepMediaEqu',
        'PrepMedianaEqu',
        'PrepGaussianoEqu'
    ]
    titulos = ['AlgoritmoHistograma', 'AlgoritmoTexturas', 'AlgoritmoOrientaciones', 'AlgoritmoNuestro']
    for i, res_alg in enumerate([resultados_alg_hist, resultados_alg_texturas, resultados_alg_orientaciones, resultados_alg_nuestro]):
        accs = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        for res in res_alg:
            if res['predictor'] == 'PredictorAdaboost':
                if res['preprocesador'] == 'PrepMedia':
                    accs[0][0] = res['acc']
                elif res['preprocesador'] == 'PrepMediana':
                    accs[1][0] = res['acc']
                elif res['preprocesador'] == 'PrepGaussiano':
                    accs[2][0] = res['acc']
                elif res['preprocesador'] == 'PrepMediaEqu':
                    accs[3][0] = res['acc']
                elif res['preprocesador'] == 'PrepMedianaEqu':
                    accs[4][0] = res['acc']
                elif res['preprocesador'] == 'PrepGaussianoEqu':
                    accs[5][0] = res['acc']
            elif res['predictor'] == 'PredictorRandomForest':
                if res['preprocesador'] == 'PrepMedia':
                    accs[0][1] = res['acc']
                elif res['preprocesador'] == 'PrepMediana':
                    accs[1][1] = res['acc']
                elif res['preprocesador'] == 'PrepGaussiano':
                    accs[2][1] = res['acc']
                elif res['preprocesador'] == 'PrepMediaEqu':
                    accs[3][1] = res['acc']
                elif res['preprocesador'] == 'PrepMedianaEqu':
                    accs[4][1] = res['acc']
                elif res['preprocesador'] == 'PrepGaussianoEqu':
                    accs[5][1] = res['acc']

        dibujar_grafico_barras(accs, labels, titulos[i], titulo_leyenda)

if __name__ == '__main__':
    generar_graficos_barras('AllPrep(3x3)_AllAlg_AllPred.pkl', 'Preprocesador (3x3)')
    generar_graficos_barras('AllPrep(7x7)_AllAlg_AllPred.pkl', 'Preprocesador (7x7)')