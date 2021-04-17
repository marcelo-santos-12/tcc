'''
Modulo que contem a funcao de calculo dos descritores do dataset, cujo retorno
sera os arrays contendo os dados escolhido para treinamento e teste
'''
import cv2
import numpy as np
from tqdm import tqdm
import glob
from sklearn.preprocessing import normalize
import os

def compute_features(path_dataset, descriptor, P, R, method, size_train, size_val, size_test, norm='l2'):
    '''
    Calcula os descritores de todas as imagens inseridas no diretorio do dataset

    Parameters
    ----------
        path_dataset: (str)
            Diretorio que contem as imagens a serem classificadas

        descriptor: (function)
            Funcao que calcula os descritores das imagens

        P,R: (int)
            Numero de pontos dentro de uma vizinhanca e \
            Raio de localizacao dos pontos da vizinhanca, respectivamente

        method: (string)
            Versao do LBP a ser calculada

        size_train, size_val, size_test: (float)
            Tamanho considerado para o dataset de treinamento, validação e teste, respectivemente.
            O valor somando das variaveis deve ser igual a 1.0
    '''

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
    for index, path_class in enumerate(glob.glob(path_dataset + '/*')):
        print('Classe {}: indice {}...'.format(path_class, index))

        dir_iter = glob.glob(path_class + '/*')
        length_dir = len(dir_iter)
        ind_max_train = int(size_train * length_dir)
        ind_max_val = ind_max_train + int(size_val * length_dir)

        for i, name_img in enumerate(tqdm(dir_iter)):

            # gerar excecao caso o arquivo lido nao seja imagem
            try:
                img = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
            except:
                print('Não é possível ler arquivo:', name_img)
                continue
            
            # calcula o histograma (array) descritor da imagem atual
            feature = descriptor(img, P=P, R=R, method=method,)

            # normalizacao l2 calculada no histograma
            feature = normalize(feature.reshape(1, -1), norm=norm).reshape(-1,)

            # adiciona ao conjunto de treinamento
            if i < ind_max_train:
                x_train.append(list(feature))
                y_train.append(index)

            # adiciona ao conjunto de validacao
            elif ind_max_train < i < ind_max_val+1:
                x_val.append(list(feature))
                y_val.append(index)

            # adiciona ao conjunto de teste
            else:
                x_test.append(list(feature))
                y_test.append(index)

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val), np.asarray(x_test), np.asarray(y_test)
