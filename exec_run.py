# modulos basicos
import numpy as np
import pandas as pd
import os
import argparse
import warnings

# algoritmos classificadores do sklearn
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# descritores de textura analisados
from lbp_module.texture import original_lbp, improved_lbp, hamming_lbp, completed_lbp, extended_lbp
from utils.features import compute_features
from utils.metrics import MyGridSearch
from utils.plotting import plot_results

# ignorando mensagens de aviso no terminal
warnings.filterwarnings("ignore")

# Para referencias aos descritores
ALGORITHM = {
    'original_lbp': [original_lbp, 'ORIGINAL LBP'],
    'improved_lbp': [improved_lbp, 'IMPROVED LBP'],
    'hamming_lbp': [hamming_lbp, 'HAMMING LBP'],
    'completed_lbp': [completed_lbp, 'COMPLETED LBP'],
    'extended_lbp': [extended_lbp, 'EXTENDED LBP'],
}

def add_to_df(name_clf, variant, method, P, R, param, matthews, f1score, accuracy, matrix, auc_roc, n_features, is_validation):
    _dic_df = {}
    _dic_df['classifier'] = name_clf
    _dic_df['variant'] = variant
    _dic_df['method'] = method
    _dic_df['(P, R)'] = (P, R)
    _dic_df['parameters'] = param
    _dic_df['best_matthews'] = np.round(matthews*100, 2)
    _dic_df['fscore'] = np.round(f1score*100, 2)
    _dic_df['accuracy'] = np.round(accuracy* 100, 2)
    _dic_df['confusion_matrix'] = matrix
    _dic_df['auc_roc'] = np.round(auc_roc * 100, 2)
    _dic_df['n_features'] = n_features
    _dic_df['is_validation'] = is_validation
    return _dic_df

def run(dataset, variant, method, P, R, size_train_percent, size_val_percent, size_test_percent, load_descriptors, output):
    ################ computing image descriptors #######################################
    print(30 * ' * ')
    print('Experimento: Descritor={}, Method={}, P = {}, R = {}'.format(ALGORITHM[variant][1], method, P, R))

    path_cur = os.path.join(output, 'descriptors', ALGORITHM[variant][1], method.upper() + '_' + str(P) +'_' +str(R)) 
    if not load_descriptors: # Computar os descritores de cada imagem do dataset
        print('Computando recursos...')
        x_train, y_train, x_val, y_val, x_test, y_test = compute_features(path_dataset=dataset, \
            descriptor=ALGORITHM[variant][0], P=P, R=R, method=method, \
            size_train=size_train_percent, size_val=size_val_percent, size_test=size_test_percent)
        
        if not os.path.exists(path_cur):
            os.makedirs(path_cur)

        # concatenando conjunto de treino, validacao e teste em um unico arquivo .txt
        all_descriptors_train = np.concatenate([x_train, y_train.reshape(-1, 1)], axis=1)
        all_descriptors_val = np.concatenate([x_val, y_val.reshape(-1, 1)], axis=1)
        all_descriptors_test = np.concatenate([x_test, y_test.reshape(-1, 1)], axis=1)
        all_descriptors = np.concatenate([all_descriptors_train, all_descriptors_val, all_descriptors_test], axis=0)
        np.savetxt(path_cur + '/features.txt', all_descriptors)

    else: # carregar recursos pre-computados
        print('Carregando recursos...')
        if os.path.exists(path_cur):
            data = np.loadtxt(path_cur + '/features.txt')

            # separando conjunto de treinamento, validacao e teste
            length_train = int(size_train_percent * data.shape[0])
            length_val = length_train + int(size_val_percent * data.shape[0])
            # separando linhas
            train = data[:length_train]
            val = data[length_train:length_val]
            test = data[length_val:]

            # separando colunas
            x_train, y_train = train[:, :-1], train[:, -1]
            x_val, y_val = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

        else:
            print("Descritores não computados. Insira um caminho valido.")
            quit()
    
    path_roc_variant = os.path.join(output, 'ARR_ROC', ALGORITHM[variant][1], 'y_true_val')
    if not os.path.exists(path_roc_variant):
        os.makedirs(path_roc_variant)

    path_roc_variant = os.path.join(output, 'ARR_ROC', ALGORITHM[variant][1], 'y_true_test')
    if not os.path.exists(path_roc_variant):
        os.makedirs(path_roc_variant)
    
    # armazenando Y true utilizado para validacao e test 
    arr_filename_true_val = os.path.join(output, 'ARR_ROC', ALGORITHM[variant][1], 'y_true_val', '{}_{}_{}.txt'.format(method.upper(), P, R))
    np.savetxt(arr_filename_true_val, y_val)
    arr_filename_true_test = os.path.join(output,'ARR_ROC', ALGORITHM[variant][1], 'y_true_test', '{}_{}_{}.txt'.format(method.upper(), P, R))
    np.savetxt(arr_filename_true_test, y_test)
    
    # Dataframe onde sera guardado os resultados dos experimentos
    # sempre sera adicionado novos valores ao final do arquivo, 
    # independente se estes resultados já existem
    # caso o arquivo de resultados nao exista, sera criado um novo automaticamente
    n_features = x_train[0].shape[0]
    if os.path.exists(output + '/results.csv'):
        df_results = pd.read_csv(output + '/results.csv')
    else:
        columns = ['variant', 'classifier', 'method', '(P, R)', 'parameters', \
                   'best_matthews','fscore', 'accuracy', 'confusion_matrix','auc_roc', 'n_features', 'is_validation']
        df_results = pd.DataFrame(columns=columns)

    ############### Defining classifiers object and its parameters for GridSearch ############
    mlp = MLPClassifier()
    svm = SVC()
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()

    mlp_parameters = {
        'hidden_layer_sizes': [(10, 10), (50, 50), (100, 100)],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'learning_rate_init': [0.01, 0.001, 0.0001],
        'solver': ['adam', 'sgd'],
        'max_iter': [50, 100, 200],
    }
    svm_parameters = {
        'C': [2, 10, 50, 100],
        'kernel': ['linear', 'rbf', 'sigmoid',],
        'gamma': [0.001, 0.0001, 0.0001, 0.00001],
    }
    knn_parameters = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights' : ['uniform', 'distance'],
        'algorithm': ['kd_tree', 'ball_tree'],
    }
    dt_parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 20, 30, 50],
    }
    rf_parameters = {
        'n_estimators': [5, 11, 51, 101],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 50, 100, 200],
    }

    classifiers =  [['MLP', mlp, mlp_parameters],                \
                    ['SVM', svm, svm_parameters],                \
                    ['K-Nearest Neighbor', knn, knn_parameters], \
                    ['Decision Trees', dt, dt_parameters],       \
                    ['Random Forest', rf, rf_parameters],        \
                    ]

    ################### Starting the train of models ##################################
    for name_clf, clf, parameters in classifiers:
        np.random.seed(10)
        res_curr_val = {} # resultados do experimentos com o conjunto de validacao
        res_curr_test = {} # resultados do experimentos com o conjunto de test
        print(35 * ' # ')
        print('Classificando com {}...'.format(name_clf))

        # CROSS-VALIDATION HOLD OUT --> train: 0.8, val:0.1, test: 0.1
        
        ################# Executing GridSearch ###################################

        clf_search = MyGridSearch(classifier=clf, grid_parameters=parameters,)

        res_search_val, res_search_test = clf_search.fit(x_train, y_train, x_val, \
                                            y_val, x_test, y_test)

        ################### Computing Performance #################################
        # Compute ROC curve and area the curve
        plot_results(name_clf, res_search_val['best_clf'], x_val, y_val, method.upper(), ALGORITHM[variant][1], P, R, output, 'val')

        plot_results(name_clf, res_search_val['best_clf'], x_test, y_test, method.upper(), ALGORITHM[variant][1], P, R, output, 'test')

        # Get AUC, F1Score and Accuracy metrics from Validation and Test 
        print(25 * '-')
        print('Melhor Parametro: ', res_search_val['best_parameter']) 
        
        # VALIDACAO
        auc_roc_val = res_search_val['auc']
        f1score_val = res_search_val['f1score']
        accuracy_val = res_search_val['accuracy']
        matthews_val = res_search_val['best_matthews']
        matrix_val = res_search_val['confusion_matrix']

        print('F1Score Validation:' , f1score_val)
        print('Accuracy Validation:', accuracy_val)
        print('AUC Validation:', auc_roc_val)
        print('Matthews Validation:', matthews_val)

        header = (name_clf, ALGORITHM[variant][1], method, P, R, res_search_val['best_parameter'])
        
        # adicionando resultados do conjunto de validacao aos resultados do melhor modelo
        def_res_val = add_to_df(*header, matthews_val, f1score_val, \
                    accuracy_val, matrix_val, auc_roc_val, n_features, True)

        df_results = df_results.append(def_res_val, ignore_index=True)

        # TEST
        auc_roc_test = res_search_test['auc']
        f1score_test= res_search_test['f1score']
        accuracy_test = res_search_test['accuracy']
        matthews_test = res_search_test['best_matthews']
        matrix_test = res_search_test['confusion_matrix']

        print('F1Score Test:' , f1score_test)
        print('Accuracy Test:', accuracy_test)
        print('AUC Test:', auc_roc_test)
        print('Matthews Validation:', matthews_test)

        # adicionando resultados do conjunto de teste ao dataframe, baseado no melhor modelo
        def_res_test = add_to_df(*header, matthews_test, f1score_test, \
                    accuracy_test, matrix_test, auc_roc_test, n_features, False)

        df_results = df_results.append(def_res_test, ignore_index=True)

    df_results.to_csv(output + '/results.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', '-d', help='Dataset that contains the images', type=str, \
        required=True)
    
    # variant
    parser.add_argument('--variant', '-v',
        help='Descritor LBP variant accepted: original_lbp, improved_lbp, \
              hamming_lbp, completed_lbp, extended_lbp', type=str, \
        required=True)

    # method
    parser.add_argument('--method', '-m', help='Method Accepted: `nri_uniform` and `uniform`', type=str,\
        default='uniform')
    
    # P
    parser.add_argument('--points', '-p', help='Number of points at neighboorhood', type=int, \
        default=8)
    
    # R
    parser.add_argument('--radius', '-r', help='Radius of points at neighboorhood', type=int, \
        default=1)
    
    # size train from dataset
    parser.add_argument('--size_train', '-s', help='Length of train dataset', type=float, \
        default=.8)

    # size validation from dataset
    parser.add_argument('--size_val', '-s', help='Length of validation dataset', type=float, \
        default=.1)

    # size test from dataset
    parser.add_argument('--size_test', '-s', help='Length of test dataset', type=float, \
        default=.1)

    # path_results
    parser.add_argument('--output', '-o', help='Path to output results', type=str, \
        default='results')

    # path_results
    parser.add_argument('--load', '-l', help='Save descriptors computed', type=bool, \
        default=False)

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print('Invalid Path to dataset...')
        quit()

    run(dataset=args.dataset, variant=args.variant, \
        method=args.method, P=args.points, R=args.radius,\
        size_train_percent=args.size_train, size_val_percent=args.size_val, \
        size_test_percent=args.size_test, load_descriptors=args.load, output=args.output)
