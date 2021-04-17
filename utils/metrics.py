'''
Modulo que implementa o GridSearch
'''
import numpy as np
import os
from itertools import product
from sklearn.metrics import (roc_curve, f1_score, auc,
                            accuracy_score, matthews_corrcoef,
                            confusion_matrix)

from sklearn.metrics._plot.base  import _get_response

from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle


class MyGridSearch():

    def __init__(self, classifier, grid_parameters):
        self.classifier = classifier
        self.grid_parameters = grid_parameters
        self.all_parameter_comb = self._get_all_parameter_comb(self.grid_parameters)

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.results_val = {}
        self.results_test = {}
        self.results_val['best_matthews'] = -1

        if isinstance(self.classifier, MLPClassifier): # embaralha o conjunto de dados quando o classificador for MLP
            x_train, y_train = shuffle(x_train, y_train, random_state=100)

        for (i, parameters_i) in enumerate(self.all_parameter_comb):

            if isinstance(self.classifier, MLPClassifier) and len(parameters_i) == 6:
                parameters_i[0] = (parameters_i[0], parameters_i[1])
                parameters_i.pop(1)

            parameters_to_clf = dict(zip(self.grid_parameters.keys(), parameters_i))

            clf_i = clone(self.classifier)

            clf_i.set_params(**parameters_to_clf)

            clf_i.fit(X=x_train, y=y_train)

            y_pred_val = clf_i.predict(x_val)
            
            matthews_val = matthews_corrcoef(y_val, y_pred_val)

            if self.results_val['best_matthews'] < matthews_val:

                self.results_val['best_matthews'] = matthews_val
                self.results_val['best_clf'] = clf_i
                y_pred_val_final = y_pred_val

        # RESULTADOS NO MELHOR CONJUNTO DE VALIDACAO
        y_pred_roc_val, _ = _get_response(x_val, self.results_val['best_clf'], 'auto', pos_label=None)
        fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_roc_val, pos_label=None, sample_weight=None, drop_intermediate=True)
        self.results_val['f1score'] = f1_score(y_val, y_pred_val_final)
        self.results_val['auc'] = auc(fpr_val, tpr_val)
        self.results_val['accuracy'] = accuracy_score(y_val, y_pred_val_final)
        self.results_val['confusion_matrix'] = confusion_matrix(y_val, y_pred_val_final)
        self.results_val['best_parameter'] = dict(zip(self.grid_parameters.keys(), parameters_i))

        # resultados do conjunto de teste utilizando o melhor classificador encontrado
        y_pred_test = self.results_val['best_clf'].predict(x_test)

        # salvar y_pred_test em um arquivo
        
        y_pred_roc_test, _ = _get_response(x_test, self.results_val['best_clf'], 'auto', pos_label=None)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_roc_test, pos_label=None, sample_weight=None, drop_intermediate=True)

        self.results_test['best_matthews'] = matthews_corrcoef(y_test, y_pred_test)
        self.results_test['f1score'] = f1_score(y_test, y_pred_test)
        self.results_test['auc'] = auc(fpr_test, tpr_test)
        self.results_test['accuracy'] = accuracy_score(y_test, y_pred_test)
        self.results_test['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)

        return self.results_val, self.results_test

    def _get_all_parameter_comb(self, grid_parameters):
        list_comb = []
        for i, k in enumerate(grid_parameters.keys()):
            if i == 0:
                list_comb = grid_parameters[k]
                continue
            list_comb = [(x,y) for x, y in product(list_comb, grid_parameters[k])]

        def formated(tuple_arr):

            def just_list(_list):
                if isinstance(_list, (list, tuple)):
                    return [sub_elem for elem in _list for sub_elem in just_list(elem)]
                else:
                    return [_list]

            _list_formated = []
            for _list in tuple_arr:
                _list_formated.extend(just_list(_list))
            return _list_formated

        return list(map(formated, list_comb))


if __name__ == '__main__':

    svm_parameters = {
        'C': [2, 10, 50, 100],
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly2', 'poly3'],
        'gamma': [0.1, 0.0001, 0.000001, 0.0000001],
    }

    gs = MyGridSearch(classifier=(SVC()), grid_parameters=svm_parameters)

    print(gs._get_all_parameter_comb(svm_parameters))
