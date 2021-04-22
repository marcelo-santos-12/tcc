'''
Modulo que implementa as funcoes de plot da curva roc e armazenamento das predicoes
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import plot_roc_curve

plt.style.use('ggplot')


def plot_results(name_clf, best_clf, x_data, y_data, method, variant, P, R, output, t_data):

    path_pred  = output + '/ARR_ROC/' + variant + '/y_pred_' + t_data

    if not os.path.exists(path_pred):
        os.makedirs(path_pred)

    # salvando array para o calculo das metricas
    y_predict = best_clf.predict(x_data)
    filename_arr_pred = path_pred + '/{}_{}_{}_{}.txt'.format(method.upper(), str(P), str(R), name_clf.replace(' ', ''))
    np.savetxt(filename_arr_pred, y_predict)

    plot_roc_curve(best_clf, x_data, y_data)
    
    #PLOTANDO LINHA DIAGONAL --> y = x
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g', alpha=.8)

    # PLOTANDO INFORMACOES BASICA DO GRAFICO
    plt.title('{} - {}/{} - P: {}, R:{}'.format(name_clf, method.upper(), variant, P, R))
    plt.legend(loc="lower right")
    
    path_graphic_roc = output + '/ROC/'+variant+'_'+t_data
    if not os.path.exists(path_graphic_roc):
        os.makedirs(path_graphic_roc)

    plt.savefig(path_graphic_roc+'/{}_{}_{}_{}.png'.format(variant, method.upper(), name_clf.replace(' ', ''), P, R))
    plt.close()