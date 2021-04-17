import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import auc, roc_curve

plt.style.use('seaborn')

TITLE_SIZE = 18
AXIS_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 22

plt.rc('axes', titlesize=TITLE_SIZE, labelsize=AXIS_SIZE)
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('legend', fontsize=LEGEND_SIZE)

def main():

    method_form = {
        'uniform': 'LBP(riu2)' ,
        'nri_uniform': 'LBP(u2)'
    }

    variant_form = {
        'ORIGINAL LBP': '',
        'IMPROVED LBP': 'i',
        'HAMMING LBP': 'h',
        'COMPLETED LPB': 'c',
        'EXTENDED LBP': 'e'
    }

    VARIANT = { 'ORIGINAL LBP': 'original_lbp',
                'IMPROVED LBP': 'improved_lbp',
                'HAMMING LBP': 'hamming_lbp',
                'COMPLETED LPB': 'completed_lbp',
                'EXTENDED LBP': 'extended_lbp'
              }

    variant = 'ORIGINAL LBP'
    classifier = 'MLP'
    method = 'nri_uniform'
    P, R = (16, 2)

    path_true = os.path.join('../result_tcc/ARR_ROC', VARIANT[variant] + '_y_true_test_' + method + '_'+str(P)+'_'+str(R)+'.txt')
    path_roc  = os.path.join('../result_tcc/ARR_ROC', variant, 'y_pred_test_roc', method.upper()+ '_' +classifier+ '_'+str(P)+'_'+str(R)+'.txt')

    y_true = np.loadtxt(path_true)
    y_roc  = np.loadtxt(path_roc)

    title = variant_form[variant] + method_form[method] + ' - P(' +str(P)+') / R(' +str(R) + ') - ' + classifier
    
    fpr, tpr, _ = roc_curve(y_true, y_roc)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label= 'AUC {}'.format(np.round(auc(fpr, tpr), 2)))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.title(title)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.legend(loc=4)

    file_path = variant+'_'+method+'_'+str(P)+'_'+str(R)+'.png'
    plt.savefig(file_path)
    plt.close()


if __name__ == '__main__':

    main()
    