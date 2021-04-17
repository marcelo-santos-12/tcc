import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    path_res = '../result_paper/results.csv'
    
    df_res = pd.read_csv(path_res)

    map_method = {
        'nri_uniform': 'u2',
        'uniform':'riu2'
        }
    def ret_format(method):
        return map_method[method]

    df_res['method'] = list(map(ret_format,df_res['method']))

    df_res.boxplot(by='variant', column=['best_matthews'])
    plt.ylim([0, 100])
    plt.title('MCC (%)')
    plt.xlabel('Descriptor')
    plt.savefig('variant_boxplot.png')
    plt.show()

    df_res.boxplot(by='classifier', column=['best_matthews'])
    plt.ylim([0, 100])
    plt.title('MCC (%)')
    plt.xlabel('Classifier')
    plt.savefig('classifier_boxplot.png')
    plt.show()

    df_res.boxplot(by='method', column=['best_matthews'])
    plt.ylim([0, 100])
    plt.title('MCC (%)')
    plt.xlabel('Method')
    plt.savefig('method_boxplot.png')
    plt.show()

    df_res.boxplot(by='(P, R)', column=['best_matthews'])
    plt.ylim([0, 100])
    plt.title('MCC (%)')
    plt.xlabel('P, R')
    #plt.savefig('p_r_boxplot.png')
    plt.show()


if __name__ == '__main__':

    main()