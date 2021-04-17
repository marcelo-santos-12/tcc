from exec_run import run
import time

if __name__ == '__main__':

    ################ defining useful parameters #######################################
    DATASET = '2_SAMPLE_10_DATASET_GASTRIC_256'
    size_train = 0.7
    size_val   = 0.15
    size_test  = 0.15

    load = False

    t0 = time.time()

    print('Inicio dos experimentos...')
    for VARIANT in ['original_lbp', 'improved_lbp', 'hamming_lbp', 'completed_lbp', 'extended_lbp']:
        for METHOD in ['nri_uniform','uniform']:
 
            for P,R in [(8, 1), (16, 2), (24, 3)]:
                # inicia treinamento

                if (P,R) == (24,3) and VARIANT == 'hamming_lbp':
                    print('A variante Hamming LBP tem sua carga de processamento acrescida exponencialmente com o número de vizinhos observados.')
                    print('Isso torna a análise inviável para P=24')
                    continue

                run(dataset=DATASET, variant=VARIANT, method=METHOD, P=P, R=R,\
                    size_train_percent=size_train, size_val_percent=size_val, \
                    size_test_percent=size_test, load_descriptors=load, output='result_tcc')

    print('Fim dos Experimentos')

    runtime = (time.time() - t0) / 60 / 60
    print('Tempo Total: {}h'.format(round(runtime, 2)))

    from utils.facetgrid import plot_facetgrid

    #plot_facetgrid(output='result_paper')
