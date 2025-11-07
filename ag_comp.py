import ag_utils as ag
import numpy as np
import gmknn as gk
import pandas as pd


def run_test(dataset_name):
    X_test,Y_test,X_train,Y_train = ag.get_data(dataset_name,fill_invalid_with=np.nan,used_for='reg')

    ## gknn
    results_gknn = []
    k_gknn = range(1,31)
    n_gknn = range(3,31)
    w_id_gknn = [0,1]
    for i in w_id_gknn:
        for j  in n_gknn:
            for l in k_gknn:
                (mae_gknn,pct80_gknn) = gk.gknn(X_test,Y_test,X_train,Y_train,n=j,k=l,w_id=i)
                results_gknn.append({'n':j, 'k':l, 'weighted knn':i, 'MAE': mae_gknn, '80 percentile error': pct80_gknn})

    results_gknn_df =pd.DataFrame(results_gknn)
    file_path_gknn = 'results/gknn_'+dataset_name+'.csv'
    results_gknn_df.to_csv(file_path_gknn)
    ##knn
    results_knn = []
    k_knn = range(1,3)
    w_id_knn = [0,1]
    for i in w_id_knn:
        for j in k_knn:
            (mae_knn,pct80_knn) = gk.knn(X_test,Y_test,X_train,Y_train,k=j,wknn_id=i)
            results_knn.append({'k':j, 'weighted knn':i, 'MAE': mae_knn, '80 percentile error': pct80_knn})
    results_knn_df =pd.DataFrame(results_knn)
    file_path_knn = 'results/knn_'+dataset_name+'.csv'
    results_knn_df.to_csv(file_path_knn)

datasets=['DSI','LIB','TUT1','TUT2','TUT3','Uji']
for dataset_name in datasets:
    run_test(dataset_name)
    print(dataset_name+' finished')