#%%
import ag_utils as ag
import numpy as np

def gknn(X_test,Y_test,X_train,Y_train,groupm_n=8,gm_k=1,w_id = 1):

    Pred_list = np.zeros_like(Y_test)
    for (i, row) in enumerate(X_test):
        x_train, y_train = ag.get_fp(row, X_train, Y_train, n=groupm_n, threshold=1)
        x_train = ag.nan_replace(x_train, replace_with=-100)
        row = ag.nan_replace(row, replace_with=-100)
        KNN_reg = ag.KNNRegression(k=gm_k,wknn_id = w_id)
        KNN_reg.fit(x_train, y_train)
        y_pred = KNN_reg.predict(row.reshape(1, -1))
        Pred_list[i, :] = y_pred.reshape(1, -1)

    errors = ag.euclidean_distance(Y_test, Pred_list)
    mae = np.mean(errors)
    sorted_errors = np.sort(errors)
    percentile_80 = np.percentile(sorted_errors, 80)
    print("MAE", mae)
    print('percentile_80:',percentile_80)
    return (mae, percentile_80)


def knn(X_test,Y_test,X_train,Y_train,k=1,wknn_id = 1):

    X_train = ag.nan_replace(X_train, replace_with=-100)
    X_test = ag.nan_replace(X_test, replace_with=-100)

    if wknn_id == 1:
        w = 'distance'
    else:
        w = 'uniform'
    model = ag.KNNRegression(k=k,wknn_id = w)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    errrors = ag.euclidean_distance(Y_test,y_pred)
    mae = np.mean(errrors)
    sorted_errors = np.sort(errrors)
    percentile_80 = np.percentile(sorted_errors, 80)
    return (mae, percentile_80)