# functions used for algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor



def get_data(dataset_name, fill_invalid_with = np.nan, used_for= 'reg'):
#     in the original data, the rss value from the ap that was 
#     not detected is shown as 100, we can determined how it replace 
#     the invalid with value that is resonable

# time stamps indicator - 0/1
# multi-floor indicator - 0/1
# multi-building indicator - 0/1
# nan-value -- values
# floor information -- 'nan' -  no such information
#                      'rel' - relative position
#                      'abs' - absolute position
# used for cls for reg

    folder = r'data/'
    file_name = dataset_name + '.npz'
    path = folder + file_name
    data = np.load(path)
    tr_rss = data['tr_rss']
    ts_rss = data['ts_rss']
    tr_crd = data['tr_crd']
    ts_crd = data['ts_crd']
    invalid_value = data['nan_value']
    multi_flr_id = data['multi_fl_id']
    multi_bd_id = data['multi_bd_id']
    fl_type = data['fl_type']


    if multi_flr_id == 1:
        if multi_bd_id == 1:
            fl_ind = -2
        else:
            fl_ind = -1

        # print(fl_ind)

        if used_for == 'reg' and fl_type == 'rel':
            # print('go with reg')

            new_fl_tr = fl_cls2reg(tr_crd[:,fl_ind],dataset_name)
            new_fl_ts = fl_cls2reg(ts_crd[:,fl_ind],dataset_name)
            # print(new_fl_tr)
            tr_crd[:,fl_ind] = new_fl_tr
            ts_crd[:,fl_ind] = new_fl_ts
        elif used_for == 'cls' and fl_type == 'abs':

            # print('go with cls')

            new_fl_tr = fl_reg2cls(tr_crd[:,fl_ind])
            new_fl_ts = fl_reg2cls(ts_crd[:,fl_ind])
            tr_crd[:,fl_ind] = new_fl_tr
            ts_crd[:,fl_ind] = new_fl_ts

    if multi_bd_id == 1 and used_for == 'reg':
        tr_crd = tr_crd[:, :3]
        ts_crd = ts_crd[:, :3]

    if fill_invalid_with != "No_Op":
        ts_rss[ts_rss == invalid_value] = fill_invalid_with
        tr_rss[tr_rss == invalid_value] = fill_invalid_with


    return (ts_rss, ts_crd, tr_rss, tr_crd)


def fl_cls2reg(fl,dataset_name):
    fl_high = {
        'TUT2': #{fill_your_data},
    }
    dif = fl_high[dataset_name]
    min_value = np.min(fl)
    scaled_categories = (np.array(fl) - min_value) * dif
    return scaled_categories

def fl_reg2cls(fl):
    unique_values = np.unique(fl)
    value_to_category = {value: index for index, value in enumerate(unique_values)}
    categories = np.array([value_to_category[value] for value in fl])
    return categories

# calculate Euclidean distance
def euclidean_distance(data1,data2):
    distances = np.linalg.norm(data1 - data2, axis=1)
    return distances


# get the index of the n strongest siganl except nan in row
def max_n_indices(row, n):
    non_nan_indices = np.where(~np.isnan(row))[0]
    non_nan_values = row[non_nan_indices]
    sorted_indices = np.argsort(non_nan_values)[::-1][:n]
    return non_nan_indices[sorted_indices]

# get the index of the n strongest siganl except nan in each row
def get_top_n_indices(data, n):
    num_rows, num_cols = data.shape
    result = np.full((num_rows, n), np.nan)  # Initialize with NaN values
    
    for i in range(num_rows):
        indices = max_n_indices(data[i], n)
        if len(indices) < n:
            missing_value = n - len(indices)
            tem_indx = np.concatenate((indices, [np.nan] * missing_value), axis=0) 
            result[i] = tem_indx
        else:
            result[i] = indices  
    
    return result

# find the index of the rows in a matrix whose element are all in the target list
def find_rows_with_all_elements(matrix, target_list):

    matching_indices = []
    for i, row in enumerate(matrix):
        if all([element in target_list for element in row]):
            matching_indices.append(i)

    return matching_indices

# key procedure of the group matching algorithm
def get_fp(signal,raw_fp,raw_cord,n=8,threshold=1):
    
    tg = max_n_indices(signal,n)
    mt = get_top_n_indices(raw_fp,n)
    
    # new added
    if len(tg)<n:
        n = len(tg)

    loop_count = 0
  
    indx_stack = []
    while loop_count<n:
        mt_new = mt[:,:loop_count+1]
        indx = find_rows_with_all_elements(mt_new,tg)
        if len(indx)>=threshold:
            mt = raw_fp[indx]
            indx_stack.append(indx)
            loop_count += 1
        else:
            break
    
    if len(indx_stack)>0:
        final_indx =  indx_stack[-1]
        new_fp = raw_fp[final_indx,:]
        new_cord = raw_cord[final_indx,:]
    else:
        new_fp = raw_fp
        new_cord = raw_cord
        
    return (new_fp, new_cord)

# KNN algorithm
class KNNRegression:

    def __init__(self, k=3, wknn_id = 1):
        self.k = k
        self.model = None
        self.wknn_id = wknn_id
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        if len(self.X_train)<self.k:
            self.k = len(self.X_train)

        if self.wknn_id == 1:
            self.model = KNeighborsRegressor(n_neighbors=self.k, weights='distance')
        else:
            self.model = KNeighborsRegressor(n_neighbors=self.k, weights='uniform')

        self.model.fit(self.X_train, self.y_train)
        prediction = self.model.predict(X)
        # print('prediction shape', prediction.shape)

        return np.array(prediction).reshape(1, -1)


# remove nan values
def nan_replace(data, replace_with = np.nan):
    
    data[np.isnan(data)] = replace_with
    
    return data


# cdf plot
def cdf_plot(data):
    sorted_data = np.sort(data)

    # Calculate the cumulative probabilities
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Create the CDF plot
    plt.plot(sorted_data, cdf, marker='o')
    plt.xlabel('Error(m)')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF)')
    # Find the 80th percentile error
    percentile_80 = np.percentile(sorted_data, 80)
    
    # Mark the 80th percentile on the plot
    plt.axvline(x=percentile_80, color='red', linestyle='--', label='80th Percentile')
    
    plt.legend()  # Add legend to the plot
    plt.grid(True)
    plt.show()
    print(f'The 80th percentile error is: {percentile_80} meters')
    return

