# Here are some functions for the the data analysis in this project

import numpy as np
import matplotlib.pyplot as plt
import csv
import random


# return testing rss/coordinates training rss/coordinates
def get_data(path):
    
    ts_rss_path = path + '\Test_rss_21Aug17.csv'
    ts_cord_path = path + '\Test_coordinates_21Aug17.csv'
    tr_rss_path = path + '\Training_rss_21Aug17.csv' 
    tr_cord_path = path + '\Training_coordinates_21Aug17.csv'
    
    ts_rss = np.genfromtxt(ts_rss_path, delimiter=',')
    ts_cord = np.genfromtxt(ts_cord_path, delimiter=',')
    tr_rss = np.genfromtxt(tr_rss_path, delimiter=',')
    tr_cord = np.genfromtxt(tr_cord_path, delimiter=',')
    
    #fill nan value (100) with "0" 
    ts_rss[ts_rss==100] = np.nan   
    tr_rss[tr_rss==100] = np.nan  
    return (ts_rss, ts_cord, tr_rss, tr_cord)


# show histgram for valid data
def hist_show(data,data_for_what,ax):
    
    stat_stack = np.zeros(data.shape[1], dtype=np.int)
    
    for idx, rss_ap in enumerate(data.T):
        a = np.logical_not(np.isnan(rss_ap))
        stat_stack[idx] = np.sum(a)

    ax.bar(range(data.shape[1]), stat_stack, width=1)
    ax.set_title('Fingerprints per Access Point for: ' + data_for_what)
    ax.set_xlabel('Access point ID')
    ax.set_ylabel('Number of fingerprints')
    return

# access point analysis
def ap_stat(data1,data2):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    hist_show(data1,'testing data',ax1)
    hist_show(data2,'training data',ax2)
    plt.tight_layout()
    plt.show()
    return

# load the phone list
def load_phone(path):
    ts_path = path + '/Test_device_21Aug17.csv'
    tr_path = path + '/Training_device_21Aug17.csv'
    
    with open(ts_path, 'r') as ts_file:
        ts_reader = csv.reader(ts_file)
        ts_phone = [ts_row[0] for ts_row in ts_reader]
    
    with open(tr_path, 'r') as tr_file:
        tr_reader = csv.reader(tr_file)
        tr_phone = [tr_row[0] for tr_row in tr_reader]
    
    return (ts_phone, tr_phone)


# histgram for phone
def phone_hist(data,data_for_what,ax):
    stat_data = {}
    for phone in data:
        stat_data[phone] = stat_data.get(phone,0)+1
        
    phone_name = stat_data.keys()
    sorted_phone_names = sorted(phone_name)

    ax.bar(range(len(sorted_phone_names)), [stat_data[name] for name in sorted_phone_names])
    ax.set_ylabel('Count')
    ax.set_title('Phone Models Used in: ' + data_for_what)
    ax.set_xticks(range(len(sorted_phone_names)))
    ax.set_xticklabels(sorted_phone_names, rotation=75, ha='right')
    return 


# stat analysis for phones
def phone_stat(data1,data2):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    phone_hist(data1,'testing data',ax1)
    phone_hist(data2,'training data',ax2)
    plt.tight_layout()
    plt.show()
    return


# find the ap whose vaild data is more than a threshold
def pt_list(data,threshold):
    
    stat_stack = np.zeros(data.shape[1], dtype=np.int)
    
    for idx, rss_ap in enumerate(data.T):
        a = np.logical_not(np.isnan(rss_ap))
        stat_stack[idx] = np.sum(a)
    
    indx_over_threshold = np.where(stat_stack>=threshold)[0]
    over_threshold = stat_stack[indx_over_threshold]
         
    return (indx_over_threshold, over_threshold)


def threshold_diff_analysis(data,threshold_list,ax,type_of_data):
    
    stat_stack = np.zeros(len(threshold_list),dtype=np.int)
    
    for indx,threshold in enumerate(threshold_list):
        results = pt_list(data,threshold)
        stat_stack[indx] = results[0].shape[0]
    
    ax.plot(threshold_list, stat_stack, marker='o')
    ax.set_title('No# sample APs detected VS No# APs in ' + type_of_data)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Numbers of APs')
    ax.set_xticks(threshold_list)   
    return


# stat analysis for threshold
def threshold_stat(data1,data2,threshold_list):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    threshold_diff_analysis(data1,threshold_list, ax1, 'training data')
    threshold_diff_analysis(data2,threshold_list, ax2, 'testing data')
    plt.tight_layout()
    plt.show()  
    return


# plot positions of the rps
def rp_plot(cord1, cord2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': '3d'})
    
    # Plot for training RPs
    scatter1 = ax1.scatter(cord1[:, 0], cord1[:, 1], cord1[:, 2], c='b', marker='o')
    ax1.set_title('Positions of training RPs')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_zlabel('Height (m)')
    
    # Plot for testing RPs
    scatter2 = ax2.scatter(cord2[:, 0], cord2[:, 1], cord2[:, 2], c='b', marker='o')
    ax2.set_title('Positions of testing RPs')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.set_zlabel('Height (m)')
    
    plt.tight_layout()
    plt.show()

# get most first x ap with largest valid value
def first_x(data,num,threshold=100):
    
    indx_over_x, data_over_x = pt_list(data,threshold)     
    sorted_indx = sorted(range(len(data_over_x)), key=lambda i: data_over_x[i], reverse=True)
    ap_indx = indx_over_x[sorted_indx[:num]]
    
    return ap_indx


# plot_heat_map(data,cord)
def heatmap(data,cord,ap,ax):
    
    data = np.nan_to_num(data, nan=-100)
    p = ax.scatter(cord[:,0],cord[:,1], cord[:,2], c=data[:,ap], cmap='jet', marker='o')
    ax.set_title('Heatmap of ap ' + repr(ap+1))
    ax.set_xlabel('Eastingasting (m)')
    ax.set_ylabel('Northingorthing (m)')
    ax.set_zlabel('Height (m)')
    cbar = plt.colorbar(p, ax=ax)
    cbar.set_label('Signal Strength')
    return


# plot heatmap for the data set using 8 aps that was randomly picked
def x_heatmap_plot(data,cord,threshold=100,num=8):

    random.seed(42)
    id  = [random.randint(0, 99) for _ in range(8)]
    tem_indx = first_x(data,num=100,threshold=100)
    indx = tem_indx[id]

    fig, axs = plt.subplots(4, 2, figsize=(18, 36), subplot_kw={'projection': '3d'}, tight_layout=True)
    for i in range(4):
        for j in range(2):
            heatmap(data,cord,ap=indx[i*2+j],ax=axs[i,j])
            
    plt.show()
    return

# plot rss value analysis - in the whole training/testing set
def rss_in_set(data1, data2, bin_size = 30):

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    axs[0].hist(data1.flatten(), bins=bin_size, color='b', alpha=0.7)
    axs[0].set_title('Histogram of RSS in Training Set')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(data2.flatten(), bins=bin_size, color='g', alpha=0.7)
    axs[1].set_title('Histogram of RSS in Testing Set')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    return
