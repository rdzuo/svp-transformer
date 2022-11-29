import torch
import numpy as np
import math
import argparse
import os
from sklearn.cluster import KMeans
import torch.nn.functional as F



def load_raw_ts(path, dataset, tensor_format=True):
    path = path + dataset + "/"
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')
    x_train = np.transpose(x_train, axes=(0, 2, 1))
    x_test = np.transpose(x_test, axes=(0, 2, 1))

    nclass = int(np.amax(y_train)) + 1
    dim_length = x_train.shape[2]
    dim_v = x_train.shape[1]

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        x_train = torch.FloatTensor(np.array(x_train))
        x_test = torch.FloatTensor(np.array(x_test))
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)


    return x_train, x_test, y_train, y_test, dim_length, dim_v, nclass


def euclid_dist(x_1, x_2) :
    a = x_1.size(0)
    if a == x_2.size(0):
        y = 0
        for i in range(0, a):
            y_temp = (x_1[i]-x_2[i]) ** 2
            y = y_temp + y
        out = y**0.5
        out = torch.tensor([out])  # 把out转为指定维度，以便后续append
    else:print('the input does not match')
    return out

def slide_TS(window_size, X):
    '''
    tensor version
    slide the multivariate time series tensor from 3D to 2D
    add the dimension label to each variate
    add step to reduce the num of new TS
    '''
    dim_length = X.shape[1]
    X_alpha = X[:, 0 : window_size]

    # determine step
    if (dim_length <= 50) :
        step = 1
    elif (dim_length > 50 and dim_length <= 100):
        step = 2
    elif (dim_length > 100 and dim_length <= 300):
        step = 3
    elif (dim_length > 300 and dim_length <= 1000):
        step = 4
    elif (dim_length > 1000 and dim_length <= 1500):
        step = 5
    elif (dim_length > 1500 and dim_length <= 2000):
        step = 7
    elif (dim_length > 2000 and dim_length <= 3000):
        step = 10
    else:
        step = 1000

    # determine step number
    step_num = int(math.ceil((dim_length -window_size)/step))

    # still slide to 2D
    for k in range(1, dim_length-window_size+1, step):

        X_temp = X[:, k : window_size + k]
        X_alpha = torch.cat((X_alpha, X_temp), dim = 1)


    # numpy reshape (number of instances, windowsize * number of subsequence ) to (number of instances* number of subsequence, windowsize)
    X_beta = torch.reshape(X_alpha,( (X_alpha.shape[0]) * (step_num+1), window_size))
    return X_beta

def cluster_variable_window(train_data, window_size, dim_v, num_center_window) :
    cluster_centers = []
    for v in range(0,dim_v):
        train_data_v = train_data[:, v, :]
        for i in range(0,3):
            x_subsequence = slide_TS(window_size[i],train_data_v)
            kmeans = KMeans(n_clusters=num_center_window[i])
            kmeans.fit(x_subsequence)
            cluster_centers_temp = kmeans.cluster_centers_
            cluster_centers_temp = torch.from_numpy(cluster_centers_temp)
            cluster_centers.append(cluster_centers_temp)
    return cluster_centers


def select_nearest(x,center,v):
    a = center.shape[0]
    d = x[0:0+a]
    b = euclid_dist(d,center)
    c = torch.tensor([x.shape[0],0,a-1,v]) #last point is a-1, not a
    if (a <= 20) :
        step = 1
    elif (a > 40 and a <= 60):
        step = 2
    elif (a > 60 and a <= 80):
        step = 3
    elif (a > 80 and a <= 100):
        step = 4
    elif (a > 100 and a <= 150):
        step = 5
    elif (a > 150 and a <= 200):
        step = 6
    elif (a > 200 and a <= 300):
        step = 8
    else:
        step = 10
    for i in range (1,x.shape[0] - a + 1,step):
        x_one = x[i:i+a]
        if euclid_dist(x_one,center)<b:
            b = euclid_dist(x_one,center)
            d = x_one
            c[1] = i
            c[2] = i+a-1
    d = torch.unsqueeze(d, 0)
    c = torch.unsqueeze(c, 0)
    return d, c




def get_single_features(x, center,v):
    a = torch.empty(0,center.shape[1])
    b = torch.empty(0,4)
    for i in range(0, center.shape[0]):
        subsequence,position  = select_nearest(x, center[i, :], v)
        a = torch.cat((a, subsequence), dim=0)
        b = torch.cat((b, position), dim=0)
    return a, b





def get_local_features(x, center, tol_sub):
    N = x.shape[0]
    M = x.shape[1]
    num_window = int((len(center)) / M)
    max_S = center[num_window - 1].shape[1]

    features = torch.empty(0, tol_sub, max_S)
    positions = torch.empty(0, tol_sub, 4)
    for i in range(0, N):
        feature = torch.empty(0, max_S)
        position = torch.empty(0, 4)
        for k in range(0, len(center)):
            feature_temp, position_temp = get_single_features(x[i, int(k / num_window), :], center[k], int(k / num_window))
            feature_temp = F.pad(feature_temp,(0,max_S-feature_temp.shape[1],0,0))
            feature = torch.cat((feature, feature_temp), dim=0)
            position = torch.cat((position, position_temp), dim=0)
        print('finished samples %d' %(i))
        features_temp = torch.unsqueeze(feature, 0)
        positions_temp = torch.unsqueeze(position, 0)
        features = torch.cat((features, features_temp), dim=0)
        positions = torch.cat((positions, positions_temp), dim=0)

    return features, positions

def save_tensor_as_numpy(x1,x2,dataset,type):
    x1 = x1.numpy()
    x2 = x2.numpy()
    file = "./featureset/" + dataset + "/" + type
    path = "./featureset/" + dataset + "/"
    if type == "train":
          os.mkdir(path)
    np.savez(file,x1,x2)

def get_window(l):
    if l <= 200:
        window_size = [int(0.1 * l), int(0.2 * l), int(0.3 * l)]
    elif ( l>200 and l<=500) :
        window_size = [int(0.05 * l), int(0.1 * l), int(0.2 * l)]
    elif ( l>500 and l <= 1000):
        window_size = [int(0.05 * l), int(0.1 * l), int(0.2 * l)]
    elif (l > 1000):
        window_size = [50, 100, 200]
    return window_size

def preprocess(x, dim_v, dim_length):
    n_ins = x.shape[0]
    n_sub = x.shape[1]
    y = torch.zeros(n_ins,n_sub,3)
    for i in range(0, n_ins):
        for k in range(0, n_sub):
                y[i, k, 0] = x[i, k, 1]/dim_length
                y[i, k, 1] = x[i, k, 2]/dim_length
                y[i, k, 2] = (x[i, k, 3] - 0)/ (dim_v - 1)
    return y
















