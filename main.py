import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pickle

import preprocessing
import dataset
import model
import training
import test


def load_encoded_data():

    save_path = 'encoded_feature.pkl'  # Encoded data pkl file
    with open(save_path, 'rb') as ff:
        encoded_data = pickle.load(ff)
        ff.close()
    return encoded_data


def training_setup( data, para, time_list, random_seed):
    acc = []
    acc_model = []
    for slice_time in time_list:
        acc_val, acc_training = training.model_training(data, para,  slice_time, random_seed)
        acc_model.append([acc_val, acc_training])
    acc.append(acc_model)
    print(acc)



if __name__ == '__main__':

    encoding_data = load_encoded_data()

    training_parameter = {
        'Batch_size': 10,
        'Learning_rate': 1e-4,
        'epcochs': 10000,
        'patience': 500
    }


    slice_time_list = [20]
    rand_seed = 20


    # training_setup(encoding_data, training_parameter, slice_time_list, rand_seed)



    result = []
    for ipca_acum in [0.999, 0.9, 0.85, 0.8, 0.5, 0.2, 0.1, 0]:
        pca_acum_engi = ipca_acum
        metrics = test.model_val(encoding_data, training_parameter, slice_time_list, pca_acum_engi)
        print(metrics)
        result.append(np.array(metrics))

    flat_arr =  np.array(result).reshape(-1,  np.array(result).shape[-1])
    np.savetxt('output_new.txt', flat_arr, fmt='%.2f')

    with open("results_new.pickle", "wb") as f:
        pickle.dump(result, f)




