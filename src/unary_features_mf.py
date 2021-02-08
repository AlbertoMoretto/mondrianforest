#!/usr/bin/env python

import numpy as np
import pprint as pp     # pretty printing module
from matplotlib import pyplot as plt        # required only for plotting results
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal 
from mondrianforest import process_command_line, MondrianForest
import pandas as pd

def load_toy_mf_data():
    n_dim = 2
    n_class = 3
    x_train = np.array([-0.5,-1, -2,-2, 1,0.5, 2,2, -1,1, -1.5, 1.5]) + 0.
    y_train = np.array([0, 0, 1, 1, 2, 2], dtype='int')
    x_train.shape = (6, 2)
    x_test = x_train

    print x_train.shape
    print y_train.shape

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    y_test = np.array([0, 0, 1, 1, 2, 2], dtype='int')
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data

unary_features = ['normal_angle', 'normal_deviation', 'l_CIELAB', 'a_CIELAB', 'b_CIELAB',
                    'standard_l_CIELAB', 'standard_a_CIELAB', 'standard_b_CIELAB',
                    'MIN_height', 'MAX_height', 'bounding_box_width', 'bounding_box_height',
                    'bounding_box_depth', 'bounding_box_vertical_area', 'bounding_box_horizontal_area',
                    'vertical_elongation', 'horizontal_elongation', 'volumeness']

x_df = pd.read_csv('/home/alberto/tesi/dataset/trained_semseg_data/entangled_csv/unary_csv/0000.csv', usecols=unary_features)
y_df = pd.read_csv('/home/alberto/tesi/dataset/trained_semseg_data/entangled_csv/labels_csv/0000.csv')

x_train = x_df.to_numpy()
y_train = y_df.to_numpy()

y_train.shape  = (y_train.shape[0],)

print x_train.shape
print y_train.shape

if x_train.shape[0] != y_train.shape[0]:
    print "ERROR! INCONSISTENT TRAINING DATA"
    exit

n_train = x_train.shape[0]
n_dim = x_train.shape[1]
n_labels = np.amax(y_train)+1

x_test = x_train
y_test = y_train

n_test = x_test.shape[0]

data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_labels, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}


toy_data = load_toy_mf_data()





