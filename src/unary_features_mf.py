#!/usr/bin/env python

import numpy as np
import pprint as pp
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal 
from mondrianforest import process_command_line, MondrianForest
import pandas as pd
import os

def load_dataset(settings):
    ''' Import unaryfeatures and labels for dataset from csv files
        (based on method load_datas() on mondrianforest_utils.py)
        Params: settings -- setting imported from command line
        Return: data -- dicitionary containing a dataset for the Mondrian Forest 
    '''

    # All the unary features computed for each cluster from the input point clouds
    unary_features = ['normal_angle', 'normal_deviation', 'l_CIELAB', 'a_CIELAB', 'b_CIELAB',
                    'standard_l_CIELAB', 'standard_a_CIELAB', 'standard_b_CIELAB',
                    'MIN_height', 'MAX_height', 'bounding_box_width', 'bounding_box_height',
                    'bounding_box_depth', 'bounding_box_vertical_area', 'bounding_box_horizontal_area',
                    'vertical_elongation', 'horizontal_elongation', 'volumeness']

    # from the input directory path geths the paths to the
    # subdirectory condaining unary features csv and labels
    # csv files (assuming that the files are divided into the
    # corresponding subfolders)
    uf_dir = settings.data_path + '/unary_csv'
    labels_dir = settings.data_path + '/labels_csv'
    file_list = os.listdir(labels_dir)

    first_time = True
    for file_name in file_list:
        curr_uf_csv = uf_dir+ '/'+ file_name
        curr_lables_csv = labels_dir + '/'+ file_name
        x_df = pd.read_csv(curr_uf_csv,usecols=unary_features)
        y_df = pd.read_csv(curr_lables_csv,dtype=int)
        if first_time:
            x_train = x_df.to_numpy()
            y_train = y_df.to_numpy()
            y_train.shape = (y_train.shape[0],)
            first_time = False
        else:
            x_curr = x_df.to_numpy()
            y_curr = y_df.to_numpy()
            y_curr.shape = (y_curr.shape[0],)
            x_train = np.append(x_train,x_curr, axis=0)
            y_train = np.append(y_train,y_curr, axis=0)

    y_train.shape  = (y_train.shape[0],)

    n_train = x_train.shape[0]
    n_dim = x_train.shape[1]
    n_labels = np.amax(y_train)+1

    x_test = x_train
    y_test = y_train
    n_test = x_test.shape[0]

    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_labels, \
                'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
                'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    
    try:
        if settings.normalize_features == 1:
            min_d = np.minimum(np.min(data['x_train'], 0), np.min(data['x_test'], 0))
            max_d = np.maximum(np.max(data['x_train'], 0), np.max(data['x_test'], 0))
            range_d = max_d - min_d
            idx_range_d_small = range_d <= 0.   # find columns where all features are identical
            if data['n_dim'] > 1:
                range_d[idx_range_d_small] = 1e-3   # non-zero value just to prevent division by 0
            elif idx_range_d_small:
                range_d = 1e-3
            data['x_train'] -= min_d + 0.
            data['x_train'] /= range_d
            data['x_test'] -= min_d + 0.
            data['x_test'] /= range_d
    except AttributeError:
        # backward compatibility with code without normalize_features argument
        pass
    if settings.select_features:
        if settings.optype == 'real':
            scores, _ = feature_selection.f_regression(data['x_train'], data['y_train'])
        else:
            raise Exception('select_features currently supported only for regression')
        scores[np.isnan(scores)] = 0.   # FIXME: setting nan scores to 0. Better alternative?
        scores_sorted, idx_sorted = np.sort(scores), np.argsort(scores)
        flag_relevant = scores_sorted > (scores_sorted[-1] * 0.05)  # FIXME: better way to set threshold? 
        idx_feat_selected = idx_sorted[flag_relevant]
        assert len(idx_feat_selected) >= 1
        print scores
        print scores_sorted
        print idx_sorted
        
        if False:
            data['x_train'] = data['x_train'][:, idx_feat_selected]
            data['x_test'] = data['x_test'][:, idx_feat_selected]
        else:
            data['x_train'] = np.dot(data['x_train'], np.diag(scores)) 
            data['x_test'] = np.dot(data['x_test'], np.diag(scores))
        data['n_dim'] = data['x_train'].shape[1]
    # ------ beginning of hack ----------
    is_mondrianforest = True
    n_minibatches = settings.n_minibatches
    if is_mondrianforest:
        # creates data['train_ids_partition']['current'] and data['train_ids_partition']['cumulative'] 
        #    where current[idx] contains train_ids in minibatch "idx", cumulative contains train_ids in all
        #    minibatches from 0 till idx  ... can be used in gen_train_ids_mf or here (see below for idx > -1)
        data['train_ids_partition'] = {'current': {}, 'cumulative': {}}
        train_ids = np.arange(data['n_train'])
        try:
            draw_mondrian = settings.draw_mondrian
        except AttributeError:
            draw_mondrian = False
        if is_mondrianforest and (not draw_mondrian):
            reset_random_seed(settings)
            np.random.shuffle(train_ids)
            # NOTE: shuffle should be the first call after resetting random seed
            #       all experiments would NOT use the same dataset otherwise
        train_ids_cumulative = np.arange(0)
        n_points_per_minibatch = data['n_train'] / n_minibatches
        assert n_points_per_minibatch > 0
        idx_base = np.arange(n_points_per_minibatch)
        for idx_minibatch in range(n_minibatches):
            is_last_minibatch = (idx_minibatch == n_minibatches - 1)
            idx_tmp = idx_base + idx_minibatch * n_points_per_minibatch
            if is_last_minibatch:
                # including the last (data[n_train'] % settings.n_minibatches) indices along with indices in idx_tmp
                idx_tmp = np.arange(idx_minibatch * n_points_per_minibatch, data['n_train'])
            train_ids_current = train_ids[idx_tmp]
            # print idx_minibatch, train_ids_current
            data['train_ids_partition']['current'][idx_minibatch] = train_ids_current
            train_ids_cumulative = np.append(train_ids_cumulative, train_ids_current)
            data['train_ids_partition']['cumulative'][idx_minibatch] = train_ids_cumulative
    return data

# Import settings from command line
settings = process_command_line()
print 'Current settings:'
pp.pprint(vars(settings))

# Resetting random seed
reset_random_seed(settings)

# Loading data
data = load_dataset(settings)

param, cache = precompute_minimal(data,settings)

mf = MondrianForest(settings, data)

print '\nminibatch\tmetric_train\tmetric_test\tnum_leaves'

for idx_minibatch in range(settings.n_minibatches):
    train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
    if idx_minibatch == 0:
        # Batch training for first minibatch
        mf.fit(data, train_ids_current_minibatch, settings, param, cache)
    else:
        # Online update
        mf.partial_fit(data, train_ids_current_minibatch, settings, param, cache)

    # Evaluate
    weights_prediction = np.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians
    train_ids_cumulative = data['train_ids_partition']['cumulative'][idx_minibatch]
    pred_forest_train, metrics_train = \
        mf.evaluate_predictions(data, data['x_train'][train_ids_cumulative, :], \
        data['y_train'][train_ids_cumulative], \
        settings, param, weights_prediction, False)
    pred_forest_test, metrics_test = \
        mf.evaluate_predictions(data, data['x_test'], data['y_test'], \
        settings, param, weights_prediction, False)
    name_metric = settings.name_metric     # acc or mse
    metric_train = metrics_train[name_metric]
    metric_test = metrics_test[name_metric]
    tree_numleaves = np.zeros(settings.n_mondrians)
    for i_t, tree in enumerate(mf.forest):
        tree_numleaves[i_t] = len(tree.leaf_nodes)
    forest_numleaves = np.mean(tree_numleaves)
    print '%9d\t%.3f\t\t%.3f\t\t%.3f' % (idx_minibatch, metric_train, metric_test, forest_numleaves)

print '\nFinal forest stats:'
tree_stats = np.zeros((settings.n_mondrians, 2))
tree_average_depth = np.zeros(settings.n_mondrians)
for i_t, tree in enumerate(mf.forest):
    tree_stats[i_t, -2:] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
    tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
print 'mean(num_leaves) = %.1f, mean(num_non_leaves) = %.1f, mean(tree_average_depth) = %.1f' \
        % (np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))
print 'n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f' \
        % (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))

