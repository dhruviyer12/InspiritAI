# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def label_to_numpy(labels):
    final_labels = np.zeros((len(labels), 4))
    for i in range(len(labels)):
        label = labels[i]
        if label == 'Attentive':
            final_labels[i,:] = np.array([1, 0, 0, 0])
        if label == 'DrinkingCoffee':
            final_labels[i,:] = np.array([0, 1, 0, 0])
        if label == 'UsingMirror':
            final_labels[i,:] = np.array([0, 0, 1, 0])
        if label == 'UsingRadio':
            final_labels[i,:] = np.array([0, 0, 0, 1])
    return final_labels


class pkg:
    def get_metadata(metadata_path, which_splits=['train', 'test']):
        metadata = pd.read_csv(metadata_path)
        metadata = metadata[metadata['split'].isin(which_splits)]

        df_coffee_train = metadata[(metadata['class'] == 'DrinkingCoffee') & (metadata['split'] == 'train')]
        df_coffee_test = metadata[(metadata['class'] == 'DrinkingCoffee') & (metadata['split'] == 'test')]
        df_mirror_train = metadata[(metadata['class'] == 'UsingMirror') & (metadata['split'] == 'train')]
        df_mirror_test = metadata[(metadata['class'] == 'UsingMirror') & (metadata['split'] == 'test')]
        df_attentive_train = metadata[(metadata['class'] == 'Attentive') & (metadata['split'] == 'train')]
        df_attentive_test = metadata[(metadata['class'] == 'Attentive') & (metadata['split'] == 'test')]
        df_radio_train = metadata[(metadata['class'] == 'UsingRadio') & (metadata['split'] == 'train')]
        df_radio_test = metadata[(metadata['class'] == 'UsingRadio') & (metadata['split'] == 'test')]

        num_samples_train = min(df_coffee_train.shape[0], df_mirror_train.shape[0],
                                df_attentive_train.shape[0], df_radio_train.shape[0])
        num_samples_test = min(df_coffee_test.shape[0], df_mirror_test.shape[0],
                               df_attentive_test.shape[0], df_radio_test.shape[0])

        metadata_train = pd.concat([
            df_coffee_train.sample(num_samples_train),
            df_mirror_train.sample(num_samples_train),
            df_attentive_train.sample(num_samples_train),
            df_radio_train.sample(num_samples_train)
        ])

        metadata_test = pd.concat([
            df_coffee_test.sample(num_samples_test),
            df_mirror_test.sample(num_samples_test),
            df_attentive_test.sample(num_samples_test),
            df_radio_test.sample(num_samples_test)
        ])

        metadata = pd.concat([metadata_train, metadata_test])
        return metadata

    def get_data_split(split_name, flatten, all_data, metadata, image_shape):
        df_coffee = metadata[(metadata['class'] == 'DrinkingCoffee') & (metadata['split'] == split_name)]
        df_mirror = metadata[(metadata['class'] == 'UsingMirror') & (metadata['split'] == split_name)]
        df_attentive = metadata[(metadata['class'] == 'Attentive') & (metadata['split'] == split_name)]
        df_radio = metadata[(metadata['class'] == 'UsingRadio') & (metadata['split'] == split_name)]

        num_samples = min(df_coffee.shape[0], df_mirror.shape[0],
                          df_attentive.shape[0], df_radio.shape[0])

        metadata_balanced = pd.concat([
            df_coffee.sample(num_samples),
            df_mirror.sample(num_samples),
            df_attentive.sample(num_samples),
            df_radio.sample(num_samples)
        ])

        index = metadata_balanced['index'].values
        labels = metadata_balanced['class'].values
        data = all_data[index,:]

        if flatten:
            data = data.reshape([-1, np.product(image_shape)])

        return data, labels

    def get_train_data(flatten, all_data, metadata, image_shape):
        return p
