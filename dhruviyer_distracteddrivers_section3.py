import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
!pip install tf-keras-vis tensorflow

import zipfile
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import model_selection
from collections import Counter
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, DenseNet121
from imgaug import augmenters
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from matplotlib import cm
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D
from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

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

def augment(data, augmenter):
    if len(data.shape) == 3:
        return augmenter.augment_image(data)
    if len(data.shape) == 4:
        return augmenter.augment_images(data)

def rotate(data, rotate):
    fun = augmenters.Affine(rotate=rotate)
    return augment(data, fun)

def shear(data, shear):
    fun = augmenters.Affine(shear=shear)
    return augment(data, fun)

def scale(data, scale):
    fun = augmenters.Affine(scale=scale)
    return augment(data, fun)

def flip_left_right(data):
    fun = augmenters.Fliplr()
    return augment(data, fun)

def flip_up_down(data):
    fun = augmenters.Flipud()
    return augment(data, fun)

def remove_color(data, channel):
    new_data = data.copy()
    if len(data.shape) == 3:
        new_data[:,:,channel] = 0
        return new_data
    if len(data.shape) == 4:
        new_data[:,:,:,channel] = 0
        return new_data

class pkg:
    def get_metadata(metadata_path, which_splits=['train', 'test']):
        metadata = pd.read_csv(metadata_path)
        keep_idx = metadata['split'].isin(which_splits)
        metadata = metadata[keep_idx]

        df_coffee_train = metadata[(metadata['class'] == 'DrinkingCoffee') & (metadata['split'] == 'train')]
        df_coffee_test = metadata[(metadata['class'] == 'DrinkingCoffee') & (metadata['split'] == 'test')]
        df_mirror_train = metadata[(metadata['class'] == 'UsingMirror') & (metadata['split'] == 'train')]
        df_mirror_test = metadata[(metadata['class'] == 'UsingMirror') & (metadata['split'] == 'test')]
        df_attentive_train = metadata[(metadata['class'] == 'Attentive') & (metadata['split'] == 'train')]
        df_attentive_test = metadata[(metadata['class'] == 'Attentive') & (metadata['split'] == 'test')]
        df_radio_train = metadata[(metadata['class'] == 'UsingRadio') & (metadata['split'] == 'train')]
        df_radio_test = metadata[(metadata['class'] == 'UsingRadio') & (metadata['split'] == 'test')]

        num_samples_train = min(df_coffee_train.shape[0], df_mirror_train.shape[0], df_attentive_train.shape[0], df_radio_train.shape[0])
        num_samples_test = min(df_coffee_test.shape[0], df_mirror_test.shape[0], df_attentive_test.shape[0], df_radio_test.shape[0])

        metadata_train = pd.concat([df_coffee_train.sample(num_samples_train), df_mirror_train.sample(num_samples_train), df_attentive_train.sample(num_samples_train), df_radio_train.sample(num_samples_train)])
        metadata_test = pd.concat([df_coffee_test.sample(num_samples_test), df_mirror_test.sample(num_samples_test), df_attentive_test.sample(num_samples_test), df_radio_test.sample(num_samples_test)])

        metadata = pd.concat([metadata_train, metadata_test])
        return metadata

    def get_data_split(split_name, flatten, all_data, metadata, image_shape):
        df_coffee_train = metadata[(metadata['class'] == 'DrinkingCoffee') & (metadata['split'] == 'train')]
        df_coffee_test = metadata[(metadata['class'] == 'DrinkingCoffee') & (metadata['split'] == 'test')]
        df_mirror_train = metadata[(metadata['class'] == 'UsingMirror') & (metadata['split'] == 'train')]
        df_mirror_test = metadata[(metadata['class'] == 'UsingMirror') & (metadata['split'] == 'test')]
        df_attentive_train = metadata[(metadata['class'] == 'Attentive') & (metadata['split'] == 'train')]
        df_attentive_test = metadata[(metadata['class'] == 'Attentive') & (metadata['split'] == 'test')]
        df_radio_train = metadata[(metadata['class'] == 'UsingRadio') & (metadata['split'] == 'train')]
        df_radio_test = metadata[(metadata['class'] == 'UsingRadio') & (metadata['split'] == 'test')]

        num_samples_train = min(df_coffee_train.shape[0], df_mirror_train.shape[0], df_attentive_train.shape[0], df_radio_train.shape[0])
        num_samples_test = min(df_coffee_test.shape[0], df_mirror_test.shape[0], df_attentive_test.shape[0], df_radio_test.shape[0])

        metadata_train = pd.concat([df_coffee_train.sample(num_samples_train), df_mirror_train.sample(num_samples_train), df_attentive_train.sample(num_samples_train), df_radio_train.sample(num_samples_train)])
        metadata_test = pd.concat([df_coffee_test.sample(num_samples_test), df_mirror_test.sample(num_samples_test), df_attentive_test.sample(num_samples_test), df_radio_test.sample(num_samples_test)])

        metadata = pd.concat([metadata_train, metadata_test])

        sub_df = metadata[metadata['split'].isin([split_name])]
        index  = sub_df['index'].values
        labels = sub_df['class'].values
        data = all_data[index,:]
        if flatten:
            data = data.reshape([-1, np.product(image_shape)])
        return data, labels

    def get_train_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('train', flatten, all_data, metadata, image_shape)

    def get_test_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('test', flatten, all_data, metadata, image_shape)

    def get_field_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('field', flatten, all_data, metadata, image_shape)

class helpers:
    def plot_image(data, num_ims, figsize=(8,6), labels=[], index=None, image_shape=[64,64,3]):
        num_dims   = len(data.shape)
        num_labels = len(labels)

        if num_dims == 1:
            data = data.reshape(target_shape)
        if num_dims == 2:
            data = data.reshape(-1,image_shape[0],image_shape[1],image_shape[2])
        num_dims   = len(data.shape)

        if num_dims == 3:
            if num_labels > 1:
                return
            label = labels
            if num_labels == 0:
                label = ''
            image = data

        if num_dims == 4:
            image = data[index, :]
            label = labels[index]

        nrows=int(np.sqrt(num_ims))
        ncols=int(np.ceil(num_ims/nrows))
        count=0
        
        if nrows==1 and ncols==1:
            plt.imshow(image)
            plt.show()
        else:
            fig = plt.figure(figsize=figsize)
            for i in range(nrows):
                for j in range(ncols):
                    if count<num_ims:
                        fig.add_subplot(nrows,ncols,count+1)
                        plt.imshow(image[count])
                        count+=1
            fig.set_size_inches(18.5, 10.5)
            plt.show()

    def get_misclassified_data(data, labels, predictions):
        missed_index     = np.where(np.abs(predictions.squeeze() - labels.squeeze()) > 0)[0]
        missed_labels    = labels[missed_index]
        missed_data      = data[missed_index,:]
        predicted_labels = predictions[missed_index]
        return missed_data, missed_labels, predicted_labels, missed_index

    def combine_data(data_list, labels_list):
        return np.concatenate(data_list, axis=0), np.concatenate(labels_list, axis=0)

    def model_to_string(model):
        import re
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))

        for layer in model.layers:
            if hasattr(layer,"activation"):
                stringlist.append(str(layer.activation))

        sms = "\n".join(stringlist)
        sms = re.sub('_\d\d\d','', sms)
        sms = re.sub('_\d\d','', sms)
        sms = re.sub('_\d','', sms)
        return sms

    def plot_acc(history, ax=None, xlabel='Epoch #'):
        history = history.history
        history.update({'epoch':list(range(len(history['val_accuracy'])))})
        history = pd.DataFrame.from_dict(history)

        best_epoch = history.sort_values(by='val_accuracy', ascending=False).iloc[0]['epoch']

        if not ax:
            f, ax = plt.subplots(1,1)
        sns.lineplot(x='epoch', y='val_accuracy', data=history, label='Validation', ax=ax)
        sns.lineplot(x='epoch', y='accuracy', data=history, label='Training', ax=ax)
        ax.axhline(0.25, linestyle='--', color='red', label='Chance')
        ax.axvline(x=best_epoch, linestyle='--', color='green', label='Best Epoch')
        ax.legend(loc=1)
        ax.set_ylim([0.01, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Accuracy (Fraction)')
        plt.show()

class models:
    def DenseClassifier(hidden_layer_sizes, nn_params, dropout=0.5):
        model = Sequential()
        model.add(Flatten(input_shape=nn_params['input_shape']))
        for ilayer in hidden_layer_sizes:
            model.add(Dense(ilayer, activation='relu'))
            if dropout:
                model.add(Dropout(dropout))
        model.add(Dense(units=nn_params['output_neurons'], activation=nn_params['output_activation']))
        model.compile(loss=nn_params['loss'],
                      optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                      metrics=['accuracy'])
        return model

    def CNNClassifier(num_hidden_layers, nn_params, dropout=0.5):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=nn_params['input_shape'], padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for i in range(num_hidden_layers-1):
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=nn_params['output_neurons'], activation=nn_params['output_activation']))

        opt = tensorflow.keras.optimizers.RMSprop(learning_rate=1e-4)
        model.compile(loss=nn_params['loss'],
                      optimizer=opt,
                      metrics=['accuracy'])
        return model

    def TransferClassifier(name, nn_params, trainable=True):
        expert_dict = {'VGG16': VGG16,
                       'VGG19': VGG19,
                       'ResNet50': ResNet50,
                       'DenseNet121': DenseNet121}

        expert_conv = expert_dict[name](weights='imagenet',
                                        include_top=False,
                                        input_shape=nn_params['input_shape'])
        for layer in expert_conv.layers:
            layer.trainable = trainable

        expert_model = Sequential()
        expert_model.add(expert_conv)
        expert_model.add(GlobalAveragePooling2D())
        expert_model.add(Dense(128, activation='relu'))
        expert_model.add(Dropout(0.3))
        expert_model.add(Dense(64, activation='relu'))
        expert_model.add(Dense(nn_params['output_neurons'], activation=nn_params['output_activation']))
        expert_model.compile(loss=nn_params['loss'],
                             optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                             metrics=['accuracy'])
        return expert_model


image_data_path      = './image_data.npy'
metadata_path        = './metadata.csv'
image_shape          = (64, 64, 3)

nn_params = {}
nn_params['input_shape']       = image_shape
nn_params['output_neurons']    = 4
nn_params['loss']              = 'categorical_crossentropy'
nn_params['output_activation'] = 'softmax'

!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Driver%20Distraction%20Detection/metadata.csv'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Driver%20Distraction%20Detection/image_data.npy'

_all_data = np.load('image_data.npy')
_metadata = pkg.get_metadata(metadata_path, ['train','test','field'])

get_data_split  = pkg.get_data_split
get_metadata    = lambda :                  pkg.get_metadata(metadata_path, ['train','test'])
get_train_data  = lambda flatten=False : pkg.get_train_data(flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape)
get_test_data   = lambda flatten=False : pkg.get_test_data(flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape)
get_field_data  = lambda flatten=False : pkg.get_field_data(flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape)

plot_image      = lambda data, num_ims, figsize=(8,6), labels=[], index=None: helpers.plot_image(data=data, num_ims=num_ims, figsize=figsize, labels=labels, index=index, image_shape=image_shape)
plot_acc        = lambda history: helpers.plot_acc(history)
model_to_string        = lambda model: helpers.model_to_string(model)
get_misclassified_data = helpers.get_misclassified_data
combine_data           = helpers.combine_data

DenseClassifier      = lambda hidden_layer_sizes: models.DenseClassifier(hidden_layer_sizes=hidden_layer_sizes, nn_params=nn_params)
CNNClassifier        = lambda num_hidden_layers: models.CNNClassifier(num_hidden_layers, nn_params=nn_params)
TransferClassifier   = lambda name: models.TransferClassifier(name=name, nn_params=nn_params)

monitor = ModelCheckpoint('./model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

train_data, train_labels = get_train_data(flatten=True)
test_data, test_labels = get_test_data(flatten=True)

train_data = train_data.reshape([-1, 64, 64, 3])
test_data = test_data.reshape([-1, 64, 64, 3])

train_labels = label_to_numpy(train_labels)
test_labels = label_to_numpy(test_labels)

dense = DenseClassifier(hidden_layer_sizes=(128,64))
cnn = CNNClassifier(num_hidden_layers=5)

dense.fit(train_data, train_labels, epochs=50, validation_data=(test_data, test_labels), shuffle=True, callbacks=[monitor])
cnn.fit(train_data, train_labels, epochs=50, validation_data=(test_data, test_labels), shuffle=True, callbacks=[monitor])

plot_acc(dense.history)
plot_acc(cnn.history)


train_data, train_labels = get_train_data(flatten=True)
test_data, test_labels = get_test_data(flatten=True)

train_data = train_data.reshape([-1, 64, 64, 3])
test_data = test_data.reshape([-1, 64, 64, 3])

train_labels_strings = train_labels
test_labels_strings = test_labels

train_labels = label_to_numpy(train_labels)
test_labels = label_to_numpy(test_labels)

replace2linear = ReplaceToLinear()

images = []
for i, title in enumerate(test_labels_strings):
    dim = (64, 64)
    img = np.array(cv2.resize(test_data[i], dim))
    images.append(img)
images = np.asarray(images)

def getImageSamples():
    image_samples = []
    image_samples_labels = []
    idx = random.randint(0, 230)
    for i in range(4):
        image_samples.append(images[idx])
        image_samples_labels.append(test_labels_strings[idx])
        idx = idx + 230
    image_samples = np.asarray(image_samples)
    return image_samples, image_samples_labels

def plot_vanilla_saliency_of_a_model(model, X_input, image_titles):
    score = CategoricalScore(list(range(X_input.shape[0])))
    saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
    saliency_map = saliency(score, X_input)

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(X_input[i])
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(saliency_map[i], cmap='jet')
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

imgs, imgs_labels = getImageSamples()
plot_vanilla_saliency_of_a_model(cnn, imgs, imgs_labels)

train_data, train_labels = get_train_data(flatten=True)
test_data, test_labels = get_test_data(flatten=True)

train_data = train_data.reshape([-1, 64, 64, 3])
test_data = test_data.reshape([-1, 64, 64, 3])

train_labels_strings = train_labels
test_labels_strings = test_labels

train_labels = label_to_numpy(train_labels)
test_labels = label_to_numpy(test_labels)

vgg_expert = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

vgg_model = Sequential()
vgg_model.add(vgg_expert)
vgg_model.add(GlobalAveragePooling2D())
vgg_model.add(Dense(1024, activation='relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(4, activation='softmax'))

vgg_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                  metrics=['accuracy'])

vgg_model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels), shuffle=True)

predictions = vgg_model.predict(test_data)
predictions = np.argmax(predictions, axis=1)

final_labels = []
for label in test_labels:
    if label[0] == 1:
        final_labels.append(0)
    else:
        final_labels.append(1)

binary_predictions = []
for label in predictions:
    if label == 1:
        binary_predictions.append(0)
    else:
        binary_predictions.append(1)

confusion = confusion_matrix(final_labels, binary_predictions)

tp = confusion[1][1]
tn = confusion[0][0]
fp = confusion[0][1]
fn = confusion[1][0]

sns.heatmap(confusion, annot=True, fmt='d', cbar_kws={'label': 'count'})
plt.ylabel('Actual')
plt.xlabel('Predicted')
