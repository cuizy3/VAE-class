import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

from imageio import imwrite
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical 
from train_mnist_VAE_class import VAE


num_preds = 2
num_classes = 10
model_name = 'trained_mnist_weights_class'

vae_cls = VAE()
vae = vae_cls.build_model()

vae.load_weights('trained_mnist_weights_class.h5')
vae.summary()

save_folder = 'sample_results\\mnist\\' + model_name + '_results'
if not os.path.exists(save_folder):
	os.makedirs(save_folder)
(x_train, y_train_o), (x_test, y_test_o) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = np.array(to_categorical(y_train_o, num_classes=num_classes))
y_test = np.array(to_categorical(y_test_o, num_classes=num_classes))

# Writes outputs in sample_results folder
for i in range(num_preds):
	result = vae.predict([(x_test).reshape(x_test.shape + (1,)), y_test])
	picture = np.zeros((result.shape[1], result.shape[2] * 2))

	for index in range(result.shape[0]):
		picture[:, 0:result.shape[2]] = x_test[index].reshape((result.shape[1], result.shape[2]))
		picture[:, result.shape[2]:result.shape[2] * 2] = result[index].reshape((result.shape[1], result.shape[2]))
		f_name = os.path.join(save_folder, f'mnist_eval_{index}_{y_test_o[index]}_posterior_predictive_sample_{i}.jpg')
		imwrite(f_name, (picture * 255).astype(np.uint8))

