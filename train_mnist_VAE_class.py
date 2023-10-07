import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, save_model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical 

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


img_r, img_c, img_ch = 28, 28, 1 # mnist
num_classes = 10
train_mode = '' # 'classes'
latent_dim = 64

class VAE():
    def build_model(self):
        # input image dimensions
        img_rows, img_cols, img_chns = img_r, img_c, img_ch
        # number of convolutional filters to use
        filters = 64
        # convolution kernel size
        num_conv = 3

        #batch_size = 8
        batch_size = 16
        if K.image_data_format() == 'channels_first':
            original_img_size = (img_chns, img_rows, img_cols)
        else:
            original_img_size = (img_rows, img_cols, img_chns)
        
        intermediate_dim = 128
        epsilon_std = 1.0

        x = Input(shape=original_img_size)
        conv_1 = Conv2D(img_chns,
                        kernel_size=(2, 2),
                        padding='same', activation='relu')(x)
        conv_2 = Conv2D(filters,
                        kernel_size=(2, 2),
                        padding='same', activation='relu',
                        strides=(2, 2))(conv_1)
        conv_3 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_2)
        conv_4 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_3)
        flat = Flatten()(conv_4)
        hidden1 = Dense(intermediate_dim, activation='relu')(flat)
        hidden2 = Dense(intermediate_dim, activation='relu')(hidden1)


        if train_mode == 'classes':
            x_c = Input(shape=(num_classes)) # one-hot encoding
            hiddenc1 = Dense(32, activation='relu')(x_c)
            hiddenc2 = Dense(64, activation='relu')(hiddenc1) # FC networks for one-hot encodings

            zprior_mean = Dense(latent_dim)(hiddenc2)
            zprior_log_var = Dense(latent_dim)(hiddenc2)
            
            z_mean = Dense(latent_dim)(hidden2)
            z_log_var = Dense(latent_dim)(hidden2)
        else:
            z_mean = Dense(latent_dim)(hidden2)
            z_log_var = Dense(latent_dim)(hidden2)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=tf.shape(z_mean),
                                    mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        z = Lambda(sampling, output_shape=tf.shape(z_mean))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(filters * 14 * 14, activation='relu')

        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, 14, 14)
        else:
            output_shape = (batch_size, 14, 14, filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Conv2DTranspose(filters,
                                        kernel_size=num_conv,
                                        padding='same',
                                        strides=1,
                                        activation='relu')
        decoder_deconv_2 = Conv2DTranspose(filters, num_conv,
                                        padding='same',
                                        strides=1,
                                        activation='relu')
        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, 29, 29)
        else:
            output_shape = (batch_size, 29, 29, filters)
        decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                                kernel_size=(3, 3),
                                                strides=(2, 2),
                                                padding='valid',
                                                activation='relu')
        decoder_mean_squash = Conv2D(img_chns,
                                    kernel_size=2,
                                    padding='valid',
                                    activation='sigmoid')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


        def vae_loss(x, x_decoded_mean):
            # NOTE: binary_crossentropy expects a batch_size by dim
            # for x and x_decoded_mean, so we MUST flatten these!

            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean)
            #kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            # original kl_loss
            # zprior_mean, zprior_log_var
            kl_loss = 0.5 * K.sum(-1 + zprior_log_var - z_log_var + K.exp(z_log_var) / K.exp(zprior_log_var) + K.square(z_mean - zprior_mean) / K.exp(zprior_log_var), axis=-1)
            # 1 + essentially adds 1 to each dimension = number of dimensions
            return xent_loss + kl_loss


        if train_mode == 'classes':
            vae = Model([x, x_c], x_decoded_mean_squash)
        else:
            vae = Model(x, x_decoded_mean_squash)

        # I need the outputs to be there how do I make sure the size is correct?? But [x, x_c] was ok?  they have same batch size....I think the batch size of conv2d_4 is missing 

        vae.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.001), loss=vae_loss)# lr = 0.001 # 0.01 was too big.... I tried several times and gave NAN
        
        return vae


if __name__ == "__main__":
    epochs = 10
    batch_size = 16
    img_rows, img_cols, img_chns = 28, 28, 1
    if K.image_data_format() == 'channels_first':
        original_img_size = (img_chns, img_rows, img_cols)
    else:
        original_img_size = (img_rows, img_cols, img_chns)

    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    y_train = np.array(to_categorical(y_train, num_classes=num_classes))
    y_test = np.array(to_categorical(y_test, num_classes=num_classes))

    print('x_train.shape:', x_train.shape)
    print('x_test.shape:', x_test.shape)

    start = time.time()
    vae_cls = VAE()
    vae = vae_cls.build_model()
    vae.summary()


    model_checkpoint = ModelCheckpoint("model_weights_mnist.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='loss', verbose=0,
    save_best_only=False, save_weights_only=True, mode='auto', period=3)
    def lr_scheduler(epoch, lr):
        decay_rate = 0.7
        decay_step = 5
        if epoch % decay_step == 0 and epoch and lr>5e-6: #stop making it smaller if lr is too small
            return lr * pow(decay_rate, np.floor(epoch / decay_step))
        return lr
    learning_scheduler= tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard('logs\\')

    if train_mode == 'classes':
        vae.fit([x_train, y_train], x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([x_test, y_test], x_test),
                callbacks=[model_checkpoint])
    else:
        vae.fit(x_train, x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test),
                validation_steps = x_test.shape[0] // batch_size,
                callbacks=[model_checkpoint])

    # vae.save('saved_models/untrained_model')
    done = time.time()
    elapsed = done - start
    print('took time: ', elapsed)
    vae.save_weights('trained_mnist_weights.h5')