import tensorflow as tf
import numpy as np
import os


path = '/m100_work/uTS23_Roncoron/data_attilio/MC_data/'
save_path = './model/'


X_train_full = np.load(path + 'X_train_bin.npy')
y_train_full = np.load(path + 'y_train.npy')

X_train_full = np.reshape(X_train_full, (X_train_full.shape[0], X_train_full.shape[1], X_train_full.shape[2], 1))
y_train_full = np.reshape(y_train_full, (y_train_full.shape[0], y_train_full.shape[1]*y_train_full.shape[2]))


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return tf.keras.layers.Dropout(p)(input_tensor, training=True)
    else:
        return tf.keras.layers.Dropout(p)(input_tensor)
def get_model(mc=False, act="relu"):
    inp = tf.keras.Input((20, 16, 1))
    x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides = (1,1), padding = "same", activation=act)(inp)
    x = get_dropout(x, p=0.5, mc=mc)
    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides = (1,1), padding = "same", activation=act)(x)
    x = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides = (1,1), padding = "same", activation=act)(x)
    #x = get_dropout(x, p=0.25, mc=mc)
    x = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides = (1,1), padding = "same", activation=act)(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(48, activation=act)(x)
  
    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(loss= 'mse',
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


batch_size = 32

mc_model = get_model(mc = True, act='relu')

#model = tf.keras.models.load_model(save_path + 'model_trained_easy')

history_weight = mc_model.fit(X_train_full, y_train_full, epochs=200, validation_split = 0.1)

mc_model.save(save_path + 'model_trained_complex')
np.save(save_path + 'loss_complex', history_weight.history['loss'])
np.save(save_path + 'val_loss_complex', history_weight.history['val_loss'])


