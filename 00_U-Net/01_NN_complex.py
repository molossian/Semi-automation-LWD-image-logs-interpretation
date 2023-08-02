import tensorflow as tf
import numpy as np
import os



path = './data/'
save_path = './model/'


X_train_full = np.load(path + 'X_train_complex.npy')
y_train_full = np.load(path + 'y_train_complex.npy')

X_train_full = np.reshape(X_train_full, (X_train_full.shape[0], X_train_full.shape[1], X_train_full.shape[2], 1))
y_train_cat = tf.keras.utils.to_categorical(np.expand_dims(y_train_full, axis=-1))


def unet1(input_size = (None, None, 1)):
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, Dense, Dropout

    

    inputs = Input(input_size)
    conv1 = Conv2D(2, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,1))(conv1)

    conv2 = Conv2D(4, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(4, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,1))(conv2)

    conv4 = Conv2D(8, (3,3), activation='relu', padding='same')(pool2)
    conv4 = Conv2D(8, (3,3), activation='relu', padding='same')(conv4)

    up6 = concatenate([UpSampling2D(size=(2,1))(conv4), conv2], axis=-1)

    conv6 = Conv2D(4, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(4, (3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2,1))(conv6), conv1], axis=-1)
    conv7 = Conv2D(2, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(2, (3,3), activation='relu', padding='same')(conv7)

    conv8 = Dense(2, activation='softmax')(conv7)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv8])
    model.summary()
    
    model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', weighted_metrics=["accuracy"])

    return model



checkpoint_path = path + "cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

model = unet1((20,16,1))

model = tf.keras.models.load_model(save_path + 'model_trained_easy')

history_weight = model.fit(X_train_full, y_train_cat, epochs=10, validation_split = 0.1)

model.save(save_path + 'model_trained_complex')
np.save(save_path + 'loss_complex', history_weight.history['loss'])
np.save(save_path + 'val_loss_complex', history_weight.history['val_loss'])
