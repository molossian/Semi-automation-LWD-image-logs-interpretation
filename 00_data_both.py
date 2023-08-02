import numpy as np
import scipy
import cv2 as cv
import scipy.signal
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# number of samples
n_sample = 20



# number of recording channel
n_channel = 16

# noise before convolution
noise_factor_0 = 0.05

# noise after convolution
noise_factor_1 = 0.15

# number of slices
it = 50

def test_func(x, a, b):
    return a * np.sin(b + x)


def create_synthetics(mio_array, n_interf, n_sample, n_channel):
    """Prende un array di 0 e genera i dati di training"""
    new_array = mio_array.copy()
    val = np.random.uniform(0.25, 1, n_interf)
    idx = np.random.randint(2, 18, n_interf)
    output_obj = np.zeros((mio_array.shape[0] , mio_array.shape[1] ))
    for i in range(n_interf):
        shift = (np.sin(np.linspace(0, 2 * np.pi, n_channel + 1)[:n_channel] + np.random.uniform(0, 2 * np.pi, 1) )*np.random.randint(1, 10)).astype(int)
        for j in range(n_channel):
            if idx[i] + shift[j] < n_sample and idx[i] + shift[j] >= 0:
                new_array[idx[i] + (shift[j]), j] = val[i]
                # for the binary output
                output_obj[idx[i] + (shift[j]), j] = i + 1
    for j in range(n_channel):
        new_array[:, j] = new_array[:, j] + np.random.randn(n_sample) * noise_factor_1/2
        new_array[:, j] = scipy.signal.convolve(new_array[:, j], np.ones(n_sample))[n_sample - 1:n_sample + n_sample]


    #post convolutional noise
    new_array = new_array[:, :] + 1 + np.random.randn(n_sample, n_channel) * noise_factor_1
    #gaussian kernel or smoothing--> crucial for derivative
    kernel = np.ones((5, 5), np.float32) / 25
    new_array  = cv.filter2D(new_array [:, :], -1, kernel)

    return new_array, output_obj


#-----------------------------------------------------------------------------------------------------------------------

X_train_full= np.zeros((it, n_sample, n_channel))
y_sep = np.zeros((it, n_sample, n_channel))
y_full = np.zeros((it, n_sample, n_channel))

for i in tqdm(range(it)):
    # number of interfaces
    n_interf = np.random.randint(0, 3)    
    test = np.zeros((n_sample, n_channel))
    raw_data, target = create_synthetics(test,n_interf, test.shape[0], test.shape[1])
    y_k = target.copy()
    y_k[np.where(y_k>0)]= 1
    if np.random.randint(0,2)==0:
        X_train_full[i, :, :] = -raw_data
    else:
        X_train_full[i, :, :] = raw_data
    y_sep[i, :, :] = target
    y_full[i, :, :] = y_k
    
np.save('./data/X_train_both', X_train_full)
np.save('./data/y_train_both', y_full)
np.save('./data/y_keep_both' , y_sep)
