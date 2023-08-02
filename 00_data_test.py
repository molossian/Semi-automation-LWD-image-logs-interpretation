import numpy as np
import scipy
import cv2 as cv
import scipy.signal
from tqdm import tqdm
import random

# Max sinusoidal shift
max_shift = np.random.randint(2, 4)

# number of samples
n_sample = 20



# number of recording channel
n_channel = 16

# noise before convolution
noise_factor_0 = 0.05

# noise after convolution
noise_factor_1 = 0.15

# number of slices
it = 25

def test_func(x, a, b):
    return a * np.sin(b + x)


def create_synthetics(mio_array, max_shift,n_sample, n_channel):
    # number of interfaces
    n_interf = np.random.randint(0, 4)
    """Prende un array di 0 e genera i dati di training"""
    new_array = mio_array.copy()
    val = np.random.uniform(0.25, 1, n_interf)
    idx = random.sample(range(max_shift//2, (n_sample - max_shift - 1)//2), 2)*2
    output_obj = np.zeros((mio_array.shape[0] , mio_array.shape[1] ))
    for i in range(n_interf):
        shift = (np.sin(np.linspace(0, 2 * np.pi, n_channel + 1)[:n_channel] + np.random.uniform(0, 2 * np.pi)) * np.random.randint(1, max_shift)).astype(int)
        for j in range(n_channel):
            new_array[idx[i] + (shift[j] + np.random.randint(0, 4)), j] = val[i]
            # for the binary output
            output_obj[idx[i] + (shift[j] + np.random.randint(0, 4)), j] = i + 1
    for j in range(n_channel):
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

    test = np.zeros((n_sample, n_channel))
    raw_data, target = create_synthetics(test, max_shift,
                                                         test.shape[0], test.shape[1])

    y_k = target.copy()
    y_k[np.where(y_k>0)]= 1
    print(y_k)
    X_train_full[i, :, :] = raw_data
    y_sep[i, :, :] = target
    y_full[i, :, :] = y_k

np.save('./data/X_test', X_train_full)
np.save('./data/y_test', y_full)
