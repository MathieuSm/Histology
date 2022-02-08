"""
Example code for image segmentation using PCNN and neuron single activation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))

I = np.ones((256, 256)) * 230
I[65:192, 65:192] = 205
I[:, 128:256] = I[:, 128:256] * 0.5

S = I

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(S, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
plt.close(Figure)

Rows, Columns = S.shape
Y = np.zeros((Rows,Columns))
T = Y

W = GaussianKernel(7,1)

F = S
Beta = 2

Theta = 255 * np.ones((Rows,Columns))
dT = 1
Vt = 400

FireNumber = 0
N = 0
while FireNumber < S.size:

    N += 1

    L = correlate(Y, W, output='float', mode='reflect')
    Theta = Theta - dT + Vt * Y
    Fire = 1

    while Fire == 1:
        Q = Y
        U = F * (1 + Beta * L)
        Y = (U > Theta) * 1

        if np.array_equal(Q,Y):
            Fire = 0
        else:
            L = correlate(Y, W, output='float', mode='reflect')

    T = T + N * Y
    FireNumber += sum(sum(Y))

T = 256 - T

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(T, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
plt.close(Figure)