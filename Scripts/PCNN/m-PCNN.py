"""
Script aimed to test multi channel PCNN (m-PCNN)
for color image fusion
Based on:
Wang, Z., Ma, Y. (2008)
Medical image fusion using m-PCNN
Information Fusion, 9(2), 176–185
https://doi.org/10.1016/j.inffus.2007.04.003
"""

import os
import numpy as np
import SimpleITK as sitk
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


CurrentDirectory = os.getcwd()
DataDirectory = os.path.join(CurrentDirectory,'Scripts/PCNN/')

# Read input
Input_Image = sitk.ReadImage(DataDirectory + 'Lena.jpg')
Input_Array = sitk.GetArrayFromImage(Input_Image)
Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input_Array)
plt.axis('off')
plt.title('Input')
plt.tight_layout()
plt.show()
plt.close(Figure)

R, G, B = Input_Array[:, :, 0], Input_Array[:, :, 1], Input_Array[:, :, 2]
Figure, Axes = plt.subplots(1, 3)
Axes[0].imshow(R, cmap='gray')
Axes[0].set_title('R channel')
Axes[0].axis('off')
Axes[1].imshow(G, cmap='gray')
Axes[1].set_title('G channel')
Axes[1].axis('off')
Axes[2].imshow(B, cmap='gray')
Axes[2].set_title('B channel')
Axes[2].axis('off')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Pulse-Connected Neural Network (PCNN) algorithm
Rows, Columns = R.shape
Y = np.zeros((Rows,Columns))
T = Y

W = GaussianKernel(3,1)

S0 = R / R.max()
S1 = G / G.max()
S2 = B / B.max()

Beta0 = 1/4
Beta1 = 1/4
Beta2 = 1/1
BetaWeights = np.array([Beta0,Beta1,Beta2])
Beta0, Beta1, Beta2 = BetaWeights / BetaWeights.sum()


Theta = 255 * np.ones((Rows,Columns))
dT = 1
Vt = 400

Sigma = 1

FireNumber = 0
N = 0
while FireNumber < S.size:

    N += 1

    H0 = correlate(Y, W, output='float', mode='reflect') + S0
    H1 = correlate(Y, W, output='float', mode='reflect') + S1
    H2 = correlate(Y, W, output='float', mode='reflect') + S2

    U = (1 + Beta0*H0) * (1 + Beta1*H1) * (1 + Beta2*H2) + Sigma
    U = U / U.max() * 255

    Theta = Theta - dT + Vt * Y

    Y = (U > Theta) * 1

    T += N * Y

    FireNumber += sum(sum(Y))

T = 256 - T
T = (T - T.min()) / (T.max() - T.min()) * 255

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(T, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Output')
plt.tight_layout()
plt.show()
plt.close(Figure)
