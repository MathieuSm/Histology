"""
This script aims to test different Pulse-Coupled Neural Network (PCNN) algorithm
Based on :
Zhan, K., Shi, J., Wang, H. et al.
Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
Arch Computat Methods Eng 24, 573–588 (2017).
https://doi.org/10.1007/s11831-016-9182-3
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

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

# Generate artificial input image
Input = np.ones((256, 256)) * 230
Input[64:192, 64:192] = 205
Input[:, 128:] = np.round(Input[:, 128:] * 0.5)

# import SimpleITK as sitk
# Input_Image = sitk.ReadImage(DataDirectory + 'Toluidinblue_protocol2A_20.jpg')
# Input_Array = sitk.GetArrayFromImage(Input_Image)
# Input = Input_Array[2000:2500,2000:2500,2]

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Input')
plt.show()

# Algorithm 5, image segmentation - Initialization
S = Input

Rows, Columns = S.shape
Y = np.zeros((Rows, Columns))
T = np.zeros((Rows, Columns))
W = GaussianKernel(Length=7, Sigma=1.)

## Feeding input
F = S
AlphaF = 0.2754
VF = -0.5186
WF =

## Linking input
AlphaL = -0.0851
VL = -0.42187
WL =

# Linking part
Beta = -0.4904

# Dynamic threshold
Theta = 255 * np.ones((Rows, Columns))
Alpha_t = 0.0361
V_t = 0.3903        # ?????????????

FiredNumber = 0
N = 0

while FiredNumber < Rows * Columns:

    N += 1
    L = F * np.exp(AlphaL) + VL * correlate(Y, WL, output='float', mode='reflect')
    F = S + F * np.exp(AlphaF) + VF * correlate(Y, WF, output='float', mode='reflect')
    Theta = Theta * np.exp(Alpha_t) + V_t * Y
    Fire = 1

    while Fire == 1:

        Q = Y
        U = F * (1 + Beta*L)
        Y = (U > Theta) * 1
        if np.array_equal(Q,Y):
            Fire = 0
        else:
            L = F * np.exp(AlphaL) + VL * correlate(Y, WL, output='float', mode='reflect')
            # L = correlate(Y, W, output='float', mode='reflect')

    T = T + N*Y
    FiredNumber = FiredNumber + sum(sum(Y))

Output = 256 - T

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Output,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Output')
plt.show()
