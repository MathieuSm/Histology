"""
This script aims to test different Pulse-Coupled Neural Network (PCNN) algorithm
Based on :
Xinzheng, X., Shifei, D., Zhongzhi, S., Zuopeng, Z., & Hong, Z. (2011).
Particle swarm optimization for automatic parameters determination of pulse coupled neural network.
Journal of Computer, 6 (8), 1546–1553.
https://doi.org/10.4304/jcp.6.8.1546-1553
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

def RGB2Gray(RGBImage):
    """
    This function convert color image to gray scale image
    based on matplotlib linear approximation
    """

    R, G, B = RGBImage[:,:,0], RGBImage[:,:,1], RGBImage[:,:,2]
    Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    return Gray
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
Input = RGB2Gray(Input_Array)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Input')
plt.show()
plt.close(Figure)


# Pulse-Connected Neural Network (PCNN) algorithm
S = Input / Input.max() * 255

Rows, Columns = S.shape
Y = np.zeros((Rows, Columns))
T = np.zeros((Rows, Columns))
W = GaussianKernel(1,1)

## Feeding input
F = S
AlphaF = 5.8779
VF = -0.7217
WF = W

## Linking input
L = Y
AlphaL = 1.9851
VL = 5.7931
WL = W

# Linking part
Beta = 3.8884

# Dynamic threshold
Theta = 255 * np.ones((Rows, Columns))
AlphaT = 1.7626
VT = 87.0318

# Loop
IterationNumber = 5
N = 0

for Iteration in range(IterationNumber):

    N += 1
    F = S + F * np.exp(-AlphaF) + VF * correlate(Y, WF, output='float', mode='reflect')
    L = L * np.exp(-AlphaL) + VL * correlate(Y, WL, output='float', mode='reflect')
    U = F * (1 + Beta*L)
    Y = (U > Theta) * 1
    Theta = Theta * np.exp(-AlphaT) + VT * Y

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    # Axes.imshow(Y, cmap='gray', vmin=0, vmax=1)
    # plt.title('Y')
    # plt.axis('off')
    # plt.show()
    # plt.close(Figure)

# Results
Segmented_Image = S.copy()
Segmented_Image[Segmented_Image < Theta] = 0
Segmented_Image[Segmented_Image >= Theta] = 1

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Segmented_Image,cmap='gray')
plt.axis('off')
plt.title('Segmented Image')
plt.show()
plt.close(Figure)
