"""
Script aimed to test image segmentation using PCNN and
threshold adaptation using watershed methodology
Based on:
Min Li and Wei Cai and Xiao-Yan Li (2006)
An Adaptive Image Segmentation Method Based on a Modified Pulse Coupled Neural Network
LNCS (4221), 471-474
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
def PCNN_Histogram(GrayImage):

    """
    Return histogram of grayscale image using PCNN
    Based on :
    Zhan, K., Shi, J., Wang, H. et al.
    Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
    Arch Computat Methods Eng 24, 573–588 (2017).
    https://doi.org/10.1007/s11831-016-9182-3
    """

    # Initialize
    Theta = 255
    Delta = 1
    Vt = 256
    Y = np.zeros(GrayImage.shape)
    H = np.zeros(256)

    # Perform analysis
    for N in range(256):
        U = GrayImage
        Theta = Theta - Delta + Vt * Y
        Y = np.where((U - Theta) > 0, 1, 0)
        H[255-N] = Y.sum()

    return H
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


# Input of PCNN
S = (Input-Input.min()) / (Input.max()-Input.min()) * 255
S = np.round(S).astype('uint8')

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(S, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Input')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Compute histogram
H = PCNN_Histogram(S)
npHist, npEdges = np.histogram(S, bins=255)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(npEdges[:-1]+0.5,npHist,color=(0,0,1), label='Numpy')
Axes.plot(np.arange(0,256), H,linestyle='none',marker='x',color=(1,0,0), label='PCNN')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Smooth histogram
KernelRadius = 10
Sigma = 7
W = GaussianKernel(2*KernelRadius+1,Sigma)[KernelRadius,:]
W = W / W.sum()
HPadded = np.pad(H,pad_width=KernelRadius,mode='reflect')
HSmooth = np.zeros(H.shape)
for i in range(len(H)):
    HSmooth[i] = np.sum(W * HPadded[i:i+2*KernelRadius+1])
HSmooth = HSmooth / HSmooth.sum()

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.hist(S.flatten(),255, density=True, color=(0,0,1),label='Histogram')
Axes.plot(HSmooth,color=(1,0,0),label='Smoothed')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Find local minima
Minima = np.array([])
Maxima = np.array([])
for x in range(1,len(HSmooth)-1):
    p = HSmooth[x]
    b = HSmooth[x-1]
    a = HSmooth[x+1]
    if p < b and p < a:
        Minima = np.concatenate([Minima,np.array([x])]).astype('uint8')
    elif p > b and p > a:
        Maxima = np.concatenate([Maxima,np.array([x])]).astype('uint8')


Theta = 0.001
LeftSidePeak = np.array([])
for Minimum in Minima:

    Area = 0
    N = 0
    while Area < Theta:
        N += 1
        if Minimum - N < 0:
            LeftArea = 0
        else:
            LeftArea = N * (HSmooth[Minimum-N] - HSmooth[Minimum])
        if Minimum + N < 256:
            RightArea = N * (HSmooth[Minimum+N] - HSmooth[Minimum])
        else:
            RightArea = 0
        Area += LeftArea + RightArea

        Index = Maxima[Maxima < Minimum - N][-1]

    LeftSidePeak = np.concatenate([LeftSidePeak,np.array([Minimum+N])])


