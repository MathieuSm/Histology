"""
Code for testing PCNN methods used for face detection
Based on:
Lim Young-Wan and Na Jin-Hee and Choi Jin-Young (2004)
Role of linking parameters in Pulse-Coupled Neural Network for face detection
Control Robot System Society, 1048-1052
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

CurrentDirectory = os.getcwd()
DataDirectory = os.path.join(CurrentDirectory,'Scripts/PCNN/')
def RGB2YUV(RGBArray):

    Conversion = np.array([[0.299, 0.587, 0.114],
                           [-0.14713, -0.28886, 0.436],
                           [0.615, -0.51499, -1.0001]])
    YUV = np.dot(RGBArray, Conversion)
    YUV[:, :, 1:] += 128.0

    return np.round(YUV).astype('uint8')
def RGB2Gray(RGBImage):
    """
    This function convert color image to gray scale image
    based on matplotlib linear approximation
    """

    R, G, B = RGBImage[:,:,0], RGBImage[:,:,1], RGBImage[:,:,2]
    Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    return Gray


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


# Convert RGB to YUV encoding
YUV = RGB2YUV(Input_Array)
Y, U, V = YUV[:,:,0], YUV[:,:,1], YUV[:,:,2]
Figure, Axes = plt.subplots(1, 3)
Axes[0].imshow(Y, cmap='gray')
Axes[0].set_title('Y channel')
Axes[0].axis('off')
Axes[1].imshow(U, cmap='gray')
Axes[1].set_title('U channel')
Axes[1].axis('off')
Axes[2].imshow(V, cmap='gray')
Axes[2].set_title('V channel')
Axes[2].axis('off')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Load segmented face
Seg_Image = sitk.ReadImage(DataDirectory + 'Lena_Seg.jpg')
Seg_Array = sitk.GetArrayFromImage(Seg_Image)
GrayImage = RGB2Gray(Seg_Array)
GrayImage[GrayImage < 245] = 0
GrayImage[GrayImage > 245] = 1

# Extract face pixels mean and variance
U_Mean = np.mean(U[GrayImage == 1])
U_Var = np.var(U[GrayImage == 1],ddof=1)

V_Mean = np.mean(V[GrayImage == 1])
V_Var = np.var(V[GrayImage == 1],ddof=1)

S_U = 255 / np.exp((U - U_Mean)**2 / U_Var)
S_U = np.round(S_U).astype('uint8')

S_V = 255 / np.exp((V - V_Mean)**2 / V_Var)
S_V = np.round(S_V).astype('uint8')


Beta_U = 0.5
Beta_V = 0.5

Theta = 255 * np.ones(S_U.shape)
dT = 1
Vt = 400

FireNumber = 0
N = 0
F = S_U
Beta = Beta_U
W = np.array([[0.5, 1, 0.5],
              [1, 0, 1],
              [0.5, 1, 0.5]])
Gamma = 0.2
T = np.zeros(S_U.shape)
while FireNumber < F.size:

    N += 1

    L = correlate(Y, W, output='float', mode='reflect')
    H = correlate(Y, W, output='float', mode='reflect')
    U = F * (1 + Beta*L) * (1 - Gamma*H)
    Theta = Theta - dT + Vt * Y
    Y = (U > Theta) * 1
    T += N * Y
    FireNumber += sum(sum(Y))

T = 256 - T
T = (T - T.min()) / (T.max() - T.min()) * 255


Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(T,cmap='gray')
plt.axis('off')
plt.title('Segmented Face')
plt.tight_layout()
plt.show()
plt.close(Figure)
