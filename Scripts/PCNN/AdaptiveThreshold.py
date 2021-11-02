"""
This script aims to test different Pulse-Coupled Neural Network (PCNN) algorithm
Based on :
Pai, Y. T., Chang, Y. F., Ruan, S. J. (2010)
Adaptive thresholding algorithm: Efficient computation technique
based on intelligent block detection for degraded document images
Pattern Recognition, 43 (9), 3177–3187
https://doi.org/10.1016/j.patcog.2010.03.014
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
def OTSU_Algorithm(GrayScale_Array):
    i = np.unique(GrayScale_Array)

    ## Pixel value probability
    Z_i = np.array([])
    for z_i in i:
        N_z = np.array(np.count_nonzero(GrayScale_Array == z_i))
        Z_i = np.concatenate([Z_i, N_z.reshape(1)])

    P_i = Z_i / GrayScale_Array.size

    ## Class occurrence probability
    W_0 = np.array([0])
    for Threshold in i[1:]:
        W_0i = np.sum(P_i[i < Threshold])
        W_0 = np.concatenate([W_0, W_0i.reshape(1)])
    W_0 = np.concatenate([W_0, np.array([1]).reshape(1)])
    W_1 = 1 - W_0

    ## Class mean level
    Mu_0 = np.array([])
    for Threshold in i[1:]:
        Indices = i < Threshold
        Mu_0i = np.sum(i[Indices] * P_i[Indices] / W_0[1:][Indices][-1])
        Mu_0 = np.concatenate([Mu_0, Mu_0i.reshape(1)])

    Mu_1 = np.array([])
    for Threshold in i[1:]:
        Indices = i >= Threshold
        Mu_1i = np.sum(i[Indices] * P_i[Indices] / W_1[:-1][Indices][0])
        Mu_1 = np.concatenate([Mu_1, Mu_1i.reshape(1)])

    Mu_T = np.mean(W_0[1:-1] * Mu_0 + W_1[1:-1] * Mu_1)

    ## Class variances
    V_0 = np.array([])
    for Threshold in i[1:]:
        Indices = i < Threshold
        Nominator = (i[Indices] - Mu_0[Indices[:-1]][-1]) ** 2 * P_i[Indices]
        V_0i = np.sum(Nominator / W_0[1:][Indices][-1])
        V_0 = np.concatenate([V_0, V_0i.reshape(1)])

    V_1 = np.array([])
    for Threshold in i[1:]:
        Indices = i >= Threshold
        Nominator = (i[Indices] - Mu_1[Indices[1:]][0]) ** 2 * P_i[Indices]
        V_1i = np.sum(Nominator / W_1[:-1][Indices][0])
        V_1 = np.concatenate([V_1, V_1i.reshape(1)])

    ## Within-class variance
    V_w = W_0[1:-1] * np.sqrt(V_0) + W_1[1:-1] * np.sqrt(V_1)

    ## Between-class variance
    V_b = W_0[1:-1] * (Mu_0 - Mu_T) ** 2 + W_1[1:-1] * (Mu_1 - Mu_T) ** 2

    ## Total variance
    V_T = np.sum((i - Mu_T) ** 2 * P_i)

    # Discrimination analysis
    N = V_b / V_w
    Threshold = i[N.argmax() + 1]

    return Threshold


CurrentDirectory = os.getcwd()
DataDirectory = os.path.join(CurrentDirectory,'Scripts/PCNN/')

# Read image
Image = sitk.ReadImage(DataDirectory + 'Lena.jpg')
Array = sitk.GetArrayFromImage(Image)
GrayScale_Array = RGB2Gray(Array)


Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(GrayScale_Array,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Input')
plt.show()
plt.close(Figure)

# Manual Otsu algorithm
Threshold = OTSU_Algorithm(GrayScale_Array)

Segmented_Array = GrayScale_Array.copy()
Segmented_Array[Segmented_Array < Threshold] = 0
Segmented_Array[Segmented_Array >= Threshold] = 1

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Segmented_Array,cmap='gray',vmin=0,vmax=1)
plt.axis('off')
plt.title('Manual Otsu')
plt.show()
plt.close(Figure)

# Original Otsu
OtsuFilter = sitk.OtsuThresholdImageFilter()
OtsuFilter.SetInsideValue(1)
OtsuFilter.SetOutsideValue(0)
OtsuFilter.Execute(sitk.GetImageFromArray(GrayScale_Array))
Threshold2 = OtsuFilter.GetThreshold()

Segmented_Array = GrayScale_Array.copy()
Segmented_Array[Segmented_Array < Threshold2] = 0
Segmented_Array[Segmented_Array >= Threshold2] = 1

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Segmented_Array,cmap='gray',vmin=0,vmax=1)
plt.axis('off')
plt.title('Manual Otsu')
plt.show()
plt.close(Figure)
