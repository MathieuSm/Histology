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
