"""
Script aimed to test multi channel PCNN (MPCNN)
for color image segmentation
Based on:
Zhuang, H., Low, K. S., Yau, W. Y. (2012)
Multichannel pulse-coupled-neural-network-based color image segmentation for object detection
IEEE Transactions on Industrial Electronics, 59(8), 3299–3308
https://doi.org/10.1109/TIE.2011.2165451
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

def ManhattanDistance(x,c):
    """
    Compute Manhattan (or taxicab) distance between vectors
    :param x: n-dimensional vector (numpy array)
    :param c: n-dimensional vector (numpy array)
    :return: d: distance between vectors
    """

    d = sum(np.abs(x-c))

    return d

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

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


