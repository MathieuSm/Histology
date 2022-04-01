import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology
import pandas as pd

def RGBThreshold(RGBArray, Threshold):
    R, G, B = RGBArray[:, :, 0], RGBArray[:, :, 1], RGBArray[:, :, 2]

    R_Filter = R < Threshold[0]
    G_Filter = G < Threshold[1]
    B_Filter = B > Threshold[2]

    BinArray = np.zeros((RGBArray.shape[0], RGBArray.shape[1]))
    BinArray[R_Filter & G_Filter & B_Filter] = 1

    return BinArray

# Set path
CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Tests/Osteons/HumanBone/'

# Read images
P1 = sitk.ReadImage(ImageDirectory + 'Stained1_Registered.png')
P2 = sitk.ReadImage(ImageDirectory + 'Stained2_Registered.png')
S1 = sitk.ReadImage(ImageDirectory + 'Stained1_Seg.png')

# Load arrays
P1_Array = sitk.GetArrayFromImage(P1)
P2_Array = sitk.GetArrayFromImage(P2)
S1_Array = sitk.GetArrayFromImage(S1)

# Build binary segmented image
S1_Bin = np.zeros(S1_Array.shape)
F_R = S1_Array[:,:,0] == 255
F_G = S1_Array[:,:,1] == 0
F_B = S1_Array[:,:,2] == 0
S1_Bin[F_R & F_G & F_B] = 1

Disk = morphology.disk(5)
S1_Bin = morphology.binary_dilation(S1_Bin[:,:,0],Disk)
S1_Bin = morphology.binary_erosion(S1_Bin,Disk)

Figure, Axes = plt.subplots(1,1)
Axes.imshow(S1_Bin[:20,30:190],cmap='binary',vmin=0,vmax=1)
Axes.axis('off')
plt.show()

# Collect protocols values
V1 = P1_Array * np.repeat(S1_Bin,4).reshape((S1_Bin.shape[0],S1_Bin.shape[1],4))
V2 = P2_Array * np.repeat(S1_Bin,4).reshape((S1_Bin.shape[0],S1_Bin.shape[1],4))

V1_DF = pd.DataFrame({'R':V1[:,:,0].flatten(),
                      'G':V1[:,:,1].flatten(),
                      'B':V1[:,:,2].flatten(),
                      'A':V1[:,:,3].flatten()})
V2_DF = pd.DataFrame({'R':V2[:,:,0].flatten(),
                      'G':V2[:,:,1].flatten(),
                      'B':V2[:,:,2].flatten(),
                      'A':V2[:,:,3].flatten()})


Figure, Axes = plt.subplots(1,1)
Axes.boxplot(V1_DF.drop().flatten())
# Axes.axis('off')
plt.show()
