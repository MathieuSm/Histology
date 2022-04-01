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

Figure, Axes = plt.subplots(1,1,figsize=(55.44,55.97))
Axes.imshow(S1_Array)
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

Figure, Axes = plt.subplots(1,1,figsize=(20,20))
Axes.imshow(S1_Array[600:2600,:2000])
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

Figure, Axes = plt.subplots(1,1,figsize=(15,30))
Axes.imshow(S1_Array[1400:1700,1600:1750])
plt.subplots_adjust(0,0,1,1)
Axes.axis('off')
plt.show()

# Collect protocols values
V1 = P1_Array * np.repeat(S1_Bin,4).reshape((S1_Bin.shape[0],S1_Bin.shape[1],4))
V2 = P2_Array * np.repeat(S1_Bin,4).reshape((S1_Bin.shape[0],S1_Bin.shape[1],4))

Figure, Axes = plt.subplots(1,1)
Axes.imshow(V2)
Axes.axis('off')
plt.show()

V1_DF = pd.DataFrame({'R':V1[:,:,0].flatten(),
                      'G':V1[:,:,1].flatten(),
                      'B':V1[:,:,2].flatten(),
                      'A':V1[:,:,3].flatten()})
V2_DF = pd.DataFrame({'R':V2[:,:,0].flatten(),
                      'G':V2[:,:,1].flatten(),
                      'B':V2[:,:,2].flatten(),
                      'A':V2[:,:,3].flatten()})

# Filter dataframes
FR_Min = V1_DF['R'] > 0
FG_Min = V1_DF['G'] > 0
FB_Min = V1_DF['B'] > 0

FR_Max = V1_DF['R'] < 255
FG_Max = V1_DF['G'] < 255
FB_Max = V1_DF['B'] < 255

F_V1 = V1_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]
F_V2 = V2_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]

# Boxplot of the values
WhiskerProps = dict(linestyle='--', linewidth=1, color=(0,0,0))
FlierProps = dict(marker='o', markerfacecolor=(0,0,0,0), markersize=5, markeredgecolor=(0,0,0))
Positions = np.array([1,2,3])
Offset = 0.2

Figure, Axes = plt.subplots(1,1)
BoxProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
MedianProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
Axes.boxplot(F_V1[['R','G','B']], positions=Positions-Offset,
             boxprops=BoxProps, flierprops=FlierProps,
             medianprops=MedianProps, whiskerprops=WhiskerProps)
BoxProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
MedianProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
Axes.boxplot(F_V2[['R','G','B']], positions=Positions+Offset,
             boxprops=BoxProps, flierprops=FlierProps,
             medianprops=MedianProps, whiskerprops=WhiskerProps)
Axes.plot([],color=(0,0,1), label='Protocol 1')
Axes.plot([],color=(1,0,0), label='Protocol 2')
Axes.set_xticks(Positions,['R','G','B'])
Axes.set_ylim([0,255])
plt.legend(loc='lower right')
plt.show()


## Step 2: take neighboring area to compare values

T1 = np.zeros(P1_Array.shape).astype('int')
T1[:,:,0][S1_Bin == 1] = 255
T1[:,:,3][S1_Bin == 1] = 50

Figure, Axes = plt.subplots(1, 1)
Axes.imshow(P1_Array[1400:1700, 1600:1750])
Axes.imshow(T1[1400:1700, 1600:1750])
Axes.axis('off')
plt.show()

for i in [5,10,15]:

    Disk = morphology.disk(i)
    S1_Big = morphology.binary_dilation(S1_Bin,Disk)

    T1 = np.zeros(P1_Array.shape).astype('int')
    T1[:, :, 0][S1_Big == 1] = 255
    T1[:, :, 3][S1_Big == 1] = 50

    Figure, Axes = plt.subplots(1, 1)
    Axes.imshow(P1_Array[1400:1700, 1600:1750])
    Axes.imshow(T1[1400:1700, 1600:1750])
    Axes.axis('off')
    plt.show()

    # Collect protocols values
    V1_B = P1_Array * np.repeat(S1_Big,4).reshape((S1_Big.shape[0],S1_Big.shape[1],4))
    V2_B = P2_Array * np.repeat(S1_Big,4).reshape((S1_Big.shape[0],S1_Big.shape[1],4))

    # Figure, Axes = plt.subplots(1,1)
    # Axes.imshow(V2)
    # Axes.axis('off')
    # plt.show()

    V1_B_DF = pd.DataFrame({'R':V1_B[:,:,0].flatten(),
                            'G':V1_B[:,:,1].flatten(),
                            'B':V1_B[:,:,2].flatten(),
                            'A':V1_B[:,:,3].flatten()})
    V2_B_DF = pd.DataFrame({'R':V2_B[:,:,0].flatten(),
                            'G':V2_B[:,:,1].flatten(),
                            'B':V2_B[:,:,2].flatten(),
                            'A':V2_B[:,:,3].flatten()})

    # Filter dataframes
    FR_Min = V1_B_DF['R'] > 0
    FG_Min = V1_B_DF['G'] > 0
    FB_Min = V1_B_DF['B'] > 0

    FR_Max = V1_B_DF['R'] < 255
    FG_Max = V1_B_DF['G'] < 255
    FB_Max = V1_B_DF['B'] < 255

    F_V1_B = V1_B_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]
    F_V2_B = V2_B_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]

    # Boxplot of the values

    Figure, Axes = plt.subplots(1,1)
    BoxProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
    MedianProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
    Axes.boxplot(F_V1_B[['R','G','B']], positions=Positions-Offset,
                 boxprops=BoxProps, flierprops=FlierProps,
                 medianprops=MedianProps, whiskerprops=WhiskerProps)
    BoxProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
    MedianProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
    Axes.boxplot(F_V2_B[['R','G','B']], positions=Positions+Offset,
                 boxprops=BoxProps, flierprops=FlierProps,
                 medianprops=MedianProps, whiskerprops=WhiskerProps)
    Axes.plot([],color=(0,0,1), label='Protocol 1')
    Axes.plot([],color=(1,0,0), label='Protocol 2')
    Axes.set_xticks(Positions,['R','G','B'])
    Axes.set_ylim([0,255])
    plt.legend(loc='lower right')
    # Axes.axis('off')
    plt.show()

