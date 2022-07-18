import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import morphology, color
import time

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPipeline'))
import PSO
from Filtering import FFT2D

def PlotArray(Array, Title, CMap='gray', ColorBar=False):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    CBar = Axes.imshow(Array, cmap=CMap)
    if ColorBar:
        plt.colorbar(CBar)
    plt.title(Title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return

# Set path and variables
CurrentDirectory = Path.cwd()
ImageDirectory = CurrentDirectory / 'Tests/Osteons/Sensitivity/'

PixelLength = 1.0460251046025104 # Computed with 418 RM
ROISize = 1000 # Size in um
SemiLength = int(round(ROISize/PixelLength/2))

DataFrame = pd.read_csv(str(ImageDirectory / 'Data.csv'))
N = 2
SampleData = DataFrame.loc[N]
Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '_Seg.jpg')

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / Name))
Array = sitk.GetArrayFromImage(Image)[:,:,:3]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array)
plt.show()

# Extract ROI
Point = [2000,6000] # y and xx coordinate, respectively
Area = [[Point[0] - SemiLength, Point[0] + SemiLength],
        [Point[1] - SemiLength, Point[1] + SemiLength]]
ROI = Array[Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(ROI)
plt.show()

# Mark areas where there is bone
Filter1 = ROI[:,:,0] < 190
Filter2 = ROI[:,:,1] < 190
Filter3 = ROI[:,:,2] < 235
Bone = Filter1 & Filter2 & Filter3

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Bone,cmap='binary')
plt.show()

# Erode and dilate to remove small bone parts
Disk = morphology.disk(2)
Dilated = morphology.binary_dilation(Bone,Disk)
Bone = morphology.binary_erosion(Dilated,Disk)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Bone,cmap='binary')
plt.show()



# Filter image to extract manual segmentation
Filter1 = ROI[:,:,0] > 100
Filter2 = ROI[:,:,1] < 90
Filter3 = ROI[:,:,2] > 100

Bin = np.zeros(Filter1.shape)
Bin[Filter1 & Filter2 & Filter3] = 1

# Dilate to link extracted segmentation
Disk = morphology.disk(5)
BinDilate = morphology.binary_dilation(Bin,Disk)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(BinDilate,cmap='binary')
plt.show()

# Skeletonize to obtain 1 pixel thickness
Skeleton = morphology.skeletonize(BinDilate)
Figure, Axis = plt.subplots(1,1)
Axis.imshow(Skeleton,cmap='binary')
plt.show()

# Compute ROI cement line density
CMD = Skeleton.sum() / Bone.sum()


# Read non-segmented image
Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '.jpg')

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / Name))
Array = sitk.GetArrayFromImage(Image)[:,:,:3][Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array)
plt.show()

Lab = color.rgb2lab(Array)
PlotArray(Lab[:,:,1],'Original Image')
Filtered = FFT2D(Lab[:,:,1],CutOff=1/10,Sharpness=50,PassType='Low')
PlotArray(Filtered,'Filtered Image')

def PrintTime(Tic, Toc):
    """
    Print elapsed time in seconds to time in HH:MM:SS format
    :param Tic: Actual time at the beginning of the process
    :param Toc: Actual time at the end of the process
    """

    Delta = Toc - Tic

    Hours = np.floor(Delta / 60 / 60)
    Minutes = np.floor(Delta / 60) - 60 * Hours
    Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

    print('Process executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

Filtered_b = 0.02644803
nonFiltered_b = 0.06711096
nonFiltered_a = 0.07100479