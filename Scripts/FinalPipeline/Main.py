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
def NormalizeValues(Image):

    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max()-Image.min())

    return N_Image

def PCNN(Image, Beta=2, AlphaF=1., VF=0.5, AlphaL=1., VL=0.5, AlphaT=0.05, VT=100):
    """
    Segment image using single neuron firing and fast linking implementation
    Based on:
    Zhan, K., Shi, J., Wang, H. et al.
    Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
    Arch Computat Methods Eng 24, 573–588 (2017).
    https://doi.org/10.1007/s11831-016-9182-3

    :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
    :param Delta: Linear decay factor for threshold level
    :param Vt: Dynamic threshold amplitude
    :return: H: Image histogram in numpy array
    """

    Tic = time.time()
    print('\nImage segmentation...')

    # Initialize parameters
    S = NormalizeValues(Image)
    Rows, Columns = S.shape
    F = np.zeros((Rows, Columns))
    L = np.zeros((Rows, Columns))
    Y = np.zeros((Rows, Columns))
    T = np.zeros((Rows, Columns))
    W = np.array([[0.5, 1, 0.5],
                  [1, 0, 1],
                  [0.5, 1, 0.5]])
    Theta = np.ones((Rows, Columns))

    FiredNumber = 0
    N = 0
    Condition = FiredNumber < S.size

    # Perform segmentation
    while Condition:
        N += 1
        F = S + F * np.exp(-AlphaF) + VF * correlate(Y, W, output='float', mode='reflect')
        L = L * np.exp(-AlphaL) + VL * correlate(Y, W, output='float', mode='reflect')
        Theta = Theta * np.exp(-AlphaT) + VT * Y

        U = F * (1 + Beta * L)
        Y = (U > Theta) * 1

        T = T + N * Y
        FiredNumber = FiredNumber + sum(sum(Y))
        Condition = FiredNumber < S.size

    Output = 1 - NormalizeValues(T)

    # Print time elapsed
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Output

def Function2Optimize(Parameters=np.array([2.,1.,0.5,1.,0.5,0.05])):

    Beta, AlphaF, VF, AlphaL, VL, AlphaT = Parameters
    Segmented = PCNN(Filtered, Beta, AlphaF, VF, AlphaL, VL, AlphaT)
    Values = np.unique(Segmented)
    DCs = np.zeros(len(Values))
    i = 0
    for Value in Values:
        Bin = (Segmented == Value) * 1
        DCs[i] = 2*np.sum(Bin * Skeleton) / (Bin.sum() + Skeleton.sum())
        i += 1

    return 1 - DCs.max()

class Arguments:
    pass
Arguments.Function = Function2Optimize
Arguments.Ranges = np.array([[0,4],[1E-2,10],[1E-2,1],[1E-2,10],[1E-2,1],[1E-2,1]])
Arguments.Population = 20
Arguments.Cs = [0.1,0.1]
Arguments.MaxIt = 10
Arguments.STC = 1E-3
Parameters = PSO.Main(Arguments)

Beta, AlphaF, VF, AlphaL, VL, AlphaT = Parameters
Segmented = PCNN(Filtered, Beta, AlphaF, VF, AlphaL, VL, AlphaT)
Values = np.unique(Segmented)
DCs = np.zeros(len(Values))
i = 0
for Value in Values:
    Bin = (Segmented == Value) * 1
    DCs[i] = 2 * np.sum(Bin * Skeleton) / (Bin.sum() + Skeleton.sum())
    i += 1

Bin = (Segmented == Values[DCs.argmax()]) * 1
PlotArray(Bin,'Segment ' + str(DCs.argmax()))

PlotArray(Skeleton, 'Skeleton')

Filtered_b = 0.02644803
nonFiltered_b = 0.06711096
nonFiltered_a = 0.07100479