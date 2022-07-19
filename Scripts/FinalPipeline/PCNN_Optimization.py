#!/usr/bin/env python3

import sys
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import statsmodels.formula.api as smf
from scipy.stats.distributions import t
from skimage import morphology, measure, color

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPipeline'))

import PSO

plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    Script used to optimize PCNN parameters and obtain good cement line density
    correlation between manual and automatic segmentation

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: July 2022
    """


class Parameters:

    def __init__(self, ImageNumber, Threshold=0.88):
        self.N = ImageNumber
        self.Directory = Path.cwd() / 'Tests/Osteons/Sensitivity/'
        self.Threshold = Threshold
        self.MinCost = 1.

class PSOArgs:

    def __init__(self, Function2Optimize, Ranges, Population, Cs, MaxIt=10, STC=1E-3):
        self.Function = Function2Optimize
        self.Ranges = Ranges
        self.Population = Population
        self.Cs = Cs
        self.MaxIt = MaxIt
        self.STC = STC


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

def PixelSize(Image, Length, Plot=False):
    """
    Determine physical size of a pixel
    :param Image: Image region containing the scalebar as numpy array r x c x 3
    :param Length: Physical length of the scalebar as integer
    :param Plot: Plot intermediate results, boolean value
    :return: Physical size of a pixel
    """

    Tic = time.time()
    print('Compute physical pixel size ...')

    Filter1 = Image[:, :, 0] < 100
    Filter2 = Image[:, :, 1] < 100
    Filter3 = Image[:, :, 2] < 100

    Bin = np.zeros(Filter1.shape, 'int')
    Bin[Filter1 & Filter2 & Filter3] = 1

    if Plot:
        Figure, Axis = plt.subplots(1, 1)
        Axis.imshow(Bin, cmap='binary')
        plt.show()

    RegionProps = measure.regionprops(Bin)[0]
    Pixels = RegionProps.coords[:, 1].max() - RegionProps.coords[:, 1].min()
    PixelLength = Length / Pixels

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return PixelLength

def ReadImage(Params, Plot=True):
    # Read image and plot it
    Directory = Params.Directory
    DataFrame = pd.read_csv(str(Directory / 'Data.csv'))
    SampleData = DataFrame.loc[Params.N]
    Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '.jpg')
    Image = sitk.GetArrayFromImage(sitk.ReadImage(str(Directory / Name)))[:, :, :3]
    Name = Name[:-4] + '_Seg.jpg'
    SegImage = sitk.GetArrayFromImage(sitk.ReadImage(str(Directory / Name)))[:, :, :3]

    if Plot:
        Shape = np.array(Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Store name, image, and pixel length in parameters class
    Params.Name = Name[:-7]
    Params.Image = Image
    Params.SegImage = SegImage

    if Params.Name[:5] == '418RM':
        Params.PixelLength = PixelSize(Params.Image[9400:-400, 12500:-300], 2000, Plot=True)
    else:
        Params.PixelLength = 1.0460251046025104  # Computed with 418 RM

def SegmentBone(Image, Plot=False):
    """
    Segment bone structure
    :param Image: RGB numpy array dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Labelled bone image
    """

    Tic = time.time()
    print('Segment bone area ...')

    # Mark areas where there is bone
    Filter1 = Image[:, :, 0] < 190
    Filter2 = Image[:, :, 1] < 190
    Filter3 = Image[:, :, 2] < 235
    Bone = Filter1 & Filter2 & Filter3

    if Plot:
        Shape = np.array(Image.shape) / max(Image.shape) * 10
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Erode and dilate to remove small bone parts
    Disk = morphology.disk(2)
    Dilated = morphology.binary_dilation(Bone, Disk)
    Bone = morphology.binary_erosion(Dilated, Disk)

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Bone

def RandCoords(Coords, ROINumber, TotalNROIs):

    XCoords, YCoords = Coords

    XRange = XCoords.max() - XCoords.min()
    Width = XRange / (TotalNROIs + 1)
    RandX = int((ROINumber + 1) * XRange / (TotalNROIs + 1) + np.random.randn() * Width**(1 / 2))
    YCoords = YCoords[XCoords == RandX]
    YRange = YCoords.max() - YCoords.min()
    RandY = int(np.median(YCoords) + np.random.randn() * (Width * YRange/XRange)**(1 / 2))

    return [RandX, RandY]

def ExtractROIs(Bone, XCoords, YCoords, ROISize, NROIs=1, Plot=False):

    ROIs = np.zeros((NROIs,ROISize,ROISize,3)).astype('int')
    BoneROIs = np.zeros((NROIs,ROISize,ROISize)).astype('int')
    Xs = np.zeros((NROIs,2)).astype('int')
    Ys = np.zeros((NROIs,2)).astype('int')

    for i in range(NROIs):
        RandX, RandY = RandCoords([XCoords, YCoords], i, NROIs)
        X1, X2 = RandX - int(ROISize / 2), RandX + int(ROISize / 2)
        Y1, Y2 = RandY - int(ROISize / 2), RandY + int(ROISize / 2)
        BoneROI = Bone[Y1:Y2, X1:X2]
        BVTV = BoneROI.sum() / BoneROI.size

        while BVTV < Params.Threshold:
            RandX, RandY = RandCoords([XCoords, YCoords], i, NROIs)
            X1, X2 = RandX - int(ROISize / 2), RandX + int(ROISize / 2)
            Y1, Y2 = RandY - int(ROISize / 2), RandY + int(ROISize / 2)
            BoneROI = Bone[Y1:Y2, X1:X2]
            BVTV = BoneROI.sum() / BoneROI.size

        ROIs[i] += Params.Image[Y1:Y2, X1:X2]
        BoneROIs[i] += Bone[Y1:Y2, X1:X2]
        Xs[i] += [X1, X2]
        Ys[i] += [Y1, Y2]

    if Plot:

        Shape = np.array(Params.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Params.Image)
        Axis.plot([X1, X2], [Y1, Y1], color=(1, 0, 0))
        Axis.plot([X2, X2], [Y1, Y2], color=(1, 0, 0))
        Axis.plot([X2, X1], [Y2, Y2], color=(1, 0, 0))
        Axis.plot([X1, X1], [Y2, Y1], color=(1, 0, 0))
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
        Axis.imshow(ROIs[i])
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    return ROIs, BoneROIs, Xs, Xs

def ExtractSkeleton(Image, Plot=False):
    """
    Extract skeleton of manually segmented image
    :param Image: Numpy image dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Skeleton of the segmentation
    """

    Tic = time.time()
    print('Extract manual segmentation skeleton ...')

    Filter1 = Image[:, :, 0] > 100
    Filter2 = Image[:, :, 1] < 90
    Filter3 = Image[:, :, 2] < 150

    Bin = np.zeros(Filter1.shape)
    Bin[Filter1 & Filter2 & Filter3] = 1

    # Dilate to link extracted segmentation
    Disk = morphology.disk(5)
    BinDilate = morphology.binary_dilation(Bin, Disk)

    # Skeletonize to obtain 1 pixel thickness
    Skeleton = morphology.skeletonize(BinDilate)

    if Plot:
        Shape = np.array(Image.shape) / max(Image.shape) * 10
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Shape = np.array(Skeleton.shape) / max(Skeleton.shape) * 10
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Skeleton, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Skeleton

def NormalizeValues(Image):
    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max() - Image.min())

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

def Function2Optimize(Parameters=np.array([2., 1., 0.5, 1., 0.5, 0.05])):

    Beta, AlphaF, VF, AlphaL, VL, AlphaT = Parameters

    for i in range(Params.ROIs.shape[0]):
    Segmented = PCNN(Params.ROIs[i], Beta, AlphaF, VF, AlphaL, VL, AlphaT)
    Values = np.unique(Segmented)
    DensityDiff = np.zeros(len(Values))
    for Index, Value in enumerate(Values):
        Bin = (Segmented == Value) * 1
        DensityDiff[Index] = abs(Params.Reference - Bin.sum() / Bin.size) / Params.Reference

    if DensityDiff.min() < Params.MinCost:
        Params.MinCost = DensityDiff.min()
        Params.SegMin = DensityDiff.argmin()
    return DensityDiff.min()

def PlotRegressionResults(Model,Alpha=0.95):

    print(Model.summary())

    ## Plot results
    Y_Obs = Model.model.endog
    Y_Fit = Model.fittedvalues
    N = int(Model.nobs)
    C = np.matrix(Model.cov_params())
    X = np.matrix(Model.model.exog)
    X_Obs = np.sort(np.array(X[:,1]).reshape(len(X)))


    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / Model.df_resid)
    TSS = np.sum((Model.model.endog - Model.model.endog.mean()) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    ## Compute CI lines
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0


    ## Plots
    DPI = 100
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI, sharey=True, sharex=True)
    Axes.plot(X[:,1], Y_Fit, color=(1,0,0), label='Fit')
    Axes.fill_between(X_Obs, np.sort(CI_Line_o), np.sort(CI_Line_u), color=(0, 0, 0), alpha=0.1, label=str(int(Alpha*100)) + '% CI')
    Axes.plot(X[:,1], Y_Obs, linestyle='none', marker='o', color=(0,0,0), fillstyle='none')
    Axes.annotate(r'$N$  : ' + str(N), xy=(0.8, 0.175), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.8, 0.1), xycoords='axes fraction')
    Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.8, 0.025), xycoords='axes fraction')
    Axes.set_ylabel('Mineral Density (mg HA/cm$^3$)')
    Axes.set_xlabel('Gray Value (-)')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend()
    plt.show()
    plt.close(Figure)


# Read Image
Params = Parameters(2)
ReadImage(Params)

# Segment bone and extract coordinate
Bone = SegmentBone(Params.SegImage, Plot='Full')
Y, X = np.where(Bone)

# Set ROI pixel size
PhysicalSize = 1000
ROISize = int(round(PhysicalSize / Params.PixelLength))

# Filter positions too close to the border
F1 = X > ROISize / 2
F2 = X < Bone.shape[1] - ROISize / 2
FilteredX = X[F1 & F2]
FilteredY = Y[F1 & F2]

F1 = FilteredY > ROISize / 2
F2 = FilteredY < Bone.shape[0] - ROISize / 2
FilteredY = FilteredY[F1 & F2]
FilteredX = FilteredX[F1 & F2]

# Extract random ROI and verify validity
NROIs = 2
ROIs, BoneROIs, Xs, Ys = ExtractROIs(Bone, FilteredX, FilteredY, ROISize, NROIs=NROIs, Plot=True)

# Extract manual segmentation and compute CM density
Skeletons = np.zeros(BoneROIs.shape)
References = np.zeros(NROIs)
for i in range(NROIs):
    Skeletons[i] += ExtractSkeleton(Params.SegImage[Ys[i,0]:Ys[i,1], Xs[i,0]:Xs[i,1]], Plot=True)
    References[i] += Skeletons[i].sum() / BoneROIs[i].sum()
Params.Reference = References

# Convert ROI into gray scale image
Grays = np.zeros(BoneROIs.shape)
for i in range(NROIs):
    Grays[i] = color.rgb2gray(ROIs[i])
Params.ROIs = Grays


# Run PSO for PCNN parameters
Ranges = np.array([[0,4],[1E-2,10],[1E-2,1],[1E-2,10],[1E-2,1],[1E-2,1]])
Population = 20
Cs = [0.15, 0.1]
Arguments = PSOArgs(Function2Optimize, Ranges, Population, Cs, MaxIt=20)
PSOResults = PSO.Main(Arguments, Evolution=True)

# Plot results
Beta, AlphaF, VF, AlphaL, VL, AlphaT = PSOResults
Segmented = PCNN(Params.ROI, Beta, AlphaF, VF, AlphaL, VL, AlphaT)
Values = np.unique(Segmented)
Bin = (Segmented == Values[Params.SegMin]) * 1
PlotArray(Bin, 'Segment ' + str(Params.SegMin))

print('Reference value: ' + str(Params.Reference))
print('Final value: ' + str(Bin.sum() / Bin.size))
print('Relative difference: ' + str(Params.MinCost))



# Verifiy correlation between automatic and manual segmentation
Manuals = Bin.sum() / Bin.size
Automatic = Params.Reference

for i in range(10):
    ROI, BoneROI, [X1,X2], [Y1,Y2] = ExtractROI(Bone, FilteredX, FilteredY)
    Skeleton = ExtractSkeleton(Params.SegImage[Y1:Y2, X1:X2], Plot=True)
    PlotArray(Skeleton, 'Skeleton')
    Manuals = np.append(Manuals, Skeleton.sum() / BoneROI.sum())

    Params.ROI = color.rgb2lab(ROI)[:, :, 2]
    Segmented = PCNN(Params.ROI, Beta, AlphaF, VF, AlphaL, VL, AlphaT)
    Values = np.unique(Segmented)
    Bin = (Segmented == Values[Params.SegMin]) * 1
    PlotArray(Bin, 'Segment ' + str(Params.SegMin))
    Automatic = np.append(Automatic, Bin.sum() / Bin.size)

# Built data frame with mean values and corresponding mineral densities (see pdf)
Data2Fit = pd.DataFrame({'Manual': Manuals,
                         'Automatic': Automatic})

FitResults = smf.ols('Automatic ~ 1 + Manual', data=Data2Fit).fit()
PlotRegressionResults(FitResults)