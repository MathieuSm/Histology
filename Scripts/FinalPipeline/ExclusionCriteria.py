#!/usr/bin/env python3

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats.distributions import t
from skimage import io, morphology, measure, filters
from PIL import Image

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPipeline'))
from Utilities import *

Image.MAX_IMAGE_PIXELS = None

class ParametersClass:

    def __init__(self, ImageNumber, Threshold=0.88, SubArea=[[1800, 2200], [7800, 8200]]):
        self.N = ImageNumber
        self.Directory = Path.cwd() / 'Tests/Osteons/Sensitivity/'
        self.Threshold = Threshold
        self.SubArea = SubArea

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

    Filter1 = Image[:,:,0] < 100
    Filter2 = Image[:,:,1] < 100
    Filter3 = Image[:,:,2] < 100

    Bin = np.zeros(Filter1.shape,'int')
    Bin[Filter1 & Filter2 & Filter3] = 1

    if Plot:
        Figure, Axis = plt.subplots(1,1)
        Axis.imshow(Bin,cmap='binary')
        plt.show()

    RegionProps = measure.regionprops(Bin)[0]
    Pixels = RegionProps.coords[:,1].max() - RegionProps.coords[:,1].min()
    PixelLength = Length / Pixels

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return PixelLength

def ReadImage(Plot=True):

    # Read image and plot it
    Directory = Parameters.Directory
    DataFrame = pd.read_csv(str(Directory / 'Data.csv'))
    SampleData = DataFrame.loc[Parameters.N]
    Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '_Seg.jpg')
    Image = io.imread(str(Directory / Name))[:, :, :3]

    if Plot:
        Shape = np.array(Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Store name, image, and pixel length in parameters class
    Parameters.Name = Name[:-7]
    Parameters.SegImage = Image
    Parameters.Image = io.imread(str(Directory / Name[:-8]) + '.jpg')

    if Parameters.Name[:5] == '418RM':
        Parameters.PixelLength = PixelSize(Parameters.Image[9400:-400, 12500:-300], 2000, Plot=True)
    else:
        Parameters.PixelLength = 1.0460251046025104  # Computed with 418 RM

def SegmentBone(Image, Plot=False, SubArea=None):

    """
    Segment bone structure
    :param Image: RGB numpy array dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Labelled bone image
    """

    Tic = time.time()
    print('Segment bone area ...')

    if not SubArea:
        SubArea = [[0, 1], [0, 1]]

    # Mark areas where there is bone
    Filter1 = Image[:, :, 0] < 190
    Filter2 = Image[:, :, 1] < 190
    Filter3 = Image[:, :, 2] < 235
    Bone = Filter1 & Filter2 & Filter3

    if Plot == 'Full':
        Shape = np.array(Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    elif Plot == 'Sub':
        Shape = np.array([SubArea[1][1]-SubArea[1][0], SubArea[0][1]-SubArea[0][0]]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone[SubArea[0][0]:SubArea[0][1],
                         SubArea[1][0]:SubArea[1][1]], cmap='binary')
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

def ExtractROIs(Bone, XCoords, YCoords, ROISize, NROIs=1, Plot=False, ROIsPlot=False):

    Tic = time.time()
    print('\nBegin ' + str(NROIs) + ' ROIs extraction ...')

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

        j = 0
        while BVTV < Parameters.Threshold and j < 100:
            RandX, RandY = RandCoords([XCoords, YCoords], i, NROIs)
            X1, X2 = RandX - int(ROISize / 2), RandX + int(ROISize / 2)
            Y1, Y2 = RandY - int(ROISize / 2), RandY + int(ROISize / 2)
            BoneROI = Bone[Y1:Y2, X1:X2]
            BVTV = BoneROI.sum() / BoneROI.size
            j += 1
            if j == 100:
                print('No ROI found after 100 iterations')

        ROIs[i] += Parameters.Image[Y1:Y2, X1:X2]
        BoneROIs[i] += Bone[Y1:Y2, X1:X2]
        Xs[i] += [X1, X2]
        Ys[i] += [Y1, Y2]

        if ROIsPlot:
            Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
            Axis.imshow(ROIs[i])
            Axis.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            plt.show()

    if Plot:
        Shape = np.array(Parameters.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Parameters.Image)

        for i in range(len(Xs)):
            Axis.plot([Xs[i,0], Xs[i,1]], [Ys[i,0], Ys[i,0]], color=(1, 0, 0))
            Axis.plot([Xs[i,1], Xs[i,1]], [Ys[i,0], Ys[i,1]], color=(1, 0, 0))
            Axis.plot([Xs[i,1], Xs[i,0]], [Ys[i,1], Ys[i,1]], color=(1, 0, 0))
            Axis.plot([Xs[i,0], Xs[i,0]], [Ys[i,1], Ys[i,0]], color=(1, 0, 0))
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic,Toc)

    return ROIs.astype('uint8'), BoneROIs, Xs, Ys

def ExtractSkeleton(Image, Plot=False):
    """
    Extract skeleton of manually segmented image
    :param Image: Numpy image dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Skeleton of the segmentation
    """

    Tic = time.time()
    print('\nExtract manual segmentation skeleton ...')

    Filter1 = Image[:, :, 0] > 110
    Filter2 = Image[:, :, 1] < 90
    Filter3 = Image[:, :, 2] < 140

    Bin = np.zeros(Filter1.shape)
    Bin[Filter1 & Filter2 & Filter3] = 1

    # Dilate to link extracted segmentation
    Disk = morphology.disk(5)
    BinDilate = morphology.binary_dilation(Bin, Disk)

    # Skeletonize to obtain 1 pixel thickness
    Skeleton = morphology.skeletonize(BinDilate)

    if Plot:
        Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
        Axis.imshow(Skeleton, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Skeleton

def FitData(DataFrame):

    Formula = DataFrame.columns[1] + ' ~ ' + DataFrame.columns[0]
    FitResults = smf.ols(Formula, data=DataFrame).fit()

    # Calculate R^2, p-value, 95% CI, SE, N
    Y_Obs = FitResults.model.endog
    Y_Fit = FitResults.fittedvalues

    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / FitResults.df_resid)

    N = int(FitResults.nobs)
    R2 = FitResults.rsquared
    p = FitResults.pvalues[1]

    CI_l = FitResults.conf_int()[0][1]
    CI_r = FitResults.conf_int()[1][1]

    X = np.matrix(FitResults.model.exog)
    X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))
    C = np.matrix(FitResults.cov_params())
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    Alpha = 0.95
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * B_0
    Sorted_CI_u = CI_Line_u[np.argsort(FitResults.model.exog[:,1])]
    Sorted_CI_o = CI_Line_o[np.argsort(FitResults.model.exog[:,1])]

    NoteYPos = 0.925
    NoteYShift = 0.075

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
    Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
    Axes.fill_between(X_Obs, Sorted_CI_o, Sorted_CI_u, color=(0, 0, 0), alpha=0.1,
                      label=str(int(Alpha * 100)) + '% CI')
    Axes.plot(X[:, 1], Y_Obs, linestyle='none', fillstyle='none', marker='o', color=(0, 0, 1), label='Data')
    Axes.annotate('Slope 95% CI [' + str(CI_l.round(2)) + r'$,$ ' + str(CI_r.round(2)) + ']',
                  xy=(0.05, NoteYPos), xycoords='axes fraction')
    # Axes.annotate(r'$N$ : ' + str(N), xy=(0.05, NoteYPos),
    #               xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + str(R2.round(2)), xy=(0.05, NoteYPos - NoteYShift),
                  xycoords='axes fraction')
    Axes.annotate(r'$\sigma_{est}$ : ' + str(SE.round(5)), xy=(0.05, NoteYPos - NoteYShift*2),
                  xycoords='axes fraction')
    Axes.annotate(r'$p$ : ' + str(p.round(3)), xy=(0.05, NoteYPos - NoteYShift*3),
                  xycoords='axes fraction')
    Axes.set_ylabel(DataFrame.columns[1])
    Axes.set_xlabel(DataFrame.columns[0])
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend(loc='lower right')
    plt.show()

    # Add fitted values and residuals to data
    DataFrame['Fitted Value'] = Y_Fit
    DataFrame['Residuals'] = E

    return DataFrame, FitResults, R2, SE, p, [CI_l, CI_r]

# Read image
Parameters = ParametersClass(2)
ReadImage()

# Segment bone and extract coordinate
Bone = SegmentBone(Parameters.Image, Plot=True)
Y, X = np.where(Bone)

# Set ROI pixel size
ROISize = int(round(1000 / Parameters.PixelLength))
if np.mod(ROISize, 2) == 1:
    ROISize = ROISize + 1

# Filter positions too close to the border
F1 = X > ROISize / 2
F2 = X < Bone.shape[1] - ROISize / 2
FilteredX = X[F1 & F2]
FilteredY = Y[F1 & F2]

F1 = FilteredY > ROISize / 2
F2 = FilteredY < Bone.shape[0] - ROISize / 2
FilteredY = FilteredY[F1 & F2]
FilteredX = FilteredX[F1 & F2]

# Extract ROIs
N = 10
ROIs, BoneROIs, Xs, Ys = ExtractROIs(Bone, FilteredX, FilteredY, ROISize, NROIs=N, Plot=False)


BHist, Bins = np.histogram(ROIs[0][:, :, 0], density=True, bins=20, range=(0, 255))
for i in range(1,N):
    Hist, Bins = np.histogram(ROIs[i][:, :, 0], density=True, bins=20, range=(0, 255))
    BHist = np.vstack([BHist,Hist])

Mean = np.mean(BHist, axis=0)
SD = np.std(BHist, axis=0, ddof=1)

Width = Bins[1]
Bins = 0.5 * (Bins[1:] + Bins[:-1])

Figure, Axis = plt.subplots(2,2)
Axis[0].imshow(ROIs[-1])
Axis[1].bar(Bins, BHist[-1], width=Width , edgecolor=(0, 0, 1), color=(1, 1, 1, 0))
Axis[1].bar(Bins, Mean, width=Width, color=(0.5, 0.5, 0.5, 0.5), yerr=SD)
plt.show()
