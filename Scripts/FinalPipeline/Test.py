#!/usr/bin/env python3

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from scipy.signal import find_peaks
import statsmodels.formula.api as smf
from scipy.stats.distributions import t
from skimage import io, morphology, color, filters, segmentation, measure

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPipeline'))

import PSO
import Filtering

plt.rc('font', size=12)


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

class PSOArgs:

    def __init__(self, Function2Optimize, Ranges, Population, Cs, MaxIt=10, STC=1E-3):
        self.Function = Function2Optimize
        self.Ranges = Ranges
        self.Population = Population
        self.Cs = Cs
        self.MaxIt = MaxIt
        self.STC = STC

def PlotImage(Array):

    Figure, Axis = plt.subplots(1,1,figsize=(10,10))
    if Array.shape[-1] == 3:
        Axis.imshow(Array)
    else:
        Axis.imshow(Array, cmap='binary_r')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()

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

def GetNeighbours(Array2D, N=1, Print=False):
    """
    Function used to get values of the neighbourhood pixels (based on numpy.roll)
    :param Array2D: Row x Column numpy array
    :param N: Number of neighbours offset (1 or 2 usually)
    :return: Neighbourhood pixels values
    """

    # Define a map for the neighbour index computation
    Map = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]])

    # number of neighbours
    Neighbours = (2*N+1)**2 - 1

    if len(Array2D.shape) > 2:
        YSize, XSize = Array2D.shape[:-1]
        Dimension = Array2D.shape[-1]
        Neighbourhood = np.zeros((YSize, XSize, Neighbours, Dimension))

        # Pad the array to avoid border effects
        Array2D = np.pad(Array2D, ((1, 1), (1, 1), (0, 0)), 'symmetric')
    else:
        YSize, XSize = Array2D.shape
        Neighbourhood = np.zeros((YSize, XSize, Neighbours))

        # Pad the array to avoid border effects
        Array2D = np.pad(Array2D, 1, 'symmetric')

    if Print:
        print('\nGet neighbours ...')
        Tic = time.time()

    i = 0
    for Shift in [-1, 1]:
        for Axis in [0, 1]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[1:-1,1:-1]
            i += 1

    for Shift in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        for Axis in [(0, 1)]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[1:-1,1:-1]
            i += 1

    if N == 2:

        # Pad again the array to avoid border effects
        if len(Array2D.shape) > 2:
            Array2D = np.pad(Array2D, ((1, 1), (1, 1), (0, 0)), 'symmetric')
        else:
            Array2D = np.pad(Array2D, 1, 'symmetric')

        for Shift in [-2, 2]:
            for Axis in [0, 1]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

        for Shift in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

        for Shift in [(-2, -1), (2, -1), (-2, 1), (2, 1)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

        for Shift in [(-1, -2), (1, -2), (-1, 2), (1, 2)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

    if Print:
        Toc = time.time()
        PrintTime(Tic, Toc)

    return Neighbourhood, Map

def ComputeGradients(Array2D, N=1):
    """
    Function used to get max spectral distance between pixel and its neighbourhood
    :param Array2D: Row x Column numpy array
    :return: Maximum distance
    """

    Neighbours, Map = GetNeighbours(Array2D, N)

    print('\nCompute gradients ...')
    Tic = time.time()

    Distances = np.zeros((Neighbours.shape[:-1]))
    Dimension = (2*N+1)**2 - 1
    for i in range(Dimension):
        Distances[:, :, i] = np.linalg.norm(Array2D - Neighbours[:, :, i], axis=2)

    Toc = time.time()
    PrintTime(Tic, Toc)

    return Distances, Map

def ComputeDistances(Array2D):
    """
    Function used to get max spectral distance between pixel and its neighbourhood
    :param Array2D: Row x Column numpy array
    :return: Maximum distance
    """

    Neighbours, Map = GetNeighbours(Array2D)

    print('\nCompute distances ...')
    Tic = time.time()

    Distances = np.zeros((Array2D.shape[0], Array2D.shape[1], 8, 3))
    for i in range(8):
        Distances[:, :, i] = np.abs(Array2D - Neighbours[:, :, i])

    Toc = time.time()
    PrintTime(Tic, Toc)

    return Distances, Map

def NormalizeValues(Image):
    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max() - Image.min())

    return N_Image

def SPCNN_Edges(Image, Beta=2, VT=100, GT=0.3, C=0.01):

    """
    Image edge detection using simplified PCNN and single neuron firing
    Based on:
    Shi, Z., Hu, J. (2010)
    Image edge detection method based on A simplified PCNN model with anisotropic linking mechanism
    Proceedings of the 2010 10th International Conference on Intelligent Systems Design and Applications, ISDA’10, 330–335
    https://doi.org/10.1109/ISDA.2010.5687242

    :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
    :param VT: Dynamic threshold amplitude
    :param GT: Gradient threshold ratio with respect to maximum gradient
    :param C: Small constant ensuring all neurons firing at first iteration
    :return: 1 - normalized time matrix
    """

    Tic = time.time()
    print('\nPCNN edges detection ...')

    # Initialize parameters
    Rows, Columns = Image.shape[:-1]
    Y = np.zeros((Rows, Columns))

    # Compute anisotropic weight matrix
    Gradients3, Map = ComputeGradients(Image, N=1)

    GradientThreshold = GT * np.max(Gradients3)
    A = (Gradients3 > GradientThreshold) * 1

    # Deal with non-edge pixels
    Gradients5 = ComputeGradients(Image, N=2)[0]
    Sum33 = np.sum(Gradients3 > GradientThreshold, axis=2)
    Sum55 = np.sum(Gradients5 > GradientThreshold, axis=2)
    NonEdgePixels = Sum33 == Sum55
    A[NonEdgePixels] = np.ones(8)

    # Perform analysis
    for N in range(2):

        N += 1
        YNeighbours = GetNeighbours(Y)[0]
        F = np.sum(YNeighbours * A, axis=2) + C
        L = np.sum(YNeighbours * A, axis=2)
        Theta = VT * Y

        U = F * (1 + Beta * L)
        Y = (U > Theta) * 1

    # Print time elapsed
    Toc = time.time()
    print('\nEdge detection done!')
    PrintTime(Tic, Toc)

    return Y

def SPCNN_Edges(Image,Beta=2,Delta=1/255,VT=100):

    """
    Image edge detection using simplified PCNN and single neuron firing
    Based on:
    Shi, Z., Hu, J. (2010)
    Image edge detection method based on A simplified PCNN model with anisotropic linking mechanism
    Proceedings of the 2010 10th International Conference on Intelligent Systems Design and Applications, ISDA’10, 330–335
    https://doi.org/10.1109/ISDA.2010.5687242

    :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
    :param Delta: Linear decay factor for threshold level
    :param VT: Dynamic threshold amplitude
    :return: H: Image histogram in numpy array
    """

    Tic = time.time()
    print('\nPCNN edges detection ...')

    # Initialize parameters
    S = NormalizeValues(Image)
    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    T = np.zeros((Rows, Columns))
    W = np.array([[0.5, 1, 0.5],
                  [1, 0, 1],
                  [0.5, 1, 0.5]])
    Theta = np.ones((Rows, Columns))

    FiredNumber = 0
    N = 0

    # Perform analysis
    while FiredNumber < S.size:

        N += 1
        F = S
        L = correlate(Y, W, output='float', mode='reflect')
        Theta = Theta - Delta + VT * Y

        U = F * (1 + Beta * L)
        Y = (U > Theta) * 1

        FiredNumber = FiredNumber + sum(sum(Y))

        MedianFilter = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9
        Y = correlate(Y,MedianFilter,output='int',mode='reflect')

        T = T + N * Y


    Output = 1 - NormalizeValues(T)

    # Print time elapsed
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Output

def BetweenClassVariance(GrayScale, Segmented):

    Ignited_Neurons = Segmented == 1

    N0 = np.sum(~Ignited_Neurons)
    N1 = np.sum(Ignited_Neurons)

    if N0 == 0 or N1 == 0:
        Vb = 0

    else:
        w0 = N0 / Segmented.size
        w1 = N1 / Segmented.size

        u0 = GrayScale[~Ignited_Neurons].mean()
        u1 = GrayScale[Ignited_Neurons].mean()

        Vb = w0 * w1 * (u0 - u1) ** 2

    return Vb

def SPCNN(Image,Beta=2,Delta=1/255,VT=100):

    """
    Segment image using simplified PCNN, single neuron firing and fast linking implementation
    Based on:
    Zhan, K., Shi, J., Wang, H. et al.
    Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
    Arch Computat Methods Eng 24, 573–588 (2017).
    https://doi.org/10.1007/s11831-016-9182-3

    :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
    :param Delta: Linear decay factor for threshold level
    :param VT: Dynamic threshold amplitude
    :param Nl_max: Max number of iteration for fast linking
    :return: H: Image histogram in numpy array
    """

    Tic = time.time()
    print('\nImage segmentation...')

    # Initialize parameters
    S = NormalizeValues(Image)
    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    T = np.zeros((Rows, Columns))
    W = np.ones((Rows, Columns, 8)) * np.array([1,1,1,1,0.5,0.5,0.5,0.5])
    Theta = np.ones((Rows, Columns))

    FiredNumber = 0
    N = 0

    # Perform segmentation
    while FiredNumber < S.size:

        N += 1
        F = S
        L = np.sum(GetNeighbours(Y)[0] * W, axis=2)
        Theta = Theta - Delta + VT * Y
        U = F * (1 + Beta * L)
        Y = (U > Theta) * 1

        T = T + N * Y
        FiredNumber = FiredNumber + sum(sum(Y))

    Output = 1 - NormalizeValues(T)

    # Print time elapsed
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Output

def SegmentsColors(Image, Segmented):

    ColorSegments = np.zeros(Image.shape, 'uint8')

    for Value in np.unique(Segmented):
        Filter = Segmented == Value
        R = Image[:, :, 0][Filter]
        G = Image[:, :, 1][Filter]
        B = Image[:, :, 2][Filter]
        MeanColor = np.mean([R, G, B], axis=1)
        Filter3 = np.repeat(Filter, 3).reshape(Image.shape)
        ColorSegments += Filter3 * np.round(MeanColor).astype('uint8')

    return ColorSegments

def FuseSegments(Segmented, Seg2Fuse):
    Fused = np.zeros(Segmented.shape)
    Segments = np.unique(Segmented)
    for j in Seg2Fuse:

        if len(j) == 2:
            Start, Stop = j

            for i in range(Start, Stop + 1):
                Filter = Segmented == Segments[i]
                Fused[Filter] += Start
        else:
            Filter = Segmented == Segments[j]
            Fused[Filter] += j

    return Fused

def DiceCoefficient(Bin1, Bin2):

    return np.sum(Bin1 * Bin2) / np.sum(Bin1 + Bin2)

def Function2Optimize(Parameters=np.array([2., 100., 0.3])):

    Beta, VT, GT = Parameters

    Segmented = SPCNN_Edges(Params.Gray, Beta, VT, GT)
    Cost = 1 - DiceCoefficient(Segmented, Params.Seg)

    if Cost < Params.Cost:
        Params.Cost = Cost

    return Cost

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

# Compute density manual vs segmented
Density = []
for i in range(N):
    SegROI = Parameters.SegImage[Ys[i,0]:Ys[i,1], Xs[i,0]:Xs[i,1]]
    ROISkeleton = ExtractSkeleton(SegROI, Plot=False)
    Density.append(ROISkeleton.sum() / ROISkeleton.size)

Data = []
for i in range(len(ROIs)):
    Gray = color.rgb2gray(ROIs[i])
    Segmented = SPCNN(Gray, Beta=2, Delta=1 / 4, VT=1000)
    Segments = Segmented == np.unique(Segmented)[3]
    Data.append(np.sum(Segments) / Segments.size)

Data2Fit = pd.DataFrame({'Manual':Density,'Automatic':Data})
Data2Fit = Data2Fit[Data2Fit['Manual'] > 2E-4].reset_index()
Data2Fit, FitResults, R2, SE, p, CI = FitData(Data2Fit[['Automatic','Manual']])

Figure, Axis = plt.subplots(1,1)
Axis.plot(Data2Fit['Residuals'],linestyle='none',marker='o')
plt.show()

Drop = [0,2,8]
Keep = [2,3,4,5,7]
FitData(Data2Fit.drop(Drop).reset_index(drop=True))

for i in Drop:
    SegROI = Parameters.SegImage[Ys[i,0]:Ys[i,1], Xs[i,0]:Xs[i,1]]
    PlotImage(SegROI)

for i in Keep:
    SegROI = Parameters.SegImage[Ys[i,0]:Ys[i,1], Xs[i,0]:Xs[i,1]]
    PlotImage(SegROI)

















# Read images
Array = io.imread('TestROI.png')
PlotImage(Array)

# Figure, Axis = plt.subplots(1,1)
# Axis.imshow(Array[470:520,350:400])
# plt.show()

# Filter image
Filtered = filters.gaussian(Array,sigma=5,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

# Find harvesian canals
Gray = color.rgb2gray(Filtered)
PlotImage(Gray)
Segmented = SPCNN(Gray,Beta=2,Delta=1/5,VT=100)
PlotImage(Segmented)
Harvesian = Segmented == np.unique(Segmented)[-1]
Harvesian = morphology.binary_dilation(Harvesian,morphology.disk(5))
PlotImage(Harvesian)

# Distances
Axis, Distances = morphology.medial_axis(1-Harvesian, return_distance=True)
PlotImage(Distances)

# Define Edges
Test = Array.copy()
F1 = Test[:,:,0] < 120
F2 = Test[:,:,1] < 120
F3 = Test[:,:,2] < 180
F4 = Test[:,:,2] > 160
Edges = 1-F1*F2*F3*F4
PlotImage(Edges)

Gray = color.rgb2gray(Array)
Segmented = SPCNN(Gray,Beta=2,Delta=1/4,VT=1000)
CSegmented = SegmentsColors(Array,Segmented)
PlotImage(CSegmented)
PlotImage(Segmented == np.unique(Segmented)[3])
Mask = Segmented == np.unique(Segmented)[3]

# Mask
PlotImage(Gray*Distances)

# Watershed with mask
Markers = measure.label(Harvesian)
Segmented = segmentation.watershed(Gray*Distances,markers=Markers,mask=Mask)
PlotImage(Segmented)

Filtered = filters.gaussian(Array,sigma=5,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

Gradients, Map = ComputeGradients(Filtered)
MaxGrad = Gradients.max()
Bin = np.sum(Gradients > 0.1 * MaxGrad,axis=2)
Borders = Bin > 2
NoNoise = Bin < 8
PlotImage(1 - Borders * NoNoise)


def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq
HSV = color.rgb2hsv(Filtered)
I = np.round(HSV[:,:,2]/HSV[:,:,2].max()*255).astype('uint8')
Enhanced = enhance_contrast(I)/255
Enhanced = color.hsv2rgb(np.dstack([HSV[:,:,0], HSV[:,:,1],Enhanced]))
Enhanced = np.round(NormalizeValues(Enhanced) * 255).astype('uint8')
PlotImage(Enhanced)

Gray = color.rgb2gray(Array)
PlotImage(Gray)
Segmented = SPCNN(Gray,Beta=2,Delta=1/5,VT=1000)
CSegmented = SegmentsColors(Array,Segmented)
PlotImage(CSegmented)

# Segments selection
Colors = np.unique(CSegmented.reshape(-1, CSegmented.shape[2]), axis=0)
PlotImage((CSegmented == Colors[10])[:,:,0])

# Compute segments color differences
Differences = (Colors - np.roll(Colors,1,axis=0).astype('float'))[1:]
Figure, Axis = plt.subplots(1,1)
Axis.plot(np.arange(len(Colors)-1)+0.5, Differences[:,0], color=(1,0,0))
Axis.plot(np.arange(len(Colors)-1)+0.5, Differences[:,1], color=(0,1,0))
Axis.plot(np.arange(len(Colors)-1)+0.5, Differences[:,2], color=(0,0,1))
Axis.plot(np.arange(len(Colors)-1)+0.5,np.linalg.norm(Differences,axis=1), color=(0,0,0))
for Segment, Color in enumerate(Colors):
    Axis.plot([],marker='s',linestyle='none',color=Color/255, label=Segment)
plt.xticks(range(len(Colors)))
plt.legend(loc='best', ncol=2)
plt.show()

# Find segments to fuse
def Segments2Fuse(Differences):
    Padded = np.pad(Differences, 1)
    Peaks = find_peaks(Padded)[0]

    Start = 0
    Segments = np.arange(len(Differences) + 1)
    Seg2Fuse = []

    for Peak in Peaks:
        Stop = Segments[Segments < Peak][-1]
        if Stop > Start:
            Seg2Fuse.append([Start, Stop])
        else:
            Seg2Fuse.append([Start])
        Start = Stop + 1

    Stop = Segments[-1]
    if Stop > Start:
        Seg2Fuse.append([Start, Stop])
    else:
        Seg2Fuse.append([Start])

    return Seg2Fuse

Seg2Fuse = Segments2Fuse(np.linalg.norm(Differences,axis=1))

# Fuse similar segments
Seg2Fuse = [[0,3],[4,8],[9,15],[16,19]]
Fused = FuseSegments(Segmented, Seg2Fuse)
PlotImage(SegmentsColors(Enhanced,Fused))

# Clean segment
Disk = morphology.disk(2)
Eroded = morphology.binary_erosion((CSegmented == Colors[1])[:,:,0], Disk)
# PlotImage(Eroded)
Dilated = morphology.binary_dilation(Eroded, Disk)
PlotImage(Dilated)

Clean = (CSegmented == Colors[1])[:,:,0] - Dilated*1
Clean[Clean < 0] = 0
PlotImage(Clean)



PlotImage(Gradients.max(axis=2))
Thresholds = np.unique(Gradients.max(axis=2))
j = 0
Threshold = Thresholds[j]

F = Gradients.max(axis=2) <= Thresholds.max()/25
Y = F * 1

for i in range(10):
    Y0, X0 = np.where(Y)
    XShifts = np.repeat(Map[:,1],len(Y0)).reshape((len(Map),len(Y0)))
    YShifts = np.repeat(Map[:,0],len(Y0)).reshape((len(Map),len(Y0)))
    Y_Copy = Y.copy()
    Fire = Gradients[Y0,X0] <= Threshold
    Y[Y0 - YShifts * Fire.T, X0 - XShifts * Fire.T] = 1
    while np.alltrue(Y == Y_Copy):
        j += 1
        Threshold = Thresholds[j]

        Fire = Gradients[Y0, X0] <= Threshold
        XShifts = np.repeat(Map[:,1], len(Y0)).reshape((len(Map), len(Y0)))
        YShifts = np.repeat(Map[:,0], len(Y0)).reshape((len(Map), len(Y0)))
        Y[Y0 - YShifts * Fire.T, X0 - XShifts * Fire.T] = 1
    PlotImage(Y)


Seg = io.imread('TestROI_Seg.png')
PlotImage(Seg)

Skeleton = ExtractSkeleton(Seg) * 1
Skeleton = morphology.binary_dilation(Skeleton, morphology.disk(5))
PlotImage(Skeleton)

Filtered = filters.gaussian(Array,sigma=1,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

Norm = np.linspace(-5,5)
Figure, Axis = plt.subplots(1,1)
Axis.plot(Norm, 1 - 1 / (1 + np.exp(-1 * (Norm - 0))), color=(1,0,0))
Axis.plot(Norm, 1 - 1 / (1 + np.exp(-2 * (Norm - 0))), color=(0,1,0))
Axis.plot(Norm, 1 - 1 / (1 + np.exp(-1 * (Norm - 1))), color=(0,0,1))
plt.show()

# Transform to gray level
Gray = color.rgb2gray(Array)
PlotImage(Gray)
Gradients, Map = ComputeGradients(Array)
PlotImage(Gradients.max(axis=2) > 30)

# Pass arguments to parameter class
Params = ParameterClass(Filtered, Skeleton)

# Run PSO for SPCNN parameters
Ranges = np.array([[0,4],[1,200],[0.1,0.9]])
Population = 20
Cs = [0.15, 0.1]
Arguments = PSOArgs(Function2Optimize, Ranges, Population, Cs, MaxIt=20)
PSOResults = PSO.Main(Arguments, Evolution=True)

# PCNN edges
Beta, VT, GT = PSOResults
PCNN_Edges = 1-SPCNN_Edges(Enhanced, 5, 100, 0.3)
PlotImage(PCNN_Edges)
Edges = np.zeros(PCNN_Edges.shape)
Edges[PCNN_Edges == Params.PCNN_Seg] = 1
PlotImage(Edges)