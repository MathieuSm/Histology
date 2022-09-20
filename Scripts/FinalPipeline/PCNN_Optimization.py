#!/usr/bin/env python3

import sys
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats.distributions import t
from matplotlib.colors import LinearSegmentedColormap
from skimage import morphology, measure, color, exposure

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


class ParameterClass:

    def __init__(self, ImageNumber, Threshold=0.88):
        self.N = ImageNumber
        self.Directory = Path.cwd() / 'Tests/Osteons/Sensitivity/'
        self.Threshold = Threshold

class PSOArgs:

    def __init__(self, Function2Optimize, Ranges, Population, Cs, MaxIt=10, STC=1E-3):
        self.Function = Function2Optimize
        self.Ranges = Ranges
        self.Population = Population
        self.Cs = Cs
        self.MaxIt = MaxIt
        self.STC = STC

class ResultsClass:

    def __init__(self, NROis):
        self.PCNN = np.zeros(NROis)
        self.MinCosts = np.ones(NROIs)
        self.SE = 1
        self.SegMin = 2


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

def PlotOverlay(ROI,Seg, Save=False, FileName=None):

    CMapDict = {'red':((0.0, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
                'green': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0)),
                'blue': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                'alpha': ((0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}
    CMap = LinearSegmentedColormap('MyMap',CMapDict)

    Figure, Axis = plt.subplots(1,1, figsize=(10,10))
    Axis.imshow(ROI)
    Axis.imshow(Seg*1, cmap=CMap, alpha=0.3)
    Axis.plot([], color=(1,0,0), lw=1, label='Segmentation')
    Axis.axis('off')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15))
    plt.subplots_adjust(0,0,1,1)
    if Save:
        plt.savefig(FileName)
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

def ReadImage(Parameters, Plot=True):
    # Read image and plot it
    Directory = Parameters.Directory
    DataFrame = pd.read_csv(str(Directory / 'Data.csv'))
    SampleData = DataFrame.loc[Parameters.N]
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
    Parameters.Name = Name[:-7]
    Parameters.Image = Image
    Parameters.SegImage = SegImage

    if Parameters.Name[:5] == '418RM':
        Parameters.PixelLength = PixelSize(Parameters.Image[9400:-400, 12500:-300], 2000, Plot=True)
    else:
        Parameters.PixelLength = 1.0460251046025104  # Computed with 418 RM

def SegmentBone(Image, Plot=False):
    """
    Segment bone structure
    :param Image: RGB numpy array dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Labelled bone image
    """

    Tic = time.time()
    print('\nSegment bone area ...')

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

def NormalizeValues(Image):
    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max() - Image.min())

    return N_Image

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

def RBFUnit(Array2D, Plot=False):
    """
    Function used to get max spectral distance between pixel and its neighbourhood
    :param Array2D: Row x Column numpy array
    :return: Maximum distance
    """

    Neighbours = GetNeighbours(Array2D)

    print('\nCompute distances ...')
    Tic = time.time()

    Distances = np.zeros((Array2D.shape[0], Array2D.shape[1], 8))
    for i in range(8):
        Distances[:, :, i] = np.linalg.norm(Array2D - Neighbours[:, :, i], axis=2)
    Distances = np.max(Distances, axis=2)

    if Plot:
        Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
        Axis.imshow(Distances, cmap='binary')
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    Toc = time.time()
    PrintTime(Tic, Toc)

    return Distances

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
    W = np.ones((Rows, Columns, 8)) * np.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])
    Theta = np.ones((Rows, Columns))

    FiredNumber = 0
    N = 0
    Condition = FiredNumber < S.size

    # Perform segmentation
    while Condition:
        N += 1
        YN = GetNeighbours(Y)[0]
        F = S + F * np.exp(-AlphaF) + VF * np.sum(YN * W, axis=2)
        L = L * np.exp(-AlphaL) + VL * np.sum(YN * W, axis=2)
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

def DiceCoefficient(Bin1, Bin2):

    return np.sum(Bin1 * Bin2) / np.sum(Bin1 + Bin2)

def Function2Optimize(Parameters=np.array([2., 1., 0.5, 1., 0.5, 0.5])):

    Beta, AlphaF, VF, AlphaL, VL, AlphaT = Parameters

    # Extract manual segmentation values
    Dict = Results.Dict
    Keys = Dict.keys()

    # Set arrays for average best segment
    NROIs = Dict['391LM']['ROI'].shape[0]
    Dices = np.zeros((NROIs, len(Keys), 1000))
    BinDensities = np.zeros((NROIs, len(Keys), 1000))

    Values = np.zeros((NROIs,len(Keys)))
    for iKey, Key in enumerate(Keys):
        Values[:,iKey] = Dict[Key]['Manual']

        for i in range(NROIs):
            Gray = color.rgb2gray(Dict[Key]['ROI'][i])
            Segmented = PCNN(Gray, Beta, AlphaF, VF, AlphaL, VL, AlphaT)

            SegValues = np.unique(Segmented)
            for Index, Value in enumerate(SegValues):
                Bin = (Segmented == Value) * 1
                Dices[i,iKey,Index] = DiceCoefficient(Bin, Dict[Key]['Skeleton'][i])
                BinDensities[i,iKey,Index] = Bin.sum() / Dict[Key]['Bone'][i].sum()
    Results.Manuals = Values

    # # Take individual best dice performance
    # MaxDicesSeg = np.argmax(Dices,axis=2)
    # PCNN_Values = np.zeros(BinDensities[:,:,0].shape)
    # for i in range(3):
    #     for j in range(5):
    #         PCNN_Values[i,j] = BinDensities[i, j, MaxDicesSeg[i,j]]

    # Sum dices to take average better performance segment
    SumDices = np.sum(Dices,axis=(1,0))
    MaxDicesSeg = np.argmax(SumDices)


    # Built data frame with mean values and corresponding mineral densities (see pdf)
    Data2Fit = pd.DataFrame({'Manual': Values.mean(axis=0),
                             'PCNN': BinDensities[:, :, MaxDicesSeg].mean(axis=0)})
    # Data2Fit = pd.DataFrame({'Manual': Values.mean(axis=0),
    #                          'PCNN': PCNN_Values.mean(axis=0)})
    Data2Fit, FitResults, R2, SE, p, CI = FitData(Data2Fit[['PCNN','Manual']], Plot=False)

    # if FitResults.params[1] <= 0:
    #     SE = Results.SE

    # Store results
    if SE < Results.SE:
        Results.SegMin = MaxDicesSeg
        Results.Automatics = BinDensities[:, :, MaxDicesSeg]
        # Results.Automatics = PCNN_Values.mean(axis=0)
        Results.Data = Data2Fit
        Results.Fit = FitResults
        Results.SE = SE

    return SE

def FitData(DataFrame, Plot=True):

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

    if Plot:
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
        plt.subplots_adjust(left=0.2, bottom=0.15)
        plt.legend(loc='lower right')
        plt.show()

    # Add fitted values and residuals to data
    DataFrame = pd.concat([DataFrame,pd.DataFrame(Y_Fit,columns=['Fitted Value'])], axis=1)
    DataFrame = pd.concat([DataFrame,pd.DataFrame(E,columns=['Residuals'])], axis=1)

    return DataFrame, FitResults, R2, SE, p, [CI_l, CI_r]

# Set parameters
Medial = [0,1,2,3,4]
NROIs = 3
PhysicalSize = 2000

Dict = {}
for Sample in range(len(Medial)):

    SampleDict = {}

    # Read Image
    Parameters = ParameterClass(Sample)
    ReadImage(Parameters)

    # Segment bone and extract coordinate
    Bone = SegmentBone(Parameters.SegImage, Plot=None)
    Y, X = np.where(Bone)

    # Set ROI pixel size
    ROISize = int(round(PhysicalSize / Parameters.PixelLength))

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
    ROIs, BoneROIs, Xs, Ys = ExtractROIs(Bone, FilteredX, FilteredY, ROISize, NROIs=NROIs, Plot=None)

    # Extract manual segmentation and compute CM density
    Skeletons = np.zeros(BoneROIs.shape)
    Manuals = np.zeros(NROIs)
    for i in range(NROIs):
        Skeletons[i] += ExtractSkeleton(Parameters.SegImage[Ys[i,0]:Ys[i,1], Xs[i,0]:Xs[i,1]], Plot=False)
        Manuals[i] += Skeletons[i].sum() / BoneROIs[i].sum()

    # Store data in dictionary
    SampleDict['ROI'] = ROIs
    SampleDict['Bone'] = BoneROIs
    SampleDict['Skeleton'] = Skeletons
    SampleDict['Manual'] = Manuals
    Dict[Parameters.Name[:-1]] = SampleDict

Results = ResultsClass(NROIs)
Results.Dict = Dict

# Compute mean ROI histogram
nBins = 255
Histograms = np.zeros((len(Dict.keys()),3,3,nBins))
for RGB in range(3):
    for nKey, Key in enumerate(Dict.keys()):
        for nROI in range(3):
            ROI = Dict[Key]['ROI'][nROI]
            Hists, Bins = np.histogram(ROI[:, :, RGB], density=False, bins=nBins, range=(0, 255))
            Histograms[nKey,nROI,RGB] = Hists
MeanHist = np.mean(Histograms,axis=(0,1)).round().astype('int')

Figure, Axis = plt.subplots(1,1)
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[0], edgecolor=(1,0,0), color=(0,0,0,0), width=Bins[1])
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[1], edgecolor=(0,1,0), color=(0,0,0,0), width=Bins[1])
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[2], edgecolor=(0,0,1), color=(0,0,0,0), width=Bins[1])
plt.show()

Start = 0
Stop = 0
Reference = np.ones(ROI.shape,'int').ravel()
for i, nPixels in enumerate(MeanHist.ravel()):
    Stop += nPixels
    Reference[Start:Stop] = np.tile(Bins,3)[i].astype('int')
    Start = Stop
Reference = np.reshape(Reference,ROI.shape,order='F')
PlotArray(Reference,'Reference')

for Key in Dict.keys():
    for nROI in range(NROIs):
        ROI = Dict[Key]['ROI'][nROI]
        ROI = np.round(exposure.match_histograms(ROI,Reference)).astype('uint8')
        Results.Dict[Key]['ROI'][nROI] = ROI

# Run PSO for PCNN parameters
Ranges = np.array([[0,4],[1E-2,10],[1E-2,1],[1E-2,10],[1E-2,1],[1E-1,5]])
Population = 50
Cs = [0.5, 0.1]
Arguments = PSOArgs(Function2Optimize, Ranges, Population, Cs, MaxIt=10, STC=1E-5)
PSOResults = PSO.Main(Arguments, Evolution=True)

# a = np.array([1])
# for a in range(10):
#     a = np.concatenate([a,np.array([a[-1]*np.exp(-1)])])
# Figure, Axis = plt.subplots(1,1)
# Axis.plot(a,color=(1,0,0))
# plt.show()

# Check PSO results
Beta, AlphaF, VF, AlphaL, VL, AlphaT = PSOResults
Beta, AlphaF, VF, AlphaL, VL, AlphaT = np.array([0.78640221, 4.55145663, 0.46926488, 7.22837865, 0.19984479, 2.33408457])
Beta, AlphaF, VF, AlphaL, VL, AlphaT = np.array([0.36657598, 2.71302182, 0.89146428, 0.01, 0.61970243, 0.1113785])
Results.SegMin = 12
Beta, AlphaF, VF, AlphaL, VL, AlphaT = np.array([0.92248269, 9.40340224, 0.39982255, 0.83549087, 0.94182256, 0.30280017])
Results.SegMin = 5

for Key in Dict.keys():
    Automatic = np.zeros(NROIs)
    for i in range(NROIs):
        Gray = color.rgb2gray(Dict[Key]['ROIs'][i])
        Segmented = PCNN(Gray, Beta, AlphaF, VF, AlphaL, VL, AlphaT)
        Values = np.unique(Segmented)
        if len(Values) > Results.SegMin:
            Bin = (Segmented == Values[Results.SegMin]) * 1
            # BinDensity = Bin.sum() / Dict[Key]['Bone'][i].sum()
            # Automatic[i] = BinDensity
            FileName = Path.cwd() / 'Scripts' / 'RandomForest' / str('PCNN_S' + str(Key) + '_ROI' + str(i) + '.jpg')
            PlotOverlay(Dict[Key]['ROIs'][i], Bin, Save=True, FileName=FileName)
    Dict[Key]['PCNN'] = np.array(Automatic)

# Plot final results
Values = np.zeros((NROIs,len(Dict.keys())))
for iKey, Key in enumerate(Dict.keys()):
    Values[0,iKey] = np.min(Dict[Key]['Manual'])
    Values[1,iKey] = np.mean(Dict[Key]['Manual'])
    Values[2,iKey] = np.max(Dict[Key]['Manual'])
Results.Manuals = Values

Values = np.zeros((NROIs,len(Dict.keys())))
for iKey, Key in enumerate(Dict.keys()):
    Values[0,iKey] = np.min(Dict[Key]['PCNN'])
    Values[1,iKey] = np.mean(Dict[Key]['PCNN'])
    Values[2,iKey] = np.max(Dict[Key]['PCNN'])
Results.Automatics = Values

Values = np.zeros((NROIs,len(Dict.keys())))
for iKey, Key in enumerate(Dict.keys()):
    Values[:,iKey] = Dict[Key]['Manual']
Results.Manuals = Values

Values = np.zeros((NROIs,len(Dict.keys())))
for iKey, Key in enumerate(Dict.keys()):
    Values[:,iKey] = Dict[Key]['PCNN']
Results.Automatics = Values

Data2Fit = pd.DataFrame({'Manual': Results.Manuals.ravel(),
                         'PCNN': 0.22-Results.Automatics.ravel()})
Data2Fit, FitResults, R2, SE, p, CI = FitData(Data2Fit[['PCNN','Manual']], Plot=True)

X = Results.Automatics.mean(axis=1)
XError = np.abs(X - np.array([Results.Automatics.min(axis=1), Results.Automatics.max(axis=1)]))
Y = Results.Manuals[1,:]
YError = np.abs(Y - np.array([Results.Manuals[0,:],Results.Manuals[2,:]]))

# Calculate R^2, p-value, 95% CI, SE, N
Y_Obs = Results.Fit.model.endog
Y_Fit = Results.Fit.fittedvalues

E = Y_Obs - Y_Fit
RSS = np.sum(E**2)
SE = np.sqrt(RSS / Results.Fit.df_resid)

N = int(Results.Fit.nobs)
R2 = Results.Fit.rsquared
p = Results.Fit.pvalues[1]

CI_l = Results.Fit.conf_int()[0][1]
CI_r = Results.Fit.conf_int()[1][1]

X = np.matrix(Results.Fit.model.exog)
X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))
C = np.matrix(Results.Fit.cov_params())
B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
Alpha = 0.95
t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
CI_Line_u = Y_Fit + t_Alpha[0] * B_0
CI_Line_o = Y_Fit + t_Alpha[1] * B_0
Sorted_CI_u = CI_Line_u[np.argsort(Results.Fit.model.exog[:, 1])]
Sorted_CI_o = CI_Line_o[np.argsort(Results.Fit.model.exog[:, 1])]

NoteYPos = 0.925
NoteYShift = 0.075

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
Axes.fill_between(X_Obs, Sorted_CI_o, Sorted_CI_u, color=(0, 0, 0), alpha=0.1,
                  label=str(int(Alpha * 100)) + '% CI')
Axes.errorbar(X[:, 1],Y_Fit,xerr=XError,yerr=YError,fmt='o',color=(0,0,1),mfc=(1,1,1),ecolor=(0,0,1,0.5), label='Data')
Axes.annotate('Slope 95% CI [' + str(CI_l.round(2)) + r'$,$ ' + str(CI_r.round(2)) + ']',
              xy=(0.05, NoteYPos), xycoords='axes fraction')
Axes.annotate(r'$R^2$ : ' + str(R2.round(2)), xy=(0.05, NoteYPos - NoteYShift),
              xycoords='axes fraction')
Axes.annotate(r'$\sigma_{est}$ : ' + str(SE.round(5)), xy=(0.05, NoteYPos - NoteYShift * 2),
              xycoords='axes fraction')
Axes.annotate(r'$p$ : ' + str(p.round(3)), xy=(0.05, NoteYPos - NoteYShift * 3),
              xycoords='axes fraction')
Axes.set_ylabel('Manual segmentation')
Axes.set_xlabel('PCNN segmentation')
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.legend(loc='lower right')
plt.show()

import pickle
with open('OptimizationData.pkl', 'wb') as f:
    pickle.dump(Dict, f)