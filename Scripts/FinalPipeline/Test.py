#!/usr/bin/env python3

import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, morphology, color, filters

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPipeline'))

import PSO
import Filtering

plt.rc('font', size=12)


class ParameterClass:

    def __init__(self, Gray, Seg):
        self.Gray = Gray
        self.Seg = Seg
        self.Cost = 1

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

def GetNeighbours(Array2D, N=1):
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
    else:
        YSize, XSize = Array2D.shape
        Neighbourhood = np.zeros((YSize, XSize, Neighbours))

    print('\nGet neighbours ...')
    Tic = time.time()
    i = 0
    for Shift in [-1, 1]:
        for Axis in [0, 1]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
            i += 1

    for Shift in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        for Axis in [(0, 1)]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
            i += 1

    if N == 2:
        for Shift in [-2, 2]:
            for Axis in [0, 1]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
                i += 1

        for Shift in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
                i += 1

        for Shift in [(-2, -1), (2, -1), (-2, 1), (2, 1)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
                i += 1

        for Shift in [(-1, -2), (1, -2), (-1, 2), (1, 2)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
                i += 1


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

    # # Deal with non-edge pixels
    # Gradients5 = ComputeGradients(Image, N=2)[0]
    # Sum33 = np.sum(Gradients3 > GradientThreshold, axis=2)
    # Sum55 = np.sum(Gradients5 > GradientThreshold, axis=2)
    # NonEdgePixels = Sum33 == Sum55
    # A[NonEdgePixels] = np.ones(8)

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

def DiceCoefficient(Bin1, Bin2):

    return np.sum(Bin1 * Bin2) / np.sum(Bin1 + Bin2)

def Function2Optimize(Parameters=np.array([2., 100., 0.3])):

    Beta, VT, GT = Parameters

    Segmented = SPCNN_Edges(Params.Gray, Beta, VT, GT)
    Cost = 1 - DiceCoefficient(Segmented, Params.Seg)

    if Cost < Params.Cost:
        Params.Cost = Cost

    return Cost


# Read images
Array = io.imread('TestROI.png')
PlotImage(Array)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array[470:520,350:400])
plt.show()

ROI = Array[470:520,350:400]
F1 = ROI[:,:,0] == 74
F2 = ROI[:,:,1] == 81
F3 = ROI[:,:,2] == 159
Y0, X0 = np.where(F1*F2*F3)

# PCNN
Gradients, Map = ComputeGradients(Array)

Y = np.zeros(ROI.shape[:-1])
Y[Y0,X0] = 1
PlotImage(ROI)
PlotImage(Y)

Min = Gradients[Y0,X0] == Gradients[Y0,X0].min()
Y[Y0 + Map[Min[0],0], X0 + Map[Min[0],1]] = 1
PlotImage(Y)

for i in range(10):
    Y0, X0 = np.where(Y)
    Min = Gradients[Y0,X0] == Gradients[Y0,X0].min()
    Shifts = np.repeat(Map[:,0],len(Y0)).reshape((len(Map),len(Y0)))
    Y[Y0 + Shifts * Min.T, X0 + Shifts * Min.T] = 1
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
PCNN_Edges = SPCNN_Edges(Array, 10, 50, 0.1)
PlotImage(PCNN_Edges)
Edges = np.zeros(PCNN_Edges.shape)
Edges[PCNN_Edges == Params.PCNN_Seg] = 1
PlotImage(Edges)