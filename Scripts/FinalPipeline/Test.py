import time
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

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

def GetNeighbours(Array2D):
    """
    Function used to get values of the neighbourhood pixels (based on numpy.roll)
    :param Array2D: Row x Column numpy array
    :return: Neighbourhood pixels values
    """

    YSize, XSize = Array2D.shape[:-1]
    Dimension = Array2D.shape[-1]

    print('\nGet neighbours ...')
    Tic = time.time()
    Neighbourhood = np.zeros((YSize, XSize, 8, Dimension))
    i = 0
    for Shift in [-1, 1]:
        for Axis in [0, 1]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
            i += 1

    for Shift in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        for Axis in [(0, 1)]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)
            i += 1
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Neighbourhood

def ComputeDistances(Array2D):
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

    Toc = time.time()
    PrintTime(Tic, Toc)

    return Distances

def ComputePSPs(Distances,Threshold=1/52):

    """
    Function used to compute discrete Post-Synaptic Potentials
    :param Distances: MxNxO-dimensional numpy array containing inter-pixel distances
    :param Threshold: Constant threshold reached by all synapses
    :return: PSPs: Discretes Post-Synaptic Potentials
    """

    # Record elapsed time
    Tic = time.time()
    print('Compute PSPs ...')

    # Compute time necessary for all synapses to fire
    N = Distances.shape[-1]
    Time = Threshold/N + Distances.sum(axis=2)/N
    MaxTime = np.ceil(Time.max()).astype('int')

    # Compute PSPs at discrete times
    Size = Distances.shape[0]
    PSPs = np.zeros((Size,Size,MaxTime+1,N))
    for Time in range(1,MaxTime+1):
        PSPs[:,:,Time][Distances > Time] = Time - Distances[Distances > Time]

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic,Toc)

    return PSPs

def NormalizeValues(Image):
    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max() - Image.min())

    return N_Image

def Histogram(Array,NBins=256,Plot=False):

    """
    Compute image histogram
    Based on:
    Zhan, K., Shi, J., Wang, H. et al.
    Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
    Arch Computat Methods Eng 24, 573–588 (2017).
    https://doi.org/10.1007/s11831-016-9182-3

    :param: NBins: Number of histogram bins
    :return: H: Image histogram in numpy array
    """

    Tic = time.time()
    print('\nCompute image histogram...')

    # Initialize PCNN
    MaxS = Array.max()
    S = NormalizeValues(Array)
    Theta = 1
    Delta = 1 / (NBins - 1)
    Vt = 1 + Delta
    Y = np.zeros(S.shape)
    U = S
    H = np.zeros(NBins)

    # Perform histogram analysis
    for N in range(1,NBins+1):
        Theta = Theta - Delta + Vt * Y
        Y = np.where((U - Theta) > 0, 1, 0)
        H[NBins - N] = Y.sum()

    # Print time elapsed
    Toc = time.time()
    PrintTime(Tic, Toc)

    Bins = np.arange(0,MaxS+Delta,Delta*MaxS)

    if Plot:
        Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
        Axes.bar(x=Bins, height=H / H.sum(), width=Bins.max()/len(Bins), color=(1, 0, 0))
        Axes.set_xlabel('Values (-)')
        Axes.set_ylabel('Density (-)')
        plt.subplots_adjust(left=0.175)
        plt.show()
        plt.close(Figure)

    return H, Bins


Array = ROIs[0]
PlotImage(Array)

Filtered = filters.gaussian(Array,sigma=2,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

Distances = ComputeDistances(Filtered)

PSPs = ComputePSPs(Distances)
Figure, Axis = plt.subplots(1,1)
Axis.plot(PSPs.sum(axis=(3,0,1)),marker='o')
plt.show()

MaxDistances = np.max(Distances,axis=2)
Histogram(MaxDistances, Plot=True)
PlotImage(MaxDistances)

Seeded = np.zeros(MaxDistances.shape)
Threshold = 10
Seeded[MaxDistances < Threshold] = 1
PlotImage(Seeded)

# Reproduce MPCNN step from paper