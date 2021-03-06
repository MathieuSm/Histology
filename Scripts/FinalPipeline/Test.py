import time
import numpy as np
from skimage import io, filters, measure
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

    # Define a map for the neighbour index computation
    Map = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]])

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

    return Neighbourhood, Map

def ComputeDistances(Array2D):
    """
    Function used to get max spectral distance between pixel and its neighbourhood
    :param Array2D: Row x Column numpy array
    :return: Maximum distance
    """

    Neighbours, Map = GetNeighbours(Array2D)

    print('\nCompute distances ...')
    Tic = time.time()

    Distances = np.zeros((Array2D.shape[0], Array2D.shape[1], 8))
    for i in range(8):
        Distances[:, :, i] = np.linalg.norm(Array2D - Neighbours[:, :, i], axis=2)

    Toc = time.time()
    PrintTime(Tic, Toc)

    return Distances, Map

def ComputePSPs(Distances,Threshold=256/52):

    """
    Function used to compute discrete PostSynaptic Potentials
    :param Distances: MxNxO-dimensional numpy array containing inter-pixel distances
    :param Threshold: Constant threshold reached by all synapses
    :return: PSPs: Discretes Post-Synaptic Potentials
    """

    # Record elapsed time
    Tic = time.time()
    print('\nCompute PSPs ...')

    # Compute time necessary for all synapses to fire
    Ni = 3
    Time = Threshold/Ni + Distances/Ni
    MaxTime = np.ceil(Time.max()).astype('int')

    # Compute PSPs at discrete times
    Size = Distances.shape[0]
    PSPs = np.zeros((Size,Size,MaxTime+1,N))
    for Time in range(1,MaxTime+1):
        PSPs[:,:,Time][Time >= Distances] = Time - Distances[Time >= Distances]

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
    Arch Computat Methods Eng 24, 573???588 (2017).
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

Array = np.round(50 * np.random.randn(5,5,3) + 128).astype('uint8')
Array[1:4,1:4] = [200,200,200]
Array[0,4] = [200, 200, 200]
Array = io.imread('TestROI.png')
PlotImage(Array)

Filtered = filters.gaussian(Array,sigma=3,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

Distances, Map = ComputeDistances(Filtered)

Threshold = 256 / 52
Seeded = np.max(Distances,axis=2) < Threshold
PlotImage(Seeded)

Py, Px = np.where(Seeded)
Linked = np.zeros(Seeded.shape).astype('bool')
Linked[Py,Px] = 1
Ly = np.repeat(Py,len(Map)) + np.tile(Map.T,len(Py))[0,:]
Lx = np.repeat(Px,len(Map)) + np.tile(Map.T,len(Px))[1,:]
Ly[Ly == Array.shape[0]] = 0
Lx[Lx == Array.shape[1]] = 0
Linked[Ly,Lx] = 1
PlotImage(Linked)

Array[Py,Px]


PSPs = ComputePSPs(Distances)
PlotImage(PSPs.sum(axis=3)[:,:,-1])

Figure, Axis = plt.subplots(1,1)
Axis.plot(PSPs.sum(axis=(3,0,1)),marker='o')
plt.show()

Links = PSPs[:,:,-1] == np.max(PSPs[:,:,-1])
Seeded = Links.sum(axis=2) == 8
PlotImage(Seeded)

Py, Px, Shifts = np.where(Links)
Linked = np.zeros(Seeded.shape).astype('bool')
Linked[Py,Px] = 1
PlotImage(Linked)

Groups = measure.label(Linked,connectivity=2)
Labels = np.unique(Groups)[1:]
Means = np.zeros((len(Labels),3)).astype('uint8')
Tic = time.time()
for Label in Labels:
    Group = Array[Groups == Label]
    Means[Label-1] = np.mean(Group, axis=0).round().astype('uint8')
    Array[Groups == Label] = Means[Label-1]
Toc = time.time()
PrintTime(Tic,Toc)
PlotImage(Array)


LP = np.array([Py,Px]).T+Map[Shifts]
LP[LP == Links.shape[0]] = 0

Py, Px = np.where(Seeded)
Shifts = np.repeat(Links[Py,Px],2).reshape((8,2))
LP = np.array([Py,Px]).T+Map*Shifts
Linked = np.zeros(Seeded.shape).astype('bool')
Linked[LP[:,0],LP[:,1]] = 1
PlotImage(Linked)

Merge = np.zeros(Seeded.shape)
Merge[Seeded | Linked] = 1
PlotImage(Merge)

P = np.array([82,417])
Positions = np.repeat(Links[P[0], P[1]],2).reshape((8,2)) * (P + Map)


