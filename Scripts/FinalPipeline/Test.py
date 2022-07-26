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

    Distances = np.zeros((Array2D.shape[0], Array2D.shape[1], 8, 3))
    for i in range(8):
        Distances[:, :, i] = np.abs(Array2D - Neighbours[:, :, i])

    Toc = time.time()
    PrintTime(Tic, Toc)

    return Distances, Map

def PSP_j(t, Distances):

    PSP = np.zeros(Distances.shape)
    PSP[t >= Distances] = t - Distances[t >= Distances]
    return PSP.sum(axis=1)

def FiringTimes(Distances,Threshold=1):

    """
    Function used to compute firing times of individual neighbour neurons
    :param Distances: MxNxO-dimensional numpy array containing inter-pixel distances
    :param Threshold: Constant threshold reached by all synapses
    :return: Times: Individual neighbour neurons firing times
    """

    # Record elapsed time
    Tic = time.time()
    print('\nCompute firing times ...')

    # Compute time necessary for all synapses to fire
    Ni = Distances.shape[-1]
    Times = Threshold/Ni + Distances.sum(axis=3)/Ni

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic,Toc)

    return Times

def FastLinking(Seeded, Map):

    """
    Perform fast linking of direct neighbours using np.where (faster than morphology.binary_dilation from skimage)
    :param Seeded: 2D numpy array containing seeded pixels, background is labelled 0
    :param Map: Mapping of the neighbours to link
    :return: Linked: 2D numpy array with all linked pixels labelled as 1
    """

    # Record elapsed time
    Tic = time.time()
    print('\nPerform fast linking ...')

    # Create linked array
    Linked = np.zeros(Seeded.shape)

    # Extract seeded pixel positions and find corresponding neighbours
    Py, Px = np.where(Seeded)
    Positions = np.repeat(np.vstack([Py,Px]).T,len(Map),axis=0)
    Links = Positions + np.tile(Map.T, len(Py)).T

    # Set out-of-border neighbours to array start index
    Links[:,0][Links[:,0] == Seeded.shape[0]] = 0
    Links[:,1][Links[:,1] == Seeded.shape[1]] = 0

    # Label linked pixels
    Linked[Links[:,0], Links[:,1]] = 1

    # Reshape coordinate vectors
    Positions = np.reshape(Positions,(len(Py),8,2))
    Links = np.reshape(Links,(len(Py),8,2))

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Positions, Links, Linked

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

Array = np.round(50 * np.random.randn(5,5,3) + 128).astype('uint8')
Array[1:4,1:4] = [200,200,200]
Array[0,4] = [200, 200, 200]
Array = np.zeros((5,5))
Array[2,2] = 1

Array = io.imread('TestROI.png')
PlotImage(Array)

Filtered = filters.gaussian(Array,sigma=3,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

# Compute distances in individual RGB dimensions
Distances, Map = ComputeDistances(Filtered)

# Compute neurons firing time to assess rank order
Times = FiringTimes(Distances)

# Define seeded pixels and neurons using Manhattan distance
Threshold = 1 / 52
Manhattan = Distances.sum(axis=3)
Seeded = np.max(Manhattan,axis=2) < Threshold
PlotImage(Seeded)

# Perform fast linking method 1
Positions, Links, Linked = FastLinking(Seeded, Map)
PlotImage(Linked)

# Compute mean group value
Groups = measure.label(Linked,connectivity=2)
Labels = np.unique(Groups)[1:]
Means = np.zeros((len(Labels),3)).astype('uint8')
Tic = time.time()
Segmented = Filtered.copy()
for Label in Labels:
    Group = Filtered[Groups == Label]
    Means[Label-1] = np.mean(Group, axis=0).round().astype('uint8')
    Segmented[Groups == Label] = Means[Label-1]
Toc = time.time()
PrintTime(Tic,Toc)
PlotImage(Segmented)

# Unlinked neuron having a linked neighbour
UnPositions, UnLinks, UnLinked = FastLinking(Linked, Map)
UnLinked = UnLinked - Linked*1
PlotImage(UnLinked)

# Unlinked neurons values
NeuronsValues = Times[UnLinked.astype('bool')]
NeuronsDistances, Map = ComputeDistances(Segmented)
NeuronsDistances[UnLinked.astype('bool')][0]

# Extract linked groups
LinkedGroups = Groups[UnLinks[0,:,0],UnLinks[0,:,1]]
# Find labelled group with minimal difference
ClosestGroup = np.argmin(NeuronsValues[0][LinkedGroups.astype('bool')])
# Attribute label of the closest group to the linked neuron
Groups[UnLinks[0,0,0],UnLinks[0,0,1]] = LinkedGroups[LinkedGroups.astype('bool')][ClosestGroup]




# Extract latest output pulse timing
SeededNeurons = np.repeat(Seeded,8).reshape((Times.shape))
MaxTime = np.max(Times * SeededNeurons)
Linked = np.sum(Times <= MaxTime,axis=2).astype('bool')
PlotImage(Linked)



Array[Py,Px]




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


