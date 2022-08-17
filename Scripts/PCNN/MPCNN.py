"""
Script aimed to test multi channel PCNN (MPCNN)
for color image segmentation
Based on:
Zhuang, H., Low, K. S., Yau, W. Y. (2012)
Multichannel pulse-coupled-neural-network-based color image segmentation for object detection
IEEE Transactions on Industrial Electronics, 59(8), 3299–3308
https://doi.org/10.1109/TIE.2011.2165451
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.future import graph
from skimage.feature import canny
from scipy.ndimage import correlate
from skimage import io, color,  filters, measure, morphology

def PlotImage(Array, Cmap='binary_r'):

    Figure, Axis = plt.subplots(1,1,figsize=(10,10))
    if Array.shape[-1] == 3:
        Axis.imshow(Array)
    else:
        Axis.imshow(Array, cmap=Cmap)
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

    if len(Array2D.shape) > 2:
        YSize, XSize = Array2D.shape[:-1]
        Dimension = Array2D.shape[-1]
        Neighbourhood = np.zeros((YSize, XSize, 8, Dimension))
    else:
        YSize, XSize = Array2D.shape
        Neighbourhood = np.zeros((YSize, XSize, 8))

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

def FastLinking(Seeded, Map, NewLinksOnly=False):

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
    Links[:,0][Links[:,0] == -1] = 0
    Links[:,1][Links[:,1] == -1] = 0

    # Set out-of-border neighbours to array end index
    Links[:, 0][Links[:, 0] == Seeded.shape[0]] = Seeded.shape[0] - 1
    Links[:, 1][Links[:, 1] == Seeded.shape[1]] = Seeded.shape[1] - 1

    # Label linked pixels
    Linked[Links[:,0], Links[:,1]] = 1

    # Unlabel seeded pixels
    if NewLinksOnly:
        Linked = Linked - Seeded * 1
        Linked[Linked < 0] = 0

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Linked

def RegionGrowing(Linked, Groups, Times, Edges):

    # Record elapsed time
    Tic = time.time()
    print('\nPerform region growing ...')

    DistanceThreshold = 0
    DeltaDistance = 1
    i = 0
    Sum = 0
    while Linked.sum() < Linked.size:

        # Unlinked neuron having a linked neighbour
        UnLinked = FastLinking(Linked, Map, NewLinksOnly=True)

        # # Remove edge pixels
        # if Linked.size - Edges.sum() > Linked.sum() > Sum:
        #     UnLinked = UnLinked - Edges*1
        #     UnLinked[UnLinked < 0] = 0

        # Unlinked neurons values
        UnlinkTimes = Times[UnLinked.astype('bool')]

        # Extract linked groups
        NeighbourGroups = GetNeighbours(Groups)[0]
        LinkedGroups = NeighbourGroups[UnLinked.astype('bool')]
        LinkedGroups[LinkedGroups == 0] = np.nan

        # Find labelled group with minimal difference
        ClosestGroup = np.nanargmin(UnlinkTimes * LinkedGroups, axis=1)

        # Check that distance between the closest group and unlinked pixel is not too high
        ClosestDistance = UnlinkTimes[np.arange(len(ClosestGroup)), ClosestGroup]
        DistanceThreshold += DeltaDistance
        CloseEnough = np.where(ClosestDistance < DistanceThreshold)[0]

        # Attribute label of the closest group to the linked neuron
        ClosestGroups = LinkedGroups[np.arange(len(ClosestGroup)), ClosestGroup]
        Ly, Lx = np.where(UnLinked)
        Groups[Ly[CloseEnough], Lx[CloseEnough]] = ClosestGroups[CloseEnough]

        # Set those neurons as linked
        Sum = Linked.sum()
        Linked[Ly[CloseEnough], Lx[CloseEnough]] = 1

        # Print elapsed time
        Toc = time.time()
        print('\nIteration number: ' + str(i))
        PrintTime(Tic, Toc)

        # Update iteration number
        i += 1

    return Groups

def MeansGroupsValues(Groups, Array):

    """
    Function to compute mean values of labelled groups in a 2D image
    :param Groups: 2D numpy array of labels
    :param Array: 2D numpy array of the image
    :return: Mean values for each group and the resulting segmented image
    """

    # Record elapsed time
    Tic = time.time()
    print('\nCompute mean value of each group ...')

    Segmented = Array.copy()
    Labels = np.unique(Groups)
    Means = np.zeros((len(Labels), 3)).astype('uint8')

    for Label in Labels:
        Group = Array[Groups == Label]
        Means[Label-1] = np.mean(Group, axis=0).round().astype('uint8')
        Segmented[Groups == Label] = Means[Label-1]

    Segmented[Groups == 0] = [0,0,0]

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Means, Segmented

def NormalizeValues(Image):
    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max() - Image.min())

    return N_Image

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

def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))

def Enhancement(Image, d=2, h=2E10, g=0.9811, Alpha=0.01, Beta=0.03):

    """
    Perform image enhancement using Feature Linking Model (FLM)
    Based on:
    Zhan, K., Shi, J., Wang, H. et al.
    Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
    Arch Computat Methods Eng 24, 573–588 (2017).
    https://doi.org/10.1007/s11831-016-9182-3

    :param: d: Inhibition factor for linkin wave
    :param: h: Amplitude factor for threshold excitation
    :param: g: Decay factor for threshold
    :param: Alpha: Decay factor for linkin input
    :param: Beta: Decay factor for external input
    :return: Y: Restored Image
    """

    Tic = time.time()
    print('\nPerform image enhancement...')

    # Initialization
    S = NormalizeValues(Image) + 1/255
    W = np.array([[0.7, 1, 0.7], [1, 0, 1], [0.7, 1, 0.7]])
    Y = np.zeros(S.shape)
    U = np.zeros(S.shape)
    T = np.zeros(S.shape)
    SumY = 0
    N = 0

    Laplacian = np.array([[1 / 6, 2 / 3, 1 / 6], [2 / 3, -10 / 3, 2 / 3], [1 / 6, 2 / 3, 1 / 6]])
    Theta = 1 + correlate(S, Laplacian, mode='reflect')
    f = 0.75 * np.exp(-S ** 2 / 0.16) + 0.05
    G = GaussianKernel(3, 1)
    f = correlate(f, G, mode='reflect')

    # Analysis
    while SumY < S.size:
        N += 1

        K = correlate(Y, W, mode='reflect')
        Wave = Alpha * K + Beta * S * (K - d)
        U = f * U + S + Wave
        Theta = g * Theta + h * Y
        Y = (U > Theta) * 1
        T += N * Y
        SumY += sum(sum(Y))

    T_inv = T.max() + 1 - T

    # Print time elapsed
    Toc = time.time()
    PrintTime(Tic, Toc)

    return T_inv


Array = io.imread('TestROI.png')
PlotImage(Array)

Test = Array.copy()
F1 = Test[:,:,0] < 120
F2 = Test[:,:,1] < 120
F3 = Test[:,:,2] < 180
F4 = Test[:,:,2] > 160
Edges = 1-F1*F2*F3*F4
PlotImage(Edges)

Filtered = filters.gaussian(Array,sigma=2,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

Filtered = Filtered * np.repeat(1-Edges,3).reshape(Filtered.shape)


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
HSV = color.rgb2hsv(Array)
I = np.round(HSV[:,:,2]/HSV[:,:,2].max()*255).astype('uint8')
PlotImage(I)
Enhanced = enhance_contrast(I)/255
PlotImage(Enhanced)
Enhanced = color.hsv2rgb(np.dstack([HSV[:,:,0], HSV[:,:,1],Enhanced]))
PlotImage(Enhanced)

Filtered = filters.gaussian(Enhanced,sigma=2,multichannel=True)
Filtered = np.round(Filtered / Filtered.max() * 255).astype('int')
PlotImage(Filtered)

Filtered = Filtered * np.repeat(1-Edges,3).reshape(Filtered.shape)


# Canny edge detection
Gray = color.rgb2gray(Array)
PlotImage(Gray)
Otsu = filters.threshold_otsu(Gray)
Edges = canny(Gray, sigma=1, high_threshold=Otsu, low_threshold=0)
LargeEdges = morphology.binary_dilation(Edges,morphology.disk(2))
PlotImage(Edges)

# PCNN edges
PCNN_Edges = SPCNN_Edges(1-Gray,Beta=2,Delta=1/5,VT=100)
PlotImage(PCNN_Edges,Cmap='viridis')
Edges = np.zeros(PCNN_Edges.shape)
Edges[PCNN_Edges == np.unique(PCNN_Edges)[0]] = 1
PlotImage(Edges)

# Compute distances in individual RGB dimensions
Distances, Map = ComputeDistances(Array)
PlotImage(np.linalg.norm(Distances,axis=3).max(axis=2))

MinDistances = np.linalg.norm(Distances,axis=3).min(axis=2)
Figure, Axis = plt.subplots(1,1)
Axis.plot(np.unique(MinDistances))
plt.show()

# Compute neurons firing time to assess rank order
Times = FiringTimes(Distances)
Figure, Axis = plt.subplots(1,1)
Axis.plot(np.unique(Times))
plt.show()

# Define seeded pixels and neurons using Manhattan distance
RGBThreshold = Filtered.max() / 52
Manhattan = Distances.sum(axis=3)

Seeded = np.max(Manhattan,axis=2) < Manhattan.max() / 52
PlotImage(Seeded)

# Perform fast linking
Linked = FastLinking(Seeded, Map)

# Remove edge pixels
Linked = Linked - Edges*1
Linked[Linked < 0] = 0
PlotImage(Linked)

# Label different groups and perform region growing
Groups = measure.label(Linked,connectivity=2)
Groups = RegionGrowing(Linked, Groups, Times, Edges)

Means, Segmented = MeansGroupsValues(Groups, Array)
PlotImage(Segmented)

SegDistances = np.linalg.norm(ComputeDistances(Segmented)[0],axis=3)
PlotImage(np.max(SegDistances,axis=2))


# Perform region merging using RAG threholding
Size = 100
Props = ['label','area',]
Regions = pd.DataFrame(measure.regionprops_table(Groups, properties=Props))
Labels = Regions[Regions['area'] < Size]

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}
def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])
RAG = graph.rag_mean_color(Filtered,Groups)
Merged = graph.merge_hierarchical(Groups,RAG,15,True,False,merge_mean_color,_weight_mean_color)
Means, MergedImage = MeansGroupsValues(Merged, Filtered)
PlotImage(MergedImage)