"""
Script aimed to test multi channel PCNN (MPCNN)
for color image segmentation
Based on:
Zhuang, H., Low, K. S., Yau, W. Y. (2012)
Multichannel pulse-coupled-neural-network-based color image segmentation for object detection
IEEE Transactions on Industrial Electronics, 59(8), 3299–3308
https://doi.org/10.1109/TIE.2011.2165451
"""

import os
import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import measure


def NormalizeValues(Image):

    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max()-Image.min())

    return N_Image
def ManhattanDistance(x,c):
    """
    Compute Euclidian distance between vectors
    :param x: n-dimensional vector (numpy array)
    :param c: n-dimensional vector (numpy array)
    :return: d: distance between vectors
    """

    d = sum(np.abs(x-c))

    return d
def PlotArray(Array, Title, ColorBar=False):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    CBar = Axes.imshow(Array, cmap='gray')
    if ColorBar:
        plt.colorbar(CBar)
    plt.title(Title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return


desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

CurrentDirectory = os.getcwd()
DataDirectory = os.path.join(CurrentDirectory,'Scripts/PCNN/')

# Read input
Input_Image = sitk.ReadImage(DataDirectory + 'Lena.jpg')
Input_Array = sitk.GetArrayFromImage(Input_Image).astype('int')
Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input_Array)
plt.axis('off')
plt.title('Input')
plt.tight_layout()
plt.show()
plt.close(Figure)

Tic = time.time()
print('\nImage segmentation...')

# Mark seeded neurons
D_Threshold = 2
I = Input_Array.copy()
I_Padded = np.pad(I,2,mode='reflect')
I_Padded = I_Padded[:,:,2:5]
Distances = np.zeros((I.shape[0],I.shape[1],9))
for Row in range(I.shape[0]):
    for Column in range(I.shape[1]):

        Vector = I[Row,Column]
        k = 0
        for i in range(-1,2):
            for j in range(-1,2):
                Pi = Row + i + 2
                Pj = Column + j + 2
                Distances[Row,Column,k] = ManhattanDistance(Vector,I_Padded[Pi,Pj])
                k += 1
D = Distances.max(axis=2)
Seeded = (D < D_Threshold) * 1
PlotArray(Seeded, 'Seeded')

W = np.ones((3,3))
L = correlate(Seeded,W,mode='reflect',output='int')
L[L > 0] = 1
Labels = measure.label(L)
Labels_Padded = np.pad(Labels,pad_width=2,mode='constant',constant_values=0)
L[Seeded == 1] = 0
PlotArray(L, 'Linked')


# Find unlinked neighbours
U = correlate(L,W,mode='reflect',output='int')
U[U > 0] = 1
U[(Seeded == 1) | (L == 1)] = 0
PlotArray(U,'U')

Sigmas = np.zeros((len(np.unique(Labels)),3)).astype('int')
FiredNeurons = 0
Threshold_delta = 1
while np.sum(L) > FiredNeurons:

    FiredNeurons = np.sum(L)

    for i in range(1,len(np.unique(Labels))):
        Label = np.unique(Labels)[i]

        # Compute mean vector value
        Sigmas[i] = np.round(I[Labels == Label].mean(axis=0)).astype('int')
        I[Labels == Label] = Sigmas[i]
        I_Padded = np.pad(I, 2, mode='reflect')
        I_Padded = I_Padded[:, :, 2:5]

        # Find unlinked neighbours
        U = correlate((Labels == Label)*1,W,mode='reflect',output='int')
        U[U > 0] = 1
        U[(Seeded == 1) | (L == 1)] = 0

        # Go through every unlinked pixel
        Rows, Columns = np.where(U == 1)
        for PixelNumber in range(len(Rows)):
            I_Row, I_Column = Rows[PixelNumber], Columns[PixelNumber]
            Pixel = I[I_Row,I_Column]

            # Extract labels of all pixel neighbours
            Neighbours_L = Labels_Padded[2+I_Row-1:2+I_Row+2,2+I_Column-1:2+I_Column+2]

            # Filter neighbours to keep only linked neighbours
            NL_Rows, NL_Columns = np.where(Neighbours_L > 0)
            LinkedNeighbours_R = NL_Rows + I_Row - 1
            LinkedNeighbours_C = NL_Columns + I_Column - 1
            LinkedNeighbours = I[LinkedNeighbours_R, LinkedNeighbours_C]

            Pixel_Distances = np.zeros(len(LinkedNeighbours))
            N_Labels = np.zeros(len(LinkedNeighbours))
            for j in range(len(Pixel_Distances)):
                N_Labels[j] = Neighbours_L[NL_Rows[j],NL_Columns[j]]
                Pixel_Distances[j] = ManhattanDistance(Pixel,LinkedNeighbours[j])

            MinDistanceIndex = np.argsort(Pixel_Distances)[0]
            if Pixel_Distances[MinDistanceIndex] < D_Threshold:
                Labels[I_Row,I_Column] = N_Labels[MinDistanceIndex]
                Labels_Padded[I_Row+2,I_Column+2] = N_Labels[MinDistanceIndex]
                L[I_Row,I_Column] = 1

    D_Threshold += Threshold_delta
PlotArray(Labels,'Linked')


# Initialize parameters
S = NormalizeValues(NormArray)
Y = np.zeros(S.shape)
Labels = np.zeros(S.shape)
T = np.zeros(S.shape)
W = np.array([[0.5, 1, 0.5],
              [1, 0, 1],
              [0.5, 1, 0.5]])
Theta = np.ones(S.shape)

FiredNumber = 0
N = 0

Delta = 1/255
VT = 10
Beta = 2

# Perform segmentation
while FiredNumber < S.size:

    N += 1
    F = S
    L = correlate(Y, W, output='float', mode='reflect')
    Theta = Theta - Delta + VT * Y
    U = F * (1 + Beta * L)

    Y = (U > Theta) * 1

    T = T + N * Y
    FiredNumber = FiredNumber + sum(sum(Y))

Output = 1 - NormalizeValues(T)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Output, cmap='gray')
plt.axis('off')
plt.title('Output')
plt.tight_layout()
plt.show()
plt.close(Figure)


# Print time elapsed
Toc = time.time()
PrintTime(Tic, Toc)

















################################## Tests

# Norm array
R, G, B = Input_Array[:,:,0], Input_Array[:,:,1], Input_Array[:,:,2]
NormArray = np.sqrt(R**2 + G**2 + B**2)
Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(NormArray, cmap='gray')
plt.axis('off')
plt.title('Norm values')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Distance array
Input_Array_Padded = np.pad(Input_Array,2,mode='reflect')
Input_Array_Padded = Input_Array_Padded[:,:,2:5]

Distances = np.zeros((NormArray.shape[0],NormArray.shape[1],9))
for Row in range(Input_Array.shape[0]):
    for Column in range(Input_Array.shape[1]):

        Vector = Input_Array[Row,Column]
        k = 0
        for i in range(-1,2):
            for j in range(-1,2):
                Pi = Row + i + 2
                Pj = Column + j + 2
                Distances[Row,Column,k] = EuclidianDistance(Vector,Input_Array_Padded[Pi,Pj])
                k += 1
D = NormalizeValues(Distances.mean(axis=2))

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(D, cmap='gray')
plt.axis('off')
plt.title('Distances values')
plt.tight_layout()
plt.show()
plt.close(Figure)



# Initialize parameters
S = NormalizeValues(NormArray)
Y = np.zeros(S.shape)
Labels = np.zeros(S.shape)
T = np.zeros(S.shape)
W = np.array([[0.5, 1, 0.5],
              [1, 0, 1],
              [0.5, 1, 0.5]])
Theta = np.ones(S.shape)

FiredNumber = 0
N = 0

Delta = 1/255
VT = 10
Beta = 2

FL_Threshold = 10
RG_Threshold = 10

Labels_Number = 1
# Perform segmentation
while FiredNumber < S.size:

    N += 1
    F = S
    L = correlate(Y, W, output='float', mode='reflect')
    Theta = Theta - Delta + VT * Y
    U = F * (1 + Beta * L)

    OldFired = sum(sum(Y))
    Y = (U > Theta) * 1
    NewFired = sum(sum(Y))

    Rows, Columns = np.where(Y == 1)
    Pixels_list = list(Input_Array[Rows,Columns])
    Pixels_Labels = Labels[Rows,Columns]

    for Pixel in Pixels_list.pop(0):

        if OldFired == 0 and NewFired > OldFired:
            Label = Labels_Number

        Pixels_Distances = np.zeros(len(Pixels_Labels))
        for j in range(len(Pixels_Labels)):
            Pixels_Distances[j] = EuclidianDistance(Pixel,Pixels_list[j])

        Pixels2Label = Pixels_Labels[Pixels_Distances < RG_Threshold]
        Pixels2Label[Pixels2Label == 0] = Label

        ### Continue to do labelling correctly

    # Fast linking
    while OldFired != NewFired:

        OldFired = sum(sum(Y))

        # Find fired neurons and distances with their neighbours
        Rows, Columns = np.where(Y == 1)
        Neighbours = Distances[Rows, Columns]

        # Keep neurons with small enough spectral distance
        FastLinked = (Neighbours < FL_Threshold) * 1

        # Compute indices of these neurons with small spectral distances
        Neighbours = np.array([-1,0,1])
        RowsNeighbours = np.tile(np.tile(Neighbours,3),len(Rows))
        ColumnsNeighbours = np.tile(np.repeat(Neighbours,3),len(Columns))
        RowsN = np.reshape(np.repeat(Rows,9) + RowsNeighbours,(len(Rows),9))
        ColumnsN = np.reshape(np.repeat(Columns,9) + ColumnsNeighbours,(len(Columns),9))

        # Fire neurons with small spectral distance from fired neuron
        RowsLinked = RowsN[FastLinked == 1]
        ColumnsLinked = ColumnsN[FastLinked == 1]

        # Remove index out of bounds
        RowsBool = ((RowsLinked >= 0) & (RowsLinked < S.shape[0]))
        ColumnsBool = ((ColumnsLinked >= 0) & (ColumnsLinked < S.shape[1]))
        BoolIndices = RowsBool & ColumnsBool
        Y[RowsLinked[BoolIndices],ColumnsLinked[BoolIndices]] = 1
        NewFired = sum(sum(Y))

    T = T + N * Y
    FiredNumber = FiredNumber + sum(sum(Y))

Output = 1 - NormalizeValues(T)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Output, cmap='gray')
plt.axis('off')
plt.title('Output')
plt.tight_layout()
plt.show()
plt.close(Figure)


# Print time elapsed
Toc = time.time()
PrintTime(Tic, Toc)