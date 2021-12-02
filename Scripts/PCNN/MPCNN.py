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

def NormalizeValues(Image):

    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max()-Image.min())

    return N_Image
def EuclidianDistance(x,c):
    """
    Compute Euclidian distance between vectors
    :param x: n-dimensional vector (numpy array)
    :param c: n-dimensional vector (numpy array)
    :return: d: distance between vectors
    """

    d2 = sum((x-c)**2)

    return np.sqrt(d2)

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

Tic = time.time()
print('\nImage segmentation...')

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
            Pixels_Distances[j] = EuclidianDistance(Pixel,Pixels[j])

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