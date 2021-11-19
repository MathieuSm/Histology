"""
Code for testing PCNN methods used for face detection
Based on:
Lim Young-Wan and Na Jin-Hee and Choi Jin-Young (2004)
Role of linking parameters in Pulse-Coupled Neural Network for face detection
Control Robot System Society, 1048-1052
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import exposure, morphology, filters, feature, segmentation, measure
import matplotlib as mpl

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

def SaveImage(Image,Name):

    from skimage import io
    io.imsave(Name, Image.astype(np.uint8))

    return
def NormalizeArray(Array):

    Normalized_Array = (Array - Array.min()) / (Array.max() - Array.min()) * 255

    return np.round(Normalized_Array).astype('uint8')
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
def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))
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
def PCNN_Segmentation(Image,ParametersDictionary,MaxIteration=10):

    # Initialization
    AlphaF = ParametersDictionary['AlphaF']
    AlphaL = ParametersDictionary['AlphaL']
    AlphaT = ParametersDictionary['AlphaT']

    VF = ParametersDictionary['VF']
    VL = ParametersDictionary['VL']
    VT = ParametersDictionary['VT']

    Beta = ParametersDictionary['Beta']

    # Input parameters
    S = Image / Image.max()

    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    Vb, New_Vb = 0, 0
    # W = GaussianKernel(1, 1)
    W = np.array([[0.5,1,0.5],[1,0,1],[0.5,1,0.5]])

    ## Feeding input
    F = S
    WF = W

    ## Linking input
    L = Y
    WL = W

    # Dynamic threshold
    Theta = np.ones((Rows, Columns))

    N = 0
    while New_Vb >= Vb and N < MaxIteration:

        N += 1
        F = S + F * np.exp(-AlphaF) + VF * correlate(Y, WF, output='float', mode='reflect')
        L = L * np.exp(-AlphaL) + VL * correlate(Y, WL, output='float', mode='reflect')
        U = F * (1 + Beta*L)

        Theta = Theta * np.exp(-AlphaT) + VT * Y
        Y = (U > Theta) * 1

        # Update variance
        Vb = New_Vb
        New_Vb = BetweenClassVariance(S, Y)

        if New_Vb >= Vb:
            Best_Y = Y

    return Best_Y
def Mutual_Information(Image1, Image2):

    """ Mutual information for joint histogram """

    Hist2D = np.histogram2d(Image1.ravel(), Image2.ravel(), bins=20)[0]

    # Convert bins counts to probability values
    pxy = Hist2D / float(np.sum(Hist2D))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
def DiceCoefficient(Image1,Image2):

    Dice = 2 * np.sum(Image1 * Image2) / np.sum(Image1 + Image2)

    return Dice
def PlotROI(ImageArray, RegionLabel, Labels):

    C = np.array([[0, 0, 0, 0], [1, 0, 0, 0.5]])
    ColorMap = mpl.colors.ListedColormap(C)
    Region = (Labels == RegionLabel)

    E = measure.EllipseModel()
    X, Y = np.where(Labels == RegionLabel)
    XY = np.concatenate(np.array([X, Y]).T, axis=0).reshape((len(X), 2))
    E.estimate(XY)
    Y0, X0, R1, R2, OrientationAngle = E.params

    Radians = np.linspace(0, 2 * np.pi, 100)
    Ellipse = np.array([R2 * np.cos(Radians), R1 * np.sin(Radians)])
    R = np.array([[np.cos(OrientationAngle), -np.sin(OrientationAngle)],
                  [np.sin(OrientationAngle), np.cos(OrientationAngle)]])
    Ellipse_R = np.dot(R, Ellipse)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(ImageArray, cmap='gray')
    Axes.imshow(Region, cmap=ColorMap, vmin=0.25, vmax=0.75)
    Axes.plot([], color=(1,0,0,0.5), linestyle='none', marker='s', label='Region')
    Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='centroid')
    Axes.plot(X0 + Ellipse_R[0, :], Y0 - Ellipse_R[1, :], color=(0, 1, 0), label='Fitted ellipse')
    plt.title('Region ' + str(RegionLabel))
    plt.axis('off')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    plt.show()
    plt.close(Figure)

    return



# Set path
CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Scripts/PCNN/'

# Open manually segmented image
SegmentedImage = sitk.ReadImage(ImageDirectory + 'PCNN_Test2_Seg.png')
SegmentedArray = sitk.GetArrayFromImage(SegmentedImage)
# SegmentedArray = SegmentedArray[3839:3839+1182,4724:4724+1182]

CementLines = SegmentedArray[:,:,0].copy()
Harvesian = SegmentedArray[:, :, 1].copy()
Osteocytes = SegmentedArray[:,:,2].copy()
Threshold = 200
CementLines[CementLines < Threshold] = 0
CementLines[CementLines >= Threshold] = 1
Harvesian[Harvesian < Threshold] = 0
Harvesian[Harvesian >= Threshold] = 1
Osteocytes[Osteocytes < Threshold] = 0
Osteocytes[Osteocytes >= Threshold] = 1

F0 = SegmentedArray[:,:,0] > 180
F1 = SegmentedArray[:,:,1] > 180
F2 = SegmentedArray[:,:,2] > 180
CementLines[F1] = 0
CementLines[F2] = 0
Harvesian[F0] = 0
Harvesian[F2] = 0
Osteocytes[F0] = 0
Osteocytes[F1] = 0

Segments = CementLines + Harvesian + Osteocytes
Disk = morphology.disk(2)
Segments = morphology.binary_erosion(Harvesian,Disk)
Disk = morphology.disk(5)
Segments = morphology.binary_dilation(CementLines,Disk)
PlotArray(Segments, 'Manual Segmentation')

# Open image to segment
Image = sitk.ReadImage(ImageDirectory + 'PCNN_Test2.png')
Array = sitk.GetArrayFromImage(Image)
PlotArray(Array, 'RGB Image')

# Decompose RGB image and equalize histogram
R, G, B = Array[:,:,0], Array[:,:,1], Array[:,:,2]

Figure, Axes = plt.subplots(1, 3)
Axes[0].imshow(R, cmap='gray')
Axes[0].set_title('R channel')
Axes[0].axis('off')
Axes[1].imshow(G, cmap='gray')
Axes[1].set_title('G channel')
Axes[1].axis('off')
Axes[2].imshow(B, cmap='gray')
Axes[2].set_title('B channel')
Axes[2].axis('off')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Match histograms for better phase difference and clarity and rescale
GS = exposure.match_histograms(R, B)
GS_Rescaled = NormalizeArray(GS)
PlotArray(GS_Rescaled, 'Grayscale Image')


Gamma, Gain = 2, 1
GS_Contrast = exposure.adjust_gamma(GS_Rescaled,Gamma,Gain)
PlotArray(GS_Contrast, 'Gamma:' + str(Gamma) + ' Gain:' + str(Gain))

Sigma = 5
GS_Gauss = filters.gaussian(GS_Rescaled,sigma=Sigma)
GS_Gauss = NormalizeArray(GS_Gauss)
PlotArray(GS_Gauss, 'Gauss sigma:' + str(Sigma))


# Particle Swarm Optimization (PSO) algorithm
Ps = 20         # Population size
t = 0           # Iteration number
Max_times = 10  # Max iteration number
Omega = 0.9 - 0.5 * t/Max_times     # Inertia factor
Average_FV_std = 1E-3   # Second PSO termination condition
Image = 1-R / R.max()

# PSO step 1 - Initialization
AlphaF_Range = np.array([-1,1])*3
AlphaL_Range = np.array([-1,1])*3
AlphaT_Range = np.array([-1,1])*3

VF_Range = np.array([-1,1])*3
VL_Range = np.array([-1,1])*3
VT_Range = np.array([-1,1])*3

Beta_Range = np.array([-1,1])*3

Ranges = np.array([AlphaF_Range,AlphaL_Range,AlphaT_Range,
                   VF_Range,VL_Range,VT_Range,Beta_Range])
Dimensions = len(Ranges)

C1, C2 = 2, 2

RangeAmplitudes = Ranges[:,1] - Ranges[:,0]
X = np.random.uniform(0,1,(Ps,Dimensions)) * RangeAmplitudes + Ranges[:,0]
V = C1 * np.random.uniform(-1,1,(Ps,Dimensions)) * RangeAmplitudes \
  + C2 * np.random.uniform(-1,1) * RangeAmplitudes



# PSO step 2 - Initial evaluation
ParameterList = ['AlphaF','AlphaL','AlphaT','VF','VL','VT','Beta']
Initial_DSCs = np.zeros((Ps, 1))
for ParticleNumber in range(Ps):
    ParametersDictionary = {}
    for ParameterNumber in range(Dimensions):
        ParameterName = ParameterList[ParameterNumber]
        ParameterValue = X[ParticleNumber,ParameterNumber]
        ParametersDictionary[ParameterName] = ParameterValue

    Y = PCNN_Segmentation(Image,ParametersDictionary)

    # Compute dice similarity coefficient with manual segmentation
    Initial_DSCs[ParticleNumber, 0] = DiceCoefficient(Y, Segments)


# Set initial best values
G_Best_Value = Initial_DSCs.max()
G_Best_Index = np.where(Initial_DSCs == G_Best_Value)[0][0]
G_Best = X[G_Best_Index]

P_Best_Values = Initial_DSCs.copy()
P_Best = X.copy()


## Start loop
Average_FVs = np.array([1,0,0])
NIteration = 0
while NIteration < 100 and Average_FVs.std() >= Average_FV_std:

    ## PSO step 3 - Update positions and velocities
    R1, R2 = np.random.uniform(0, 1, 2)
    V = Omega * V + C1 * R1 * (P_Best - X) + C2 * R2 * (G_Best - X)
    X = X + V
    # If new position exceed limits, set to limit
    X[X < Ranges[:,0]] = np.tile(Ranges[:,0],Ps).reshape((Ps,Dimensions))[X < Ranges[:,0]]
    X[X > Ranges[:,1]] = np.tile(Ranges[:,1],Ps).reshape((Ps,Dimensions))[X > Ranges[:,1]]



    ## PSO step 4 - Evaluation of the updated population
    New_DSCs = np.zeros((Ps, 1))
    for ParticleNumber in range(Ps):
        ParametersDictionary = {}
        for ParameterNumber in range(Dimensions):
            ParameterName = ParameterList[ParameterNumber]
            ParameterValue = X[ParticleNumber,ParameterNumber]
            ParametersDictionary[ParameterName] = ParameterValue

        Y = PCNN_Segmentation(Image,ParametersDictionary)

        # Compute dice similarity coefficient with manual segmentation
        New_DSCs[ParticleNumber, 0] = DiceCoefficient(Y, Segments)

    # Update best values if better than previous
    if New_DSCs.max() > G_Best_Value:
        G_Best_Value = New_DSCs.max()
        G_Best_Index = np.where(New_DSCs == G_Best_Value)[0][0]
        G_Best = X[G_Best_Index]

    ImprovedValues = New_DSCs > P_Best_Values
    P_Best_Values[ImprovedValues] = New_DSCs[ImprovedValues]
    Reshaped_IP = np.tile(ImprovedValues,Dimensions).reshape((Ps,Dimensions))
    P_Best[Reshaped_IP] = X[Reshaped_IP]



    ## PSO step 5 - Update and check if terminal condition is satisfied
    NIteration += 1
    Average_FVs = np.concatenate([Average_FVs[1:],np.array(G_Best_Value).reshape(1)])


## PSO step 6 - Output results
ParametersDictionary = {}
for ParameterNumber in range(Dimensions):
    ParameterName = ParameterList[ParameterNumber]
    ParameterValue = G_Best[ParameterNumber]
    ParametersDictionary[ParameterName] = ParameterValue
Y = PCNN_Segmentation(Image,ParametersDictionary)
PlotArray(Y, 'PCNN Segmentation')