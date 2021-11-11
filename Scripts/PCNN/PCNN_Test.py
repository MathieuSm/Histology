"""
Code for testing PCNN-PSO-AT with different inputs or fitness function
"""""

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
SegmentedImage = sitk.ReadImage(ImageDirectory + 'PCNN_Test_Seg.png')
SegmentedArray = sitk.GetArrayFromImage(SegmentedImage)

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
PlotArray(Segments, 'Manual Segmentation')

# Open image to segment
Image = sitk.ReadImage(ImageDirectory + 'PCNN_Test.jpg')
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
GS_Rescaled = sitk.ReadImage(ImageDirectory + 'PCNN_Test.png')
GS_Rescaled = sitk.GetArrayFromImage(GS_Rescaled)


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
Average_FV = 0.99   # Second PSO termination condition
Image = R[4:,3:] / R[4:,3:].max()

# PSO step 1 - Initialization
AlphaF_Range = [-1,1]
AlphaL_Range = [-1,1]
AlphaT_Range = [-1,1]

VF_Range = [-1,1]
VL_Range = [-1,1]
VT_Range = [-1,1]

Beta_Range = [-1,1]

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
Average_FVs = np.zeros(3)
NIteration = 0
while NIteration < 100 and Average_FVs.mean() <= Average_FV:

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

ParametersDictionary = {'AlphaF': -0.2381907530225938,
                        'AlphaL': -0.2398532814016563,
                        'AlphaT': 0.7220370952618878,
                        'VF': -0.30581795422704827,
                        'VL': 1.0,
                        'VT': 0.5418764530883855,
                        'Beta': -1.0}

Disk = morphology.disk(2)
BW_Dilate = morphology.binary_dilation(Y,Disk)
PlotArray(BW_Dilate, 'Dilated segmentation')

Labels = measure.label(BW_Dilate,connectivity=2)
Properties = ('label', 'area', 'orientation', 'euler_number')
PropertiesTable = pd.DataFrame(measure.regionprops_table(Labels,properties=Properties))

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(PropertiesTable['area'], marker='o',linestyle='none',color=(1,0,0),fillstyle='none')
plt.show()
plt.close(Figure)

PropertiesTable.sort_values('area')
RegionsLabels = [3, 20, 31]

for Region in Regions:
    PlotROI(Y, Region, Labels)










# Compare with initial segments
C1 = np.array([[0, 0, 0, 0], [1, 0, 0, 1]])
ColorMap1 = mpl.colors.ListedColormap(C1)
C2 = np.array([[0, 0, 0, 0], [0, 1, 0, 1]])
ColorMap2 = mpl.colors.ListedColormap(C2)
C3 = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
ColorMap3 = mpl.colors.ListedColormap(C3)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
# Axes.imshow(ImageArray,cmap='gray')
Axes.imshow(CementLines, cmap=ColorMap1, vmin=0.2, vmax=0.8)
Axes.imshow(Harvesian, cmap=ColorMap2, vmin=0.2, vmax=0.8)
Axes.imshow(Osteocytes, cmap=ColorMap3, vmin=0.2, vmax=0.8)
Axes.axis('off')
Axes.set_title('Segments')
plt.tight_layout()
plt.show()
plt.close(Figure)












GS_Gradient = filters.rank.gradient(GS_Rescaled, morphology.disk(2))
PlotArray(GS_Gradient, 'Grayscale Gradient')

Threshold = filters.threshold_otsu(GS_Gradient)
GS_Gradient_Seg = (GS_Gradient > Threshold) * 1
PlotArray(GS_Gradient_Seg, 'Otsu segmentation')

Combine = (Y + GS_Gradient_Seg) / 2
Combine[Combine < 1] = 0
PlotArray(Combine, 'Combine segmentation')




GS_Enhanced = exposure.match_histograms(GS, Y)
GS_Enhanced = NormalizeArray(GS_Enhanced)
PlotArray(GS_Enhanced, 'Grayscale Image')


Threshold = 250
BW = T.copy()
BW = (BW >= Threshold) * 1
PlotArray(BW, 'Threshold Segmentation')

MedialAxis, Distances = morphology.medial_axis(1-BW, return_distance=True)
PlotArray(Distances, 'Distances', ColorBar=True)

Combine = (256-T)*Distances/Distances.max()
Combine = NormalizeArray(Combine)
PlotArray(Combine, 'T x Distances', ColorBar=True)

Threshold = filters.threshold_otsu(Combine)
Combine_Seg = (Combine > Threshold*0.8) * 255
PlotArray(Combine_Seg, 'Otsu segmentation')

WS_Limits = (Combine_Seg + Combine) / 2
# WS_Limits[WS_Limits > 255] = 255
PlotArray(WS_Limits, 'Test', True)

Labels = BW
WS = segmentation.watershed(WS_Limits,Labels,connectivity=2)
PlotArray(WS, 'Watershed segmentation')
WS_Edges = segmentation.find_boundaries(WS) * 255







# 3D visualization of the picture
XGrid = np.arange(GS_Rescaled.shape[1])
YGrid = np.arange(GS_Rescaled.shape[0])
XGrid, YGrid = np.meshgrid(XGrid,YGrid)

Figure, Axes = plt.subplots(subplot_kw={"projection": "3d"})
Axes.plot_surface(XGrid,YGrid,GS_Rescaled,cmap='jet')
plt.show()
plt.close(Figure)

W = GaussianKernel(10,7)
GS_Smooth = correlate(GS_Rescaled, W, output='float', mode='reflect')
PlotArray(GS_Smooth,'Grayscale Filtered')

Figure, Axes = plt.subplots(subplot_kw={"projection": "3d"})
Axes.plot_surface(XGrid, YGrid, GS_Smooth, cmap='gray')
plt.show()

GS_Gradient = filters.rank.gradient(GS_Rescaled, morphology.disk(2))
PlotArray(GS_Gradient, 'Grayscale Gradient')

GS_Edges_F = filters.frangi(GS_Rescaled, sigmas=range(1,5,1), mode='reflect')
PlotArray(GS_Edges_F, 'Grayscale Edges (Frangi)')

GS_Edges_S = filters.sobel(GS_Rescaled, mode='reflect')
PlotArray(GS_Edges_S, 'Grayscale Edges (Sobel)')

GS_Edges_C = feature.canny(GS_Rescaled,sigma=2)
PlotArray(GS_Edges_C, 'Grayscale Edges (Canny)')

GS_Edges_R = filters.roberts(GS_Rescaled)
PlotArray(GS_Edges_R, 'Grayscale Edges (Roberts)')

Threshold = filters.threshold_otsu(GS_Rescaled)
GS_Seg = (GS_Edges_F > Threshold) * 1
PlotArray(GS_Seg, 'Otsu segmentation')

Thresholds = filters.threshold_multiotsu(GS_Gradient,classes=2)
GS_Seg = np.digitize(GS_Gradient, bins=Thresholds)
PlotArray(GS_Seg, '3 Otsu segmentation')



XR_min = 256 - 150
XR_max = 256 - 40

XB_min = 256 - 215
XB_max = 256 - 100

Wl = np.array([[0.25, 1, 0.25],[1, 0, 1],[0.25, 1, 0.25]])

LR_min = np.sum(Wl[:int(round(Wl.shape[0]/2)),:int(round(Wl.shape[1]/2))])
LB_min = np.sum(Wl[:int(round(Wl.shape[0]/2)),:int(round(Wl.shape[1]/2))])
LB_max = np.sum(Wl[:int(round(Wl.shape[0]/2)),:int(round(Wl.shape[1]/2))])

Beta_max = ((XR_max/XB_max)-1)/LB_max
Beta_min = max(((XR_max/XR_min)-1)/LR_min, ((XB_max/XB_min)-1)/LB_min)

Wh = Wl

T_S = PCNN_Segmentation(GS_Contrast, Wl, Wh, Beta=0.7, Gamma=0.05, dT=1)
PlotArray(T_S, 'Inhibitted PCNN')

def PCNN_Segmentation(GS_Image, Wl, Wh, Beta=0.7, Gamma=0.00, dT=1):

    S = GS_Image

    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    T = Y

    F = S

    Theta = 255 * np.ones((Rows, Columns))
    Vt = 400

    FireNumber = 0
    N = 0
    while FireNumber < S.size:

        N += 1

        L = correlate(Y, Wl, output='float', mode='reflect')
        H = correlate(Y, Wh, output='float', mode='reflect')
        Theta = Theta - dT + Vt * Y
        Fire = 1

        while Fire == 1:
            Q = Y
            U = F * (1 + Beta * L) * (1 - Gamma * H)
            Y = (U > Theta) * 1

            if np.array_equal(Q, Y):
                Fire = 0
            else:
                L = correlate(Y, Wl, output='float', mode='reflect')
                # H = correlate(Y, Wh, output='float', mode='reflect')

        T = T + N * Y
        FireNumber += sum(sum(Y))

    T = 256 - T

    return T
