"""
This script aims to segment cement lines using Pulse-Coupled Neural Network (PCNN) algorithm
Optimized using particle swarm optimization and an adaptive threshold
Based on :
Hage, I., Hamade, R. (2015)
Automatic Detection of Cortical Bones Haversian Osteonal Boundaries.
AIMS Medical Science, 2(4), 328–346.
https://doi.org/10.3934/medsci.2015.4.328

Josephson, T. (2020)
A microstructural analysis of the mechanical behavior of cortical bone through histology and image processing
Master thesis, Drexel university
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology, measure
from scipy.ndimage import correlate
import matplotlib as mpl


def PlotImage(Image):
    Spacing = Image.GetSpacing()

    Image_Array = sitk.GetArrayFromImage(Image)

    X_Positions = np.arange(Image_Array.shape[1]) * Spacing[1]
    Y_Positions = np.arange(Image_Array.shape[0]) * Spacing[0]

    N_XTicks = round(len(X_Positions) / 5)
    N_YTicks = round(len(Y_Positions) / 5)
    TicksSize = min(N_XTicks,N_YTicks)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(Image_Array)
    Axes.set_xlim([0, Image_Array.shape[1]])
    Axes.set_ylim([0, Image_Array.shape[0]])
    Axes.set_xlabel('X ($\mu$m)')
    Axes.set_ylabel('Y ($\mu$m)')
    plt.xticks(np.arange(0, Image_Array.shape[1])[::TicksSize], np.round(X_Positions[::TicksSize]).astype('int'))
    plt.yticks(np.arange(0, Image_Array.shape[0])[::TicksSize], np.round(Y_Positions[::TicksSize]).astype('int'))
    plt.show()

    return Image_Array
def PlotArray(Array,Spacing=(0.08466666666666667, 0.08466666666666667),Title=None):

    X_Positions = np.arange(Array.shape[1]) * Spacing[1]
    Y_Positions = np.arange(Array.shape[0]) * Spacing[0]

    N_XTicks = round(len(X_Positions) / 5)
    N_YTicks = round(len(Y_Positions) / 5)
    TicksSize = min(N_XTicks, N_YTicks)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(Array,cmap='gray')
    Axes.set_xlim([0, Array.shape[1]])
    Axes.set_ylim([0, Array.shape[0]])
    Axes.set_xlabel('X ($\mu$m)')
    Axes.set_ylabel('Y ($\mu$m)')
    plt.title(Title)
    plt.xticks(np.arange(0, Array.shape[1])[::TicksSize], np.round(X_Positions[::TicksSize]).astype('int'))
    plt.yticks(np.arange(0, Array.shape[0])[::TicksSize], np.round(Y_Positions[::TicksSize]).astype('int'))
    plt.show()

    return
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
def PlotROI(ImageArray, RegionLabel, Labels, Spacing=(0.08466666666666667, 0.08466666666666667)):

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

    X_Positions = np.arange(ImageArray.shape[1]) * Spacing[1]
    Y_Positions = np.arange(ImageArray.shape[0]) * Spacing[0]

    N_XTicks = round(len(X_Positions) / 5)
    N_YTicks = round(len(Y_Positions) / 5)
    TicksSize = min(N_XTicks, N_YTicks)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(ImageArray, cmap='gray')
    Axes.imshow(Region, cmap=ColorMap, vmin=0.25, vmax=0.75)
    Axes.plot([], color=(1,0,0,0.5), linestyle='none', marker='s', label='Region')
    Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='centroid')
    Axes.plot(X0 + Ellipse_R[0, :], Y0 - Ellipse_R[1, :], color=(0, 1, 0), label='Fitted ellipse')
    plt.title('Region ' + str(RegionLabel))
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.275), ncol=3, frameon=False)
    Axes.set_xlim([0, ImageArray.shape[1]])
    Axes.set_ylim([0, ImageArray.shape[0]])
    Axes.set_xlabel('X ($\mu$m)')
    Axes.set_ylabel('Y ($\mu$m)')
    plt.xticks(np.arange(0, ImageArray.shape[1])[::TicksSize], np.round(X_Positions[::TicksSize]).astype('int'))
    plt.yticks(np.arange(0, ImageArray.shape[0])[::TicksSize], np.round(Y_Positions[::TicksSize]).astype('int'))
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    plt.close(Figure)

    return


CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Tests/Osteons/'
Images = [File for File in os.listdir(ImageDirectory) if File.endswith('.jpg')]
Images.sort()

Image = sitk.ReadImage(ImageDirectory+Images[1])

# Crop image (size in um) at random position
X_Crop_Size = 100
Y_Crop_Size = 100
Crop_X = round(X_Crop_Size / Image.GetSpacing()[1] + 0.5)
Crop_Y = round(Y_Crop_Size / Image.GetSpacing()[0] + 0.5)
Random_X = 779
Random_Y = 6396
Random_X = round(325 / Image.GetSpacing()[1])
Random_Y = round(400 / Image.GetSpacing()[0])
Cropping = (Image.GetSize()[0]-Random_Y-Crop_Y,Image.GetSize()[1]-Random_X-Crop_X)
SubImage = sitk.Crop(Image,(Random_Y,Random_X),Cropping)
SubImage_Array = PlotImage(SubImage)

# Decompose RGB image and equalize histogram
R, G, B = SubImage_Array[:,:,0], SubImage_Array[:,:,1], SubImage_Array[:,:,2]

Figure, Axes = plt.subplots(1, 3, figsize=(16.5, 4.5), dpi=100)
Axes[0].imshow(R, cmap='gray')
Axes[0].set_title('R channel')
Axes[0].axis('off')
Axes[1].imshow(G, cmap='gray')
Axes[1].set_title('G channel')
Axes[1].axis('off')
Axes[2].imshow(B, cmap='gray')
Axes[2].set_title('B channel')
Axes[2].axis('off')
plt.show()
plt.close(Figure)



Image = R / R.max()
# Parameters for 1 osteon
ParametersDictionary = {'AlphaF': -0.2381907530225938,
                        'AlphaL': -0.2398532814016563,
                        'AlphaT': 0.7220370952618878,
                        'VF': -0.30581795422704827,
                        'VL': 1.0,
                        'VT': 0.5418764530883855,
                        'Beta': -1.0}
# Parameters for multiple osteons
ParametersDictionary = {'AlphaF': -1.0,
                        'AlphaL': -1.0,
                        'AlphaT': 1.0,
                        'VF': -0.6907362822709551,
                        'VL': -1.0,
                        'VT': 1.0,
                        'Beta': 1.0}
ParametersDictionary = {'AlphaF': -1.0,
                        'AlphaL': -0.5171884715426962,
                        'AlphaT': 1.0,
                        'VF': -1.0,
                        'VL': -1.0,
                        'VT': 0.8293069163951987,
                        'Beta': 0.9486743832771634}
ParametersDictionary = {'AlphaF': -1.4568293766698421,
                        'AlphaL': 1.7664468143804637,
                        'AlphaT': 0.90684945230546,
                        'VF': -1.2138776497934514,
                        'VL': -2.430799349519539,
                        'VT': -1.9142988177549285,
                        'Beta': 0.2516870868740857}
ParametersDictionary = {'AlphaF': -0.4358389450874143,
                        'AlphaL': 0.42548990174264945,
                        'AlphaT': -0.0026282587881454322,
                        'VF': 0.12978216128591713,
                        'VL': -0.5343658289990307,
                        'VT': -0.6003009584815666,
                        'Beta': -0.5078775611128168}
# Parameters for Harvesian channels
ParametersDictionary = {'AlphaF': 0.559958004152834,
                        'AlphaL': 0.6362288702766017,
                        'AlphaT': -0.2964094323025681,
                        'VF': -0.6665908949885424,
                        'VL': 1.0,
                        'VT': -1.0,
                        'Beta': 1.0}
ParametersDictionary = {'AlphaF': -0.77116675769926,
                        'AlphaL': 0.3906772796273532,
                        'AlphaT': -1.0,
                        'VF': -0.48348777605091137,
                        'VL': 0.7318494869324401,
                        'VT': -0.026910000212507557,
                        'Beta': 1.0}
Y = PCNN_Segmentation(Image,ParametersDictionary)
PlotArray(Y, Title='PCNN Segmentation')

Disk = morphology.disk(2)
BW_Dilate = morphology.binary_dilation(Y,Disk)
PlotArray(BW_Dilate, Title='Dilated segmentation')

Labels = measure.label(BW_Dilate,connectivity=2)
Properties = ('label', 'area', 'orientation', 'euler_number')
PropertiesTable = pd.DataFrame(measure.regionprops_table(Labels,properties=Properties))

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(PropertiesTable['euler_number'], marker='o',linestyle='none',color=(1,0,0),fillstyle='none')
plt.show()
plt.close(Figure)

PropertiesTable.sort_values('area').iloc[-9:]['label'].values
RegionsLabels = [ 2, 76, 23, 55, 20, 32, 26, 38,  1]


for Region in RegionsLabels:
    PlotROI(Y, Region, Labels)




## Find edges using Frangi filter
R_Edges = filters.frangi(R, sigmas=range(1,5,1), mode='reflect')
Threshold = filters.threshold_otsu(R_Edges)
R_Edges = (R_Edges >= Threshold) * 1

G_Edges = filters.frangi(G, sigmas=range(1,5,1), mode='reflect')
Threshold = filters.threshold_otsu(G_Edges)
G_Edges = (G_Edges >= Threshold) * 1

B_Edges = filters.frangi(B, sigmas=range(1,5,1), mode='reflect')
Threshold = filters.threshold_otsu(B_Edges)
B_Edges = (B_Edges >= Threshold) * 1


Disk = morphology.disk(2)
R_Dilate = morphology.binary_erosion(R_Edges,Disk)
G_Dilate = morphology.binary_erosion(G_Edges,Disk)
B_Dilate = morphology.binary_erosion(B_Edges,Disk)

Disk = morphology.disk(5)
Edges = R_Dilate * G_Dilate * B_Dilate
Edges = morphology.binary_dilation(Edges,Disk)
PlotArray(Edges,Spacing=Image.GetSpacing())

Labels = measure.label(Edges,connectivity=1)
Properties = ('label','area')
PropertiesTable = pd.DataFrame(measure.regionprops_table(Labels,properties=Properties))

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(PropertiesTable['area'], marker='o',linestyle='none',color=(1,0,0),fillstyle='none')
plt.show()
plt.close(Figure)

Filter = PropertiesTable['area'] > 250
RegionLabels = PropertiesTable[Filter]['label']
FilteredRegions = np.zeros(Labels.shape)
for Label in RegionLabels.values:
    Y, X = np.where(Labels == Label)
    FilteredRegions[Y, X] = 1
PlotArray(FilteredRegions,Spacing=Image.GetSpacing())


GrayScale_Array = np.round(exposure.match_histograms(R, B)).astype('uint8')
GrayScale_Array = (GrayScale_Array - GrayScale_Array.min()) / (GrayScale_Array.max()-GrayScale_Array.min()) * 255
PlotArray(GrayScale_Array,Spacing=Image.GetSpacing())

## Find edges using Frangi filter
Edges = filters.frangi(GrayScale_Array, sigmas=range(1,5,1), mode='reflect')
PlotArray(Edges,Spacing=Image.GetSpacing())

## Threshold
Threshold = filters.threshold_otsu(Edges)
BW = Edges.copy()
BW = (BW >= Threshold) * 1
PlotArray(BW,Spacing=Image.GetSpacing())

## Erode to get rid of small structures
Disk = morphology.disk(2)
BW_Dilate = morphology.binary_dilation(BW,Disk)
PlotArray(BW_Dilate,Spacing=Image.GetSpacing())




## Try PCNN single neuron firing segmentation

