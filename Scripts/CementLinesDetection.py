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
def PlotArray(Array,Spacing):

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
    plt.xticks(np.arange(0, Array.shape[1])[::TicksSize], np.round(X_Positions[::TicksSize]).astype('int'))
    plt.yticks(np.arange(0, Array.shape[0])[::TicksSize], np.round(Y_Positions[::TicksSize]).astype('int'))
    plt.show()

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

