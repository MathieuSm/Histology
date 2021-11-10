"""
This script aims to test segmentation of osteons proposed in a master thesis
Based on :
Josephson, T. (2020)
A microstructural analysis of the mechanical behavior of cortical bone through histology and image processing
Master thesis, Drexel university
"""


import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import exposure, morphology, measure, segmentation, filters, feature
from scipy.ndimage import correlate
import matplotlib as mpl


def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))
def DistinguishableColors(N, shuffle=True):

    """ Create colormap with distinguishable colors and a background color """

    if N > 3:
        X = np.linspace(0,1,N)
        Ramp = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
        Xp = np.linspace(0,1,5)
    else :
        Ramp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Xp = np.linspace(0, 1, 3)

    R = np.interp(X, Xp, Ramp[:, 0])
    G = np.interp(X, Xp, Ramp[:, 1])
    B = np.interp(X, Xp, Ramp[:, 2])
    CMap = np.vstack([R,G,B, np.ones(N)]).T.round(3)

    # Whether to shuffle the output colors
    if shuffle:
        np.random.seed(1)
        np.random.shuffle(CMap)

    CMap = np.vstack(([0, 0, 0, 0], CMap))
    ColorMap = mpl.colors.ListedColormap(CMap)

    return ColorMap


def Watershed(GrayImage, Labels, Hmax):
    """
    Insipired from master thesis of Josephson
    Does not work, for the moment
    """

    N = len(np.unique(Labels))

    Keys = np.arange(256).astype('int')
    Q = {Key: None for Key in Keys}

    for i in range(N):
        Y, X = np.where(Labels == i)

        for j in range(len(X)):
            Key = int(round(GrayImage[Y[j], X[j]]))
            Values = [Y[j], X[j]]

            if Q[Key]:
                Q[Key] = [Q[Key][0], Values]
            else:
                Q[Key] = [Values]

    EmptyQ = False

    while not EmptyQ:

        for i in range(256):

            if not Q[i]:
                EmptyQ = True
                continue

            EmptyQ = False

            for j in range(len(Q[i])):
                X = Q[i][j][1]
                Y = Q[i][j][0]
                cX = [X - 1, X, X + 1]
                cY = [Y - 1, Y, Y + 1]
                Marker = Labels[Y, X]
                Q[i].pop(j)

                for x in cX:
                    for y in cY:

                        C1 = x == X
                        C2 = y == Y
                        C3 = Labels[y, x] > 0
                        C4 = GrayImage[y, x] > Hmax

                        if C1 and C2:
                            continue
                        if C3:
                            continue
                        if C4:
                            continue

                        Labels[y, x] = Marker
                        Key = int(round(GrayImage[y, x]))
                        Q[Key] = [Q[Key][0], [y, x]]

                break

            break

    return Labels

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Scripts/ManualSegmentation/'

Image = sitk.ReadImage(ImageDirectory + 'Toluidinblue_protocol2A_20.jpg')
Resolution = Image.GetSpacing()
ImageArray = sitk.GetArrayFromImage(Image)
ImageArray = ImageArray[3839:3839+1181+1,4724:4724+1181+1]


Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(ImageArray)
plt.axis('off')
plt.title('Image')
plt.show()
plt.close(Figure)


## Decompose image in RGB canals
R, G, B = ImageArray[:,:,0], ImageArray[:,:,1], ImageArray[:,:,2]

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


EqualizedHistogram = np.round(exposure.match_histograms(R, B)).astype('uint8')

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(EqualizedHistogram, cmap='gray')
plt.axis('off')
plt.title('Equalized Histogram')
plt.show()
plt.close(Figure)


## Label canals
Threshold = 213.5
BW = EqualizedHistogram.copy()
BW = (BW >= Threshold) * 1

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(BW, cmap='gray')
plt.axis('off')
plt.title('Binary image')
plt.show()
plt.close(Figure)

G = GaussianKernel(10,5)
G_Smooth = correlate(BW, G, output='float', mode='reflect')

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(G_Smooth, cmap='gray')
plt.axis('off')
plt.title('Binary image')
plt.show()
plt.close(Figure)


BW_Smooth = (G_Smooth >= 0.75) * 1

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(BW_Smooth, cmap='gray')
plt.axis('off')
plt.title('Binary image')
plt.show()
plt.close(Figure)

## Label
Disk = morphology.disk(40)
BW_Dilate = morphology.binary_dilation(BW_Smooth,Disk)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(BW_Dilate, cmap='gray')
plt.axis('off')
plt.title('Binary image')
plt.show()
plt.close(Figure)

Labels = measure.label(BW_Dilate,connectivity=2)
CMap = DistinguishableColors(len(np.unique(Labels))-1)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(EqualizedHistogram, cmap='gray')
Axes.imshow(Labels, cmap=CMap)
plt.axis('off')
plt.title('Labeled image')
plt.show()
plt.close(Figure)

DistanceLabels = Labels.copy()
DistanceLabels[DistanceLabels > 0] = 1
MedialAxis, Distances = morphology.medial_axis(1-DistanceLabels, return_distance=True)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Distances, cmap='gray')
plt.axis('off')
plt.title('Image gradient')
plt.show()
plt.close(Figure)

SegmentImage = (256 - EqualizedHistogram) * (Distances / Distances.max())
WS = segmentation.watershed(SegmentImage,Labels,connectivity=1)
WS_Edges = segmentation.find_boundaries(WS) * 255

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(WS, cmap=CMap)
plt.axis('off')
plt.title('Watershed labelling')
plt.show()
plt.close(Figure)

Gradient = filters.rank.gradient(EqualizedHistogram, morphology.disk(5))
NormGradient = np.round((Gradient - Gradient.min()) / (Gradient.max() - Gradient.min()) * 255).astype('uint8')

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(NormGradient, cmap='gray')
plt.axis('off')
plt.title('Image gradient')
plt.show()
plt.close(Figure)

Gradient_Edges = feature.canny(NormGradient, sigma=1) * 255

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Gradient_Edges, cmap='binary_r')
plt.axis('off')
plt.title('Gradient edges')
plt.show()
plt.close(Figure)

Disk = morphology.disk(2)
Edges_Dilate = morphology.binary_dilation(Gradient_Edges,Disk) * 255

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Edges_Dilate, cmap='binary_r')
plt.axis('off')
plt.title('Gradient edges')
plt.show()
plt.close(Figure)


Overlay = WS_Edges + Edges_Dilate + NormGradient
Overlay[Overlay > 255] = 255

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Overlay, cmap='gray')
plt.axis('off')
plt.title('Gradient edges')
plt.show()
plt.close(Figure)

WS2 = segmentation.watershed(Overlay,Labels,connectivity=1)
Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(WS2, cmap=CMap)
plt.axis('off')
plt.title('Watershed labelling')
plt.show()
plt.close(Figure)


