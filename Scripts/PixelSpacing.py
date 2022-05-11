"""
Code used to determine physical pixel spacing
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import color, measure
from skimage import morphology


desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)


# Define utilities functions
def PlotImage(Image, Title=' ', CMap=False, ColorBar=False):

    Array = sitk.GetArrayFromImage(Image)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    if CMap:
        CBar = Axes.imshow(Array)
    else:
        CBar = Axes.imshow(Array, cmap='gray')
    if ColorBar:
        plt.colorbar(CBar)
    plt.title(Title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def PlotArray(Array, Title=' ', CMap=False, ColorBar=False):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    if CMap:
        CBar = Axes.imshow(Array)
    else:
        CBar = Axes.imshow(Array, cmap='gray')
    if ColorBar:
        plt.colorbar(CBar)
    plt.title(Title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def RGBThreshold(RGBArray, Threshold):
    R, G, B = RGBArray[:, :, 0], RGBArray[:, :, 1], RGBArray[:, :, 2]

    R_Filter = R > Threshold[0]
    G_Filter = G > Threshold[1]
    B_Filter = B > Threshold[2]

    BinArray = np.zeros((RGBArray.shape[0], RGBArray.shape[1]))
    BinArray[R_Filter & G_Filter & B_Filter] = 1

    return BinArray



# Set path
CurrentDirectory = Path.cwd()
ImageDirectory = CurrentDirectory / 'Tests/Osteons/HumanBone/'

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / 'Staining3_ScaleBar.jpg'))
PlotImage(Image)

Crop = [-190,-315,-50,-50]
Array = sitk.GetArrayFromImage(Image)
PlotArray(Array[Crop[0]:Crop[2],Crop[1]:Crop[3],:])

Gray = color.rgb2gray(Array[Crop[0]:Crop[2],Crop[1]:Crop[3],:])
PlotArray(Gray)

Filter = Gray < 0.1
PlotArray(Filter)

Labels = measure.label(Filter*1)
PlotArray(Labels)

RegionsProps = measure.regionprops(Labels)
BBox = RegionsProps[-1].bbox

Figure, Axes = plt.subplots(1,1)
Axes.imshow(Filter,cmap='gray')
# Axes.plot([BBox[1], BBox[1]], [BBox[0], BBox[2]-1], color=(1,0,0))
# Axes.plot([BBox[3]-1, BBox[3]-1], [BBox[0], BBox[2]-1], color=(1,0,0))
# Axes.plot([BBox[1], BBox[3]-1], [BBox[0], BBox[0]], color=(1,0,0))
Axes.plot([BBox[1], BBox[3]-1], [BBox[2]-1, BBox[2]-1], color=(1,0,0))
plt.title(' ')
Axes.axis('off')
plt.tight_layout()
plt.show()

ScaleLength = BBox[3]-1 - BBox[1]
ScaleValue = 50
PixelSpacing = ScaleValue / ScaleLength


R_Image = sitk.ReadImage(str(ImageDirectory / 'Stained1.png'))
PlotImage(R_Image)
R_Array = sitk.GetArrayFromImage(R_Image)

P_Array = np.pad(R_Array[:,:,0],((250,0),(0,0)))

Figure, Axes = plt.subplots(1,1)
Axes.imshow(P_Array,cmap='gray')
Axes.axis('off')
plt.show()

# Register images
ParameterMap = sitk.GetDefaultParameterMap('affine')
ParameterMap['MaximumNumberOfIterations'] = [str(1000)]
ParameterMap['SP_alpha'] = [str(500)]
ParameterMap['SP_A'] = [str(0.1)]

FixedImage = sitk.GetImageFromArray(Array[:,:,0])
Subsampling = 2
MovingImage = sitk.GetImageFromArray(P_Array[::Subsampling,::Subsampling])

ElastixImageFilter = sitk.ElastixImageFilter()
ElastixImageFilter.SetParameterMap(ParameterMap)
ElastixImageFilter.SetFixedImage(FixedImage)
ElastixImageFilter.SetMovingImage(MovingImage)
ElastixImageFilter.SetOutputDirectory(str(ImageDirectory))
ElastixImageFilter.LogToConsoleOn()
ResultImage = ElastixImageFilter.Execute()
ResultImage_Array = sitk.GetArrayFromImage(ResultImage)

Figure, Axes = plt.subplots(1,1)
Axes.imshow(Array)
Axes.imshow(ResultImage_Array,cmap='gray', alpha=0.5)
Axes.axis('off')
plt.show()

TransformParameterMap = ElastixImageFilter.GetTransformParameterMap()
TransformParameters = np.array(TransformParameterMap[0]['TransformParameters']).astype('float')
Matrix = np.matrix([[TransformParameters[0],TransformParameters[1]],
                    [TransformParameters[2],TransformParameters[3]]])
Scale = Subsampling * np.linalg.det(Matrix)

# Spacing for registered images [um]
Spacing = PixelSpacing / Scale