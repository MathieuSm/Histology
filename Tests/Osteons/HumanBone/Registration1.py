import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology

def RGBThreshold(RGBArray, Threshold):
    R, G, B = RGBArray[:, :, 0], RGBArray[:, :, 1], RGBArray[:, :, 2]

    R_Filter = R > Threshold[0]
    G_Filter = G < Threshold[1]
    B_Filter = B < Threshold[2]

    BinArray = np.zeros((RGBArray.shape[0], RGBArray.shape[1]))
    BinArray[R_Filter & G_Filter & B_Filter] = 1

    return BinArray

# Set path
CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Tests/Osteons/HumanBone/'

# Read images
MedialOriginal = sitk.ReadImage(ImageDirectory + 'Original.png')
MO_Array = sitk.GetArrayFromImage(MedialOriginal)

Figure, Axes = plt.subplots(1,1)
Axes.imshow(MO_Array)
Axes.axis('off')
plt.show()

Threshold = [220, 50, 50]
MO_Bin = RGBThreshold(MO_Array,Threshold)

Figure, Axes = plt.subplots(1,1)
Axes.imshow(MO_Bin, cmap='binary')
Axes.axis('off')
plt.show()

Disk = morphology.disk(15)
MO_Bin = morphology.binary_dilation(MO_Bin,Disk)
MO_Bin = morphology.binary_erosion(MO_Bin,Disk)

MO_Labels = morphology.label(MO_Bin)
MO_Shape = np.zeros(MO_Labels.shape)
Filter1 = MO_Labels == 1
Filter2 = MO_Labels == 3
MO_Shape[Filter1] = 1

Figure, Axes = plt.subplots(1,1)
Axes.imshow(MO_Shape, cmap='binary')
Axes.axis('off')
plt.show()


MedialStained = sitk.ReadImage(ImageDirectory + 'Stained.png')
MS_Array = sitk.GetArrayFromImage(MedialStained)

Threshold = [220, 50, 50]
MS_Bin = RGBThreshold(MS_Array,Threshold)

Figure, Axes = plt.subplots(1,1)
Axes.imshow(MS_Bin, cmap='binary')
Axes.axis('off')
plt.show()

Disk = morphology.disk(12)
MS_Bin = morphology.binary_dilation(MS_Bin,Disk)
MS_Bin = morphology.binary_erosion(MS_Bin,Disk)


MS_Labels = morphology.label(MS_Bin)
MS_Shape = np.zeros(MS_Labels.shape)
Filter1 = MS_Labels == 2
Filter2 = MS_Labels == 3
MS_Shape[Filter2] = 1

Figure, Axes = plt.subplots(1,1)
Axes.imshow(MS_Shape, cmap='binary')
Axes.axis('off')
plt.show()



# Register images
ParameterMap = sitk.GetDefaultParameterMap('rigid')
ParameterMap['MaximumNumberOfIterations'] = [str(1000)]
ParameterMap['SP_alpha'] = [str(500)]
ParameterMap['SP_A'] = [str(0.1)]

FixedImage = MS_Array[:,:,0]
MovingImage = MO_Array[:,:,0]

ElastixImageFilter = sitk.ElastixImageFilter()
ElastixImageFilter.SetParameterMap(ParameterMap)
ElastixImageFilter.SetFixedImage(sitk.GetImageFromArray(FixedImage))
ElastixImageFilter.SetMovingImage(sitk.GetImageFromArray(MovingImage))
ElastixImageFilter.SetOutputDirectory(ImageDirectory)
ElastixImageFilter.LogToConsoleOn()
ResultImage = ElastixImageFilter.Execute()
ResultImage_Array = sitk.GetArrayFromImage(ResultImage)

Figure, Axes = plt.subplots(1,1)
Axes.imshow(ResultImage_Array,cmap='gray')
Axes.axis('off')
plt.show()

TransformParameterMap = ElastixImageFilter.GetTransformParameterMap()
# TransformParameterMap['Size'] = [str(S) for S in MedialStained.GetSize()]
TransformixImageFilter = sitk.TransformixImageFilter()
TransformixImageFilter.ComputeDeformationFieldOff()
TransformixImageFilter.ComputeSpatialJacobianOff()
TransformixImageFilter.ComputeDeterminantOfSpatialJacobianOff()
TransformixImageFilter.SetTransformParameterMap(TransformParameterMap)
TransformixImageFilter.SetOutputDirectory(ImageDirectory)

RGB = sitk.GetArrayFromImage(MedialOriginal)
TransformedArray = np.ones(sitk.GetArrayFromImage(MedialStained).shape)*255

for i in range(3):
    TransformixImageFilter.SetMovingImage(sitk.GetImageFromArray(RGB[:,:,i]))
    TransformedImage = TransformixImageFilter.Execute()
    TransformedArray[:,:,i] = sitk.GetArrayFromImage(TransformedImage)

# Normalize and cast to integer
TransformedArray_N = (TransformedArray - TransformedArray.min()) / (TransformedArray.max() - TransformedArray.min())
TransformedArray_R = np.round(TransformedArray_N*255).astype('int')


Figure, Axes = plt.subplots(1,1,figsize=(30.16,27.08),dpi=100)
Axes.imshow(TransformedArray_R[:-8:,32:])
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(ImageDirectory+'Original_Registered.png')
plt.show()

Figure, Axes = plt.subplots(1,1,figsize=(30.16,27.08),dpi=100)
Axes.imshow(sitk.GetArrayFromImage(MedialStained)[:-8:,32:])
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(ImageDirectory+'Stained_Registered.png')
plt.show()
