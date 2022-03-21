import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology

def sRGB2Gray(SimpleITK_Image):

    # Convert sRGB image to gray scale and rescale results to [0,255]
    channels = [sitk.VectorIndexSelectionCast(SimpleITK_Image,i, sitk.sitkFloat32)
                for i in range(SimpleITK_Image.GetNumberOfComponentsPerPixel())]
    # Linear mapping
    I = 1/255.0 * (0.2126*channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])

    # Nonlinear gamma correction
    I = I * sitk.Cast(I <= 0.0031308,sitk.sitkFloat32) * 12.92 + I**(1/2.4) * sitk.Cast(I > 0.0031308,sitk.sitkFloat32) * 1.055 - 0.055
    return sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8)

# Set path
CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Tests/Osteons/HumanBone/'

# Read images
MedialOriginal = sitk.ReadImage(ImageDirectory + '22_010_R_distal_03_med_20.jpg')
MR_Gray = sRGB2Gray(MedialOriginal)
MR_Gray_Array = sitk.GetArrayFromImage(MR_Gray)

Threshold = 245
MR_Bin = np.zeros(MR_Gray_Array.shape)
MR_Bin[MR_Gray_Array > Threshold] = 1

Disk = morphology.disk(20)
MR_Bin = morphology.binary_dilation(MR_Bin,Disk)
MR_Bin = morphology.binary_erosion(MR_Bin,Disk)

Figure, Axes = plt.subplots(1,1)
Axes.imshow(MR_Bin[6150:,4000:9000], cmap='binary')
Axes.axis('off')
plt.show()


MedialStained = sitk.ReadImage(ImageDirectory + '22_010_R_dist_medial_03_20.jpg')
MS_Gray = sRGB2Gray(MedialStained)
MS_Gray_Array = sitk.GetArrayFromImage(MS_Gray)

Threshold = 245
MS_Bin = np.zeros(MS_Gray_Array.shape)
MS_Bin[MS_Gray_Array > Threshold] = 1

Disk = morphology.disk(20)
MS_Bin = morphology.binary_dilation(MS_Bin,Disk)
MS_Bin = morphology.binary_erosion(MS_Bin,Disk)


Figure, Axes = plt.subplots(1,1)
Axes.imshow(MS_Bin[5300:,6150:], cmap='binary')
Axes.axis('off')
plt.show()


# Register images
ParameterMap = sitk.GetDefaultParameterMap('rigid')

FixedImage = MS_Bin[5300:,6150:]*1
MovingImage = MR_Bin[6150:,4000:9000]*1

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

TransformParameterMap = sitk.ReadParameterFile(ImageDirectory + '/TransformParameters.0.txt')
TransformixImageFilter = sitk.TransformixImageFilter()
TransformixImageFilter.ComputeDeformationFieldOff()
TransformixImageFilter.ComputeSpatialJacobianOff()
TransformixImageFilter.ComputeDeterminantOfSpatialJacobianOff()
TransformixImageFilter.SetTransformParameterMap(TransformParameterMap)
TransformixImageFilter.SetOutputDirectory(ImageDirectory)

RGB = sitk.GetArrayFromImage(MedialOriginal)
TransformedArray = np.zeros((MedialStained.GetWidth(),MedialStained.GetHeight(),3))

for i in range(3):
    TransformixImageFilter.SetMovingImage(sitk.GetImageFromArray(RGB[:,:,i]))
    TransformedImage = TransformixImageFilter.Execute()
    TransformedArray[:,:,i] = sitk.GetArrayFromImage(TransformedImage)

TransformedImages = [sitk.GetArrayFromImage(TransformedImage0),
                     sitk.GetArrayFromImage(TransformedImage1),
                     sitk.GetArrayFromImage(TransformedImage2)]
TransformedImage = np.zeros(MedialStained_Array.shape)
for i in range(3):
    TransformedImage[:,:,i] += TransformedImages[i]

Figure, Axes = plt.subplots(1,1)
Axes.imshow(TransformedArray)
Axes.axis('off')
plt.show()
