from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import exposure

Directory = Path.cwd() / 'Tests\Calcein'
WideField = sitk.ReadImage(str(Directory / 'Test3/image_220311_006.JPG'))
Confocal = sitk.ReadImage(str(Directory / 'Nikon/437_R/Medial_001_x20_EPI.tif'))

W_Array = sitk.GetArrayFromImage(WideField)
C_Array = sitk.GetArrayFromImage(Confocal)
C_Array = C_Array[::-1,:]

Figure, Axes = plt.subplots(1,1, figsize=(W_Array.shape[1]/100, W_Array.shape[0]/100))
Axes.imshow(W_Array)
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

Figure, Axes = plt.subplots(1,1, figsize=(C_Array.shape[1]/100, C_Array.shape[0]/100))
Axes.imshow(C_Array)
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

# Match histogram to compare images
M_Array = exposure.match_histograms(C_Array,W_Array)
Figure, Axes = plt.subplots(1,1, figsize=(M_Array.shape[1]/100, M_Array.shape[0]/100))
Axes.imshow(M_Array.astype('uint8'))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

# Perform resampling and registration on gray images
Confocal.SetSpacing((0.1454670271209604,0.1454670271209604))
WideField.SetSpacing((0.28786667, 0.28786667))

## Resample WideField image
Offset = WideField.GetOrigin()
Direction = WideField.GetDirection()
Orig_Size = np.array(WideField.GetSize()).astype('int')
Orig_Spacing = WideField.GetSpacing()

New_Spacing = Confocal.GetSpacing()

Resample = sitk.ResampleImageFilter()
Resample.SetInterpolator = sitk.sitkLinear
Resample.SetOutputDirection(Direction)
Resample.SetOutputOrigin(Offset)
Resample.SetOutputSpacing(New_Spacing)

New_Size = Orig_Size * (np.array(Orig_Spacing) / np.array(New_Spacing))
New_Size = np.ceil(New_Size).astype('int')  # Image dimensions are in integers
New_Size = [int(s) for s in New_Size]
Resample.SetSize(New_Size)

R_WideField = Resample.Execute(WideField)
R_Array = sitk.GetArrayFromImage(R_WideField)

ParameterMap = sitk.GetDefaultParameterMap('affine')
ParameterMap['MaximumNumberOfIterations'] = [str(2000)]
ParameterMap['SP_alpha'] = [str(0.6)]
ParameterMap['SP_A'] = [str(1000)]

FixedImage = sitk.GetImageFromArray(M_Array[:,:,1])
MovingImage = sitk.GetImageFromArray(R_Array[:,:,1])

ElastixImageFilter = sitk.ElastixImageFilter()
ElastixImageFilter.SetParameterMap(ParameterMap)
ElastixImageFilter.SetFixedImage(FixedImage)
ElastixImageFilter.SetMovingImage(MovingImage)
ElastixImageFilter.SetOutputDirectory(str(Directory))
ElastixImageFilter.LogToConsoleOn()
ElastixImageFilter.Execute()

ResultImage = ElastixImageFilter.GetResultImage()
TransformParameterMap = ElastixImageFilter.GetTransformParameterMap()
sitk.PrintParameterMap(TransformParameterMap)

Result_Array = sitk.GetArrayFromImage(ResultImage)
Figure, Axes = plt.subplots(1,1, figsize=(Result_Array.shape[1]/100, Result_Array.shape[0]/100))
Axes.imshow(Result_Array, cmap='gray')
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

# Transform image
Registered = np.zeros(M_Array.shape)

TransformFilter = sitk.TransformixImageFilter()
TransformFilter.SetTransformParameterMap(TransformParameterMap)

for i in range(3):
    Image = sitk.GetImageFromArray(R_Array[:,:,i])
    TransformFilter.SetMovingImage(Image)
    Registered_Image = TransformFilter.Execute()
    Registered[:,:,i] = sitk.GetArrayFromImage(Registered_Image)

Figure, Axes = plt.subplots(1,1, figsize=(Registered.shape[1]/100, Registered.shape[0]/100))
Axes.imshow(Registered.astype('uint8'))
# Axes.axis('off')
# plt.subplots_adjust(0,0,1,1)
plt.show()

# Crop image
Crop = np.array([[200,4500],[490,3264]])
C_Size = Crop[:,1] - Crop[:,0]

W_Crop = Registered[Crop[1,0]:Crop[1,1],Crop[0,0]:Crop[0,1]].astype('uint8')
C_Crop = M_Array[Crop[1,0]:Crop[1,1],Crop[0,0]:Crop[0,1]].astype('uint8')

Figure, Axes = plt.subplots(1,1, figsize=(C_Size[0]/100, C_Size[1]/100))
Axes.imshow(W_Crop)
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()


Figure, Axes = plt.subplots(1,1, figsize=(C_Size[0]/100, C_Size[1]/100))
Axes.imshow(C_Crop)
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

T = exposure.match_histograms(C_Crop,W_Crop)
Figure, Axes = plt.subplots(1,1, figsize=(C_Size[0]/100, C_Size[1]/100))
Axes.imshow(T.astype('uint8'))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()


# Add scale bars
Calibration = 0.1454670271209604/1000
# ScaleBar parameters
SB_Length = 100
SB_Start = 100
SB_X = [SB_Start, SB_Start + SB_Length/1000 / Calibration]


plt.rcParams['font.size'] = '52'
Text = str(SB_Length) + ' ' + u'\u03bc' + 'm'
Text_Y = 250
Offset = 25

Figure, Axes = plt.subplots(1,1, figsize=(C_Size[0]/100, C_Size[1]/100))
Axes.imshow(T.astype('uint8'))
Axes.plot(SB_X, [Text_Y+Offset,Text_Y+Offset], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[0],SB_X[0]], [Text_Y,Text_Y+2*Offset], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[1],SB_X[1]], [Text_Y,Text_Y+2*Offset], color=(1,0,0), linewidth=5)
Axes.annotate(Text, (SB_Start+len(Text)*35, Text_Y), color=(1,0,0))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(Directory / 'Confocal.png'))
plt.show()

Figure, Axes = plt.subplots(1,1, figsize=(C_Size[0]/100, C_Size[1]/100))
Axes.imshow(W_Crop)
Axes.plot(SB_X, [Text_Y+Offset,Text_Y+Offset], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[0],SB_X[0]], [Text_Y,Text_Y+2*Offset], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[1],SB_X[1]], [Text_Y,Text_Y+2*Offset], color=(1,0,0), linewidth=5)
Axes.annotate(Text, (SB_Start+len(Text)*35, Text_Y), color=(1,0,0))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(Directory / 'WideField.png'))
plt.show()

