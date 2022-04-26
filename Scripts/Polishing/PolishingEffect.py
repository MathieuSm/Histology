import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure

# desired_width = 320
# pd.set_option('display.width', desired_width)


# Read data
CurrentDirectory = Path.cwd()
DataPath = CurrentDirectory / 'Tests/Polishing/Protocol/'

Pictures = [File for File in os.listdir(DataPath) if File.endswith('.JPG')]
Pictures.sort()

P1200 = sitk.ReadImage(str(DataPath / Pictures[0]))

P1200_Array = sitk.GetArrayFromImage(P1200)
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(P1200_Array)
Axes.axis('off')
plt.show()


Gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
Gaussian.SetSigma((10,10,10))

G_P1200 = Gaussian.Execute(P1200)
for i in range(5):
    G_P1200 = Gaussian.Execute(G_P1200)

G_P1200_Array = sitk.GetArrayFromImage(G_P1200).astype('int')
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(G_P1200_Array)
Axes.axis('off')
plt.show()

Threshold = 100
Filter1 = G_P1200_Array[:,:,0] < Threshold
Filter2 = G_P1200_Array[:,:,1] < Threshold
Filter3 = G_P1200_Array[:,:,2] < Threshold
BinArray_1200 = np.zeros(P1200.GetSize()[::-1]).astype('uint8')
BinArray_1200[Filter1 & Filter2 & Filter3] = 1

Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(BinArray_1200, cmap='binary')
Axes.axis('off')
plt.show()


# Second image
P2400 = sitk.ReadImage(str(DataPath / Pictures[1]))

P2400_Array = sitk.GetArrayFromImage(P2400)
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(P2400_Array)
Axes.axis('off')
plt.show()


Gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
Gaussian.SetSigma((10,10,10))

G_P2400 = Gaussian.Execute(P2400)
for i in range(4):
    G_P2400 = Gaussian.Execute(G_P2400)

G_P2400_Array = sitk.GetArrayFromImage(G_P2400).astype('int')
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(G_P2400_Array)
Axes.axis('off')
plt.show()

Threshold = 120
Filter1 = G_P2400_Array[:,:,0] < Threshold
Filter2 = G_P2400_Array[:,:,1] < Threshold
Filter3 = G_P2400_Array[:,:,2] < Threshold
BinArray_P2400 = np.zeros(P2400.GetSize()[::-1]).astype('uint8')
BinArray_P2400[Filter1 & Filter2 & Filter3] = 1

Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(BinArray_P2400, cmap='binary')
Axes.axis('off')
plt.show()


# Registration
ParameterMap = sitk.GetDefaultParameterMap('rigid')
ParameterMap['MaximumNumberOfIterations'] = [str(2000)]
ParameterMap['SP_alpha'] = [str(0.6)]
ParameterMap['SP_A'] = [str(1000)]

FixedImage = sitk.GetImageFromArray(BinArray_1200)
MovingImage = sitk.GetImageFromArray(BinArray_P2400)

ElastixImageFilter = sitk.ElastixImageFilter()
ElastixImageFilter.SetParameterMap(ParameterMap)
ElastixImageFilter.SetFixedImage(FixedImage)
ElastixImageFilter.SetMovingImage(MovingImage)
ElastixImageFilter.SetOutputDirectory(str(DataPath))
ElastixImageFilter.LogToConsoleOn()
ElastixImageFilter.Execute()

ResultImage = ElastixImageFilter.GetResultImage()
TransformParameterMap = ElastixImageFilter.GetTransformParameterMap()
sitk.PrintParameterMap(TransformParameterMap)

Result_Array = sitk.GetArrayFromImage(ResultImage)
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(Result_Array, cmap='binary')
Axes.axis('off')
plt.show()


# Transform image
Registered = np.zeros(P1200_Array.shape)

TransformFilter = sitk.TransformixImageFilter()
TransformFilter.SetTransformParameterMap(TransformParameterMap)

for i in range(3):
    Image = sitk.GetImageFromArray(P2400_Array[:,:,i])
    TransformFilter.SetMovingImage(Image)
    Registered_Image = TransformFilter.Execute()
    Registered[:,:,i] = sitk.GetArrayFromImage(Registered_Image)


# Crop and equalize colors

Crop = np.array([[75,1925],[100,1300]])

P1200_Array_Crop = P1200_Array[Crop[1][0]:Crop[1][1],Crop[0][0]:Crop[0][1]]
Registered_Crop = Registered[Crop[1][0]:Crop[1][1],Crop[0][0]:Crop[0][1]]

P1200_Array_Eq = exposure.match_histograms(P1200_Array_Crop,Registered_Crop, multichannel=True)

Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(P1200_Array_Eq.astype('uint8'))
plt.show()


# Save figures
Size = np.array(Crop[:,1] - Crop[:,0]) / 100
Figure, Axes = plt.subplots(1,1,dpi=96, figsize=(Size[0],Size[1]))
Axes.imshow(Registered_Crop.astype('uint8'))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(DataPath / 'Pictures' / 'P2400.png'),dpi=96)
plt.show()

Figure, Axes = plt.subplots(1,1,dpi=96, figsize=(Size[0],Size[1]))
Axes.imshow(P1200_Array_Eq.astype('uint8'))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(DataPath / 'Pictures' / 'P1200.png'),dpi=96)
plt.show()




# Third image
P4000 = sitk.ReadImage(str(DataPath / Pictures[2]))

P4000_Array = sitk.GetArrayFromImage(P4000)
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(P4000_Array)
Axes.axis('off')
plt.show()


Gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
Gaussian.SetSigma((10,10,10))

G_P4000 = Gaussian.Execute(P4000)
for i in range(3):
    G_P4000 = Gaussian.Execute(G_P4000)

G_P4000_Array = sitk.GetArrayFromImage(G_P4000).astype('int')
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(G_P4000_Array)
Axes.axis('off')
plt.show()

Threshold = 120
Filter1 = G_P4000_Array[:,:,0] < Threshold
Filter2 = G_P4000_Array[:,:,1] < Threshold
Filter3 = G_P4000_Array[:,:,2] < Threshold
BinArray_P4000 = np.zeros(P2400.GetSize()[::-1]).astype('uint8')
BinArray_P4000[Filter1 & Filter2 & Filter3] = 1

Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(BinArray_P4000, cmap='binary')
Axes.axis('off')
plt.show()


# Registration
ParameterMap = sitk.GetDefaultParameterMap('rigid')
ParameterMap['MaximumNumberOfIterations'] = [str(2000)]
ParameterMap['SP_alpha'] = [str(0.6)]
ParameterMap['SP_A'] = [str(1000)]

FixedImage = sitk.GetImageFromArray(BinArray_1200)
MovingImage = sitk.GetImageFromArray(BinArray_P4000)

ElastixImageFilter = sitk.ElastixImageFilter()
ElastixImageFilter.SetParameterMap(ParameterMap)
ElastixImageFilter.SetFixedImage(FixedImage)
ElastixImageFilter.SetMovingImage(MovingImage)
ElastixImageFilter.SetOutputDirectory(str(DataPath))
ElastixImageFilter.LogToConsoleOn()
ElastixImageFilter.Execute()

ResultImage = ElastixImageFilter.GetResultImage()
TransformParameterMap = ElastixImageFilter.GetTransformParameterMap()
sitk.PrintParameterMap(TransformParameterMap)

Result_Array = sitk.GetArrayFromImage(ResultImage)
Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(Result_Array, cmap='binary')
Axes.axis('off')
plt.show()


# Transform image
Registered2 = np.zeros(P1200_Array.shape)

TransformFilter = sitk.TransformixImageFilter()
TransformFilter.SetTransformParameterMap(TransformParameterMap)

for i in range(3):
    Image = sitk.GetImageFromArray(P4000_Array[:,:,i])
    TransformFilter.SetMovingImage(Image)
    Registered_Image = TransformFilter.Execute()
    Registered2[:,:,i] = sitk.GetArrayFromImage(Registered_Image)


# Crop and equalize colors

Crop = np.array([[75,1925],[100,1300]])

P1200_Array_Crop = P1200_Array[Crop[1][0]:Crop[1][1],Crop[0][0]:Crop[0][1]]
Registered_Crop = Registered[Crop[1][0]:Crop[1][1],Crop[0][0]:Crop[0][1]]
Registered2_Crop = Registered2[Crop[1][0]:Crop[1][1],Crop[0][0]:Crop[0][1]]


P1200_Array_Eq = exposure.match_histograms(P1200_Array_Crop,Registered_Crop, multichannel=True)

Figure, Axes = plt.subplots(1,1,dpi=96)
Axes.imshow(Registered2_Crop.astype('uint8'))
plt.show()


# Save figures with scalebar
Size = np.array(Crop[:,1] - Crop[:,0]) / 100
DPI = 96
Calibration = 0.0013
# ScaleBar parameters
SB_Length = 0.5
SB_Start = 100
SB_X = [SB_Start, SB_Start + SB_Length / Calibration]


plt.rcParams['font.size'] = '40'
Text = str(SB_Length) + ' mm'

Figure, Axes = plt.subplots(1,1,dpi=DPI, figsize=(Size[0],Size[1]))
Axes.imshow(Registered2_Crop.astype('uint8'))
Axes.plot(SB_X, [1100,1100], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[0],SB_X[0]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[1],SB_X[1]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.annotate(str(SB_Length) + ' mm', (SB_Start+len(Text)*15, 1075), color=(1,0,0))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(DataPath / 'Pictures' / 'P4000.png'),dpi=96)
plt.show()

Figure, Axes = plt.subplots(1,1,dpi=DPI, figsize=(Size[0],Size[1]))
Axes.imshow(Registered_Crop.astype('uint8'))
Axes.plot(SB_X, [1100,1100], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[0],SB_X[0]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[1],SB_X[1]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.annotate(str(SB_Length) + ' mm', (SB_Start+len(Text)*15, 1075), color=(1,0,0))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(DataPath / 'Pictures' / 'P2400.png'),dpi=96)
plt.show()

Figure, Axes = plt.subplots(1,1,dpi=DPI, figsize=(Size[0],Size[1]))
Axes.imshow(P1200_Array_Crop.astype('uint8'))
Axes.plot(SB_X, [1100,1100], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[0],SB_X[0]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[1],SB_X[1]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.annotate(str(SB_Length) + ' mm', (SB_Start+len(Text)*15, 1075), color=(1,0,0))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(DataPath / 'Pictures' / 'P1200.png'),dpi=96)
plt.show()

Rescale = exposure.adjust_gamma(P1200_Array_Crop,gamma=0.6,gain=1)

Figure, Axes = plt.subplots(1,1,dpi=DPI, figsize=(Size[0],Size[1]))
Axes.imshow(Rescale.astype('uint8'))
Axes.plot(SB_X, [1100,1100], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[0],SB_X[0]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.plot([SB_X[1],SB_X[1]], [1075,1125], color=(1,0,0), linewidth=5)
Axes.annotate(str(SB_Length) + ' mm', (SB_Start+len(Text)*15, 1075), color=(1,0,0))
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(str(DataPath / 'Pictures' / 'P1200.png'),dpi=96)
plt.show()