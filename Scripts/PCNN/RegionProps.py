
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure, feature, exposure
import pandas as pd
import matplotlib as mpl


def PlotROI(RegionsProperties, ROINumber, X, Y):
    # M = 0
    M = ROINumber
    RegionProperties = RegionsProperties[M-1]

    Y0, X0 = RegionProperties.centroid
    R1 = RegionProperties.major_axis_length * 0.5
    R2 = RegionProperties.minor_axis_length * 0.5
    OrientationAngle = RegionProperties.orientation

    Radians = np.linspace(0, 2 * np.pi, 100)
    Ellipse = np.array([R2 * np.cos(Radians), R1 * np.sin(Radians)])
    R = np.array([[np.cos(OrientationAngle), -np.sin(OrientationAngle)],
                  [np.sin(OrientationAngle), np.cos(OrientationAngle)]])
    Ellipse_R = np.dot(R, Ellipse)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(ImageArray, cmap='gray')
    Axes.plot(X, Y, color=(1, 0, 0, 0.2), label='Region')
    Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='centroid')
    Axes.plot(X0 + Ellipse_R[0, :], Y0 + Ellipse_R[1, :], color=(0, 1, 0), label='Fitted ellipse')
    Axes.set_xlim([int((X0 + Ellipse_R[0, :].min())*0.9), int((X0 + Ellipse_R[0, :].max())*1.1)])
    Axes.set_ylim([int((Y0 + Ellipse_R[1, :].min())*0.9), int((Y0 + Ellipse_R[1, :].max())*1.1)])
    plt.title('ROI ' + str(ROINumber))
    plt.axis('off')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    plt.show()
    plt.close(Figure)

    return
def RGB2Gray(RGBImage):
    """
    This function convert color image to gray scale image
    based on matplotlib linear approximation
    """

    R, G, B = RGBImage[:,:,0], RGBImage[:,:,1], RGBImage[:,:,2]
    Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    Normalized_Gray = Gray / Gray.max()

    return Normalized_Gray



desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Scripts/PCNN/Training/'
Files = [File for File in os.listdir(ImageDirectory) if not os.path.isdir(ImageDirectory+File)]
Files.sort()

Image = sitk.ReadImage(ImageDirectory + Files[0])
ImageArray = sitk.GetArrayFromImage(Image)

OtsuFilter = sitk.OtsuThresholdImageFilter()
OtsuFilter.SetInsideValue(1)
OtsuFilter.SetOutsideValue(0)
OtsuFilter.Execute(Image)
Best_Threshold = OtsuFilter.GetThreshold()

SegmentedImage = sitk.ReadImage(ImageDirectory + 'Fiji/ClassifiedImage.png')
SegmentedImageArray = sitk.GetArrayFromImage(SegmentedImage)
GrayScaleSegments = RGB2Gray(SegmentedImageArray)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(GrayScaleSegments,cmap='gray')
plt.axis('off')
plt.title('Image')
Axes.set_ylim([0,GrayScaleSegments.shape[0]])
plt.show()
plt.close(Figure)



CementLines = SegmentedImageArray[:,:,0].copy()
Canaliculi = SegmentedImageArray[:,:,1].copy()
Threshold = 200
CementLines[CementLines < Threshold] = 0
CementLines[CementLines >= Threshold] = 1
Canaliculi[Canaliculi < Threshold] = 0
Canaliculi[Canaliculi >= Threshold] = 1

F0 = SegmentedImageArray[:,:,0] > 180
F1 = SegmentedImageArray[:,:,1] > 180
F2 = SegmentedImageArray[:,:,2] > 180
CementLines[F1] = 0
CementLines[F2] = 0
Canaliculi[F0] = 0
Canaliculi[F2] = 0

# Create custom color map
C = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
ColorMap = mpl.colors.ListedColormap(C)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(SegmentedImageArray,cmap='gray')
Axes.imshow(Canaliculi*2 + CementLines, cmap=ColorMap, vmin=0.5, vmax=1.5)
Axes.axis('off')
Axes.set_xlim([4724,4724+1181])
Axes.set_ylim([3839,3839+1181])
# Axes.set_ylim([0,Segments.shape[0]])
plt.show()
plt.close(Figure)

Labels = measure.label(CementLines,connectivity=2)
RegionsProperties = measure.regionprops(Labels, ImageArray)


R = 0
Region = RegionsProperties[R]
Y_s, X_s = Region.slice
ImageRegion = ImageArray[Y_s.start:Y_s.stop,X_s.start:X_s.stop] * Region.filled_image
CanaliculiRegion = Canaliculi[Y_s.start:Y_s.stop,X_s.start:X_s.stop]

CanaliculiProperties = measure.regionprops(CanaliculiRegion, ImageRegion)
Properties[0].filled_image*1

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(ImageRegion,cmap='gray')
Axes.imshow(CanaliculiRegion, cmap=ColorMap, vmin=0.25,vmax=0.75)
plt.axis('off')
plt.title('Region')
Axes.set_ylim([0,ImageRegion.shape[0]])
plt.show()
plt.close(Figure)

RegionContrast = exposure.equalize_hist(ImageRegion)

RegionEdges = feature.canny(RegionContrast,sigma=2)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(RegionEdges,cmap='gray')
plt.axis('off')
plt.title('Region')
Axes.set_ylim([0,RegionEdges.shape[0]])
plt.show()
plt.close(Figure)


RegionLabels = measure.label(SegmentedRegion,connectivity=2)
ObjectsProperties = measure.regionprops(RegionLabels,)


M = 3
Y, X = np.where(Labels == M)
PlotROI(RegionsProperties, M, X, Y)


ROIs = ImageArray.copy()
ROI_Files = [File for File in Files if File.endswith('csv')]

for ROINumber in [1,2,4,5,13,17]:
    Coordinates = pd.read_csv(ImageDirectory + ROI_Files[ROINumber])

    Image_ROI = np.zeros(ImageArray.shape)
    X = Coordinates.round().astype('int')['X'].values-1
    Y = Coordinates.round().astype('int')['Y'].values-1
    Image_ROI[Y,X] = 1

    RegionsProperties = measure.regionprops(Image_ROI.astype('int'))
    PlotROI(RegionsProperties,ROINumber)

