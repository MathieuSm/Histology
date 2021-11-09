
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
    Axes.plot(X, Y, color=(1, 0, 0, 0.5), label='Region')
    Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='centroid')
    Axes.plot(X0 + Ellipse_R[0, :], Y0 - Ellipse_R[1, :], color=(0, 1, 0), label='Fitted ellipse')
    # Axes.set_xlim([int((X0 + Ellipse_R[0, :].min())*0.9), int((X0 + Ellipse_R[0, :].max())*1.1)])
    # Axes.set_ylim([int((Y0 + Ellipse_R[1, :].min())*0.9), int((Y0 + Ellipse_R[1, :].max())*1.1)])
    Axes.set_ylim([0,ImageArray.shape[1]])
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
ImageArray = ImageArray[3839:3839+1181+1,4724:4724+1181+1]

SegmentedImage = sitk.ReadImage(ImageDirectory + 'SegTrainingImage.png')
SegmentedImageArray = sitk.GetArrayFromImage(SegmentedImage)

SegmentedOsteons = sitk.ReadImage(ImageDirectory + 'SegTrainingImage2.png')
OsteonsArray = sitk.GetArrayFromImage(SegmentedOsteons)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(SegmentedImageArray)
plt.axis('off')
plt.title('Segmented Image')
Axes.set_ylim([0,SegmentedImageArray.shape[0]])
plt.show()
plt.close(Figure)

CementLines = SegmentedImageArray[:,:,0].copy()
Harvesian = SegmentedImageArray[:, :, 1].copy()
Osteocytes = SegmentedImageArray[:,:,2].copy()
Osteons = OsteonsArray[:, :, 0].copy()
Threshold = 200
CementLines[CementLines < Threshold] = 0
CementLines[CementLines >= Threshold] = 1
Harvesian[Harvesian < Threshold] = 0
Harvesian[Harvesian >= Threshold] = 1
Osteocytes[Osteocytes < Threshold] = 0
Osteocytes[Osteocytes >= Threshold] = 1
Osteons[Osteons < Threshold] = 0
Osteons[Osteons >= Threshold] = 1


F0 = SegmentedImageArray[:,:,0] > 180
F1 = SegmentedImageArray[:,:,1] > 180
F2 = SegmentedImageArray[:,:,2] > 180
CementLines[F1] = 0
CementLines[F2] = 0
Harvesian[F0] = 0
Harvesian[F2] = 0
Osteocytes[F0] = 0
Osteocytes[F1] = 0

F1 = OsteonsArray[:,:,1] > 180
F2 = OsteonsArray[:,:,2] > 180
Osteons[F1] = 0
Osteons[F2] = 0

Segments = CementLines + Harvesian + Osteocytes

# Create custom color map
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
Axes.set_ylim([0,CementLines.shape[0]])
plt.show()
plt.close(Figure)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(ImageArray, cmap='gray')
Axes.imshow(OsteonsArray, cmap=ColorMap1)
plt.axis('off')
plt.title('Segmented osteons')
Axes.set_ylim([0,OsteonsArray.shape[0]])
plt.show()
plt.close(Figure)

Labels = measure.label(Osteons,connectivity=2)
RegionsProperties = measure.regionprops(Labels, ImageArray)
Properties = ('label','centroid','major_axis_length','minor_axis_length','orientation','euler_number')
PropertiesTable = pd.DataFrame(measure.regionprops_table(Labels,properties=Properties))
Columns = ['Region','X','Y','R1','R2','Alpha','Euler']
PropertiesTable.columns = Columns
a = PropertiesTable['R1']
b = PropertiesTable['R2']
PropertiesTable['Perimeter'] = np.pi * np.sqrt(2*(a**2 + b**2))
PropertiesTable['Alpha'] = PropertiesTable['Alpha'] / np.pi * 180

Filter = PropertiesTable['Euler'] < 0
Regions = PropertiesTable[Filter]['Region']
for Region in range(len(Regions)+1):
    Y, X = np.where(Labels == Region)
    PlotROI(RegionsProperties, Region, X, Y)

ROI2Drop = [0,1,7]
FilteredProperties = PropertiesTable.drop(ROI2Drop)
FilteredProperties['Alpha'].round(-1).value_counts()


Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(FilteredProperties['Perimeter'], marker='o',linestyle='none',color=(1,0,0),fillstyle='none')
plt.show()
plt.close(Figure)

FilteredProperties['Perimeter'].mean()

Targets = pd.DataFrame({'Perimeter':3000,
                        'Euler Number': 1,
                        'Orientation': 1}, index=[0])

R = 0
Region = RegionsProperties[R]
Y_s, X_s = Region.slice
ImageRegion = ImageArray[Y_s.start:Y_s.stop,X_s.start:X_s.stop] * Region.filled_image

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(ImageRegion, cmap='binary')
plt.axis('off')
plt.title('Region')
Axes.set_ylim([0,ImageRegion.shape[0]])
plt.show()
plt.close(Figure)


HarvesianRegion = Harvesian[Y_s.start:Y_s.stop, X_s.start:X_s.stop]

CanaliculiProperties = measure.regionprops(HarvesianRegion, ImageRegion)

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




CLine, Osteocyte, Matrix, Harvesian, Osteon = np.unique(GrayScaleSegments)

CementLines = np.zeros(GrayScaleSegments.shape)
CementLines[GrayScaleSegments == CLine] = 1

Canaliculis = np.zeros(GrayScaleSegments.shape)
Canaliculis[GrayScaleSegments == Harvesian] = 1

Osteocytes = np.zeros(GrayScaleSegments.shape)
Osteocytes[GrayScaleSegments == Osteocyte] = 1