
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd


def PlotROI(RegionsProperties, ROINumber):
    M = 0
    RegionProperties = RegionsProperties[M]

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
    Axes.plot(X, Y, color=(0, 1, 0), label='Contour')
    Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='centroid')
    Axes.plot(X0 + Ellipse_R[0, :], Y0 + Ellipse_R[1, :], color=(1, 0, 0), label='Fitted ellipse')
    Axes.set_xlim([0, Image_ROI.shape[1]])
    Axes.set_ylim([0, Image_ROI.shape[0]])
    plt.title('ROI ' + str(ROINumber))
    plt.axis('off')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    plt.show()
    plt.close(Figure)

    return

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Scripts/PCNN/Training/'
Files = os.listdir(ImageDirectory)
Files.sort()

Image = sitk.ReadImage(ImageDirectory + Files[-1])
ImageArray = sitk.GetArrayFromImage(Image)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(ImageArray,cmap='gray')
plt.axis('off')
plt.title('Image')
Axes.set_ylim([0,ImageArray.shape[0]])
plt.show()
plt.close(Figure)

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

