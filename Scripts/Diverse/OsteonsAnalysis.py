import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


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
    Axes.imshow(Array,cmap='bone')
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
Image_Array = PlotImage(Image)

# Crop image (size in um) at random position
X_Crop_Size = 100
Y_Crop_Size = 100
Crop_X = round(X_Crop_Size / Image.GetSpacing()[1] + 0.5)
Crop_Y = round(Y_Crop_Size / Image.GetSpacing()[0] + 0.5)
Random_X = round(np.random.uniform(0,Image.GetSize()[1]-Crop_X-1))
Random_Y = round(np.random.uniform(0,Image.GetSize()[0]-Crop_Y-1))
Cropping = (Image.GetSize()[0]-Random_Y-Crop_Y,Image.GetSize()[1]-Random_X-Crop_X)
SubImage = sitk.Crop(Image,(Random_Y,Random_X),Cropping)
SubImage_Array = PlotImage(SubImage)

GrayImage_Array = SubImage_Array[:,:,2].copy()
PlotArray(GrayImage_Array,Image.GetSpacing())
GrayImage = sitk.GetImageFromArray(GrayImage_Array)
GrayImage.SetSpacing(Image.GetSpacing())

## Segment result image
Segmented_Array = GrayImage_Array.copy()
Otsu_Filter = sitk.OtsuThresholdImageFilter()
Otsu_Filter.SetInsideValue(0)
Otsu_Filter.SetOutsideValue(1)
Segmentation = Otsu_Filter.Execute(GrayImage)
Threshold = Otsu_Filter.GetThreshold()
Segmented_Array[Segmented_Array <= Threshold] = 0
Segmented_Array[Segmented_Array > 0] = 1
PlotArray(Segmented_Array,Image.GetSpacing())


