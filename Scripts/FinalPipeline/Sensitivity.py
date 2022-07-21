#!/usr/bin/env python3

import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from skimage import morphology, measure

plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    Script used to determine size and number of ROI for the cement lines analysis
    
    Based on recomendation from:
    Grimal, Q., Raum, K., Gerisch, A., &#38; Laugier, P. (2011)
    A determination of the minimum sizes of representative volume elements
    for the prediction of cortical bone elastic properties
    Biomechanics and Modeling in Mechanobiology (6), 925–937
    https://doi.org/10.1007/s10237-010-0284-9

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: July 2022
    """


class ParametersClass:

    def __init__(self, ImageNumber, Threshold=0.88, SubArea=[[1800, 2200], [7800, 8200]]):
        self.N = ImageNumber
        self.Directory = Path.cwd() / 'Tests/Osteons/Sensitivity/'
        self.Threshold = Threshold
        self.SubArea = SubArea

def PrintTime(Tic, Toc):
    """
    Print elapsed time in seconds to time in HH:MM:SS format
    :param Tic: Actual time at the beginning of the process
    :param Toc: Actual time at the end of the process
    """

    Delta = Toc - Tic

    Hours = np.floor(Delta / 60 / 60)
    Minutes = np.floor(Delta / 60) - 60 * Hours
    Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

    print('Process executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

def PixelSize(Image, Length, Plot=False):

    """
    Determine physical size of a pixel
    :param Image: Image region containing the scalebar as numpy array r x c x 3
    :param Length: Physical length of the scalebar as integer
    :param Plot: Plot intermediate results, boolean value
    :return: Physical size of a pixel
    """

    Tic = time.time()
    print('Compute physical pixel size ...')

    Filter1 = Image[:,:,0] < 100
    Filter2 = Image[:,:,1] < 100
    Filter3 = Image[:,:,2] < 100

    Bin = np.zeros(Filter1.shape,'int')
    Bin[Filter1 & Filter2 & Filter3] = 1

    if Plot:
        Figure, Axis = plt.subplots(1,1)
        Axis.imshow(Bin,cmap='binary')
        plt.show()

    RegionProps = measure.regionprops(Bin)[0]
    Pixels = RegionProps.coords[:,1].max() - RegionProps.coords[:,1].min()
    PixelLength = Length / Pixels

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return PixelLength

def ReadImage(Plot=True):

    # Read image and plot it
    Directory = Parameters.Directory
    DataFrame = pd.read_csv(str(Directory / 'Data.csv'))
    SampleData = DataFrame.loc[Parameters.N]
    Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '_Seg.jpg')
    Image = sitk.GetArrayFromImage(sitk.ReadImage(str(Directory / Name)))[:, :, :3]

    if Plot:
        Shape = np.array(Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Store name, image, and pixel length in parameters class
    Parameters.Name = Name[:-7]
    Parameters.Image = Image

    if Parameters.Name[:5] == '418RM':
        Parameters.PixelLength = PixelSize(Parameters.Image[9400:-400, 12500:-300], 2000, Plot=True)
    else:
        Parameters.PixelLength = 1.0460251046025104  # Computed with 418 RM

def SegmentBone(Image, Plot=False, SubArea=None):

    """
    Segment bone structure
    :param Image: RGB numpy array dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Labelled bone image
    """

    Tic = time.time()
    print('Segment bone area ...')

    if not SubArea:
        SubArea = [[0, 1], [0, 1]]

    # Mark areas where there is bone
    Filter1 = Image[:, :, 0] < 190
    Filter2 = Image[:, :, 1] < 190
    Filter3 = Image[:, :, 2] < 235
    Bone = Filter1 & Filter2 & Filter3

    if Plot == 'Full':
        Shape = np.array(Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    elif Plot == 'Sub':
        Shape = np.array([SubArea[1][1]-SubArea[1][0], SubArea[0][1]-SubArea[0][0]]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone[SubArea[0][0]:SubArea[0][1],
                         SubArea[1][0]:SubArea[1][1]], cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Erode and dilate to remove small bone parts
    Disk = morphology.disk(2)
    Dilated = morphology.binary_dilation(Bone, Disk)
    Bone = morphology.binary_erosion(Dilated, Disk)

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Bone

def ValidArea(Bone, GridSize, Threshold, Plot=False):

    """
    Define valid area according to a given BV/TV threshold and a given grid size
    :param Bone: Segmented bone
    :param GridSize: Grid size to evaluate BV/TV
    :param Threshold: Minimum BV/TV to consider area as valid
    :param Plot: Plot valid area
    :param Image: Add initial image on the plot
    :return: Area with a sufficiently high BV/TV
    """

    Tic = time.time()
    print('Define valid area ...')

    NPoints = np.ceil(np.array(Bone.shape) / GridSize)
    XPoints = np.arange(NPoints[1], dtype='int') * GridSize
    YPoints = np.arange(NPoints[0], dtype='int') * GridSize
    XPoints = np.append(XPoints, Bone.shape[1])
    YPoints = np.append(YPoints, Bone.shape[0])
    XGrid, YGrid = np.meshgrid(XPoints, YPoints)

    # Compute subregion bone volume fraction
    ValidArea = np.zeros(Bone.shape).astype('int')

    for i in range(int(NPoints[1])):
        for j in range(int(NPoints[0])):
            SubRegion = Bone[YGrid[j, i]:YGrid[j + 1, i], XGrid[j, i]:XGrid[j, i + 1]]

            if SubRegion.sum() / SubRegion.size > Threshold:
                ValidArea[YGrid[j, i]:YGrid[j+1, i], XGrid[j, i]:XGrid[j, i+1]] = 1

    if Plot:
        Shape = np.array(Parameters.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Parameters.Image)
        Axis.imshow(ValidArea, cmap='Greens', alpha=1/3)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return ValidArea

def ExtractSkeleton(Image, Plot=False):
    """
    Extract skeleton of manually segmented image
    :param Image: Numpy image dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Skeleton of the segmentation
    """

    Tic = time.time()
    print('\nExtract manual segmentation skeleton ...')

    Filter1 = Image[:, :, 0] > 100
    Filter2 = Image[:, :, 1] < 90
    Filter3 = Image[:, :, 2] < 150

    Bin = np.zeros(Filter1.shape)
    Bin[Filter1 & Filter2 & Filter3] = 1

    # Dilate to link extracted segmentation
    Disk = morphology.disk(5)
    BinDilate = morphology.binary_dilation(Bin, Disk)

    # Skeletonize to obtain 1 pixel thickness
    Skeleton = morphology.skeletonize(BinDilate)

    if Plot:
        Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
        Axis.imshow(Skeleton, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Skeleton

def CMDensities(GridSize, Bone, Skeleton, Plot=False):

    """
    Compute cement line densities for a given grid size
    :param GridSize: Grid size to compute CM densities
    :param Bone: Segmented bone in the valid area
    :param Skeleton: CM skeleton extracted from manual segmentation
    :param Image: Original image to plot CM densities over it
    :return: Cement line densities of the grid
    """

    Tic = time.time()
    print('Compute cement line densities ...')

    NPoints = np.ceil(np.array(Skeleton.shape) / GridSize)
    XPoints = np.arange(NPoints[1], dtype='int') * GridSize
    YPoints = np.arange(NPoints[0], dtype='int') * GridSize
    XPoints = np.append(XPoints, Skeleton.shape[1])
    YPoints = np.append(YPoints, Skeleton.shape[0])
    XGrid, YGrid = np.meshgrid(XPoints, YPoints)

    # Compute subregion cement line density
    BVTV = np.zeros(XGrid.shape)
    Densities = np.zeros(XGrid.shape)

    for i in range(int(NPoints[1])):
        for j in range(int(NPoints[0])):
            SubRegion = Skeleton[YGrid[j, i]:YGrid[j + 1, i], XGrid[j, i]:XGrid[j, i + 1]]
            SubBone = Bone[YGrid[j, i]:YGrid[j + 1, i], XGrid[j, i]:XGrid[j, i + 1]]

            BVTV[j, i] = SubBone.sum() / SubBone.size

            if SubBone.sum() > 0.01:
                Densities[j, i] = SubRegion.sum() / SubBone.sum()
            else:
                Densities[j, i] = 0

    if Plot:
        Shape = np.array(Parameters.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Parameters.Image)
        Axis.pcolormesh(XGrid + GridSize / 2, YGrid + GridSize / 2, Densities, cmap='Greens', alpha=0.5)
        Axis.set_xlim([0, Parameters.Image.shape[1]])
        Axis.set_ylim([Parameters.Image.shape[0], 0])
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Densities

def CollectDensities(PhysicalSizes):

    Bone = SegmentBone(Parameters.Image, Plot='Full')

    Skeleton = ExtractSkeleton(Parameters.Image, Plot=True)

    # Store results for different densities in data frame
    DensityData = pd.DataFrame()

    for PhysicalSize in PhysicalSizes:

        GridSize = int(round(PhysicalSize / Parameters.PixelLength))
        Valid = ValidArea(Bone, GridSize, Parameters.Threshold, Plot=True)
        Densities = CMDensities(GridSize,Bone*Valid,Skeleton*Valid)

        if DensityData.size == 0:
            DensityData[PhysicalSize] = Densities.flatten()
        else:
            DensityData[PhysicalSize] = np.nan
            DensityData.loc[np.arange(Densities.size, dtype='int'), PhysicalSize] = Densities.flatten()

    DensityData = DensityData.replace({0: np.nan})

    Figure, Axis = plt.subplots(1, 1)
    for Column in DensityData.columns:
        Axis.boxplot(DensityData[Column].dropna(), vert=True, widths=int(Column / 5),
                     positions=[Column],
                     showmeans=False, meanline=True,
                     capprops=dict(color=(0, 0, 0)),
                     boxprops=dict(color=(0, 0, 0)),
                     whiskerprops=dict(color=(0, 0, 0), linestyle='--'),
                     flierprops=dict(color=(0, 0, 0)),
                     medianprops=dict(color=(1, 0, 0)),
                     meanprops=dict(color=(0, 1, 0)))
    Axis.plot([], linestyle='none', marker='o', fillstyle='none', color=(1, 0, 0), label='Data')
    Axis.plot([], color=(0, 0, 1), label='Median')
    Axis.set_xscale('log')
    Axis.set_xlabel('Grid Size [$\mu$m]')
    Axis.set_ylabel('Density [-]')
    plt.subplots_adjust(left=0.25, right=0.75)
    plt.show()

    # Save densities
    DensityData = DensityData.replace({np.nan: 0})
    DensityData.to_csv(str(Parameters.Directory / str(Parameters.Name + 'Densities.csv')), index=False)

    return

def RandCoords(Coords, ROINumber, TotalNROIs):

    XCoords, YCoords = Coords

    XRange = XCoords.max() - XCoords.min()
    Width = XRange / (TotalNROIs + 1)
    RandX = int((ROINumber + 1) * XRange / (TotalNROIs + 1) + np.random.randn() * Width**(1 / 2))
    YCoords = YCoords[XCoords == RandX]
    YRange = YCoords.max() - YCoords.min()
    RandY = int(np.median(YCoords) + np.random.randn() * (Width * YRange/XRange)**(1 / 2))

    return [RandX, RandY]

def ExtractROIs(Bone, XCoords, YCoords, ROISize, NROIs=1, Plot=False, ROIsPlot=False):

    Tic = time.time()
    print('\nBegin ' + str(NROIs) + ' ROIs extraction ...')

    ROIs = np.zeros((NROIs,ROISize,ROISize,3)).astype('int')
    BoneROIs = np.zeros((NROIs,ROISize,ROISize)).astype('int')
    Xs = np.zeros((NROIs,2)).astype('int')
    Ys = np.zeros((NROIs,2)).astype('int')

    for i in range(NROIs):
        RandX, RandY = RandCoords([XCoords, YCoords], i, NROIs)
        X1, X2 = RandX - int(ROISize / 2), RandX + int(ROISize / 2)
        Y1, Y2 = RandY - int(ROISize / 2), RandY + int(ROISize / 2)
        BoneROI = Bone[Y1:Y2, X1:X2]
        BVTV = BoneROI.sum() / BoneROI.size

        j = 0
        while BVTV < Parameters.Threshold and j < 100:
            RandX, RandY = RandCoords([XCoords, YCoords], i, NROIs)
            X1, X2 = RandX - int(ROISize / 2), RandX + int(ROISize / 2)
            Y1, Y2 = RandY - int(ROISize / 2), RandY + int(ROISize / 2)
            BoneROI = Bone[Y1:Y2, X1:X2]
            BVTV = BoneROI.sum() / BoneROI.size
            j += 1
            if j == 100:
                print('No ROI found after 100 iterations')

        ROIs[i] += Parameters.Image[Y1:Y2, X1:X2]
        BoneROIs[i] += Bone[Y1:Y2, X1:X2]
        Xs[i] += [X1, X2]
        Ys[i] += [Y1, Y2]

        if ROIsPlot:
            Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
            Axis.imshow(ROIs[i])
            Axis.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            plt.show()

    if Plot:
        Shape = np.array(Parameters.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Parameters.Image)

        for i in range(len(Xs)):
            Axis.plot([Xs[i,0], Xs[i,1]], [Ys[i,0], Ys[i,0]], color=(1, 0, 0))
            Axis.plot([Xs[i,1], Xs[i,1]], [Ys[i,0], Ys[i,1]], color=(1, 0, 0))
            Axis.plot([Xs[i,1], Xs[i,0]], [Ys[i,1], Ys[i,1]], color=(1, 0, 0))
            Axis.plot([Xs[i,0], Xs[i,0]], [Ys[i,1], Ys[i,0]], color=(1, 0, 0))
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic,Toc)

    return ROIs, BoneROIs, Xs, Ys


Parameters = ParametersClass(9)
ReadImage()

# Compute cement lines density for multiple grid size
PhysicalSizes = [100, 200, 500, 1000, 1500, 2000]  # Grid size in um
CollectDensities(PhysicalSizes)


# Collect densities data to compare between samples
DataFrame = pd.read_csv(str(Parameters.Directory / 'Data.csv'))
Samples = DataFrame[DataFrame['Cortex'] == 'Lateral']
Data100 = pd.DataFrame()
Data200 = pd.DataFrame()
Data500 = pd.DataFrame()
Data1000 = pd.DataFrame()
Data1500 = pd.DataFrame()
Data2000 = pd.DataFrame()
Datas = [Data100, Data200, Data500, Data1000, Data1500, Data2000]

for Index in Samples.index:
    SampleData = Samples.loc[Index]
    Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '_Densities.csv')
    Data = pd.read_csv(str(Parameters.Directory / Name))
    Data = Data.replace({0: np.nan})

    if Index == 0:
        for i in range(len(Datas)):
            Datas[i][Name[:5]] = Data[str(PhysicalSizes[i])]

    else:
        for i in range(len(Datas)):
            Datas[i][Name[:5]] = np.nan
            Datas[i][Name[:5]] = Data[str(PhysicalSizes[i])]

# Plot densities for a given sample
i = 0
Figure, Axis = plt.subplots(1, 1)
for Data in Datas:
    Axis.boxplot(Data['418RM'].dropna(), vert=True, widths=int(PhysicalSizes[i] / 5),
                 positions=[PhysicalSizes[i]],
                 showmeans=False, meanline=True,
                 capprops=dict(color=(0, 0, 0)),
                 boxprops=dict(color=(0, 0, 0)),
                 whiskerprops=dict(color=(0, 0, 0), linestyle='--'),
                 flierprops=dict(color=(0, 0, 0)),
                 medianprops=dict(color=(1, 0, 0)),
                 meanprops=dict(color=(0, 1, 0)))
    i += 1
Axis.plot([], linestyle='none', marker='o', fillstyle='none', color=(1, 0, 0), label='Data')
Axis.plot([], color=(0, 0, 1), label='Median')
Axis.set_xscale('log')
Axis.set_xlabel('Grid Size [$\mu$m]')
Axis.set_ylabel('Density [-]')
Axis.set_ylim([-0.0035, 0.065])
plt.subplots_adjust(left=0.25, right=0.75)
plt.show()

# Plot densities for a given size
j = 0
for Data in Datas:
    Figure, Axis = plt.subplots(1, 1)
    i = 0
    for Column in Data.columns:
        Axis.boxplot(Data[Column].dropna(), vert=True, widths=0.35,
                     positions=[i],
                     showmeans=False, meanline=True,
                     capprops=dict(color=(0, 0, 0)),
                     boxprops=dict(color=(0, 0, 0)),
                     whiskerprops=dict(color=(0, 0, 0), linestyle='--'),
                     flierprops=dict(color=(0, 0, 0)),
                     medianprops=dict(color=(1, 0, 0)),
                     meanprops=dict(color=(0, 1, 0)))
        i += 1
    Axis.set_xticks(np.arange(len(Data.columns)), Data.columns)
    if j > 1:
        Axis.set_ylim([-0.001, 0.021])
    j += 1
    Axis.set_xlabel('Sample [-]')
    Axis.set_ylabel('Density [-]')
    plt.subplots_adjust(left=0.25, right=0.75)
    plt.show()

# Simulate random zone selection
Size = 2
Data = Datas[Size]
GridSize = PhysicalSizes[Size]
MeansData = pd.DataFrame()
ROINumber = 10

for i in Samples.index:

    # Open image to segment
    Parameters = ParametersClass(i)
    ReadImage(Parameters)
    Shape = Parameters.Image.shape

    Sample = Samples.loc[i]
    SampleName = str(Sample['Sample']) + Sample['Side'][0] + Sample['Cortex'][0]

    M = Data[SampleName].mean()
    S = Data[SampleName].std()

    # Segment bone and extract coordinate
    Bone = SegmentBone(Parameters.Image, Plot=False)
    Y, X = np.where(Bone)

    # Set ROI pixel size
    ROISize = int(round(GridSize / Parameters.PixelLength))
    if np.mod(ROISize,2) == 1:
        ROISize = ROISize + 1

    # Filter positions too close to the border
    F1 = X > ROISize / 2
    F2 = X < Bone.shape[1] - ROISize / 2
    FilteredX = X[F1 & F2]
    FilteredY = Y[F1 & F2]

    F1 = FilteredY > ROISize / 2
    F2 = FilteredY < Bone.shape[0] - ROISize / 2
    FilteredY = FilteredY[F1 & F2]
    FilteredX = FilteredX[F1 & F2]

    # Initialize loop for ROIs selection
    BVTV = np.zeros((ROINumber, ROINumber))
    CMDensity = np.zeros((ROINumber, ROINumber))

    for j in range(1, ROINumber+1):

        # Extract random ROI and verify validity
        ROIs, BoneROIs, Xs, Ys = ExtractROIs(Bone, FilteredX, FilteredY, ROISize, NROIs=j, Plot=False)

        for k in range(j):
            ROISkeleton = ExtractSkeleton(ROIs[k], Plot=False)
            BVTV[j - 1, k] = BoneROIs[k].sum() / BoneROIs[k].size
            CMDensity[j - 1, k] = ROISkeleton.sum() / BoneROIs[k].sum()

    Filter = BVTV < Parameters.Threshold
    BVTV[Filter] = np.nan
    CMDensity[Filter] = np.nan

    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(BVTV.flatten(), CMDensity.flatten(), color=(1, 0, 0), marker='o', fillstyle='none', linestyle='none',
              label='Tests')
    Axis.plot([np.nanmin(BVTV), np.nanmax(BVTV)], [M, M], color=(0, 0, 0), linestyle='--', label='Grid Mean')
    Axis.fill_between([np.nanmin(BVTV), np.nanmax(BVTV)], [M + S, M + S], [M - S, M - S], color=(0, 0, 0, 0.2),
                      label='Standard deviation')
    Axis.set_xlabel('BV/TV (-)')
    Axis.set_ylabel('Density (-)')
    plt.legend()
    plt.show()

    # Means
    Means = np.nanmean(CMDensity, axis=1)

    Figure, Axis = plt.subplots(1, 1)
    for j in range(ROINumber):
        Axis.plot(np.repeat(j + 1, ROINumber), CMDensity[j, :] / M - 1, color=(1, 0, 0), marker='o',
                  fillstyle='none', linestyle='none')
    Axis.plot([], color=(1, 0, 0), marker='o', fillstyle='none', linestyle='none', label='Tests')
    Axis.plot(np.arange(1, ROINumber + 1), Means / M - 1, color=(0, 0, 1), linestyle='--', label='ROIs Mean')
    Axis.plot([1, ROINumber], [0, 0], color=(0, 0, 0), linestyle='--', label='Grid Mean')
    Axis.fill_between([1, ROINumber], [S / M, S / M], [-S / M, -S / M], color=(0, 0, 0, 0.15),
                      label='Standard deviation')
    Axis.set_xlabel('ROI number (-)')
    Axis.set_ylabel('Density relative error (-)')
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.17))
    plt.show()

    MeansData[SampleName] = Means / M - 1
print('Done!')

Colors = [(1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0)]

Figure, Axis = plt.subplots(1, 1)
for Index in range(len(MeansData.columns)):
    Axis.plot(np.arange(ROINumber)+1,MeansData[MeansData.columns[Index]].abs(), color=Colors[Index])
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Relative error [-]')
Axis.set_xlim([0, ROINumber+1])
# Axis.set_ylim([0,1])
plt.show()

# Perform mixed-effect linear regression
plt.rc('font', size=12)

def PlotRegressionResults(Model, Data, ROINumber):

    ## Get data from the model
    Y_Obs = Model.model.endog
    N = int(Model.nobs)
    C = np.matrix(Model.cov_params())
    X = np.matrix(Model.model.exog)
    Y_Fit = np.array(Model.params[0] + Model.params[1] * X[:, 1]).reshape(len(X))

    if not C.shape[0] == X.shape[1]:
        C = C[:-1, :-1]

    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Model.predict()
    RSS = np.sum(E**2)
    SE = np.sqrt(RSS / Model.df_resid)
    TSS = np.sum((Model.model.endog - Model.model.endog.mean())**2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    ## Plot
    Colors = [(1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0)]
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5 * 1.5, 4.5 * 1.5))
    for i in range(len(Data['Groups'].unique())):
        y = Data[Data['Groups'] == Data['Groups'].unique()[i]]['y'].values
        Axes.plot(np.arange(ROINumber)+1, y, label=Data['Groups'].unique()[i], color=Colors[i])
    Axes.plot(np.arange(10)+1,Y_Fit[:10], color=(0, 0, 0), linestyle='--', label='Fit')
    Axes.set_xlabel('Number of ROIs [-]')
    Axes.set_ylabel('Relative error [-]')
    # Axes.annotate(r'N Groups : ' + str(len(Data.groupby('Groups'))), xy=(0.65, 0.925), xycoords='axes fraction')
    # Axes.annotate(r'N Points : ' + str(N), xy=(0.65, 0.86), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.85, 0.65), xycoords='axes fraction')
    Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.85, 0.585), xycoords='axes fraction')
    Axes.set_xlim([0, ROINumber+1])
    plt.legend(ncol=1, loc='upper right')
    plt.show()

    return R2, SE

Data2Fit = pd.DataFrame()
Data2Fit['x'] = 1 / (np.arange(len(MeansData)) + 1)
Data2Fit['y'] = np.abs(MeansData[MeansData.columns[0]])
Data2Fit['Groups'] = MeansData.columns[0]

GroupedData = Data2Fit.copy()
for i in range(1, 4):
    Data2Fit['y'] = np.abs(MeansData[MeansData.columns[i]])
    Data2Fit['Groups'] = MeansData.columns[i]
    GroupedData = pd.concat([GroupedData, Data2Fit], axis=0, ignore_index=True)

LMM = smf.mixedlm('y ~ x', data=GroupedData.dropna(), groups=GroupedData.dropna()['Groups']).fit(reml=True)
PlotRegressionResults(LMM, GroupedData, ROINumber)
LMM.summary()
LMM.params[0]
LMM.params[1]

# Fit results
MedialCurve0500 = 0.04329550911421737 + 0.28248717785898364 / (np.arange(ROINumber) + 1)
MedialCurve1000 = 0.01970149959782633 + 0.1461692814902478 / (np.arange(ROINumber) + 1)
MedialCurve1500 = 0.03467395522706307 + 0.03713466489600956 / (np.arange(ROINumber) + 1)
MedialCurve2000 = 0.022140427755325202 + 0.013525341908709876 / (np.arange(ROINumber) + 1)
MedialCurves = [MedialCurve0500, MedialCurve1000, MedialCurve1500, MedialCurve2000]

LateralCurve100 = 0.27331467423722555 + 0.3656847636005985 / (np.arange(ROINumber) + 1)
LateralCurve200 = 0.10139006402062499 + 0.27522131834720104 / (np.arange(ROINumber) + 1)
LateralCurve500 = 0.1687137707060012 + 0.04178297189997549 / (np.arange(ROINumber) + 1)
LateralCurves = [LateralCurve100, LateralCurve200, LateralCurve500]

Colors = [(1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0)]
PhysicalSizes = [100, 200, 500, 1000, 1500, 2000]  # Grid size in um

Figure, Axis = plt.subplots(1, 1)
for Index in range(len(MedialCurves)):
    Axis.plot(np.arange(ROINumber)+1, MedialCurves[Index], color=Colors[Index],
              label=str(PhysicalSizes[Index + 2]))
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Fitted Curve [-]')
Axis.set_xlim([0, ROINumber+1])
Axis.set_ylim([0, 0.7])
plt.legend()
plt.show()

Figure, Axis = plt.subplots(1, 1)
for Index in range(len(LateralCurves)):
    Axis.plot(np.arange(ROINumber)+1, LateralCurves[Index], color=Colors[Index], label=str(PhysicalSizes[Index]))
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Fitted Curve [-]')
Axis.set_xlim([0, ROINumber+1])
Axis.set_ylim([0, 0.7])
plt.legend()
plt.show()