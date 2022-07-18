#!/usr/bin/env python3

import time
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
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


class Parameters:

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

def ReadImage(Params, Plot=True):

    # Read image and plot it
    Directory = Params.Directory
    DataFrame = pd.read_csv(str(Directory / 'Data.csv'))
    SampleData = DataFrame.loc[Params.N]
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
    Params.Name = Name[:-7]
    Params.Image = Image

    if Params.Name[:5] == '418RM':
        Params.PixelLength = PixelSize(Params.Image[9400:-400, 12500:-300], 2000, Plot=True)
    else:
        Params.PixelLength = 1.0460251046025104  # Computed with 418 RM

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

def ExtractSkeleton(Image, Plot=False, SubArea=None):

    """
    Extract skeleton of manually segmented image
    :param Image: Numpy image dim r x c x 3
    :param Plot: 'Full' or 'Sub' to plot intermediate results
    :param SubArea: Indices to plot smaller image of intermediate results
    :return: Skeleton of the segmentation
    """

    Tic = time.time()
    print('Extract manual segmentation skeleton ...')

    if not SubArea:
        SubArea = [[0, 1], [0, 1]]

    Filter1 = Image[:, :, 0] > 100
    Filter2 = Image[:, :, 1] < 90
    Filter3 = Image[:, :, 2] > 100

    Bin = np.zeros(Filter1.shape)
    Bin[Filter1 & Filter2 & Filter3] = 1

    # Dilate to link extracted segmentation
    Disk = morphology.disk(5)
    BinDilate = morphology.binary_dilation(Bin, Disk)

    if Plot == 'Full':
        Shape = np.array(Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(BinDilate, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    elif Plot == 'Sub':
        Shape = np.array([SubArea[1][1]-SubArea[1][0], SubArea[0][1]-SubArea[0][0]]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(BinDilate[SubArea[0][0]:SubArea[0][1],
                    SubArea[1][0]:SubArea[1][1]], cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Skeletonize to obtain 1 pixel thickness
    Skeleton = morphology.skeletonize(BinDilate)

    if Plot == 'Sub':
        Shape = np.array([SubArea[1][1] - SubArea[1][0], SubArea[0][1] - SubArea[0][0]]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Skeleton[SubArea[0][0]:SubArea[0][1],
                    SubArea[1][0]:SubArea[1][1]], cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Skeleton

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
        Shape = np.array(Params.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Params.Image)
        Axis.imshow(ValidArea, cmap='Greens', alpha=1/3)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return ValidArea

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
        Shape = np.array(Params.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Params.Image)
        Axis.pcolormesh(XGrid + GridSize / 2, YGrid + GridSize / 2, Densities, cmap='Greens', alpha=0.5)
        Axis.set_xlim([0, Params.Image.shape[1]])
        Axis.set_ylim([Params.Image.shape[0], 0])
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Densities

def CollectDensities(PhysicalSizes):

    Bone = SegmentBone(Params.Image, Plot='Full')

    Skeleton = ExtractSkeleton(Params.Image, Plot='Full')

    # Store results for different densities in data frame
    DensityData = pd.DataFrame()

    for PhysicalSize in PhysicalSizes:

        GridSize = int(round(PhysicalSize / Params.PixelLength))
        Valid = ValidArea(Bone, GridSize, Params.Threshold, Plot=True)
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
    DensityData.to_csv(str(Params.Directory / str(Params.Name + 'Densities.csv')), index=False)

    return


Params = Parameters(0)
ReadImage(Params)

# Compute cement lines density for multiple grid size
PhysicalSizes = [100, 200, 500, 1000, 1500, 2000]  # Grid size in um
CollectDensities(PhysicalSizes)


# Collect densities data to compare between samples
DataFrame = pd.read_csv(str(Params.Directory / 'Data.csv'))
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
    Data = pd.read_csv(str(Params.Directory / Name))
    Data = Data.replace({0: np.nan})

    if Index == 0:
        for i in range(len(Datas)):
            Datas[i][Name[:5]] = Data[str(PhysicalSizes[i])]

    else:
        for i in range(len(Datas)):
            Datas[i][Name[:5]] = np.nan
            Datas[i][Name[:5]] = Data[str(PhysicalSizes[i])]

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
    Params = Parameters(i)
    ReadImage(Params)
    Shape = Params.Image.shape

    Sample = Samples.loc[i]
    SampleName = str(Sample['Sample']) + Sample['Side'][0] + Sample['Cortex'][0]

    M = Data[SampleName].mean()
    S = Data[SampleName].std()

    Bone = SegmentBone(Params.Image, Plot='Full')
    Skeleton = ExtractSkeleton(Params.Image, Plot='Full')
    Valid = ValidArea(Bone, GridSize, Params.Threshold, Plot=True)

    # Random zone selection
    Size = int(round(GridSize / Params.PixelLength))
    BVTV = np.zeros((ROINumber, ROINumber))
    CMDensity = np.zeros((ROINumber, ROINumber))
    j = 1
    while j < ROINumber + 1:

        # To plot simulation results
        IShape = np.array(Params.Image.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(IShape[1], IShape[0]))
        Axis.imshow(Params.Image)

        for k in range(j):
            RandomXPos = int((k + 1) * Shape[1] / (j + 1) + np.random.randn() * Size / 4)
            RandomYPos = int(np.random.uniform(Size / 2 + 1, Shape[0] - Size / 2 - 1))

            X1, X2 = RandomXPos - int(Size / 2), RandomXPos + int(Size / 2)
            Y1, Y2 = RandomYPos - int(Size / 2), RandomYPos + int(Size / 2)

            SubRegion = (Skeleton * Valid)[Y1:Y2, X1:X2]
            SubBone = (Bone * Valid)[Y1:Y2, X1:X2]

            BoneFraction = SubBone.sum() / SubBone.size

            l = 0
            while BoneFraction < Params.Threshold:
                RandomXPos = int((k + 1) * Shape[1] / (j + 1) + np.random.randn() * Size / 2)
                RandomYPos = int(np.random.uniform(Size / 2 + 1, Shape[0] - Size / 2 - 1))

                X1, X2 = RandomXPos - int(Size / 2), RandomXPos + int(Size / 2)
                Y1, Y2 = RandomYPos - int(Size / 2), RandomYPos + int(Size / 2)

                SubRegion = (Skeleton * Valid)[Y1:Y2, X1:X2]
                SubBone = (Bone * Valid)[Y1:Y2, X1:X2]

                BoneFraction = SubBone.sum() / SubBone.size
                l += 1

                if l > 100:
                    print('No corresponding ROI found')
                    break

            Axis.plot([X1, X2], [Y1, Y1], color=(1, 0, 0))
            Axis.plot([X2, X2], [Y1, Y2], color=(1, 0, 0))
            Axis.plot([X2, X1], [Y2, Y2], color=(1, 0, 0))
            Axis.plot([X1, X1], [Y2, Y1], color=(1, 0, 0))

            BVTV[j - 1, k] = BoneFraction
            CMDensity[j - 1, k] = SubRegion.sum() / SubBone.sum()

        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()
        j += 1

    Filter = BVTV == 0
    BVTV[Filter] = np.nan
    CMDensity[Filter] = np.nan

    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(BVTV.flatten(), CMDensity.flatten(), color=(1, 0, 0), marker='o', fillstyle='none', linestyle='none',
              label='Tests')
    Axis.plot([0, np.nanmax(BVTV)], [M, M], color=(0, 0, 0), linestyle='--', label='Grid Mean')
    Axis.fill_between([0, np.nanmax(BVTV)], [M + S, M + S], [M - S, M - S], color=(0, 0, 0, 0.2),
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
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))
    plt.show()

    MeansData[SampleName] = Means / M - 1

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
    X = np.matrix(Model.model.exog)[:1000]
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
    Axes.plot(Y_Fit, color=(0, 0, 0), linestyle='--', label='Fit')
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
MedialCurve0500 = 0.0061242024263207595 + 0.3908892151023689 / (np.arange(ROINumber) + 1)
MedialCurve1000 = 0.05096539213717121 + 0.08095311891458373 / (np.arange(ROINumber) + 1)
MedialCurve1500 = 0.03726726993200507 + 0.03193766700713566 / (np.arange(ROINumber) + 1)
MedialCurve2000 = 0.05180894059961111 + 0.0004849689940706492 / (np.arange(ROINumber) + 1)
MedialCurves = [MedialCurve0500, MedialCurve1000, MedialCurve1500, MedialCurve2000]

LateralCurve0500 = 0.1382048195925299 + 0.7874752850317369 / (np.arange(ROINumber) + 1)
LateralCurve1000 = 0.14736325218702714 + 0.3527520514687957 / (np.arange(ROINumber) + 1)
LateralCurve1500 = 0.20875914756443886 + 0.01027383986497419 / (np.arange(ROINumber) + 1)
# LateralCurve2000 = 0.07483507798042248 + 0.9343716072050389 / (np.arange(ROINumber) + 1)
LateralCurves = [LateralCurve0500, LateralCurve1000, LateralCurve1500]

Colors = [(1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0)]
PhysicalSizes = [100, 200, 500, 1000, 1500, 2000]  # Grid size in um

Figure, Axis = plt.subplots(1, 1)
for Index in range(len(MedialCurves)):
    Axis.plot(np.arange(ROINumber)+1, MedialCurves[Index], color=Colors[Index],
              label=str(PhysicalSizes[Index + 2]))
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Fitted Curve [-]')
Axis.set_xlim([0, ROINumber+1])
# Axis.set_ylim([0, 1.5])
plt.legend()
plt.show()

Figure, Axis = plt.subplots(1, 1)
for Index in range(len(LateralCurves)):
    Axis.plot(np.arange(ROINumber)+1, LateralCurves[Index], color=Colors[Index], label=str(PhysicalSizes[Index + 2]))
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Fitted Curve [-]')
Axis.set_xlim([0, ROINumber+1])
Axis.set_ylim([0, 1.5])
plt.legend()
plt.show()