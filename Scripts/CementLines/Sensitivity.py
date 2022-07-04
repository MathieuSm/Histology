import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from skimage import morphology
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# Set path
CurrentDirectory = Path.cwd()
ImageDirectory = CurrentDirectory / 'Tests/Osteons/Sensitivity/'

DataFrame = pd.read_csv(str(ImageDirectory / 'Data.csv'))
N = 8
SampleData = DataFrame.loc[N]
Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '_Seg.jpg')

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / Name))
Array = sitk.GetArrayFromImage(Image)[:,:,:3]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array)
plt.show()

# Mark areas where there is bone
Filter1 = Array[:,:,0] < 190
Filter2 = Array[:,:,1] < 190
Filter3 = Array[:,:,2] < 235
Bone = Filter1 & Filter2 & Filter3

Area = [[3000,3400],[13800,14200]]
Figure, Axis = plt.subplots(1,1)
Axis.imshow(Bone[Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]],cmap='binary')
plt.show()

# Erode and dilate to remove small bone parts
Disk = morphology.disk(2)
Dilated = morphology.binary_dilation(Bone,Disk)
Bone = morphology.binary_erosion(Dilated,Disk)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Bone[Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]],cmap='binary')
plt.show()

# # Use scalebar to compute pixel size
# Figure, Axis = plt.subplots(1,1)
# Axis.imshow(Array[9400:-400,12500:-300])
# plt.show()
#
# ScaleRegion = Array[9400:-400,12500:-300]
# Filter1 = ScaleRegion[:,:,0] < 100
# Filter2 = ScaleRegion[:,:,1] < 100
# Filter3 = ScaleRegion[:,:,2] < 100
#
# Bin = np.zeros(Filter1.shape,'int')
# Bin[Filter1 & Filter2 & Filter3] = 1
#
# Figure, Axis = plt.subplots(1,1)
# Axis.imshow(Bin,cmap='binary')
# plt.show()
#
# RegionProps = measure.regionprops(Bin)[0]
# Pixels = RegionProps.coords[:,1].max() - RegionProps.coords[:,1].min()
# Length = 2000
# PixelLength = Length / Pixels
# print('Pixel size is ' + str(round(PixelLength,3)))
PixelLength = 1.0460251046025104 # Computed with 418 RM


# Filter image to extract manual segmentation
Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array[Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]])
plt.show()

Filter1 = Array[:,:,0] > 100
Filter2 = Array[:,:,1] < 90
Filter3 = Array[:,:,2] > 100

Bin = np.zeros(Filter1.shape)
Bin[Filter1 & Filter2 & Filter3] = 1

# Dilate to link extracted segmentation
Disk = morphology.disk(5)
BinDilate = morphology.binary_dilation(Bin,Disk)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(BinDilate[Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]],cmap='binary')
plt.show()

# Skeletonize to obtain 1 pixel thickness
Skeleton = morphology.skeletonize(BinDilate)
Figure, Axis = plt.subplots(1,1)
Axis.imshow(Skeleton[Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]],cmap='binary')
plt.show()

# Store results for different densities in data frame
DensityData = pd.DataFrame()

# Grid image to compute cement line density
PhysicalSizes = [100, 200, 500, 1000, 1500, 2000] # Grid size in um

for PhysicalSize in PhysicalSizes:

    Size = int(round(PhysicalSize / PixelLength))
    NPoints = np.ceil(np.array(Skeleton.shape) / Size)
    XPoints = np.arange(NPoints[1],dtype='int') * Size
    YPoints = np.arange(NPoints[0],dtype='int') * Size
    XPoints = np.append(XPoints, Skeleton.shape[1])
    YPoints = np.append(YPoints, Skeleton.shape[0])
    XGrid, YGrid = np.meshgrid(XPoints,YPoints)


    # Compute subregion cement line density
    i, j = 0, 0
    Densities = np.zeros(XGrid.shape)

    for i in range(int(NPoints[1])):
        for j in range(int(NPoints[0])):
            SubRegion = Skeleton[YGrid[j,i]:YGrid[j+1,i],XGrid[j,i]:XGrid[j,i+1]]
            SubBone = Bone[YGrid[j,i]:YGrid[j+1,i],XGrid[j,i]:XGrid[j,i+1]]

            # Figure, Axis = plt.subplots(1, 1)
            # Axis.imshow(SubRegion)
            # plt.show()

            if SubBone.sum() > 0:
                Densities[j,i] = SubRegion.sum() / SubBone.sum()
            else:
                Densities[j,i] = 0
            j += 1
        i += 1


    if DensityData.size == 0:
        DensityData[PhysicalSize] = Densities.flatten()
    else:
        DensityData[PhysicalSize] = np.nan
        DensityData.loc[np.arange(Densities.size,dtype='int'),PhysicalSize] = Densities.flatten()

    Shape = np.array(Array.shape[:-1])/1000
    Figure, Axis = plt.subplots(1,1,figsize=(Shape[1],Shape[0]))
    Axis.imshow(Array)
    Axis.pcolormesh(XGrid + Size/2, YGrid + Size/2, Densities, cmap='Greens', alpha=0.5)
    Axis.set_xlim([0,Array.shape[1]])
    Axis.set_ylim([Array.shape[0],0])
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()


DensityData = DensityData.replace({0:np.nan})
Column = 1000

Figure, Axis = plt.subplots(1,1)
for Column in DensityData.columns:
    # RandPos = np.random.normal(0, Column/50, len(DensityData[Column]))
    # Axis.plot(RandPos - RandPos.mean() + Column,DensityData[Column], linestyle='none',
    #           marker='o',fillstyle='none', color=(1,0,0))
    Axis.boxplot(DensityData[Column].dropna(), vert=True, widths=int(Column/5),
                 positions=[Column],
                 showmeans=False,meanline=True,
                 capprops=dict(color=(0,0,0)),
                 boxprops=dict(color=(0,0,0)),
                 whiskerprops=dict(color=(0,0,0),linestyle='--'),
                 flierprops=dict(color=(0,0,0)),
                 medianprops=dict(color=(1,0,0)),
                 meanprops=dict(color=(0,1,0)))
Axis.plot([], linestyle='none', marker='o',fillstyle='none', color=(1,0,0), label='Data')
Axis.plot([],color=(0,0,1), label='Median')
Axis.set_xscale('log')
# Axis.set_xticks([])
Axis.set_xlabel('Grid Size [$\mu$m]')
Axis.set_ylabel('Density [-]')
# plt.legend()
plt.subplots_adjust(left=0.25, right=0.75)
plt.show()

# Save densities
DensityData = DensityData.replace({np.nan:0})
DensityData.to_csv(str(ImageDirectory / str(Name[:-7] + 'Densities.csv')), index=False)




# Collect densities data to compare between samples
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
    Data = pd.read_csv(str(ImageDirectory / Name))
    Data = Data.replace({0:np.nan})

    if Index == 0:
        for i in range(len(Datas)):
            Datas[i][Name[:5]] = Data[str(PhysicalSizes[i])]

    else:
        for i in range(len(Datas)):
            Datas[i][Name[:5]] = np.nan
            Datas[i][Name[:5]] = Data[str(PhysicalSizes[i])]

for Data in Datas:
    Figure, Axis = plt.subplots(1,1)
    i = 0
    for Column in Data.columns:
        Axis.boxplot(Data[Column].dropna(), vert=True, widths=0.35,
                     positions=[i],
                     showmeans=False,meanline=True,
                     capprops=dict(color=(0,0,0)),
                     boxprops=dict(color=(0,0,0)),
                     whiskerprops=dict(color=(0,0,0),linestyle='--'),
                     flierprops=dict(color=(0,0,0)),
                     medianprops=dict(color=(1,0,0)),
                     meanprops=dict(color=(0,1,0)))
        i += 1
    Axis.set_xticks(np.arange(len(Data.columns)),Data.columns)
    Axis.set_ylim([-0.001, 0.021])
    Axis.set_xlabel('Sample [-]')
    Axis.set_ylabel('Density [-]')
    plt.subplots_adjust(left=0.25, right=0.75)
    plt.show()





# Simulate random zone selection
Data = Datas[5]
MeansData = pd.DataFrame()

for j in range(5):
    M = Data[Data.columns[j]].mean()
    S = Data[Data.columns[j]].std()


    # Open image to segment
    Image = sitk.ReadImage(str(ImageDirectory / str(Data.columns[j] + '_Seg.jpg')))
    Array = sitk.GetArrayFromImage(Image)[:,:,:3]

    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(Array)
    # plt.show()

    # Mark areas where there is bone
    Filter1 = Array[:,:,0] < 190
    Filter2 = Array[:,:,1] < 190
    Filter3 = Array[:,:,2] < 235
    Bone = Filter1 & Filter2 & Filter3
    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(Bone[1800:2200,4800:5200],cmap='binary')
    # plt.show()

    # Erode and dilate to remove small bone parts
    Disk = morphology.disk(2)
    Dilated = morphology.binary_dilation(Bone,Disk)
    Bone = morphology.binary_erosion(Dilated,Disk)

    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(Bone[1800:2200,4800:5200],cmap='binary')
    # plt.show()


    # Filter image to extract manual segmentation
    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(Array[1800:2200,4800:5200])
    # plt.show()

    Filter1 = Array[:,:,0] > 100
    Filter2 = Array[:,:,1] < 90
    Filter3 = Array[:,:,2] > 100

    Bin = np.zeros(Filter1.shape)
    Bin[Filter1 & Filter2 & Filter3] = 1

    # Dilate to link extracted segmentation
    Disk = morphology.disk(5)
    BinDilate = morphology.binary_dilation(Bin,Disk)

    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(BinDilate[1800:2200,4800:5200],cmap='binary')
    # plt.show()

    # Skeletonize to obtain 1 pixel thickness
    Skeleton = morphology.skeletonize(BinDilate)
    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(Skeleton[1800:2200,4800:5200],cmap='binary')
    # plt.show()

    # Random zone selection
    Size = int(round(1000 / PixelLength))
    BVTV, CMDensity = np.zeros(1000), np.zeros(1000)
    i = 0
    while i < 1000:
        RandomXPos = int(np.random.uniform(int(Size/2) + 1, Array.shape[1] - int(Size/2) - 1))
        RandomYPos = int(np.random.uniform(int(Size/2) + 1, Array.shape[0] - int(Size/2) - 1))

        SubRegion = Skeleton[RandomYPos - int(Size/2):RandomYPos + int(Size/2),
                             RandomXPos - int(Size/2):RandomXPos + int(Size/2)]
        SubBone = Bone[RandomYPos - int(Size/2):RandomYPos + int(Size/2),
                       RandomXPos - int(Size/2):RandomXPos + int(Size/2)]

        if SubRegion.sum() > 1E-4:
            BVTV[i] = SubBone.sum() / SubBone.size
            CMDensity[i] = SubRegion.sum() / SubBone.sum()
            i += 1

    Figure, Axis = plt.subplots(1,1)
    Axis.plot(BVTV,CMDensity, color=(1,0,0), marker='o', fillstyle='none', linestyle='none', label='Tests')
    Axis.plot([BVTV.min(), BVTV.max()], [M, M], color=(0,0,0), linestyle='--', label='Grid Mean')
    Axis.fill_between([BVTV.min(), BVTV.max()],[M + S, M + S],[M - S, M - S], color=(0,0,0,0.2), label='Standard deviation')
    Axis.set_xlabel('BV/TV (-)')
    Axis.set_ylabel('Density (-)')
    plt.legend()
    plt.show()


    # Means
    Means = np.zeros(1000)
    for i in range(1000):
        Means[i] = np.mean(CMDensity[:i+1])

    ROINumber = 100

    Figure, Axis = plt.subplots(1,1)
    Axis.plot(np.arange(1,ROINumber+1),CMDensity[:ROINumber] / M - 1, color=(1,0,0), marker='o', fillstyle='none', linestyle='none', label='Tests')
    Axis.plot(np.arange(1,ROINumber+1),Means[:ROINumber] / M - 1, color=(0,0,1), linestyle='--', label='ROIs Mean')
    Axis.plot([1, ROINumber], [0, 0], color=(0,0,0), linestyle='--', label='Grid Mean')
    Axis.fill_between([1, ROINumber],[S/M, S/M],[-S/M, -S/M], color=(0,0,0,0.15), label='Standard deviation')
    Axis.set_xlabel('ROI number (-)')
    Axis.set_ylabel('Density relative error (-)')
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.15))
    plt.show()

    MeansData[Data.columns[j]] = Means / M - 1


Colors = [(1,0,0),(1,0,1),(0,0,1),(0,1,1),(0,1,0)]
MaxROIs = 100

Figure, Axis = plt.subplots(1,1)
for Index in range(len(MeansData.columns)):
    Axis.plot(MeansData[MeansData.columns[Index]].abs(), color=Colors[Index])
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Relative error [-]')
Axis.set_xlim([-1,MaxROIs])
# Axis.set_ylim([0,1])
plt.show()


# Perform mixed-effect linear regression
plt.rc('font', size=12)
def PlotRegressionResults(Model, Data, NROIs=25):

    ## Get data from the model
    Y_Obs = Model.model.endog
    N = int(Model.nobs)
    C = np.matrix(Model.cov_params())
    X = np.matrix(Model.model.exog)[:1000]
    Y_Fit = np.array(Model.params[0] + Model.params[1] * X[:,1]).reshape(len(X))

    if not C.shape[0] == X.shape[1]:
        C = C[:-1,:-1]


    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Model.predict()
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / Model.df_resid)
    TSS = np.sum((Model.model.endog - Model.model.endog.mean()) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    ## Plot
    Colors = [(1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0)]
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5 * 1.5, 4.5 * 1.5))
    for i in range(len(Data['Groups'].unique())):
        y = Data[Data['Groups'] == Data['Groups'].unique()[i]]['y'].values
        Axes.plot(y, label=Data['Groups'].unique()[i], color=Colors[i])
    Axes.plot(Y_Fit, color=(0, 0, 0), linestyle='--', label='Fit')
    Axes.set_xlabel('Number of ROIs [-]')
    Axes.set_ylabel('Relative error [-]')
    # Axes.annotate(r'N Groups : ' + str(len(Data.groupby('Groups'))), xy=(0.65, 0.925), xycoords='axes fraction')
    # Axes.annotate(r'N Points : ' + str(N), xy=(0.65, 0.86), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.85, 0.65), xycoords='axes fraction')
    Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.85, 0.585), xycoords='axes fraction')
    Axes.set_xlim([-1, NROIs])
    plt.legend(ncol=1, loc='upper right')
    plt.show()

    return R2, SE

MaxROIs = 25

Data2Fit = pd.DataFrame()
Data2Fit['x'] = 1 / (np.arange(len(MeansData))+1)
Data2Fit['y'] = np.abs(MeansData[MeansData.columns[0]])
Data2Fit['Groups'] = MeansData.columns[0]

GroupedData = Data2Fit.copy()
for i in range(1,4):
    Data2Fit['y'] = np.abs(MeansData[MeansData.columns[i]])
    Data2Fit['Groups'] = MeansData.columns[i]
    GroupedData = pd.concat([GroupedData, Data2Fit], axis=0, ignore_index=True)

LMM = smf.mixedlm('y ~ x', data=GroupedData, groups=GroupedData['Groups']).fit(reml=True)
PlotRegressionResults(LMM, GroupedData, NROIs=25)
LMM.summary()
LMM.params[0]
LMM.params[1]

# Fit results
MedialCurve0500 = 0.1061014048386727 + 0.32978012713848315 / (np.arange(len(MeansData))+1)
MedialCurve1000 = 0.05142477003653866 + 0.8294294775342638 / (np.arange(len(MeansData))+1)
MedialCurve1500 = 0.10499514381506228 + 0.6086221151450486 / (np.arange(len(MeansData))+1)
MedialCurve2000 = 0.13443837628548178 + 0.4402865255409403 / (np.arange(len(MeansData))+1)
MedialCurves = [MedialCurve0500, MedialCurve1000, MedialCurve1500, MedialCurve2000]

LateralCurve0500 = 0.24594793990885896 + 0.7896431976544062 / (np.arange(len(MeansData))+1)
LateralCurve1000 = 0.07224763136122508 + 1.40477214166528 / (np.arange(len(MeansData))+1)
LateralCurve1500 = 0.10269523893867395 + 1.092066105802237 / (np.arange(len(MeansData))+1)
LateralCurve2000 = 0.07483507798042248 + 0.9343716072050389 / (np.arange(len(MeansData))+1)
LateralCurves = [LateralCurve0500, LateralCurve1000, LateralCurve1500, LateralCurve2000]

Figure, Axis = plt.subplots(1,1)
for Index in range(len(MedialCurves)):
    Axis.plot(MedialCurves[Index], color=Colors[Index], label=str(PhysicalSizes[Index+2]))
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Fitted Curve [-]')
Axis.set_xlim([-1,MaxROIs])
Axis.set_ylim([0,1.5])
plt.legend()
plt.show()

Figure, Axis = plt.subplots(1,1)
for Index in range(len(MedialCurves)):
    Axis.plot(LateralCurves[Index], color=Colors[Index], label=str(PhysicalSizes[Index+2]))
Axis.set_xlabel('Number of ROIs [-]')
Axis.set_ylabel('Fitted Curve [-]')
Axis.set_xlim([-1,MaxROIs])
Axis.set_ylim([0,1.5])
plt.legend()
plt.show()
