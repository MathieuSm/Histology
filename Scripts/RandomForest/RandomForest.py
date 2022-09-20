#!/usr/bin/env python3

"""
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
#sphx-glr-auto-examples-segmentation-plot-trainable-segmentation-py
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats.distributions import t
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LinearSegmentedColormap
from skimage.feature import multiscale_basic_features as mbf
from skimage import io, future, filters, exposure, morphology


plt.rc('font', size=12)

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPipeline'))
from Utilities import *

class FeatureNames:

    def __init__(self, SigmaMin=4, SigmaMax=32):

        # Random forest classifier
        Channels = ['R', 'G', 'B']
        # Names = ['Intensity', 'Edges', 'Hessian 1', 'Hessian 2']
        Names = ['I', 'E', 'H1', 'H2']
        NumSigma = int(np.log2(SigmaMax / SigmaMin) + 1)
        F_Names = []
        for Channel in Channels:
            for Sigma in range(NumSigma):
                SigmaValue = SigmaMin * 2 ** Sigma
                if SigmaValue >= 1:
                    SigmaValue = int(SigmaValue)
                for Name in Names:
                    F_Names.append(Channel + ' Sigma ' + str(SigmaValue) + ' ' + Name)

        self.Names = F_Names


def ExtractLabels(Seg, DilateCM=False):
    # Extract segments
    CL = Seg[:, :, 0] == 255
    OC = Seg[:, :, 1] == 255
    HC = CL * OC

    # Label cement lines segments
    Label = np.zeros(HC.shape, 'uint8')
    Label[CL] = 1
    Label[HC] = 0

    if DilateCM == True:
        Label = morphology.binary_dilation(Label, morphology.disk(1)) * 1

    # Select random pixels for tissue
    Coordinates = np.argwhere(~Label)
    np.random.shuffle(Coordinates)
    Pixels = Coordinates[:np.bincount(Label.ravel())[-1]]
    Label = Label * 1
    Label[Pixels[:, 0], Pixels[:, 1]] = 2

    # Label osteocytes and Harvesian canals
    Label[OC] = 3
    Label[HC] = 4

    Ticks = ['CL', 'IT', 'OC', 'HC']

    return Label, Ticks

def GetNeighbours(Array2D, N=1, Print=False):
    """
    Function used to get values of the neighbourhood pixels (based on numpy.roll)
    :param Array2D: Row x Column numpy array
    :param N: Number of neighbours offset (1 or 2 usually)
    :return: Neighbourhood pixels values
    """

    # Define a map for the neighbour index computation
    Map = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]])

    # number of neighbours
    Neighbours = (2*N+1)**2 - 1

    if len(Array2D.shape) > 2:
        YSize, XSize = Array2D.shape[:-1]
        Dimension = Array2D.shape[-1]
        Neighbourhood = np.zeros((YSize, XSize, Neighbours, Dimension))

        # Pad the array to avoid border effects
        Array2D = np.pad(Array2D, ((1, 1), (1, 1), (0, 0)), 'symmetric')
    else:
        YSize, XSize = Array2D.shape
        Neighbourhood = np.zeros((YSize, XSize, Neighbours))

        # Pad the array to avoid border effects
        Array2D = np.pad(Array2D, 1, 'symmetric')

    if Print:
        print('\nGet neighbours ...')
        Tic = time.time()

    i = 0
    for Shift in [-1, 1]:
        for Axis in [0, 1]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[1:-1,1:-1]
            i += 1

    for Shift in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        for Axis in [(0, 1)]:
            Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[1:-1,1:-1]
            i += 1

    if N == 2:

        # Pad again the array to avoid border effects
        if len(Array2D.shape) > 2:
            Array2D = np.pad(Array2D, ((1, 1), (1, 1), (0, 0)), 'symmetric')
        else:
            Array2D = np.pad(Array2D, 1, 'symmetric')

        for Shift in [-2, 2]:
            for Axis in [0, 1]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

        for Shift in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

        for Shift in [(-2, -1), (2, -1), (-2, 1), (2, 1)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

        for Shift in [(-1, -2), (1, -2), (-1, 2), (1, 2)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2,2:-2]
                i += 1

    if Print:
        Toc = time.time()
        PrintTime(Tic, Toc)

    return Neighbourhood, Map

def NormalizeValues(Image):
    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max() - Image.min())

    return N_Image

def SPCNN(Image,Beta=2,Delta=1/255,VT=100):

    """
    Segment image using simplified PCNN, single neuron firing and fast linking implementation
    Based on:
    Zhan, K., Shi, J., Wang, H. et al.
    Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
    Arch Computat Methods Eng 24, 573–588 (2017).
    https://doi.org/10.1007/s11831-016-9182-3

    :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
    :param Delta: Linear decay factor for threshold level
    :param VT: Dynamic threshold amplitude
    :param Nl_max: Max number of iteration for fast linking
    :return: H: Image histogram in numpy array
    """

    Tic = time.time()
    print('\nImage segmentation...')

    # Initialize parameters
    S = NormalizeValues(Image)
    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    T = np.zeros((Rows, Columns))
    W = np.ones((Rows, Columns, 8)) * np.array([1,1,1,1,0.5,0.5,0.5,0.5])
    Theta = np.ones((Rows, Columns))

    FiredNumber = 0
    N = 0

    # Perform segmentation
    while FiredNumber < S.size:

        N += 1
        F = S
        L = np.sum(GetNeighbours(Y)[0] * W, axis=2)
        Theta = Theta - Delta + VT * Y
        U = F * (1 + Beta * L)
        Y = (U > Theta) * 1

        T = T + N * Y
        FiredNumber = FiredNumber + sum(sum(Y))

    Output = 1 - NormalizeValues(T)

    # Print time elapsed
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Output

def FitData(DataFrame):

    Formula = DataFrame.columns[1] + ' ~ ' + DataFrame.columns[0]
    FitResults = smf.ols(Formula, data=DataFrame).fit()

    # Calculate R^2, p-value, 95% CI, SE, N
    Y_Obs = FitResults.model.endog
    Y_Fit = FitResults.fittedvalues

    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / FitResults.df_resid)

    N = int(FitResults.nobs)
    R2 = FitResults.rsquared
    p = FitResults.pvalues[1]

    CI_l = FitResults.conf_int()[0][1]
    CI_r = FitResults.conf_int()[1][1]

    X = np.matrix(FitResults.model.exog)
    X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))
    C = np.matrix(FitResults.cov_params())
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    Alpha = 0.95
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * B_0
    Sorted_CI_u = CI_Line_u[np.argsort(FitResults.model.exog[:,1])]
    Sorted_CI_o = CI_Line_o[np.argsort(FitResults.model.exog[:,1])]

    NoteYPos = 0.925
    NoteYShift = 0.075

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
    Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
    Axes.fill_between(X_Obs, Sorted_CI_o, Sorted_CI_u, color=(0, 0, 0), alpha=0.1,
                      label=str(int(Alpha * 100)) + '% CI')
    Axes.plot(X[:, 1], Y_Obs, linestyle='none', fillstyle='none', marker='o', color=(0, 0, 1), label='Data')
    Axes.annotate('Slope 95% CI [' + str(CI_l.round(2)) + r'$,$ ' + str(CI_r.round(2)) + ']',
                  xy=(0.05, NoteYPos), xycoords='axes fraction')
    # Axes.annotate(r'$N$ : ' + str(N), xy=(0.05, NoteYPos),
    #               xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + str(R2.round(2)), xy=(0.05, NoteYPos - NoteYShift),
                  xycoords='axes fraction')
    Axes.annotate(r'$\sigma_{est}$ : ' + str(SE.round(5)), xy=(0.05, NoteYPos - NoteYShift*2),
                  xycoords='axes fraction')
    Axes.annotate(r'$p$ : ' + str(p.round(3)), xy=(0.05, NoteYPos - NoteYShift*3),
                  xycoords='axes fraction')
    Axes.set_ylabel(DataFrame.columns[1])
    Axes.set_xlabel(DataFrame.columns[0])
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend(loc='lower right')
    plt.show()

    # Add fitted values and residuals to data
    DataFrame['Fitted Value'] = Y_Fit
    DataFrame['Residuals'] = E

    return DataFrame, FitResults, R2, SE, p, [CI_l, CI_r]

# Load dictionary
CWD = Path.cwd() / 'Scripts' / 'RandomForest'
FileName = CWD / 'ROIs.pkl'
with open(str(FileName), 'rb') as f:
    Dict = pickle.load(f)

# Compute mean ROI histogram
nBins = 255
Histograms = np.zeros((len(Dict.keys()),3,3,nBins))
for RGB in range(3):
    for nKey, Key in enumerate(Dict.keys()):
        for nROI in range(3):
            ROI = Dict[Key]['ROIs'][nROI]
            Hists, Bins = np.histogram(ROI[:, :, RGB], density=False, bins=nBins, range=(0, 255))
            Histograms[nKey,nROI,RGB] = Hists
MeanHist = np.mean(Histograms,axis=(0,1)).round().astype('int')

Figure, Axis = plt.subplots(1,1)
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[0], edgecolor=(1,0,0), color=(0,0,0,0), width=Bins[1])
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[1], edgecolor=(0,1,0), color=(0,0,0,0), width=Bins[1])
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[2], edgecolor=(0,0,1), color=(0,0,0,0), width=Bins[1])
plt.show()

Start = 0
Stop = 0
Reference = np.ones(ROI.shape,'int').ravel()
for i, nPixels in enumerate(MeanHist.ravel()):
    Stop += nPixels
    Reference[Start:Stop] = np.tile(Bins,3)[i].astype('int')
    Start = Stop
Reference = np.reshape(Reference,ROI.shape,order='F')
PlotImage(Reference)


# Load images
FileName = CWD / 'TestROI.png'
ROI = io.imread(str(FileName))
ROI = np.round(exposure.match_histograms(ROI,Reference)).astype('uint8')
PlotImage(ROI)

FileName = CWD / 'TestROI_Forest.png'
Seg_ROI = io.imread(str(FileName))
PlotImage(Seg_ROI)

ROI_Label, Ticks = ExtractLabels(Seg_ROI, DilateCM=True)
PlotImage(ROI_Label)

Smin, Smax = 0.5, 16
FNames = FeatureNames(Smin, Smax)

F_ROI = mbf(ROI, multichannel=True,intensity=True,edges=True,texture=True,
            sigma_min=Smin,sigma_max=Smax)

Features = F_ROI[ROI_Label > 0]
Label = ROI_Label[ROI_Label > 0]

Colors = [(0,0,0),(1,0,0),(0,0,1),(0,0,1),(1,1,1)]
Values = [0, 0.25, 0.5, 0.75, 1]

def PlotLabel(Seg):

    Image = np.zeros((Seg.shape[0], Seg.shape[1], 3))

    Colors = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1)]
    for iValue, Value in enumerate(np.unique(Seg)):
        Filter = Seg == Value
        Image[Filter] = Colors[iValue]


    Figure, Axis = plt.subplots(1,1, figsize=(10,10))
    Axis.imshow(Image)
    Axis.plot([], color=(1,0,0), lw=1, label='Segmentation')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()

for Key, ROINumber in [[0, 0], [4, 2]]:
    ROI = Dict[Key]['ROIs'][ROINumber]
    ROI = np.round(exposure.match_histograms(ROI, Reference)).astype('uint8')
    F_ROI = mbf(ROI, multichannel=True, intensity=True, edges=True, texture=True,
                sigma_min=Smin, sigma_max=Smax)
    FileName = CWD / str('Sample' + str(Key) + '_Seg' + str(ROINumber) + '.png')
    Seg_ROI = io.imread(str(FileName))
    ROI_Label = ExtractLabels(Seg_ROI)[0]
    PlotLabel(ROI_Label)

    Features = np.vstack([Features,F_ROI[ROI_Label > 0]])
    Label = np.concatenate([Label,ROI_Label[ROI_Label > 0]])

Classifier = RandomForestClassifier(n_jobs=-1, max_samples=0.2, class_weight='balanced')

# for Sigma in range(NumSigma):
#     SigmaValue = SigmaMin * 2 ** Sigma
#     PlotImage(filters.gaussian(ROI,SigmaValue, multichannel=True))
#
# for iKey, Key in enumerate(Dict.keys()):
#     for ROINumber in range(Dict[Key]['ROI'].shape[0]):
#         ROI = Dict[Key]['ROI'][ROINumber]
#         ROI = np.round(exposure.match_histograms(ROI,Reference)).astype('uint8')
#         ROI_Features = mbf(ROI, multichannel=True, intensity=True, edges=True, texture=True,
#                            sigma_min=SigmaMin, sigma_max=SigmaMax, num_sigma=NumSigma)
#
#         Skeleton = Dict[Key]['Skeleton'][ROINumber]
#         ROI_Label = morphology.binary_dilation(Skeleton, morphology.disk(1)) * 1
#         # Coordinates = np.argwhere(~ROI_Label)
#         # np.random.shuffle(Coordinates)
#         # Pixels = Coordinates[:np.bincount(ROI_Label.ravel())[-1]]
#         # ROI_Label[Pixels[:, 0], Pixels[:, 1]] = 1
#
#         # if iKey == 0 and ROINumber == 0:
#         #     Features = ROI_Features[ROI_Label > 0]
#         #     Label = ROI_Label[ROI_Label > 0]
#         # else:
#         if iKey == 0 and ROINumber == 0:
#             Features = Features[Label > 0]
#             Label = Label[Label > 0]
#
#         Features = np.vstack([Features,ROI_Features[ROI_Label > 0]])
#         Label = np.concatenate([Label,ROI_Label[ROI_Label > 0]])
#
# Features = mbf(ROI, multichannel=True,intensity=True,edges=True,texture=True,
#                sigma_min=SigmaMin,sigma_max=SigmaMax,num_sigma=NumSigma)
# Classifier = RandomForestClassifier(n_jobs=-1, max_samples=0.2)
#
# for i in range(int(Features.shape[-1]/4)+1):
#     PlotImage(Features[:,:,i])
#
# # # Add PCNN features
# # MoreFeatures = np.zeros((Features.shape[0],Features.shape[1],Features.shape[2] + 1))
# # MoreFeatures[:,:,:-1] = Features
# # MoreFeatures[:,:,-1] = PCNN_ROI
# # Features = MoreFeatures
#
# # Exclude H2
# LessFeatures = np.zeros((Features.shape[0],Features.shape[1],24))
# LessFeatures = Features[:,:,:24]
# LessNames = F_Names[:24]
# Features = LessFeatures

# Train model

Tic = time.time()
clf = future.fit_segmenter(Label,Features, Classifier)
Toc = time.time()
PrintTime(Tic, Toc)

# See predictions
Results = future.predict_segmenter(Features, Classifier)

# Assess model
CM = metrics.confusion_matrix(Label,Results,normalize=None)
CM2 = metrics.confusion_matrix(Label,Results,normalize='true')
CM3 = metrics.confusion_matrix(Label,Results,normalize='pred')
VSpace = 0.2

Figure, Axis = plt.subplots(1,1, figsize=(5.5,4.5))
Axis.matshow(CM3, cmap='binary', alpha=0.33)
for Row in range(CM.shape[0]):
    for Column in range(CM.shape[1]):
        Axis.text(x=Row, y=Column, position=(Row,Column), va='center', ha='center', s=CM[Row, Column])
        Axis.text(x=Row, y=Column, position=(Row,Column+VSpace), va='center', ha='center', s=round(CM2[Row, Column],2), color=(0,0,1))
        Axis.text(x=Row, y=Column, position=(Row,Column-VSpace), va='center', ha='center', s=round(CM3[Row, Column],2), color=(1,0,0))
Axis.xaxis.set_ticks_position('bottom')
Axis.set_xticks(np.arange(len(Ticks)),Ticks)
Axis.set_yticks(np.arange(len(Ticks)),Ticks)
Axis.set_ylim([-0.49,CM.shape[0]-0.5])
Axis.set_title('Total: ' + str(Label[Label > 0].size))
Axis.set_xlabel('Ground Truth',color=(0,0,1))
Axis.set_ylabel('Predictions',color=(1,0,0))
plt.show()

# Dice coefficient
Dice(Label-1,Results-1)

Report = metrics.classification_report(Label.ravel(),Results.ravel())
print(Report)

# Feature importance
def PlotFeatureImportance(Classifier, F_Names):

    FI = pd.DataFrame(Classifier.feature_importances_, columns=['Importance'])
    FI['Channel'] = [C.split()[0] for C in F_Names]
    FI['Sigma'] = [C.split()[2] for C in F_Names]
    FI['Feature'] = [C.split()[3] for C in F_Names]
    Features = FI['Feature'].unique()

    Sorted = FI.sort_values(by='Importance')
    R = Sorted[Sorted['Channel'] == 'R']
    G = Sorted[Sorted['Channel'] == 'G']
    B = Sorted[Sorted['Channel'] == 'B']

    Sigmas = FI['Sigma'].unique()

    if len(Sigmas) == 1:
        Figure, Axis = plt.subplots(1,1)
        RS = R.sort_values(by='Feature')
        GS = G.sort_values(by='Feature')
        BS = B.sort_values(by='Feature')
        Axis.bar(np.arange(len(RS)), RS['Importance'], edgecolor=(1, 0, 0), facecolor=(0, 0, 0, 0))
        Axis.bar(np.arange(len(GS)), GS['Importance'], edgecolor=(0, 1, 0), facecolor=(0, 0, 0, 0))
        Axis.bar(np.arange(len(BS)), BS['Importance'], edgecolor=(0, 0, 1), facecolor=(0, 0, 0, 0))
        Axis.set_xticks(np.arange(len(Features)), Features)

    elif len(Sigmas) < 4:
        Figure, Axis = plt.subplots(1, len(Sigmas), sharex=True, sharey=True)
        i = 0
        for Sigma in Sigmas:
            RF = R[R['Sigma'] == Sigma]
            GF = G[G['Sigma'] == Sigma]
            BF = B[B['Sigma'] == Sigma]
            RS = RF.sort_values(by='Feature')
            GS = GF.sort_values(by='Feature')
            BS = BF.sort_values(by='Feature')
            Axis[i].bar(np.arange(len(RS)), RS['Importance'], edgecolor=(1, 0, 0), facecolor=(0, 0, 0, 0))
            Axis[i].bar(np.arange(len(GS)), GS['Importance'], edgecolor=(0, 1, 0), facecolor=(0, 0, 0, 0))
            Axis[i].bar(np.arange(len(BS)), BS['Importance'], edgecolor=(0, 0, 1), facecolor=(0, 0, 0, 0))
            Axis[i].set_xticks(np.arange(len(Features)), Features)
            Axis[i].set_title('Sigma = ' + Sigma)
            i += 1

    else:
        NRows = np.floor(np.sqrt(len(Sigmas))).astype('int')
        NColumns = np.ceil(len(Sigmas)/NRows).astype('int')
        Figure, Axis = plt.subplots(NRows, NColumns, sharex=True, sharey=True)
        Columns = np.tile(np.arange(NColumns),NRows)
        Rows = np.repeat(np.arange(NRows),NColumns)
        i = 0
        for Sigma in Sigmas:
            Row = Rows[i]
            Column = Columns[i]
            RF = R[R['Sigma'] == Sigma]
            GF = G[G['Sigma'] == Sigma]
            BF = B[B['Sigma'] == Sigma]
            RS = RF.sort_values(by='Feature')
            GS = GF.sort_values(by='Feature')
            BS = BF.sort_values(by='Feature')
            Axis[Row,Column].bar(np.arange(len(RS)), RS['Importance'], edgecolor=(1,0,0), facecolor=(0,0,0,0))
            Axis[Row,Column].bar(np.arange(len(GS)), GS['Importance'], edgecolor=(0,1,0), facecolor=(0,0,0,0))
            Axis[Row,Column].bar(np.arange(len(BS)), BS['Importance'], edgecolor=(0,0,1), facecolor=(0,0,0,0))
            Axis[Row,Column].set_xticks(np.arange(len(Features)), Features)
            Axis[Row,Column].set_title('Sigma = ' + Sigma)
            i += 1

    plt.show()

    return FI
FI = PlotFeatureImportance(Classifier, FNames.Names)

Threshold = 0.02
Mask = FI['Importance'] > Threshold

# Train model again
Tic = time.time()
clf = future.fit_segmenter(Label,Features[:,Mask], Classifier)
Toc = time.time()
PrintTime(Tic, Toc)

# See new predictions
Results = future.predict_segmenter(Features[:,Mask], Classifier)

def PlotOverlay(ROI,Seg, Save=False, FileName=None):

    CMapDict = {'red':((0.0, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
                'green': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0)),
                'blue': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                'alpha': ((0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}
    CMap = LinearSegmentedColormap('MyMap',CMapDict)

    Figure, Axis = plt.subplots(1,1, figsize=(10,10))
    Axis.imshow(ROI)
    Axis.imshow(Seg*1, cmap=CMap, alpha=0.3)
    Axis.plot([], color=(1,0,0), lw=1, label='Segmentation')
    Axis.axis('off')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15))
    plt.subplots_adjust(0,0,1,1)
    if Save:
        plt.savefig(FileName)
    plt.show()

Data = pd.DataFrame(columns=Dict.keys(),index=range(3))
for Key in Dict.keys():
    for ROINumber in range(3):
        TestROI = Dict[Key]['ROIs'][ROINumber]
        Tic = time.time()
        TestROI = np.round(exposure.match_histograms(TestROI,Reference)).astype('uint8')
        F_Test = mbf(TestROI, multichannel=True, intensity=True, edges=True, texture=True,
                     sigma_min=Smin, sigma_max=Smax)
        R_Test = future.predict_segmenter(F_Test, Classifier)

        FileName = CWD / str('RF_S' + str(Key) + '_ROI' + str(ROINumber) + '.jpg')
        PlotOverlay(TestROI,R_Test == 1, Save=True, FileName=FileName)

        Data.loc[ROINumber,Key] = np.sum(R_Test == 1) / R_Test.size
        Toc = time.time()
        PrintTime(Tic, Toc)

Density = pd.DataFrame(columns=Dict.keys(),index=range(3))
for Key in Dict.keys():
    for ROINumber in range(3):
        TestSeg = Dict[Key]['Skeletons'][ROINumber]
        Density.loc[ROINumber,Key] = np.sum(TestSeg) / TestSeg.size


Data2Fit = pd.DataFrame({'Manual':Density.mean(axis=0),'Automatic':Data.mean(axis=0)})
Data2Fit = pd.DataFrame({'Manual':Density.values.ravel(),'Automatic':Data.values.ravel()})
# Data2Fit = Data2Fit[Data2Fit['Manual'] > 2E-4].reset_index()
Data2Fit, FitResults, R2, SE, p, CI = FitData(Data2Fit[['Automatic','Manual']].astype('float'))


# Fit with error bar








Figure, Axis = plt.subplots(1,1)
Axis.plot(Data2Fit['Residuals']/Data2Fit['Fitted Value'],linestyle='none',marker='o')
plt.show()

FitData(Data2Fit.drop([4,7,11]).reset_index(drop=True))
FitData(Data2Fit.drop([1,4,5,6,9]).reset_index(drop=True))


TestROI = Dict['437RM']['ROI'][0]
TestROI = Dict['418RM']['ROI'][1]
TestROI = Dict['418LM']['ROI'][2]
PlotImage(TestROI)
Histogram(TestROI)

SegROI = Parameters.SegImage[Ys[ROINumber, 0]:Ys[ROINumber, 1], Xs[ROINumber, 0]:Xs[ROINumber, 1]]
PlotImage(SegROI)

F = filters.gaussian(ROIs[ROINumber],sigma=1, multichannel=True)
R = features_func(F)
Results = future.predict_segmenter(R, clf)
PlotImage(Results == 4)

B1 = []
B1.append(TestROI)
for i in [1,4,5,6,9]:
    SegROI = Parameters.SegImage[Ys[i, 0]:Ys[i, 1], Xs[i, 0]:Xs[i, 1]]
    B1.append(SegROI)
HistoError(B1,ylim=0.02)

B2 = []
B2.append(Dict['391LM']['ROI'][2])
for i in [0,2,3,7,8]:
    SegROI = Parameters.SegImage[Ys[i, 0]:Ys[i, 1], Xs[i, 0]:Xs[i, 1]]
    B2.append(SegROI)
HistoError(B2,ylim=0.02)

B1 = np.concatenate(B1,axis=0)
B2 = np.concatenate(B2,axis=0)
NBins = 20
B1H, B2H = np.zeros((3,NBins)), np.zeros((3,NBins))
for i in range(3):
    B1H[i], Bins = np.histogram(B1[:, :, i], density=True, bins=NBins, range=(0, 255))
    B2H[i], Bins = np.histogram(B2[:, :, i], density=True, bins=NBins, range=(0, 255))

Width = Bins[1]
Bins = 0.5 * (Bins[1:] + Bins[:-1])

Figure, Axes = plt.subplots(1,1)
Axes.bar(Bins, B1H[0]-B2H[0], width=Width, color=(1, 1, 1, 0), edgecolor=(1,0,0))
Axes.bar(Bins, B1H[1]-B2H[1], width=Width, color=(1, 1, 1, 0), edgecolor=(0,1,0))
Axes.bar(Bins, B1H[2]-B2H[2], width=Width, color=(1, 1, 1, 0), edgecolor=(0,0,1))
plt.show()

[1,4,5,6,9]
[0,2,3,7,8]
i = 9
PlotImage(Parameters.SegImage[Ys[i,0]:Ys[i,1], Xs[i,0]:Xs[i,1]])
PlotImage(ROI)
