#!/usr/bin/env python3

"""
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
#sphx-glr-auto-examples-segmentation-plot-trainable-segmentation-py
"""

import os
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LinearSegmentedColormap
from skimage import io, future, exposure, morphology, color
from skimage.feature import multiscale_basic_features as mbf
from sklearn.model_selection import RandomizedSearchCV, train_test_split

plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    This script creates and train a random forest classifier for cement lines segmentation in the curse
    of the FEXHIP project. The path to the manually segmented images should be specified if it is different
    from inside the 'Pipeline' folder.

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: September 2022
    """

class FeatureNames:

    def __init__(self, SigmaMin=4, SigmaMax=32, Channels=['R','G','B'], Features=['I', 'E', 'H1', 'H2'], nSigma=None):

        if nSigma:
            NumSigma = nSigma
        else:
            NumSigma = int(np.log2(SigmaMax / SigmaMin) + 1)
        F_Names = []
        for Channel in Channels:
            for Sigma in range(NumSigma):
                if nSigma:
                    SigmaValue = np.linspace(SigmaMin, SigmaMax, nSigma)[Sigma]
                else:
                    SigmaValue = SigmaMin * 2 ** Sigma
                if SigmaValue >= 1:
                    SigmaValue = int(round(SigmaValue))
                for Feature in Features:
                    F_Names.append(Channel + ' Sigma ' + str(SigmaValue) + ' ' + Feature)

        self.Names = F_Names

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

def PlotImage(Array):

    Figure, Axis = plt.subplots(1,1,figsize=(10,10))
    if Array.shape[-1] == 3:
        Axis.imshow(Array)
    else:
        Axis.imshow(Array, cmap='binary_r')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()

def ExtractLabels(Seg, DilateCM=False, Plot=True):

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

    if Plot:
        Image = np.zeros((Seg.shape[0], Seg.shape[1], 3))

        Colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        for iValue, Value in enumerate(np.unique(Label)):
            Filter = Label == Value
            Image[Filter] = Colors[iValue]

        Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
        Axis.imshow(Image)
        Axis.plot([], color=(1, 0, 0), lw=1, label='Segmentation')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    return Label, Ticks

def ExtractFeatures(Dict):
    Features = {}
    for Key in Dict.keys():
        Lab = color.rgb2lab(Dict[Key]['ROI'])

        Smin, Smax, nS = 0.5, 8, 3
        FNames = FeatureNames(Smin, Smax, Channels=['L', 'a', 'b'], Features=['E', 'H1', 'H2'], nSigma=nS)
        ROIFeatures = mbf(Lab, multichannel=True, intensity=False, edges=True, texture=True,
                          sigma_min=Smin, sigma_max=Smax, num_sigma=3)
        Features[Key] = ROIFeatures

    return Features, FNames

def PlotConfusionMatrix(GroundTruth, Results, Ticks=None):

    CM = metrics.confusion_matrix(GroundTruth, Results, normalize=None)
    CM2 = metrics.confusion_matrix(GroundTruth, Results, normalize='true')
    CM3 = metrics.confusion_matrix(GroundTruth, Results, normalize='pred')
    VSpace = 0.2

    Figure, Axis = plt.subplots(1, 1, figsize=(5.5, 4.5))
    Axis.matshow(CM3, cmap='binary', alpha=0.33)
    for Row in range(CM.shape[0]):
        for Column in range(CM.shape[1]):
            Axis.text(x=Row, y=Column, position=(Row, Column), va='center', ha='center', s=CM[Row, Column])
            Axis.text(x=Row, y=Column, position=(Row, Column + VSpace), va='center', ha='center',
                      s=round(CM2[Row, Column], 2), color=(0, 0, 1))
            Axis.text(x=Row, y=Column, position=(Row, Column - VSpace), va='center', ha='center',
                      s=round(CM3[Row, Column], 2), color=(1, 0, 0))
    Axis.xaxis.set_ticks_position('bottom')
    if Ticks:
        Axis.set_xticks(np.arange(len(Ticks)), Ticks)
        Axis.set_yticks(np.arange(len(Ticks)), Ticks)
    Axis.set_ylim([-0.49, CM.shape[0] - 0.5])
    Axis.set_title('Total: ' + str(GroundTruth[GroundTruth > 0].size))
    Axis.set_xlabel('Ground Truth', color=(0, 0, 1))
    Axis.set_ylabel('Predictions', color=(1, 0, 0))
    plt.show()

    return CM

def PlotFeatureImportance(Classifier, F_Names=None):

    FI = pd.DataFrame(Classifier.feature_importances_, columns=['Importance'])

    if F_Names:
        FI['Channel'] = [C.split()[0] for C in F_Names]
        FI['Sigma'] = [C.split()[2] for C in F_Names]
        FI['Feature'] = [C.split()[3] for C in F_Names]
        Features = FI['Feature'].unique()

        Sorted = FI.sort_values(by='Importance')
        Channels = FI['Channel'].unique()
        CList = []
        if 'R' in Channels:
            R = Sorted[Sorted['Channel'] == 'R']
            CList.append(R)
        if 'G' in Channels:
            G = Sorted[Sorted['Channel'] == 'G']
            CList.append(G)
        if 'B' in Channels:
            B = Sorted[Sorted['Channel'] == 'B']
            CList.append(B)
        if 'H' in Channels:
            H = Sorted[Sorted['Channel'] == 'H']
            CList.append(H)
        if 'S' in Channels:
            S = Sorted[Sorted['Channel'] == 'S']
            CList.append(S)
        if 'V' in Channels:
            V = Sorted[Sorted['Channel'] == 'V']
            CList.append(V)
        if 'L' in Channels:
            L = Sorted[Sorted['Channel'] == 'L']
            CList.append(L)
        if 'a' in Channels:
            a = Sorted[Sorted['Channel'] == 'a']
            CList.append(a)
        if 'b' in Channels:
            b = Sorted[Sorted['Channel'] == 'b']
            CList.append(b)

        Sigmas = FI['Sigma'].unique()

    CMapDict = {'red': ((0.0, 1.0, 1.0),
                        (1/2, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),
                'blue': ((0.0, 0.0, 0.0),
                         (1/4, 1.0, 1.0),
                         (3/4, 1.0, 1.0),
                         (1.0, 0.0, 0.0)),
                'green': ((0.0, 0.0, 0.0),
                          (1/2, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}
    CMap = LinearSegmentedColormap('MyMap', CMapDict)

    if F_Names:

        if len(Sigmas) == 1:

            Figure, Axis = plt.subplots(1,1)

            for iC, C in enumerate(CList):
                S = C.sort_values(by='Feature')
                Axis.bar(np.arange(len(S)), S['Importance'], edgecolor=CMap(iC / len(CList)), facecolor=(0, 0, 0, 0))
                Axis.plot([],color=CMap(iC / len(CList)), label=C['Channel'].unique())
            Axis.set_xticks(np.arange(len(Features)), Features)
            Handles, Labels = Axis.get_legend_handles_labels()
            Figure.legend(Handles, Labels, loc='upper center', ncol=3)
            plt.subplots_adjust(0.1, 0.1, 0.9, 0.8)

        elif len(Sigmas) < 4:
            Figure, Axis = plt.subplots(1, len(Sigmas), sharex=True, sharey=True)

            for i, Sigma in enumerate(Sigmas):
                for iC, C in enumerate(CList):
                    F = C[C['Sigma'] == Sigma]
                    S = F.sort_values(by='Feature')
                    Axis[i].bar(np.arange(len(S)), S['Importance'], edgecolor=CMap(iC / (len(CList)-1)), facecolor=(0, 0, 0, 0))
                    Axis[i].set_xticks(np.arange(len(Features)), Features)
                    Axis[i].set_title('Sigma = ' + Sigma)
                    Axis[i].plot([], color=CMap(iC / (len(CList)-1)), label=C['Channel'].unique()[0])

                    Handles, Labels = Axis[0].get_legend_handles_labels()
                    Figure.legend(Handles, Labels, loc='upper center', ncol=3)
                    plt.subplots_adjust(0.15, 0.15, 0.9, 0.8)

        else:
            NRows = np.floor(np.sqrt(len(Sigmas))).astype('int')
            NColumns = np.ceil(len(Sigmas)/NRows).astype('int')
            Figure, Axis = plt.subplots(NRows, NColumns, sharex=True, sharey=True)
            Columns = np.tile(np.arange(NColumns),NRows)
            Rows = np.repeat(np.arange(NRows),NColumns)
            for i, Sigma in enumerate(Sigmas):
                Row = Rows[i]
                Column = Columns[i]
                Ls = []
                for iC, C in enumerate(CList):
                    F = C[C['Sigma'] == Sigma]
                    S = F.sort_values(by='Feature')
                    Axis[Row,Column].bar(np.arange(len(S)), S['Importance'], edgecolor=CMap(iC / len(CList)), facecolor=(0, 0, 0, 0))
                    Axis[Row,Column].set_xticks(np.arange(len(Features)), Features)
                    Axis[Row,Column].set_title('Sigma = ' + Sigma)
                    Axis[Row,Column].plot([], color=CMap(iC / (len(CList)-1)), label=C['Channel'].unique()[0])

            Handles, Labels = Axis[0,0].get_legend_handles_labels()
            Figure.legend(Handles, Labels, loc='upper center', ncol=3)
            plt.subplots_adjust(0.1,0.1,0.9,0.8)
        plt.show()

    else:

        Figure, Axis = plt.subplots(1,1)
        Axis.bar(np.arange(len(FI))+1, FI['Importance'], edgecolor=(1,0,0), facecolor=(0, 0, 0, 0))
        Axis.set_xlabel('Feature number (-)')
        Axis.set_ylabel('Relative importance (-)')
        plt.show()

    return FI

def PlotResults(ROI,Results, FileName=None):

    H, W = Results.shape
    SegImage = np.zeros((H, W, 4))

    Colors = [(1,0,0,0.25),(0,0,1,0.25),(0,1,0,0.25),(1,1,1,0.25)]
    for iValue, Value in enumerate(np.unique(Results)):
        Filter = Results == Value
        SegImage[Filter] = Colors[iValue]

    Figure, Axis = plt.subplots(1,1, figsize=(H/100,W/100))
    Axis.imshow(ROI)
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    if FileName:
        plt.savefig(FileName)
    plt.show()

    Figure, Axis = plt.subplots(1, 1, figsize=(H / 100, W / 100))
    Axis.imshow(ROI)
    Axis.imshow(SegImage, interpolation='none')
    Axis.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    if FileName:
        plt.savefig(FileName[:-4] + '_Seg.png')
    plt.show()



class PCNNClass:

    def __init__(self):
        self.NSegments = 10

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

    def GaussianKernel(self, Length=5, Sigma=1.):
        """
        Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
        """
        Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
        Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
        Kernel = np.outer(Gauss, Gauss)
        return Kernel / sum(sum(Kernel))

    def GetNeighbours(self, Array, N=1, Map=False):

        """
        Function used to get values of the neighbourhood pixels
        :param Array: Row x Column numpy array
        :param N: Number of neighbours offset
        :param Map: return rows and columns indices of neigbours (bool)
        :return: Neighbourhood pixels values, Map
        """

        L = 2*N+1
        S = Array.shape
        iRows, iCols = np.arange(S[0]), np.arange(S[1])
        Cols, Rows = np.meshgrid(iRows,iCols)
        Cols = np.repeat(Cols, L).reshape((S[0], S[0], L))
        Cols = np.repeat(Cols, L).reshape((S[0], S[0], L, L))
        Rows = np.repeat(Rows, L).reshape((S[0], S[0], L))
        Rows = np.repeat(Rows, L).reshape((S[0], S[0], L, L))

        Range = np.arange(L) - N
        ColRange = np.repeat(Range,L).reshape((L,L))
        RowRange = np.tile(Range,L).reshape((L,L))
        iCols = Cols + ColRange
        iRows = Rows + RowRange

        Pad = np.pad(Array,N)
        Neighbours = Pad[iRows+N,iCols+N]

        if Map:
            return Neighbours, [iRows, iCols]

        else:
            return Neighbours

    def NormalizeValues(self, Image):
        """
        Normalize image values, used in PCNN for easier parameters handling
        :param Image: Original grayscale image
        :return: N_Image: Image with 0,1 normalized values
        """

        N_Image = (Image - Image.min()) / (Image.max() - Image.min())

        return N_Image

    def Enhance(self, Image, SRange = (0.01, 1.0)):

        """
        Enhance image using PCNN, single neuron firing and fast linking implementation
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


        def FLM(Input):
            S = (Input - Input.min()) / (Input.max() - Input.min()) + 1 / 255
            W = np.array([[0.7, 1, 0.7], [1, 0, 1], [0.7, 1, 0.7]])
            Y = np.zeros(S.shape)
            U = np.zeros(S.shape)
            T = np.zeros(S.shape)
            SumY = 0
            N = 0

            Laplacian = np.array([[1 / 6, 2 / 3, 1 / 6], [2 / 3, -10 / 3, 2 / 3], [1 / 6, 2 / 3, 1 / 6]])
            Theta = 1 + np.sum(self.GetNeighbours(S) * Laplacian, axis=(3,2))
            f = 0.75 * np.exp(-S**2 / 0.16) + 0.05
            L = 3
            G = self.GaussianKernel(2*L+1, 1)
            f = np.sum(self.GetNeighbours(f,L) * G, axis=(3,2))

            h = 2E10
            d = 2
            g = 0.9811
            Alpha = 0.01
            Beta = 0.03

            while SumY < S.size:
                N += 1

                K = np.sum(self.GetNeighbours(Y) * W, axis=(3, 2))
                Wave = Alpha * K + Beta * S * (K - d)
                U = f * U + S + Wave
                Theta = g * Theta + h * Y
                Y = (U > Theta) * 1
                T += N * Y
                SumY += sum(sum(Y))

            T_inv = T.max() + 1 - T
            Time = self.NormalizeValues(T_inv)

            return Time

        Tic = time.time()
        print('\nEnhance image using PCNN')
        if len(Image.shape) == 2:
            Output = FLM(Image)

        else:
            Output = np.zeros(Image.shape)
            for i in range(Image.shape[-1]):
                Output[:,:,i] = FLM(Image[:,:,i])

        Hist, Bins = np.histogram(Output, bins=1000, density=True)
        Low = np.cumsum(Hist) / np.cumsum(Hist).max() > SRange[0]
        High = np.cumsum(Hist) / np.cumsum(Hist).max() < SRange[1]
        Values = Bins[1:][Low & High]
        Output = exposure.rescale_intensity(Output, in_range=(Values[0], Values[-1]))

        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output

    def Segment(self,Image, Beta=2, Delta=1 / 255, VT=100):

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

        if Image.shape[-1] == 3:
            Input = color.rgb2gray(Image)
        else:
            Input = Image

        # Initialize parameters
        S = self.NormalizeValues(Input)
        Rows, Columns = S.shape
        Y = np.zeros((Rows, Columns))
        T = np.zeros((Rows, Columns))
        W = np.array([[0.5, 1, 0.5], [1, 0, 1], [0.5, 1, 0.5]])
        Theta = np.ones((Rows, Columns))

        FiredNumber = 0
        N = 0

        # Perform segmentation
        while FiredNumber < S.size:
            N += 1
            F = S
            L = np.sum(self.GetNeighbours(Y) * W, axis=(3,2))
            Theta = Theta - Delta + VT * Y
            U = F * (1 + Beta * L)
            Y = (U > Theta) * 1

            T = T + N * Y
            FiredNumber = FiredNumber + sum(sum(Y))

        Output = 1 - self.NormalizeValues(T)

        if Image.shape[-1] == 3:
            Segments = np.zeros(Image.shape)
            for Value in np.unique(Output):
                Binary = Output == Value
                Color = np.mean(Image[Binary], axis=0)
                Segments[Binary] = Color
            Output = Segments

        if Image.dtype == 'uint8':
            Output = np.round(Output).astype('uint8')

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output
PCNN = PCNNClass()

# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Scripts' / 'Pipeline' / 'Data')
        self.Path = str(Path.cwd() / 'Scripts' / 'Pipeline')
        self.N = 5
        self.Pixel_S = 1.0460251046025104
        self.ROI_S = 500

        # Add margin to ROI to minimize border effects in cleaning morphological operations
        self.Clean = True
        self.Margin = 100
Arguments = ArgumentsClass()


def Main(Arguments):

    # List manually segmented pictures
    DataDirectory = str(Path(Arguments.Path, 'ManualSegmentation'))
    Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.png')]
    Pictures.sort()

    # Store training and test pictures
    Data = {}
    for iPicture, Picture in enumerate(Pictures):
        Data[Picture[:-8]] = {}
        Data[Picture[:-8]]['ROI'] = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
        Seg = io.imread(str(Path(DataDirectory, Picture)))
        Data[Picture[:-8]]['Labels'] = ExtractLabels(Seg,DilateCM=iPicture<1)[0]


    # Extract picture features
    print('\nExtract manual segmentation features')
    Tic = time.time()
    Features, FNames = ExtractFeatures(Data)
    Toc = time.time()
    PrintTime(Tic, Toc)

    # Build data arrays
    FeaturesData = []
    LabelData = []
    for Key in Features.keys():
        ROIFeatures = Features[Key]
        FeaturesData.append(ROIFeatures.reshape(-1,ROIFeatures.shape[-1]))
        Labels = Data[Key]['Labels']
        LabelData.append(Labels.ravel())
    FeaturesData = np.vstack(FeaturesData)
    LabelData = np.hstack(LabelData)

    # Filter out non labelled data
    FeaturesData = FeaturesData[LabelData > 0]
    LabelData = LabelData[LabelData > 0]

    # Store into data frame and split training and test data
    Data = pd.DataFrame(FeaturesData)
    Data['Labels'] = LabelData
    Train, Test = train_test_split(Data)
    TrainFeatures = Train.drop('Labels', axis=1)
    TestFeatures = Test.drop('Labels', axis=1)
    TrainLabels = Train['Labels']
    TestLabels = Test['Labels']

    # Create classifier, assess accuracy and look at default parameters
    Classifier = RandomForestClassifier(n_jobs=-1, verbose=1)
    Classifier.fit(TrainFeatures,TrainLabels)
    OriginalPredictions = Classifier.predict(TestFeatures)
    print('\nModel accuracy:')
    print(round(metrics.accuracy_score(TestLabels, OriginalPredictions),3))
    Classifier.get_params()

    # Create parameter grid for randomized search
    nEstimators = [int(n) for n in np.linspace(start=20, stop=200, num=4)]
    MaxSamples = [n.round(1) for n in np.linspace(start=0.2, stop=0.8, num=4)]
    RandomGrid = {'n_estimators': nEstimators,
                  'max_samples': MaxSamples}

    # Random search of parameters, using 3 folds cross validation,
    # search across 100 different combinations, and use all available cores
    RandomSCV = RandomizedSearchCV(estimator=Classifier,
                                   param_distributions=RandomGrid,
                                   n_iter=10, cv=5, n_jobs=-1, verbose=2)
    # Fit the random search model to find best parameters
    RandomSCV.fit(TrainFeatures, TrainLabels)
    Best = RandomSCV.best_estimator_
    RandomSCV.best_params_

    # Check accuracy improvement
    Classifier.fit(TrainData,TrainLabel)
    Original = Classifier.predict(TestData)
    Optimized = Best.predict(TestData)


    Report = metrics.classification_report(TestLabel[TestLabel > 0], Optimized[TestLabel > 0])
    print(Report)

    # Create and train classifier
    print('\nCreate and train random forest classifier')
    Tic = time.time()
    Classifier = future.fit_segmenter(TrainLabel, TrainData, Classifier)
    Toc = time.time()
    PrintTime(Tic, Toc)

    # Perform predictions
    print('\nPerform preditions and assess model')
    Tic = time.time()
    Results = future.predict_segmenter(TrainData, Classifier)
    Toc = time.time()
    PrintTime(Tic, Toc)

    # Assess model
    CM = PlotConfusionMatrix(TrainLabel, Results, Ticks)

    # Print report
    Report = metrics.classification_report(Label.ravel(),Results.ravel())
    print(Report)

    # Feature importance
    FI = PlotFeatureImportance(Classifier, FNames.Names)
    FI = FI.sort_values(by='Importance',ascending=False)
    FI['Cum Sum'] = FI['Importance'].cumsum()
    Figure, Axis = plt.subplots(1,1)
    Axis.plot(FI['Cum Sum'].values, color=(1,0,0), marker='o', linestyle='--', fillstyle='none')
    Axis.set_ylim([0,1.05])
    Axis.set_ylabel('Relative importance (-)')
    Axis.set_xlabel('Feature number (-)')
    plt.show()
    FI[FI['Cum Sum'] < 0.8]['Importance'].value_counts()

    # Assess model with test image
    print('\nPerform preditions of test image')
    Tic = time.time()
    Results = future.predict_segmenter(TestData, Classifier)
    Toc = time.time()
    PrintTime(Tic, Toc)

    # Assess model
    CM = PlotConfusionMatrix(TestLabel[TestLabel > 0], Results[TestLabel > 0], Ticks)
    PlotResults(Test['ROI_3']['ROI'], np.reshape(Results,Test['ROI_3']['Labels'].shape))


    if Arguments.Save:
        joblib.dump(Classifier,str(Path(Arguments.Path, 'RFC.joblib')))

    Dict = {'Confusion Matrix':CM,
            'Feature Importance': FI}
    return Dict


if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('-s', '--Save', help='Save the trained classifier', type=bool, default=False)

    # Define paths
    DataDirectory = str(Path.cwd() / 'Scripts' / 'Pipeline')
    Parser.add_argument('-Path', help='Set data directory', type=str, default=DataDirectory)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)