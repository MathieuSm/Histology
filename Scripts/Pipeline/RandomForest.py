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

    def __init__(self, SigmaMin=4, SigmaMax=32, Channels=['R','G','B'], Features=['I', 'E', 'H1', 'H2']):

        NumSigma = int(np.log2(SigmaMax / SigmaMin) + 1)
        F_Names = []
        for Channel in Channels:
            for Sigma in range(NumSigma):
                SigmaValue = SigmaMin * 2 ** Sigma
                if SigmaValue >= 1:
                    SigmaValue = int(SigmaValue)
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

def ExtractFeatures(Array):

    """
    Extract image features to train classifier and segment images
    :param Array: 3D np array (2D + RGB)
    :return: Features extracted
    """

    Smin, Smax = 0.5, 8
    FNames = FeatureNames(Smin, Smax, Channels=['R', 'G', 'B', 'H', 'S', 'V'], Features=['I', 'E', 'H1', 'H2'])

    Features = mbf(Array, multichannel=True, intensity=True, edges=True, texture=True,
                sigma_min=Smin, sigma_max=Smax)

    HSV = color.rgb2lab(Array)
    F_HSV = mbf(HSV, multichannel=True, intensity=True, edges=True, texture=True,
                sigma_min=Smin, sigma_max=Smax)
    Features = np.dstack([Features, F_HSV])


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

        elif len(Sigmas) < 4:
            Figure, Axis = plt.subplots(1, len(Sigmas), sharex=True, sharey=True)

            for i, Sigma in enumerate(Sigmas):
                for iC, C in enumerate(CList):
                    F = C[C['Sigma'] == Sigma]
                    S = F.sort_values(by='Feature')
                    Axis[i].bar(np.arange(len(S)), S['Importance'], edgecolor=CMap(iC / len(CList)), facecolor=(0, 0, 0, 0))
                    Axis[i].set_xticks(np.arange(len(Features)), Features)
                    Axis[i].set_title('Sigma = ' + Sigma)
                    Axis[i].plot([], color=CMap(iC / len(CList)), label=C['Channel'].unique())

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

    def GetNeighbours(self, Array, N=1, Map=False):

        Range = np.arange(2 * N + 1) - N
        Rows, Cols = Array.shape
        iRows, iCols = np.arange(Rows), np.arange(Cols)
        Cols, Rows = np.meshgrid(iRows,iCols)


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
        Neighbours = (2 * N + 1)**2 - 1

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
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[1:-1, 1:-1]
                i += 1

        for Shift in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            for Axis in [(0, 1)]:
                Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[1:-1, 1:-1]
                i += 1

        if N == 2:

            # Pad again the array to avoid border effects
            if len(Array2D.shape) > 2:
                Array2D = np.pad(Array2D, ((1, 1), (1, 1), (0, 0)), 'symmetric')
            else:
                Array2D = np.pad(Array2D, 1, 'symmetric')

            for Shift in [-2, 2]:
                for Axis in [0, 1]:
                    Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2, 2:-2]
                    i += 1

            for Shift in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
                for Axis in [(0, 1)]:
                    Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2, 2:-2]
                    i += 1

            for Shift in [(-2, -1), (2, -1), (-2, 1), (2, 1)]:
                for Axis in [(0, 1)]:
                    Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2, 2:-2]
                    i += 1

            for Shift in [(-1, -2), (1, -2), (-1, 2), (1, 2)]:
                for Axis in [(0, 1)]:
                    Neighbourhood[:, :, i] = np.roll(Array2D, Shift, axis=Axis)[2:-2, 2:-2]
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

    def Enhance(self, Image):

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

            Laplacian = np.array([[1 / 6, 2 / 3, 1 / 6, 2 / 3, -10 / 3, 2 / 3, 1 / 6, 2 / 3, 1 / 6]])
            Theta = 1 + np.sum(GetNeighbours(S)[0] * Laplacian, axis=2)
            f = 0.75 * np.exp(-S**2 / 0.16) + 0.05
            G = GaussianKernel(7, 1)
            f = correlate(f, G, mode='reflect')
            # f = gaussian(f,sigma=1,mode='reflect')

            h = 2E10
            d = 2
            g = 0.9811
            Alpha = 0.01
            Beta = 0.03

            while SumY < S.size:
                N += 1

                K = correlate(Y, W, mode='reflect')
                Wave = Alpha * K + Beta * S * (K - d)
                U = f * U + S + Wave
                Theta = g * Theta + h * Y
                Y = (U > Theta) * 1
                T += N * Y
                SumY += sum(sum(Y))
                # print(SumY)

            T_inv = T.max() + 1 - T
            Time = (NormalizeArray(T_inv) * 255).astype('uint8')
            Stretched = GrayStretch(Time, 0.99)

            return Stretched

        if len(Image.shape) == 2:
            Output = FLM(Image)
        else:
            Output = np.zeros(Image.shape)
            for i in range(Image.shape[-1]):
                Output[:,:,i] = FLM(Image)[:,:,i]



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

        # Initialize parameters
        S = NormalizeValues(Image)
        Rows, Columns = S.shape
        Y = np.zeros((Rows, Columns))
        T = np.zeros((Rows, Columns))
        W = np.ones((Rows, Columns, 8)) * np.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])
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
    Train = {}
    Test = {}
    for iPicture, Picture in enumerate(Pictures):

        if iPicture < len(Pictures) - 1:
            Train[Picture[:-8]] = {}
            Train[Picture[:-8]]['ROI'] = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
            Seg = io.imread(str(Path(DataDirectory, Picture)))
            Train[Picture[:-8]]['Labels'] = ExtractLabels(Seg,DilateCM=bool(1-iPicture))[0]

        else:
            Test[Picture[:-8]] = {}
            Test[Picture[:-8]]['ROI'] = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
            Seg = io.imread(str(Path(DataDirectory, Picture)))[:,:,:-1]
            Test[Picture[:-8]]['Labels'], Ticks = ExtractLabels(Seg,DilateCM=False)


    def ExtractFeatures(Dict):

        Features = {}
        for Key in Dict.keys():
            Shape = Dict[Key]['ROI'].shape
            ROIFeatures = np.zeros((Shape[0],Shape[1],9))
            ROIFeatures[:, :, :3] = Dict[Key]['ROI']
            ROIFeatures[:, :, 3:6] = color.rgb2hsv(Dict[Key]['ROI'])
            ROIFeatures[:, :, 6:] = color.rgb2lab(Dict[Key]['ROI'])
            Features[Key] = ROIFeatures

        return Features

    TrainFeatures = ExtractFeatures(Train)
    TestFeatures = ExtractFeatures(Test)

    # Build data frames
    TrainData = []
    TrainLabel = []
    for Key in TrainFeatures.keys():
        Features = TrainFeatures[Key]
        TrainData.append(Features.reshape(-1,Features.shape[-1]))
        Labels = Train[Key]['Labels']
        TrainLabel.append(Labels.ravel())
    TrainData = np.vstack(TrainData)
    TrainLabel = np.hstack(TrainLabel)

    # Filter out non labelled data
    TrainData = TrainData[TrainLabel > 0]
    TrainLabel = TrainLabel[TrainLabel > 0]

    TestData = []
    TestLabel = []
    for Key in TestFeatures.keys():
        Features = TestFeatures[Key]
        TestData.append(Features.reshape(-1, Features.shape[-1]))
        Labels = Test[Key]['Labels']
        TestLabel.append(Labels.ravel())
    TestData = np.vstack(TestData)
    TestLabel = np.hstack(TestLabel)



    # Extract picture features
    print('\nExtract manual segmentation features')
    Tic = time.time()
    for iPicture, Picture in enumerate(Pictures[:-1]):
        ROI = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
        Seg_ROI = io.imread(str(Path(DataDirectory, Picture)))

        if iPicture == 0:
            Label, Ticks = ExtractLabels(Seg_ROI, DilateCM=True)
            Features, FNames = ExtractFeatures(ROI)

            Features = Features[Label > 0]
            Label = Label[Label > 0]

        else:
            NewLabel = ExtractLabels(Seg_ROI)[0]
            NewFeatures = ExtractFeatures(ROI)[0]

            Features = np.vstack([Features, NewFeatures[NewLabel > 0]])
            Label = np.concatenate([Label, NewLabel[NewLabel > 0]])

    Toc = time.time()
    PrintTime(Tic, Toc)

    # Create and train classifier
    print('\nCreate and train random forest classifier')
    Tic = time.time()
    Classifier = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_samples=1/3, class_weight='balanced')
    Classifier = future.fit_segmenter(Label, Features, Classifier)
    Toc = time.time()
    PrintTime(Tic, Toc)

    # Perform predictions
    print('\nPerform preditions and assess model')
    Tic = time.time()
    Results = future.predict_segmenter(Features, Classifier)
    Toc = time.time()
    PrintTime(Tic, Toc)

    # Assess model
    CM = PlotConfusionMatrix(Label, Results, Ticks)

    # Print report
    Report = metrics.classification_report(Label.ravel(),Results.ravel())
    print(Report)

    # Feature importance
    FI = PlotFeatureImportance(Classifier, FNames.Names)
    FI = FI.sort_values(by='Importance',ascending=False)
    FI['Cum Sum'] = FI['Importance'].cumsum()
    Figure, Axis = plt.subplots(1,1)
    Axis.plot(FI['Cum Sum'].values, color=(1,0,0), marker='o', linestyle='--', fillstyle='none')
    Axis.set_ylim([0,1])
    Axis.set_ylabel('Relative importance (-)')
    Axis.set_xlabel('Feature number (-)')
    plt.show()
    FI[FI['Cum Sum'] < 0.8]['Feature'].value_counts()

    # Assess model with test image
    Picture = Pictures[-1]
    ROI = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
    Seg_ROI = io.imread(str(Path(DataDirectory, Picture)))

    TestLabel = ExtractLabels(Seg_ROI)[0]
    TestFeatures = ExtractFeatures(ROI)[0]

    print('\nPerform preditions of test image')
    Tic = time.time()
    Results = future.predict_segmenter(TestFeatures, Classifier)
    Toc = time.time()
    PrintTime(Tic, Toc)

    # Assess model
    CM = PlotConfusionMatrix(TestLabel[TestLabel > 0], Results[TestLabel > 0], Ticks)
    PlotResults(ROI, Results)


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