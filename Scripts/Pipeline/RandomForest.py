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

    HSV = color.rgb2hsv(Array)
    F_HSV = mbf(HSV, multichannel=True, intensity=True, edges=True, texture=True,
                sigma_min=Smin, sigma_max=Smax)
    Features = np.dstack([Features, F_HSV])


    return Features, FNames

def PlotConfusionMatrix(GroundTruth, Results, Ticks):

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
    Axis.set_xticks(np.arange(len(Ticks)), Ticks)
    Axis.set_yticks(np.arange(len(Ticks)), Ticks)
    Axis.set_ylim([-0.49, CM.shape[0] - 0.5])
    Axis.set_title('Total: ' + str(GroundTruth[GroundTruth > 0].size))
    Axis.set_xlabel('Ground Truth', color=(0, 0, 1))
    Axis.set_ylabel('Predictions', color=(1, 0, 0))
    plt.show()

    return CM

def PlotFeatureImportance(Classifier, F_Names):

    FI = pd.DataFrame(Classifier.feature_importances_, columns=['Importance'])
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

    return FI

def Main(Arguments):

    # List manually segmented pictures
    DataDirectory = str(Path(Arguments.Path, 'ManualSegmentation'))
    Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.png')]
    Pictures.sort()

    # Extract picture features
    print('\nExtract manual segmentation features')
    Tic = time.time()
    for iPicture, Picture in enumerate(Pictures):
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
    Classifier = RandomForestClassifier(n_jobs=-1, max_samples=1/3, class_weight='balanced')
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