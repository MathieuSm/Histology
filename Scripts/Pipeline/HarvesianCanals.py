#!/usr/bin/env python3

"""
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
#sphx-glr-auto-examples-segmentation-plot-trainable-segmentation-py
"""

import os
import pickle
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LinearSegmentedColormap
from skimage import io, future, exposure, morphology, color, transform
from skimage.feature import multiscale_basic_features as mbf
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV

plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    To write.

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: September 2022
    """


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

def DataAugmentation(Image,Label,N):

    ISize = Image.shape[:-1]
    ASize = (Arguments.Size, Arguments.Size)

    # Normalize data
    Image = Image / 255
    # Label = (Label - Label.min()) / (Label.max() - Label.min())

    Data = []
    Labels = []

    Tic = time.time()
    print('\nStart data augmentation')
    for iN in range(N):

        Rot = np.random.randint(0, 360)
        rImage = transform.rotate(Image, Rot)
        rLabel = transform.rotate(Label, Rot, order=0, preserve_range=True)

        Flip = np.random.binomial(1, 0.5, 2)
        if sum(Flip) == 0:
            fImage = rImage
            fLabel = rLabel
        if Flip[0] == 1:
            fImage = rImage[::-1, :, :]
            fLabel = rLabel[::-1, :]
        if Flip[1] == 1:
            fImage = rImage[:, ::-1, :]
            fLabel = rLabel[:, ::-1]

        X1 = np.random.randint(0, ISize[1] - ASize[1] - 1)
        Y1 = np.random.randint(0, ISize[0] - ASize[0] - 1)
        X2, Y2 = X1 + ASize[1], Y1 + ASize[0]
        cImage = fImage[Y1:Y2,X1:X2]
        cLab = fLabel[Y1:Y2,X1:X2]

        Data.append(cImage)
        Labels.append(cLab)

    Toc = time.time()
    PrintTime(Tic,Toc)

    return Data, Labels

def ExtractFeatures(Images):

    Widths = [5, 10, 20, 50]
    Features = []
    print('\nExtract ROI features')
    Tic = time.time()
    for ROI in Images:
        ROIFeatures = np.zeros((ROI.shape[0], ROI.shape[1], len(Widths) + 6))
        ROIFeatures[:, :, :3] = ROI
        ROIFeatures[:, :, 3:6] = color.rgb2hsv(ROI)
        for iW, WindowHalfWidth in enumerate(Widths):
            Values = ROI.sum(axis=-1)
            pROI = np.pad(Values, WindowHalfWidth, 'symmetric')
            VaW = view_as_windows(pROI, WindowHalfWidth * 2 + 1)
            Convolution = VaW.sum(axis=(-1, -2))
            ROIFeatures[:, :, iW + 6] = Convolution
        Features.append(ROIFeatures)
    Features = np.array(Features)
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Features

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

# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Scripts' / 'Pipeline' / 'Data')
        self.Path = str(Path.cwd() / 'Scripts' / 'Pipeline')
        self.N = 5
        self.Pixel_S = 1.0460251046025104
        self.ROI_S = 500
        self.Size = 1000

        # Add margin to ROI to minimize border effects in cleaning morphological operations
        self.Clean = True
        self.Margin = 100
Arguments = ArgumentsClass()


def Main(Arguments):

    # List manually segmented pictures
    DataDirectory = str(Path(Arguments.Path, 'ManualSegmentation'))
    Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.png')]
    Pictures.sort()

    # Store pictures
    PicturesData = {}
    for iPicture, Picture in enumerate(Pictures[1:]):
        PicturesData[Picture[:-8]] = {}
        ROI = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
        PlotImage(ROI)
        Seg = io.imread(str(Path(DataDirectory, Picture)))
        PlotImage(Seg)
        PicturesData[Picture[:-8]]['ROI'] = ROI
        PicturesData[Picture[:-8]]['HC'] = (Seg[:, :, 0] == 255) * (Seg[:, :, 1] == 255)

    # Perform data augmentation
    Images = []
    Labels = []
    N = 5
    for K in PicturesData.keys():
        ROI = PicturesData[K]['ROI']
        HC = PicturesData[K]['HC'] * 1 + 1

        AugData, AugLabels = DataAugmentation(ROI, HC, N)

        for iN in range(N):
            Images.append(AugData[iN])
            Labels.append(AugLabels[iN])
    Images = np.array(Images)
    Labels = np.array(Labels).astype('int')
    Labels = np.expand_dims(Labels, -1)

    # Extract features
    Features = ExtractFeatures(Images)

    # Filter out non-labeled data
    Features = Features[:,:,:][Labels[:,:,:,0] > 0]
    FeaturesLabels = Labels[Labels > 0]

    # Store into data frame and balance sampling
    Data = pd.DataFrame(Features)
    Data['Labels'] = FeaturesLabels
    nLabels = Data.value_counts('Labels').min()
    Data = Data.groupby('Labels').sample(nLabels)

    # Split training and test data
    Train, Test = train_test_split(Data)
    TrainFeatures = Train.drop('Labels', axis=1)
    TestFeatures = Test.drop('Labels', axis=1)
    TrainLabels = Train['Labels']
    TestLabels = Test['Labels']

    # Fit classifier and record metrics
    Classifier = RandomForestClassifier(n_jobs=-1, verbose=2)
    Tic = time.time()
    Classifier.fit(TrainFeatures, TrainLabels)
    Toc = time.time()
    Predictions = Classifier.predict(TestFeatures)
    PrintTime(Tic,Toc)

    # Assess model
    print('Accuracy: ' + str(round(metrics.accuracy_score(TestLabels, Predictions),2)))
    CM = PlotConfusionMatrix(TestLabels, Predictions, ['IT','HC'])

    # Check results on images
    Random = np.random.randint(0, len(Images)-1)
    Image = Images[Random]
    Label = Labels[Random]
    ImageFeatures = ExtractFeatures([Image])[0]
    Prediction = Classifier.predict(np.reshape(ImageFeatures,(1000*1000,10)))
    Prediction = Prediction.reshape((1000,1000,1))
    Prediction[Label == 0] = 1
    Label[Label == 0] = 1

    Figure, Axis = plt.subplots(1,3)
    Axis[0].imshow(Image)
    Axis[0].set_title('Image')
    Axis[1].imshow(Label,cmap='binary_r')
    Axis[1].set_title('Ground Truth')
    Axis[2].imshow(Prediction, cmap='binary_r')
    Axis[2].set_title('Prediction')
    for i in range(3):
        Axis[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Dice coefficient
    2 * np.sum((Label-1)*(Prediction-1)) / np.sum((Label-1) + (Prediction-1))

    # Improve results with morphological operations
    pPrediction = np.pad(Prediction[:,:,0]-1, 20, 'symmetric').astype('bool')
    Clean = morphology.remove_small_objects(pPrediction,100)[20:-20,20:-20]

    Figure, Axis = plt.subplots(1,3)
    Axis[0].imshow(Image)
    Axis[0].set_title('Image')
    Axis[1].imshow(Label,cmap='binary_r')
    Axis[1].set_title('Ground Truth')
    Axis[2].imshow(Clean, cmap='binary_r')
    Axis[2].set_title('Prediction')
    for i in range(3):
        Axis[i].axis('off')
    plt.tight_layout()
    plt.show()

    PlotImage(Label[:,:,0] + Clean)

    # Save model
    joblib.dump(Classifier,str(Path(Arguments.Path, 'HCC.joblib')))




