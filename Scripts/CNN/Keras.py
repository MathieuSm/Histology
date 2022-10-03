# https://youtu.be/5ct8Yqkiioo

"""
@author: Sreenivas Bhattiprolu
"""
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from skimage import io, transform, morphology, color
from keras import layers, Model

import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
def FeatureExtractor(Filters):

    Activation = 'sigmoid'
    FeatExt = Sequential()

    Layer = layers.Conv2D(Filters[0], 3, activation=Activation, padding='same', input_shape=(Arguments.Size, Arguments.Size, 3))
    FeatExt.add(Layer)
    Layer = layers.Conv2D(Filters[0], 3, activation=Activation, padding='same', kernel_initializer='he_uniform')
    FeatExt.add(Layer)

    for Layer in FeatExt.layers:
        Layer.trainable = False

    print(FeatExt.summary())

    return FeatExt
def Unet(Filters):

    Activation = 'sigmoid'

    ConLayers = []

    Input = layers.Input(shape=(Arguments.Size, Arguments.Size, 3))
    Layer = layers.Conv2D(Filters[0], 3, activation=Activation, padding='same')(Input)
    Layer = layers.Conv2D(Filters[0], 3, activation=Activation, padding='same')(Layer)
    Pool = layers.MaxPool2D(2)(Layer)
    ConLayers.append([Layer,Pool])

    for Filter in Filters[1:-1]:
        Layer = layers.Conv2D(Filter, 3, activation=Activation, padding='same')(Pool)
        Layer = layers.Conv2D(Filter, 3, activation=Activation, padding='same')(Layer)
        Pool = layers.MaxPool2D(2)(Layer)
        ConLayers.append([Layer, Pool])

    Layer = layers.Conv2D(Filter*2, 3, activation=Activation, padding='same')(Pool)
    Layer = layers.Conv2D(Filter*2, 3, activation=Activation, padding='same')(Layer)

    for i in np.arange(len(Filters)-1)+1:
        Layer = layers.Conv2DTranspose(Filters[-i], 3, 2, padding="same")(Layer)
        Layer = layers.concatenate([Layer, ConLayers[-i][0]])
        Layer = layers.Conv2D(Filters[-i], 3, activation=Activation, padding='same')(Layer)
        Layer = layers.Conv2D(Filters[-i], 3, activation=Activation, padding='same')(Layer)

    # outputs
    Outputs = layers.Conv2D(Filters[0], 1, padding="same", activation="softmax")(Layer)

    UNetModel = Model(Input, Outputs, name="U-Net")
    UNetModel.trainable = False
    print(UNetModel.summary())

    return UNetModel
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
def PlotOverlay(ROI,Seg, FileName=None):

    H, W = Seg.shape
    SegImage = np.zeros((H, W, 4))

    Colors = [(1,0,0,0.25),(0,0,1,0.25),(0,1,0,0.25),(1,1,1,0.25)]
    for iValue, Value in enumerate(np.unique(Seg)):
        Filter = Seg == Value
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


# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Scripts' / 'Pipeline' / 'Data')
        self.Path = str(Path.cwd() / 'Scripts' / 'Pipeline')
        self.Size = 1024
Arguments = ArgumentsClass()

# List manually segmented pictures
DataDirectory = str(Path(Arguments.Path, 'ManualSegmentation'))
Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.png')]
Pictures.sort()

# Store train images and label
Train = []
Labels = []
HSV = []
for iPicture, Picture in enumerate(Pictures[:-1]):
    Image = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
    Image = transform.resize(Image,(Arguments.Size,Arguments.Size,3),anti_aliasing=True)
    Train.append(Image)
    HSV.append(color.rgb2hsv(Image))

    Image = io.imread(str(Path(DataDirectory, Picture)))
    Image, Ticks = ExtractLabels(Image, DilateCM=bool(1-iPicture))
    # Image = (Image == 1) * 1
    Image = transform.resize(Image, (Arguments.Size, Arguments.Size, 1), order=0, preserve_range=True)
    Labels.append(Image.astype('int'))
Train = np.array(Train)
HSV = np.array(HSV)
Labels = np.array(Labels)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Labels[0])
plt.show()


# Store test image
Test = np.zeros((1,Arguments.Size,Arguments.Size,3))
Image = io.imread(str(Path(DataDirectory, Pictures[-1][:-8] + '.png')))
Image = transform.resize(Image,(Arguments.Size,Arguments.Size,3),anti_aliasing=True)
Test += Image
THSV = np.expand_dims(color.rgb2hsv(Image),0)
TestLabel = np.zeros((Arguments.Size,Arguments.Size))
Image = io.imread(str(Path(DataDirectory, Pictures[-1])))
Image = ExtractLabels(Image, DilateCM=False)[0]
Image = transform.resize(Image, (Arguments.Size, Arguments.Size), order=0, preserve_range=True)
TestLabel += Image


FeatExt = FeatureExtractor(2**np.arange(5,10))
FeatExt = Unet(2**np.arange(5,10))
# feature_extractor.add(BatchNormalization())
#
# FeatureExtractor.add(Conv2D(64, 3, activation=Activation, padding='same', kernel_initializer='he_uniform'))
# FeatureExtractor.add(BatchNormalization())
# FeatureExtractor.add(MaxPooling2D())
# FeatureExtractor.add(Flatten())

print('Extract features')
X1 = FeatExt.predict(Train)
X1 = X1.reshape(-1, X1.shape[3])
X2 = FeatExt.predict(HSV)
X2 = X2.reshape(-1, X2.shape[3])
Data = pd.DataFrame(np.hstack([X1,X2]))


Y = Labels.reshape(-1)
Data['Label'] = Y

print(Data['Label'].unique())
print(Data['Label'].value_counts())
Data = Data[Data['Label'] != 0]
print(Data['Label'].value_counts())

Features = Data.drop(labels=['Label'], axis=1)
Labels = Data['Label']


Classifier = RandomForestClassifier(n_jobs=-1, max_samples=0.3, class_weight='balanced')
Tic = time.time()
Classifier.fit(Features, Labels)
Toc = time.time()
PrintTime(Tic,Toc)
Results = Classifier.predict(Features)

# Assess model
CM = PlotConfusionMatrix(Labels, Results, Ticks)

# Print report
Report = metrics.classification_report(Labels.ravel(),Results.ravel())
print(Report)

TX1 = FeatExt.predict(Test)
TX1 = TX1.reshape(-1, TX1.shape[3])
TX2 = FeatExt.predict(THSV)
TX2 = TX2.reshape(-1, TX2.shape[3])
TestFeatures = np.hstack([TX1,TX2])

Prediction = Classifier.predict(TestFeatures)
PredictionImage = Prediction.reshape((Arguments.Size,Arguments.Size))

CM = PlotConfusionMatrix(TestLabel[TestLabel > 0], PredictionImage[TestLabel > 0], Ticks)

PlotOverlay(Test[0],PredictionImage)