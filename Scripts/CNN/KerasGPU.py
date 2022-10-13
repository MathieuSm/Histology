#%%
# Initialization

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from keras import utils, layers, Model
from sklearn.model_selection import train_test_split
from skimage import io, transform, morphology


#%%
# Define functions
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
def ConvulationBlock(Input, nFilters):

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Input)
    Layer = layers.Activation("relu")(Layer)

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Layer)
    Layer = layers.Activation("relu")(Layer)

    return Layer
def EncoderBlock(Input, nFilters):
    Layer = ConvulationBlock(Input, nFilters)
    Pool = layers.MaxPool2D((2, 2))(Layer)
    return Layer, Pool
def DecoderBlock(Input, SkipFeatures, nFilters):
    Layer = layers.Conv2DTranspose(nFilters, (2, 2), strides=2, padding="same")(Input)
    Layer = layers.Concatenate()([Layer, SkipFeatures])
    Layer = ConvulationBlock(Layer, nFilters)
    return Layer
def BuildUNet(InputShape, nClasses, nFilters=[64, 128, 256, 512, 1024]):

    Input = layers.Input(InputShape)
    Block = []
    Block.append(EncoderBlock(Input, nFilters[0]))
    for i, nFilter in enumerate(nFilters[1:-1]):
        Block.append(EncoderBlock(Block[i][1], nFilter))

    Bridge = ConvulationBlock(Block[-1][1], nFilters[-1])
    D = DecoderBlock(Bridge, Block[-1][0], nFilters[-2])

    for i, nFilter in enumerate(nFilters[-3::-1]):
        D = DecoderBlock(D, Block[-i+2][0], nFilter)

    if nClasses == 2:  #Binary
      Activation = 'sigmoid'
    else:
      Activation = 'softmax'

    Outputs = layers.Conv2D(nClasses, 1, padding="same", activation=Activation)(D)
    UNet = Model(Input, Outputs, name='U-Net')
    return UNet
def PlotHistory(History):
    Loss = History.history['loss']
    ValLoss = History.history['val_loss']
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(range(1, len(Loss) + 1), Loss, color=(0, 0, 1), marker='o', linestyle='--', label='Training loss')
    Axis.plot(range(1, len(Loss) + 1), ValLoss, color=(1, 0, 0), marker='o', linestyle='--', label='Validation loss')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Loss')
    Axis.legend()
    # plt.show()

    Accuracy = History.history['accuracy']
    ValAccuracy = History.history['val_accuracy']
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(range(1, len(Accuracy) + 1), Accuracy, color=(0, 0, 1), marker='o', linestyle='--', label='Training accuracy')
    Axis.plot(range(1, len(Accuracy) + 1), ValAccuracy, color=(1, 0, 0), marker='o', linestyle='--', label='Validation accuracy')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Accuracy')
    Axis.legend()
    # plt.show()

    return
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
def PlotImage(Array):

    Figure, Axis = plt.subplots(1,1,figsize=(10,10))
    if Array.shape[-1] == 3:
        Axis.imshow(Array)
    else:
        Axis.imshow(Array, cmap='binary_r')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()

#%%
# For testing purpose
class ArgumentsClass:

    def __init__(self):
        
        self.Data = str(Path.cwd() / 'Scripts' / 'Pipeline' / 'Data')
        self.Path = str(Path.cwd() / '..' / 'Pipeline')
        self.Size = 1024
Arguments = ArgumentsClass()

#%%
# List manually segmented pictures
DataDirectory = str(Path(Arguments.Path, 'ManualSegmentation'))
Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.png')]
Pictures.sort()

# Store pictures
PicturesData = {}
for iPicture, Picture in enumerate(Pictures[1:]):
    PicturesData[Picture[:-8]] = {}
    ROI = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
    Seg = io.imread(str(Path(DataDirectory, Picture)))
    PicturesData[Picture[:-8]]['ROI'] = ROI

    # Extract segments
    CL = Seg[:, :, 0] == 255
    OC = Seg[:, :, 1] == 255
    HC = CL * OC

    # Label cement lines segments
    Label = np.zeros(HC.shape, 'uint8')
    Label[CL] = 1
    Label[OC] = 2
    Label[HC] = 3

    PicturesData[Picture[:-8]]['Seg'] = Label

#%%

# Perform data augmentation
Data = []
Labels = []
for K in PicturesData.keys():

    ROI = PicturesData[K]['ROI']
    HC = PicturesData[K]['Seg']
    N = 96
    AugData, AugLabels = DataAugmentation(ROI, HC, N)

    for iN in range(N):
        Data.append(AugData[iN])
        Labels.append(AugLabels[iN])
Data = np.array(Data)
Labels = np.array(Labels).astype('int')
Labels = np.expand_dims(Labels,-1)

#%%

# Split into train and test data
XTrain, XTest, YTrain, YTest = train_test_split(Data, Labels)
YTrainCat = utils.to_categorical(YTrain)
YTestCat = utils.to_categorical(YTest)

#%%
# Build UNet
UNet = BuildUNet(XTrain.shape[1:], YTrainCat.shape[-1])
UNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(UNet.summary())
History = UNet.fit(XTrain,YTrainCat, validation_data=(XTest,YTestCat), verbose=2, epochs=50, workers=2)
UNet.save('Unet.hdf5')
PlotHistory(History)

#%%
# Look at testing image
Random = np.random.randint(0, len(XTest)-1)
TestImage = XTest[Random]
TestLabel = YTest[Random]
Prediction = UNet.predict(np.expand_dims(TestImage,0))

Figure, Axis = plt.subplots(1,3)
Axis[0].imshow(TestImage)
Axis[1].imshow(TestLabel, cmap='binary_r')
Axis[2].imshow(Prediction[0,:,:,1] > 0.01, cmap='binary_r')
for i in range(3):
    Axis[i].axis('off')
plt.tight_layout()
plt.show()