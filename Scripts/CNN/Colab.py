#%%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from pathlib import Path
from patchify import patchify
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from skimage import io, transform, morphology
from sklearn.model_selection import train_test_split
from focal_loss import SparseCategoricalFocalLoss as SFL
from tensorflow.keras import layers, utils, Model, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# https://youtu.be/-XeKG_T6tdc



#%%

def SegmentBone(Image):

    """
    Segment bone area using simple threshold
    """

    # Mark areas where there is bone
    Filter1 = Image[:, :, 0] < 190
    Filter2 = Image[:, :, 1] < 190
    Filter3 = Image[:, :, 2] < 235
    Bone = Filter1 & Filter2 & Filter3

    # Erode and dilate to remove small bone parts
    Bone = morphology.remove_small_objects(~Bone, 15)
    Bone = morphology.binary_closing(Bone, morphology.disk(25))

    return ~Bone
def ConvulationBlock(Input, nFilters, DropRate, BatchNorm):

    Activation = 'relu'

    Layer = layers.Conv2D(nFilters, 3, padding="same", kernel_initializer='he_uniform')(Input)
    if BatchNorm:
        layers.BatchNormalization(axis=-1)(Layer)
    Layer = layers.Activation(Activation)(Layer)

    Layer = layers.Conv2D(nFilters, 3, padding="same", kernel_initializer='he_uniform')(Layer)
    if BatchNorm:
        layers.BatchNormalization(axis=-1)(Layer)
    Layer = layers.Activation(Activation)(Layer)

    if DropRate:
        Layer = layers.Dropout(DropRate)(Layer)

    return Layer
def EncoderBlock(Input, nFilters, DropRate, BatchNorm):
    Layer = ConvulationBlock(Input, nFilters, DropRate, BatchNorm)
    Pool = layers.MaxPool2D((2, 2))(Layer)
    return Layer, Pool
def GatingSignal(Input, nFilters, BatchNorm):
    """
    Resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    Layer = layers.Conv2D(nFilters, (1, 1), padding='same')(Input)
    if BatchNorm:
        Layer = layers.BatchNormalization(axis=-1)(Layer)
    Layer = layers.Activation('relu')(Layer)
    return Layer
def AttentionBlock(Layer, Gating, nFilters, BatchNorm):

    LayerShape = K.int_shape(Layer)
    GatingShape = K.int_shape(Gating)

    # Same shape fot Layer signal as for Gating signal
    Theta = layers.Conv2D(nFilters, (2,2), padding='same')(Layer)
    ThetaShape = K.int_shape(Theta)

    # Same nFilters for Gating signal as for Layer shape
    Phi = layers.Conv2D(nFilters, (1,1), padding='same')(Gating)
    Strides = (ThetaShape[1] // GatingShape[1], ThetaShape[2] // GatingShape[2])
    Phi = layers.Conv2DTranspose(nFilters, (3,3), strides=Strides, padding='same')(Phi)

    Psi = layers.add([Phi, Theta])
    Psi = layers.Activation('relu')(Psi)
    Psi = layers.Conv2D(1, (1,1), padding='same')(Psi)
    Psi = layers.Activation('softmax')(Psi)
    PsiShape = K.int_shape(Psi)
    Size = (LayerShape[1] // PsiShape[1], LayerShape[2] // PsiShape[2])
    Psi = layers.UpSampling2D(size=Size)(Psi)
    Psi = K.repeat_elements(Psi, LayerShape[-1], axis=-1)
    AttentionLayer = layers.multiply([Psi, Layer])

    AttentionLayer = layers.Conv2D(LayerShape[-1], (1,1), padding='same')(AttentionLayer)

    if BatchNorm:
        AttentionLayer = layers.BatchNormalization(axis=-1)(AttentionLayer)

    return AttentionLayer
def DecoderBlock(Input, SkipFeatures, nFilters, DropRate, BatchNorm, Type='Standard'):

    if Type == 'Standard':
        Layer = layers.Conv2DTranspose(nFilters, (2, 2), strides=2, padding="same")(Input)
    elif Type == 'Attention':
        Layer = layers.UpSampling2D((2,2), data_format='channels_last')(Input)
    Layer = layers.Concatenate()([Layer, SkipFeatures])
    Layer = ConvulationBlock(Layer, nFilters, DropRate, BatchNorm)
    return Layer
def BuildUNet(InputShape, nClasses, nFilters=[12, 32, 64, 128, 256], DropRates=[0.1,0.1,0.2,0.2,0.3], BatchNorm=True, Type='Standard'):

    Input = layers.Input(InputShape)
    Block = []
    Block.append(EncoderBlock(Input, nFilters[0], DropRates[0], BatchNorm))
    for i, nFilter in enumerate(nFilters[1:-1]):
        Block.append(EncoderBlock(Block[i][1], nFilter, DropRates[i+1], BatchNorm))

    Bridge = ConvulationBlock(Block[-1][1], nFilters[-1], DropRates[-1], BatchNorm)

    if Type == 'Standard':
        SkipFeatures = Block[-1][0]
    elif Type == 'Attention':
        Gating = GatingSignal(Bridge, nFilters[-2], BatchNorm)
        SkipFeatures = AttentionBlock(Block[-1][0], Gating, nFilters[-2], BatchNorm)

    D = DecoderBlock(Bridge, SkipFeatures, nFilters[-2], DropRates[-2], BatchNorm)

    for i, nFilter in enumerate(nFilters[-3::-1]):

        if Type == 'Standard':
            SkipFeatures = Block[-i+2][0]
        elif Type == 'Attention':
            Gating = GatingSignal(Bridge, nFilter, BatchNorm)
            SkipFeatures = AttentionBlock(Block[-i+2][0], Gating, nFilter, BatchNorm)

        D = DecoderBlock(D, SkipFeatures, nFilter, DropRates[-i+2], BatchNorm)

    if nClasses == 2:  #Binary
      Activation = 'sigmoid'
    else:
      Activation = 'softmax'

    Outputs = layers.Conv2D(nClasses, 1, padding="same")(D)
    if BatchNorm:
        Outputs = layers.BatchNormalization(axis=-1)(Outputs)
    Outputs = layers.Activation(Activation)(Outputs)
    UNet = Model(Input, Outputs, name='U-Net')
    return UNet
def PlotHistory(History):

    Loss = History.history['loss']
    ValLoss = History.history['val_loss']
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(range(1, len(Loss) + 1), Loss, color=(0, 1, 1), marker='o', linestyle='--', label='Training loss')
    Axis.plot(range(1, len(Loss) + 1), ValLoss, color=(1, 0, 0), marker='o', linestyle='--', label='Validation loss')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Loss')
    Axis.legend()
    plt.show()

    Accuracy = History.history['accuracy']
    ValAccuracy = History.history['val_accuracy']
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(range(1, len(Accuracy) + 1), Accuracy, color=(0, 1, 1), marker='o', linestyle='--', label='Training accuracy')
    Axis.plot(range(1, len(Accuracy) + 1), ValAccuracy, color=(1, 0, 0), marker='o', linestyle='--', label='Validation accuracy')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Accuracy')
    Axis.legend()
    plt.show()

    return


#%%
# Read and store pictures

DataDirectory = str(Path.cwd() / '..' / 'Pipeline' / 'ManualSegmentation')
Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.png')]
Pictures.sort()

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
# Create data with pictures patches

Data = []
Labels = []
PatchSize = 512 
# Resize images to have 256x256 shape if PatchSize is bigger than 256
# -> solve memory issue
for Key in PicturesData.keys():

    Picture = PicturesData[Key]['ROI']
    Label = PicturesData[Key]['Seg']

    PadWidth = PatchSize - np.mod(Picture.shape[0],PatchSize)
    Pad1, Pad2 = int(np.ceil(PadWidth/2)), int(np.floor(PadWidth/2))

    Padded = np.pad(Picture,((Pad1,Pad2),(Pad1,Pad2),(0,0)),mode='reflect')
    PatchedP = patchify(Padded, (PatchSize, PatchSize, 3), step=PatchSize)

    Padded = np.pad(Label,((Pad1,Pad2),(Pad1,Pad2)),mode='reflect')
    PatchedL = patchify(Padded, (PatchSize, PatchSize), step=PatchSize)
    for i in range(PatchedP.shape[0]):
        for j in range(PatchedP.shape[1]):

                if PatchSize > 256:
                    ImageP = transform.resize(PatchedP[i,j,0], (256,256,Picture.shape[-1]), preserve_range=True)
                    ImageP = np.round(ImageP).astype('uint8')
                    ImageA = transform.resize(PatchedL[i,j], (256,256), anti_aliasing=False, order=0)
                    ImageL = np.zeros(ImageA.shape)
                    for iv, v in enumerate(np.unique(ImageA)):
                        Bin = ImageA == v
                        if iv == 1:
                            Disk = 2
                        else:
                            Disk = 1
                        Bin = morphology.binary_dilation(Bin, morphology.disk(Disk))
                        Bin = morphology.binary_erosion(Bin, morphology.disk(1))
                        ImageL[Bin] = iv

                else:
                    ImageP = PatchedP[i,j,0]
                    ImageL = PatchedL[i,j,0]

                Data.append(ImageP)
                Labels.append(ImageL)
Data = np.array(Data)
Labels = np.array(Labels)
Labels = np.expand_dims(Labels,-1)


#%%
# Filter data to remove too low BVTV areas
FData = []
FLabels = []
for iImage, Image in enumerate(Data):
    Bone = SegmentBone(Image)
    BVTV = Bone.sum() / Bone.size
    if BVTV > 0.88:
        FData.append(Image)
        FLabels.append(Labels[iImage])
FData = np.array(FData)
FLabels = np.array(FLabels)

#%%
# Separate into train and test data and build unet model
TrainX, TestX, TrainY, TestY = train_test_split(FData, FLabels, random_state = 0)
TrainYCat = utils.to_categorical(TrainY)
TestYCat = utils.to_categorical(TestY)


#%%
#Sanity check, view few mages
Random = np.random.randint(0, len(TrainX))
Figure, Axis = plt.subplots(1,2)
Axis[0].imshow(TrainX[Random])
Axis[0].axis('Off')
Axis[1].imshow(TrainY[Random])
Axis[1].axis('Off')
plt.show()

#%%
# Define data augmentation
Seed=24

DataArgs = {'rescale':1/255,
            'rotation_range':90,
            'width_shift_range':0.3,
            'height_shift_range':0.3,
            'horizontal_flip':True,
            'vertical_flip':True,
            'fill_mode':'reflect',
            'featurewise_center':False,
            'samplewise_center':False,
            'featurewise_std_normalization':False,
            'samplewise_std_normalization':False}

LabelsArgs = {'rotation_range':90,
              'width_shift_range':0.3,
              'height_shift_range':0.3,
              'horizontal_flip':True,
              'vertical_flip':True,
              'fill_mode':'reflect',
              'preprocessing_function':lambda x: np.round(x).astype('int'),
              'featurewise_center':False,
              'samplewise_center':False,
              'featurewise_std_normalization':False,
              'samplewise_std_normalization':False} 

DataGenerator = ImageDataGenerator(**DataArgs)
LabelsGenerator = ImageDataGenerator(**LabelsArgs)

# DataGenerator.fit(TrainX, augment=True, seed=Seed)
# LabelsGenerator.fit(TrainY, augment=True, seed=Seed)

TrainDataGenerator = DataGenerator.flow(TrainX, seed=Seed)
TestDataGenerator = DataGenerator.flow(TestX, seed=Seed)

TrainLabelsGenerator = LabelsGenerator.flow(TrainY, seed=Seed)
TestLabelsGenerator = LabelsGenerator.flow(TestY, seed=Seed)

def MyGenerator(DataGenerator, LabelsGenerator):
    Generators = zip(DataGenerator, LabelsGenerator)
    for (Data, Labels) in Generators:
        yield (Data, Labels)

TrainGenerator = MyGenerator(TrainDataGenerator, TrainLabelsGenerator)
TestGenerator = MyGenerator(TestDataGenerator, TestLabelsGenerator)

#%%
# Check augmented data
D = TrainDataGenerator.next()
L = TrainLabelsGenerator.next()
Figure, Axis = plt.subplots(1,2)
Axis[0].imshow(np.round(D[0]*255).astype('uint8'))
Axis[0].axis('Off')
Axis[1].imshow(np.sum(L[0], axis=-1))
Axis[1].axis('Off')
plt.show()

#%%
# Compute class weigths on training data

CW = class_weight.compute_class_weight('balanced', classes=np.unique(TrainY), y=TrainY.ravel())

#%%
# Build Unet

UNet = BuildUNet(Data.shape[1:],TrainYCat.shape[-1], DropRates=[0.25,0.25,0.25,0.25,0.25], Type='Standard')
UNet.compile(optimizer='adam', loss=[SFL(gamma=3, class_weight=[0, 1/3, 1/3, 1/3])], metrics=['accuracy'])
print(UNet.summary())

#%%
# Set callbacks
File = Path('Models', 'UNet_{epoch:04d}_{val_accuracy:.3f}.hdf5')
CP = callbacks.ModelCheckpoint(str(File), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
ES = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
Callbacks = [CP,ES]
#%%
# Fit Unet and plot history

BatchSize = 32
StepsPerEpoch = 16 #3*(len(TrainX))//BatchSize
History = UNet.fit(TrainGenerator, validation_data=TestGenerator,
                   steps_per_epoch=StepsPerEpoch, validation_steps=StepsPerEpoch,
                   epochs=20, callbacks=Callbacks)
# History = UNet.fit(TrainGenerator, validation_data=TestGenerator, epochs=50)
UNet.save('Unet.hdf5')
PlotHistory(History)

#%%
# Look at testing image
Random = np.random.randint(0, len(TestX)-1)
TestImage = TestX[Random]
TestLabel = TestY[Random]

# Load best weigths and look at prediction
UNet.load_weights('Models/UNet_0009_0.542.hdf5')
Prediction = UNet.predict(np.expand_dims(TestImage,0))
PredictionClasses = np.argmax(Prediction,axis=-1)[0]

Figure, Axis = plt.subplots(1,3)
Axis[0].imshow(TestImage)
Axis[0].set_title('Image')
Axis[1].imshow(TestLabel[:,:,0], vmin=0, vmax=3, interpolation='none')
Axis[1].set_title('Labels')
Axis[2].imshow(PredictionClasses, vmin=0, vmax=3, interpolation='none')
Axis[2].set_title('Predicitons')
for i in range(3):
        Axis[i].axis('off')
plt.tight_layout()
plt.show()



# %%
def PlotResults(ROI, Labels, Results, FileName=None, ShowPlot=True):

    H, W = Labels.shape
    SegImage = np.zeros((H, W, 4))
    PredImage = np.zeros((H, W, 4))

    Colors = [(0,0,1,0.0),(1,0,0,0.5),(0,1,0,0.5),(0,1,1,0.5)]
    for iValue, Value in enumerate(np.unique(Labels)):
        Filter = Labels == Value
        SegImage[Filter] = Colors[iValue]

        Filter = Results == Value
        PredImage[Filter] = Colors[iValue]

    Figure, Axis = plt.subplots(1,3, figsize=(H/50,3.2*W/50), facecolor=(1,1,1))
    Axis[0].imshow(ROI)
    Axis[0].set_title('Image',color=(0,0,0))
    Axis[1].imshow(ROI)
    Axis[1].imshow(SegImage, interpolation='none')
    Axis[1].set_title('Labels')
    Axis[2].imshow(ROI)
    Axis[2].imshow(PredImage, interpolation='none')
    for i in range(3):
        Axis[i].axis('off')
    plt.subplots_adjust(0,0,1,1)
    if FileName:
        plt.savefig(FileName)
    if ShowPlot:
        plt.show()
    else:
        plt.close(Figure)

PlotResults(TestImage, TestLabel[:,:,0], PredictionClasses)

# %%
