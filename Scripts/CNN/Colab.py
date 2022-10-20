#%%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from pathlib import Path
from patchify import patchify
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.metrics import MeanIoU
from skimage import io, transform, morphology
from tensorflow.keras import layers, utils, Model
from sklearn.model_selection import train_test_split
from focal_loss import SparseCategoricalFocalLoss as SFL
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# https://youtu.be/-XeKG_T6tdc



#%%

def ConvulationBlock(Input, nFilters, DropRate):

    Activation = 'relu'

    Layer = layers.Conv2D(nFilters, 3, padding="same", kernel_initializer='he_uniform')(Input)
    Layer = layers.Activation(Activation)(Layer)

    if DropRate:
        Layer = layers.Dropout(DropRate)(Layer)

    Layer = layers.Conv2D(nFilters, 3, padding="same", kernel_initializer='he_uniform')(Layer)
    Layer = layers.Activation(Activation)(Layer)

    return Layer
def EncoderBlock(Input, nFilters, DropRate):
    Layer = ConvulationBlock(Input, nFilters, DropRate)
    Pool = layers.MaxPool2D((2, 2))(Layer)
    return Layer, Pool
def DecoderBlock(Input, SkipFeatures, nFilters, DropRate):
    Layer = layers.Conv2DTranspose(nFilters, (2, 2), strides=2, padding="same")(Input)
    Layer = layers.Concatenate()([Layer, SkipFeatures])
    Layer = ConvulationBlock(Layer, nFilters, DropRate)
    return Layer
def BuildUNet(InputShape, nClasses, nFilters=[12, 32, 64, 128, 256], DropRates=[0.1,0.1,0.2,0.2,0.3]):

    Input = layers.Input(InputShape)
    Block = []
    Block.append(EncoderBlock(Input, nFilters[0], DropRates[0]))
    for i, nFilter in enumerate(nFilters[1:-1]):
        Block.append(EncoderBlock(Block[i][1], nFilter, DropRates[i+1]))

    Bridge = ConvulationBlock(Block[-1][1], nFilters[-1], DropRates[-1])
    D = DecoderBlock(Bridge, Block[-1][0], nFilters[-2], DropRates[-2])

    for i, nFilter in enumerate(nFilters[-3::-1]):
        D = DecoderBlock(D, Block[-i+2][0], nFilter, DropRates[-i+2])

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
    Axis.plot(range(1, len(Loss) + 1), Loss, color=(0, 1, 1), marker='o', linestyle='--', label='Training loss')
    Axis.plot(range(1, len(Loss) + 1), ValLoss, color=(1, 0, 0), marker='o', linestyle='--', label='Validation loss')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Loss')
    Axis.legend()
    plt.show()

    Accuracy = History.history['categorical_accuracy']
    ValAccuracy = History.history['val_categorical_accuracy']
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
                    for v in np.unique(ImageA):
                        Bin = ImageA == v
                        if v == 1:
                            Disk = 2
                        else:
                            Disk = 1
                        Bin = morphology.binary_dilation(Bin, morphology.disk(Disk))
                        Bin = morphology.binary_erosion(Bin, morphology.disk(1))
                        ImageL[Bin] = v

                else:
                    ImageP = PatchedP[i,j,0]
                    ImageL = PatchedL[i,j,0]

                Data.append(ImageP)
                Labels.append(ImageL)
Data = np.array(Data)
Labels = np.array(Labels)
Labels = np.expand_dims(Labels,-1)



#%%
# Separate into train and test data and build unet model
TrainX, TestX, TrainY, TestY = train_test_split(Data, Labels, random_state = 0)
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
Axis[1].imshow(L[0,:,:,0])
Axis[1].axis('Off')
plt.show()

#%%
# Compute class weigths on training data

CW = class_weight.compute_class_weight('balanced', classes=np.unique(TrainY), y=TrainY.ravel())

#%%
# Build Unet

UNet = BuildUNet(Data.shape[1:],TrainYCat.shape[-1], DropRates=[0.3,0.3,0.3,0.3,0.3])
UNet.compile(optimizer='adam', loss=[SFL(gamma=2, class_weight=CW)], metrics=['categorical_accuracy'])
print(UNet.summary())

#%%
# Fit Unet and plot history

BatchSize = 24
StepsPerEpoch = 3*(len(TrainX))//BatchSize
History = UNet.fit(TrainGenerator, validation_data=TestGenerator, steps_per_epoch=StepsPerEpoch, validation_steps=StepsPerEpoch, epochs=600)
# History = UNet.fit(TrainGenerator, validation_data=TestGenerator, epochs=50)
UNet.save('Unet.hdf5')
PlotHistory(History)

#%%
# Look at testing image
Random = np.random.randint(0, len(TestX)-1)
TestImage = TestX[Random]
TestLabel = TestY[Random]
Prediction = UNet.predict(np.expand_dims(TestImage,0))
PredictionClasses = np.argmax(Prediction,axis=-1)[0]

Figure, Axis = plt.subplots(1,3)
Axis[0].imshow(TestImage)
Axis[0].set_title('Image')
Axis[1].imshow(TestLabel[:,:,0], vmin=0, vmax=3)
Axis[1].set_title('Labels')
Axis[2].imshow(PredictionClasses, vmin=0, vmax=3)
Axis[2].set_title('Predicitons')
for i in range(3):
        Axis[i].axis('off')
plt.tight_layout()
plt.show()



# %%
