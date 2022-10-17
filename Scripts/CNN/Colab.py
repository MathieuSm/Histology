#%%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from skimage import io
from pathlib import Path
from patchify import patchify
from matplotlib import pyplot as plt
from tensorflow.keras import layers, utils, Model
from sklearn.model_selection import train_test_split
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
    Axis.plot(range(1, len(Loss) + 1), Loss, color=(0, 0, 1), marker='o', linestyle='--', label='Training loss')
    Axis.plot(range(1, len(Loss) + 1), ValLoss, color=(1, 0, 0), marker='o', linestyle='--', label='Validation loss')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Loss')
    Axis.legend()
    plt.show()

    Accuracy = History.history['accuracy']
    ValAccuracy = History.history['val_accuracy']
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(range(1, len(Accuracy) + 1), Accuracy, color=(0, 0, 1), marker='o', linestyle='--', label='Training accuracy')
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
PatchSize = 256
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
                Data.append(PatchedP[i,j,0])
                Labels.append(PatchedL[i,j])
Data = np.array(Data)
Labels = np.array(Labels)
Labels = np.expand_dims(Labels,-1)



#%%
# Separate into train and test data and build unet model

TrainX, TestX, TrainY, TestY = train_test_split(Data, Labels, random_state = 0)
TrainYCat = utils.to_categorical(TrainY)
TestYCat = utils.to_categorical(TestY)

UNet = BuildUNet(Data.shape[1:],TrainYCat.shape[-1]) 
UNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(UNet.summary())

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

TrainLabelsGenerator = LabelsGenerator.flow(TrainYCat, seed=Seed)
TestLabelsGenerator = LabelsGenerator.flow(TestYCat, seed=Seed)

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
Label = np.zeros((PatchSize, PatchSize))
for i in range(1,L.shape[-1]):
    Label += i*L[0,:,:,i]
Figure, Axis = plt.subplots(1,2)
Axis[0].imshow(np.round(D[0]*255).astype('uint8'))
Axis[0].axis('Off')
Axis[1].imshow(Label)
Axis[1].axis('Off')
plt.show()


#%%
# Fit model and plot history
BatchSize = 16
StepsPerEpoch = 3*(len(TrainX))//BatchSize
History = UNet.fit(TrainGenerator, validation_data=TestGenerator, steps_per_epoch=StepsPerEpoch, validation_steps=StepsPerEpoch, epochs=50)
# History = UNet.fit(TrainGenerator, validation_data=TestGenerator, epochs=50)
UNet.save('Unet.hdf5')
PlotHistory(History)

#%%
# Look at testing image
Random = np.random.randint(0, len(TestX)-1)
TestImage = TestX[Random]
TestLabel = TestY[Random]
Prediction = UNet.predict(np.expand_dims(TestImage,0))

Figure, Axis = plt.subplots(2,3)
Axis[0,0].imshow(TestImage)
Axis[0,0].set_title('Image')
Axis[0,1].imshow(Prediction[0,:,:,0], cmap='binary_r')
Axis[0,1].set_title('Interstitial')
Axis[0,2].imshow(Prediction[0,:,:,1], cmap='binary_r')
Axis[0,2].set_title('Cement lines')
Axis[1,0].imshow(TestLabel[:,:,0])
Axis[1,0].set_title('Labels')
Axis[1,1].imshow(Prediction[0,:,:,2], cmap='binary_r')
Axis[1,1].set_title('Osteocytes')
Axis[1,2].imshow(Prediction[0,:,:,3], cmap='binary_r')
Axis[1,2].set_title('Harvesian canals')
for i in range(2):
    for j in range(3):
        Axis[i,j].axis('off')
plt.tight_layout()
plt.show()

# %%
