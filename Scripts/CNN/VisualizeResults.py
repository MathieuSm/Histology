import os
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage import io, transform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
def ExtractLabels(Seg, Plot=True):

    # Extract segments
    CL = Seg[:, :, 0] == 255
    OC = Seg[:, :, 1] == 255
    HC = CL * OC

    # Label cement lines segments
    Label = np.zeros(HC.shape, 'uint8')
    Label[CL] = 1
    Label[HC] = 0

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
def DataAugmentation(Image,Label,N):

    ISize = Image.shape[:-1]
    ASize = (1024, 1024)

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
def PlotImage(Array):

    Figure, Axis = plt.subplots(1,1,figsize=(10,10))
    if Array.shape[-1] == 3:
        Axis.imshow(Array)
    else:
        Axis.imshow(Array, cmap='binary_r')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()


# Define paths
ModelPath = Path.cwd() / 'Scripts' / 'CNN'
DataPath = Path.cwd() / 'Scripts' / 'Pipeline'

# List manually segmented pictures
DataDirectory = str(DataPath / 'ManualSegmentation')
Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.png')]
Pictures.sort()

# Load manual segmentations
PicturesData = {}
for iPicture, Picture in enumerate(Pictures[1:]):
    PicturesData[Picture[:-8]] = {}
    PicturesData[Picture[:-8]]['ROI'] = io.imread(str(Path(DataDirectory, Picture[:-8] + '.png')))
    Seg = io.imread(str(Path(DataDirectory, Picture)))
    PicturesData[Picture[:-8]]['Labels'] = ExtractLabels(Seg)[0]

# Perform data augmentation
Images = []
Labels = []
N = 5
for K in PicturesData.keys():
    ROI = PicturesData[K]['ROI']
    HC = PicturesData[K]['Labels']

    AugData, AugLabels = DataAugmentation(ROI, HC, N)

    for iN in range(N):
        Images.append(AugData[iN])
        Labels.append(AugLabels[iN])
Images = np.array(Images)
Labels = np.array(Labels).astype('int')
Labels = np.expand_dims(Labels, -1)

# Load UNet model and perform predictions
UNet = load_model(str(ModelPath / 'UNet'))

# Perform predictions
Tic = time.time()
Predictions = UNet.predict(Images)
Toc = time.time()
PrintTime(Tic,Toc)

# Visualize results
Random = np.random.randint(0, len(Images)-1)
RandomImage = Images[Random]
RandomLabel = Labels[Random]
Prediction = Predictions[Random]

Figure, Axis = plt.subplots(1,3)
Axis[0].imshow(RandomImage)
Axis[1].imshow(RandomLabel == 1, cmap='binary_r')
Axis[2].imshow(Prediction[:,:,1], cmap='binary_r')
for i in range(3):
    Axis[i].axis('off')
plt.tight_layout()
plt.show()

PlotImage(Prediction[:,:,3] > 0.5)

Figure, Axis = plt.subplots(1,1)
Axis.plot(np.unique(Prediction[:,:,2]), color=(1,0,0))
plt.show()