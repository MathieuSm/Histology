#!/usr/bin/env python3

"""
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
#sphx-glr-auto-examples-segmentation-plot-trainable-segmentation-py
"""

import sys
import time
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, exposure, morphology, color


plt.rc('font', size=12)

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPCNN'))
from Utilities import *

class FeatureNames:

    def __init__(self, SigmaMin=4, SigmaMax=32, Names=['I', 'E', 'H1', 'H2']):

        # Random forest classifier
        Channels = ['R', 'G', 'B']
        # Names = ['Intensity', 'Edges', 'Hessian 1', 'Hessian 2']
        Names = Names
        NumSigma = int(np.log2(SigmaMax / SigmaMin) + 1)
        F_Names = []
        for Channel in Channels:
            for Sigma in range(NumSigma):
                SigmaValue = SigmaMin * 2 ** Sigma
                if SigmaValue >= 1:
                    SigmaValue = int(SigmaValue)
                for Name in Names:
                    F_Names.append(Channel + ' Sigma ' + str(SigmaValue) + ' ' + Name)

        self.Names = F_Names


def ExtractLabels(Seg, DilateCM=False):
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

    return Label, Ticks

# Load dictionary
CWD = Path.cwd() / 'Scripts'
FileName = CWD / 'RandomForest' / 'ROIs.pkl'
with open(str(FileName), 'rb') as f:
    Dict = pickle.load(f)

# Compute mean ROI histogram
nBins = 255
Histograms = np.zeros((len(Dict.keys()),3,3,nBins))
for RGB in range(3):
    for nKey, Key in enumerate(Dict.keys()):
        for nROI in range(3):
            ROI = Dict[Key]['ROIs'][nROI]
            Hists, Bins = np.histogram(ROI[:, :, RGB], density=False, bins=nBins, range=(0, 255))
            Histograms[nKey,nROI,RGB] = Hists
MeanHist = np.mean(Histograms,axis=(0,1)).round().astype('int')

Figure, Axis = plt.subplots(1,1)
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[0], edgecolor=(1,0,0), color=(0,0,0,0), width=Bins[1])
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[1], edgecolor=(0,1,0), color=(0,0,0,0), width=Bins[1])
Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[2], edgecolor=(0,0,1), color=(0,0,0,0), width=Bins[1])
plt.show()

Start = 0
Stop = 0
Reference = np.ones(ROI.shape,'int').ravel()
for i, nPixels in enumerate(MeanHist.ravel()):
    Stop += nPixels
    Reference[Start:Stop] = np.tile(Bins,3)[i].astype('int')
    Start = Stop
Reference = np.reshape(Reference,ROI.shape,order='F')
PlotImage(Reference)


# Load images
FileName = CWD / 'RandomForest' / 'TestROI.png'
ROI = io.imread(str(FileName))
ROI = np.round(exposure.match_histograms(ROI,Reference)).astype('uint8')
PlotImage(ROI)

FileName = CWD / 'RandomForest' / 'TestROI_Forest.png'
Seg_ROI = io.imread(str(FileName))
PlotImage(Seg_ROI)

ROI_Label, Ticks = ExtractLabels(Seg_ROI, DilateCM=True)
def PlotLabel(Seg):

    Image = np.zeros((Seg.shape[0], Seg.shape[1], 3))

    Colors = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1)]
    for iValue, Value in enumerate(np.unique(Seg)):
        Filter = Seg == Value
        Image[Filter] = Colors[iValue]


    Figure, Axis = plt.subplots(1,1, figsize=(10,10))
    Axis.imshow(Image)
    Axis.plot([], color=(1,0,0), lw=1, label='Segmentation')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()
PlotLabel(ROI_Label)

Labels = np.unique(ROI_Label)[1:]
Labels = [1, 2, 4]
Stains = {}
for Label in Labels:
    Stains[Label] = ROI[ROI_Label == Label]

for Key, ROINumber in [[0, 0], [4, 2]]:
    ROI = Dict[Key]['ROIs'][ROINumber]
    ROI = np.round(exposure.match_histograms(ROI, Reference)).astype('uint8')

    FileName = CWD / 'RandomForest' / str('Sample' + str(Key) + '_Seg' + str(ROINumber) + '.png')
    Seg_ROI = io.imread(str(FileName))
    ROI_Label = ExtractLabels(Seg_ROI)[0]

    for Label in Labels:
        LabelStain = Stains[Label]
        Stains[Label] = np.vstack([LabelStain,ROI[ROI_Label == Label]])

MeanStain = np.zeros((len(Labels),3))
for iLabel, Label in enumerate(Stains.keys()):
    MeanStain[iLabel] = np.mean(Stains[Label],axis=0)/255

MeanStain = np.array([[1,0,0],[0,1,0],[0,0,1]])
SepStains = color.separate_stains(ROI,MeanStain)
PlotImage(ROI[:,:,2])