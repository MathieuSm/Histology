#!/usr/bin/env python3

"""
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
#sphx-glr-auto-examples-segmentation-plot-trainable-segmentation-py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure


plt.rc('font', size=12)

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPCNN'))
from Utilities import *

# Load dictionary
with open('OptimizationData.pkl', 'rb') as f:
    Dict = pickle.load(f)

Indices = [[],[]]
for Key in Dict.keys():
    for i in range(3):
        Indices[0].append(Key)
        Indices[1].append(i+1)


# Store ROIs properties
Columns = ['MeanR','MeanG','MeanB','StdR','StdG','StdB']
Data = pd.DataFrame(columns=Columns,index=Indices)
for Key in Dict.keys():
    for ROINumber in range(3):
        TestROI = Dict[Key]['ROI'][ROINumber]
        MeanR, MeanG, MeanB = np.mean(TestROI,axis=(0,1))
        StdR, StdG, StdB = np.std(TestROI, axis=(0,1), ddof=1)
        for iValue, Value in enumerate([MeanR,MeanB,MeanG,StdR,StdG,StdB]):
            Data.loc[Key,ROINumber+1][Columns[iValue]] = Value

# Plot ROIs data
Samples = Data.index.unique(0)
P = np.arange(len(Samples))
Colors = [(1,0,0),(0,1,0),(0,0,1)]
eColors = [(1,0,0,0.5),(0,1,0,0.5),(0,0,1,0.5)]
Offset = 0.2
Figure, Axis = plt.subplots(1,1)
for i, c in enumerate(['R','G','B']):
    for j in range(3):
        Y = Data.xs(j+1,level=1)
        Axis.errorbar(P + Offset*(j-1), Y['Mean' + c], yerr=Y['Std' + c], fmt='o', color=Colors[i], mfc=(1, 1, 1), ecolor=eColors[i])
Axis.set_ylabel('RGB mean value (-)')
Axis.set_xticks(P,Samples)
plt.show()


Figure, Axis = plt.subplots(1,1)
for i, c in enumerate(['R','G','B']):
    for j in range(3):
        Y = Data.xs(j+1,level=1)
        Axis.plot(P + Offset*(j-1), Y['Std' + c], marker='o', color=Colors[i], linestyle='none', mew=2, fillstyle='none')
Axis.set_ylabel('Standard deviation (-)')
Axis.set_xticks(P,Samples)
plt.show()

# For each RGB channel
CumulativeDiff = pd.DataFrame(data=np.zeros((len(Data),len(Data))), index=Data.index, columns=Data.index)
for i in range(3):
    # Collect histograms
    Histograms = pd.DataFrame(columns=Data.index)
    for ROI in Histograms.columns:
        TestROI = Dict[ROI[0]]['ROI'][ROI[1]-1]
        Hists, Bins = np.histogram(TestROI[:,:,i], density=True, bins=20, range=(0, 255))
        Histograms.loc[:,ROI] = Hists

    # Compute differences between histograms
    Differences = pd.DataFrame(index=Data.index, columns=Data.index)
    for Ref in Differences.columns:
        RefValues = Histograms[Ref]
        for ROI in Differences.index:
            ComparativeValues = Histograms[ROI]
            Differences.loc[Ref,ROI] = sum(np.abs(RefValues - ComparativeValues))
    Differences[Differences == 0] = 1
    CumulativeDiff = CumulativeDiff + Differences

# Build arrays with corresponding ROIs and their differences
Values = np.ravel(CumulativeDiff)
Pairs = []
for i in CumulativeDiff.index:
    for j in CumulativeDiff.columns:
        Pairs.append((i,j))

# Sort to take the minimal differences
ArgSorted = np.argsort(Values)
PairSorted = []
for i in ArgSorted:
    PairSorted.append(Pairs[i])
PairSorted = PairSorted[:-15:2]
ValuesSorted = Values[ArgSorted][:-15:2]

# Plot histograms differences
Figure, Axis = plt.subplots(1,1)
Axis.bar(np.arange(len(ValuesSorted)),ValuesSorted,edgecolor=(1,0,0),color=(0,0,0,0))
Axis.set_xlabel('Pairs n° (-)')
Axis.set_ylabel('Histogram cumulative differences')
Axis.set_ylim([0,0.3])
plt.show()


# Plot closest pair
Figure, Axis = plt.subplots(1,2)
for i in range(2):
    Axis[i].imshow(Dict[PairSorted[0][i][0]]['ROI'][PairSorted[0][i][1]-1])
    Axis[i].axis('off')
    Axis[i].set_title(PairSorted[0][i][0] + ' ' + str(PairSorted[0][i][1]))
plt.tight_layout()
plt.show()

# Plot the furthest pair
Figure, Axis = plt.subplots(1,2)
for i in range(2):
    Axis[i].imshow(Dict[PairSorted[-1][i][0]]['ROI'][PairSorted[-1][i][1]-1])
    Axis[i].axis('off')
    Axis[i].set_title(PairSorted[-1][i][0] + ' ' + str(PairSorted[-1][i][1]))
plt.tight_layout()
plt.show()


# Test histogram matching
Reference = Dict[PairSorted[0][i][0]]['ROI'][PairSorted[0][i][1]-1]

# For each RGB channel
CumulativeDiff = pd.DataFrame(data=np.zeros((len(Data),len(Data))), index=Data.index, columns=Data.index)
for i in range(3):
    # Collect histograms
    Histograms = pd.DataFrame(columns=Data.index)
    for ROI in Histograms.columns:
        TestROI = Dict[ROI[0]]['ROI'][ROI[1]-1]
        MatchedROI = exposure.match_histograms(TestROI[:,:,i],Reference[:,:,i])
        Hists, Bins = np.histogram(MatchedROI, density=True, bins=20, range=(0, 255))
        Histograms.loc[:,ROI] = Hists

    # Compute differences between histograms
    Differences = pd.DataFrame(index=Data.index, columns=Data.index)
    for Ref in Differences.columns:
        RefValues = Histograms[Ref]
        for ROI in Differences.index:
            ComparativeValues = Histograms[ROI]
            Differences.loc[Ref,ROI] = sum(np.abs(RefValues - ComparativeValues))
    Differences[Differences == 0] = 1
    CumulativeDiff = CumulativeDiff + Differences

# Build arrays with corresponding ROIs and their differences
Values = np.ravel(CumulativeDiff)
Pairs = []
for i in CumulativeDiff.index:
    for j in CumulativeDiff.columns:
        Pairs.append((i,j))

# Sort to take the minimal differences
ArgSorted = np.argsort(Values)
PairSorted = []
for i in ArgSorted:
    PairSorted.append(Pairs[i])
PairSorted = PairSorted[:-15:2]
ValuesSorted = Values[ArgSorted][:-15:2]

# Plot histograms differences
Figure, Axis = plt.subplots(1,1)
Axis.bar(np.arange(len(ValuesSorted)),ValuesSorted,edgecolor=(1,0,0),color=(0,0,0,0))
Axis.set_xlabel('Pairs n° (-)')
Axis.set_ylabel('Histogram cumulative differences')
Axis.set_ylim([0,0.3])
plt.show()
