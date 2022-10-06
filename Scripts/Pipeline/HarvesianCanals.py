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


# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Scripts' / 'Pipeline' / 'Data')
        self.Path = str(Path.cwd() / 'Scripts' / 'Pipeline')
        self.N = 5
        self.Pixel_S = 1.0460251046025104
        self.ROI_S = 500

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

    # Extract features and labels
    Features = []
    Labels = []
    for K in PicturesData.keys():
        Features.append(PicturesData[K]['ROI'])
        Labels.append(PicturesData[K]['HC'] * 1 + 1)
    Features = np.vstack(Features)
    Labels = np.hstack(Labels)