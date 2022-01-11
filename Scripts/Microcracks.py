import os
import time

import matplotlib.colors
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import exposure, morphology, filters, feature, segmentation, measure
import matplotlib as mpl

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
pd.set_option('display.width', desired_width)
plt.rc('font', size=12)

def PlotArray(Array, Title):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(Array)
    plt.title(Title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def NormalizeValues(Image):

    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max()-Image.min())

    return N_Image
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
def Histogram(Image,NBins=256,Plot=False):

        """
        Compute image histogram
        Based on:
        Zhan, K., Shi, J., Wang, H. et al.
        Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
        Arch Computat Methods Eng 24, 573–588 (2017).
        https://doi.org/10.1007/s11831-016-9182-3

        :param: NBins: Number of histogram bins
        :return: H: Image histogram in numpy array
        """

        Tic = time.time()
        print('\nCompute image histogram...')

        # Initialize PCNN
        MaxS = Image.max()
        S = NormalizeValues(Image)
        Theta = 1
        Delta = 1 / (NBins - 1)
        Vt = 1 + Delta
        Y = np.zeros(S.shape)
        U = S
        H = np.zeros(NBins)

        # Perform histogram analysis
        for N in range(1,NBins+1):
            Theta = Theta - Delta + Vt * Y
            Y = np.where((U - Theta) > 0, 1, 0)
            H[NBins - N] = Y.sum()

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        Bins = np.arange(0,MaxS+Delta,Delta*MaxS)

        if Plot:
            Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
            Axes.bar(x=Bins, height=H / H.sum(), width=Bins.max()/len(Bins), color=(1, 0, 0))
            Axes.set_xlabel('Values (-)')
            Axes.set_ylabel('Density (-)')
            plt.subplots_adjust(left=0.175)
            plt.show()
            plt.close(Figure)

        return H, Bins



# List tests directories
CurrentDirectory = os.getcwd()
Directory = os.path.join(CurrentDirectory,'Tests/Calcein/')
Tests = [Dir for Dir in os.listdir(Directory) if os.path.isdir(Directory+Dir)][1:]
Tests.sort()

# Analyze images of 1 directory
Data = pd.DataFrame()
for TestNumber in range(3):
    TestDir = os.path.join(Directory,Tests[TestNumber]) + '/'
    Parameters = pd.read_csv(TestDir + 'Parameters.txt',sep=';')

    # Choose 1 image to analyze
    if TestNumber == 0:
        Range = [2, 7, 12, 17]
    else:
        Range = [2]

    for Index in Range:
        Sample = Parameters.loc[Index,'Sample']
        Image = sitk.ReadImage(TestDir + Sample + '.BMP')
        Array = sitk.GetArrayFromImage(Image)
        # PlotArray(Array[:,:,:],Sample)

        GreenValues = Array[:, :, 1]
        # H, Bins = Histogram(GreenValues,Plot=True)
        OtsuFilter = sitk.OtsuThresholdImageFilter()
        OtsuFilter.SetInsideValue(0)
        OtsuFilter.SetOutsideValue(1)
        OtsuImage = OtsuFilter.Execute(sitk.GetImageFromArray(GreenValues))
        Threshold = OtsuFilter.GetThreshold()

        OtsuArray = sitk.GetArrayFromImage(OtsuImage)
        # PlotArray(OtsuArray * GreenValues, Sample)

        # Extract segmented image data
        Values = (OtsuArray * GreenValues)
        Values_Array = Values[Values > 0].ravel()
        Dict = {'Test':Tests[TestNumber],
                'Sample':Sample,
                'Values':Values_Array}
        Data = Data.append(Dict,ignore_index=True)


# Plot Stats
GroupedData = Data.groupby(by='Test')
Figure, Axes = plt.subplots(1, 1, figsize=(3.5, 4.5),dpi=100)
Axes.boxplot(Data.groupby(by='Test').loc[:,'Values'],vert=True,
             showmeans=True,
             boxprops=dict(linestyle='-',color=(0,0,0)),
             medianprops=dict(linestyle='-',color=(1,0,0)),
             whiskerprops=dict(linestyle='--',color=(0,0,0)),
             meanprops=dict(marker='x',markeredgecolor=(0,0,1)),
             flierprops=dict(marker='o',markeredgecolor=(0,0,0,0.01)))
Axes.plot([],linestyle='-',color=(1,0,0),label='Median')
Axes.plot([],linestyle='none',marker='x',color=(0,0,1),label='Mean')
Axes.set_ylabel('Pixel Intensity')
Axes.set_xticks([])
Axes.set_ylim([0,255])
Axes.set_title('')
plt.title('')
plt.legend(loc='lower center',ncol=2, handletextpad=0.2, frameon=False)
plt.suptitle('')
plt.subplots_adjust(0.2)
plt.show()
plt.close(Figure)

