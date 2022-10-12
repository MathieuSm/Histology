#%%
#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, morphology

Version = '01'

# Define the script description
Description = """
    Script used to compute a mean histogram to which match ROIs previous to automatic segmentation.
    The idea is to reduce the effect of different staining intensities

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

def SegmentBone(Image, Plot=False):
    """
    Segment bone structure
    :param Image: RGB numpy array dim r x c x 3
    :param Plot: Plot the results (bool)
    :return: Segmented bone image
    """

    Tic = time.time()
    print('\nSegment bone area ...')

        # Mark areas where there is bone
    Filter1 = Image[:, :, 0] < 190
    Filter2 = Image[:, :, 1] < 190
    Filter3 = Image[:, :, 2] < 235
    Bone = Filter1 & Filter2 & Filter3

    if Plot:
        PlotImage(~Bone)

    # Erode and dilate to remove small bone parts
    Bone = morphology.remove_small_objects(~Bone, 15)
    Bone = morphology.binary_closing(Bone, morphology.disk(25))

    if Plot:
        PlotImage(Bone)

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return Bone

def ValidArea(Bone, GridSize, Threshold, Plot=False):

    """
    Define valid area according to a given BV/TV threshold and a given grid size
    :param Bone: Segmented bone
    :param GridSize: Grid size to evaluate BV/TV
    :param Threshold: Minimum BV/TV to consider area as valid
    :param Plot: Plot valid area
    :param Image: Add initial image on the plot
    :return: Area with a sufficiently high BV/TV
    """

    Tic = time.time()
    print('\nDefine valid area ...')

    NPoints = np.ceil(np.array(Bone.shape) / GridSize)
    XPoints = np.arange(NPoints[1], dtype='int') * GridSize
    YPoints = np.arange(NPoints[0], dtype='int') * GridSize
    XPoints = np.append(XPoints, Bone.shape[1])
    YPoints = np.append(YPoints, Bone.shape[0])
    XGrid, YGrid = np.meshgrid(XPoints, YPoints)

    # Compute subregion bone volume fraction
    ValidArea = np.zeros(Bone.shape).astype('int')

    for i in range(int(NPoints[1])):
        for j in range(int(NPoints[0])):
            SubRegion = Bone[YGrid[j, i]:YGrid[j + 1, i], XGrid[j, i]:XGrid[j, i + 1]]

            if SubRegion.sum() / SubRegion.size > Threshold:
                ValidArea[YGrid[j, i]:YGrid[j+1, i], XGrid[j, i]:XGrid[j, i+1]] = 1

    if Plot:
        Shape = np.array(Bone.shape) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone, cmap='binary_r')
        Axis.imshow(ValidArea, cmap='Greens', alpha=1/3)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic, Toc)

    return ValidArea

#%%

# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Data')
        self.Path = str(Path.cwd())
        self.N = 5
        self.Pixel_S = 1.0460251046025104
        self.ROI_S = 500
Arguments = ArgumentsClass()

#%%

def Main(Arguments):

    # List pictures
    DataDirectory = Arguments.Data
    Pictures = [P for P in os.listdir(DataDirectory)]
    Pictures.sort()

    # Compute mean ROI histogram
    nBins = 255
    Threshold = 0.88
    Histograms = np.zeros((len(Pictures),3,nBins))
    for Index, Name in enumerate(Pictures):

        Array = io.imread(str(Path(DataDirectory, Name)))
        Bone = SegmentBone(Array, Plot=False)
        # GridSize = int(Arguments.ROI_S / Arguments.Pixel_S * 1.2)
        # BoneVA = ValidArea(Bone, GridSize, Threshold, Plot=True)

        for RGB in range(3):
            ROI = Array[:,:,RGB][~Bone]
            Hists, Bins = np.histogram(ROI, density=False, bins=nBins, range=(0, 255))
            Histograms[Index,RGB] = Hists
    MeanHist = np.mean(Histograms,axis=0).round().astype('int')

    Figure, Axis = plt.subplots(1,1)
    Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[0], edgecolor=(1,0,0), color=(0,0,0,0), width=Bins[1])
    Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[1], edgecolor=(0,1,0), color=(0,0,0,0), width=Bins[1])
    Axis.bar(Bins[:-1] + Bins[1]/2, MeanHist[2], edgecolor=(0,0,1), color=(0,0,0,0), width=Bins[1])
    plt.show()

    nPixels = MeanHist.sum(axis=1).max()
    Width = np.ceil(np.sqrt(nPixels)).astype('int')
    Height = np.ceil(nPixels / Width).astype('int')
    Reference = np.ones((Height,Width,3),'int').ravel() * np.nan

    Start = 0
    Stop = 0
    for i, nPixels in enumerate(MeanHist.ravel()):
        Stop += nPixels
        Reference[Start:Stop] = np.tile(Bins,3)[i].astype('int')
        Start = Stop
    Reference = np.reshape(Reference,(Height,Width,3), order='F')

    FileName = str(Path(Arguments.Path) / 'Reference.png')
    H, W = Reference.shape[:-1]
    Figure, Axis = plt.subplots(1, 1, figsize=(W / 500, H / 500))
    Axis.imshow(Reference.astype('uint8'))
    Axis.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(FileName)
    plt.show()

#%%

if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('Proximal', help='Set proximal scan file number (required)', type=str)
    Parser.add_argument('Sample', help='Set slice (sample) scan file number (required)', type=str)
    Parser.add_argument('-a', '--Angle', help='Set angle of the cutting lines in degrees', type=int, default=60)

    # Define paths
    DataDirectory = str(Path.cwd() / 'Tests\Osteons\Sensitivity')
    Parser.add_argument('-data', help='Set data directory', type=str, default=DataDirectory)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)
# %%
