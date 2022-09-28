#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, morphology

Version = '01'

# Define the script description
Description = """
    This script extract N regions of interest randomly and save them as well as their coordinates

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
        Shape = np.array(Image.shape) / max(Image.shape) * 10
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Erode and dilate to remove small bone parts
    Disk = morphology.disk(2)
    Dilated = morphology.binary_dilation(Bone, Disk)
    Bone = morphology.binary_erosion(Dilated, Disk)

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

def RandCoords(Coords, ROINumber, TotalNROIs):

    """
    Perform semi-random region of interest (ROI) selection by selecting random coordinate in a given set of coordinates.
    It is said semi-random because the selection is performed in restricted area for each ROI ensuring a good sampling
    of the picture
    :param Coords: Set of coordinates where to select a ROI
    :param ROINumber: Number of the ROI which is actually selected
    :param TotalNROIs: Number of ROIs to select
    :return: ROI central coordinates
    """

    XCoords, YCoords = Coords

    XRange = XCoords.max() - XCoords.min()
    Width = XRange / (TotalNROIs + 1)
    RandX = int((ROINumber + 1) * XRange / (TotalNROIs + 1) + np.random.randn() * Width**(1 / 2))
    RandX = XCoords[np.argmin(np.abs(XCoords - RandX))]
    YCoords = YCoords[XCoords == RandX]
    YRange = YCoords.max() - YCoords.min()
    RandY = int(np.median(YCoords) + np.random.randn() * (YRange/2)**(1 / 2))

    return [RandX, RandY]

def ExtractROIs(Array, N, Plot=False):

    """
    Extract regions of interest of cortical bone according to the parameters given as arguments for the Main function.
    According to Grimal et al (2011), cortical bone representative volume element should be around 1mm side length and
    presents a BV/TV of 88% at least. Therefore, a threshold of 0.88 is used to ensure that the selected ROI reaches
    this value.

    Grimal, Q., Raum, K., Gerisch, A., &#38; Laugier, P. (2011)
    A determination of the minimum sizes of representative volume elements
    for the prediction of cortical bone elastic properties
    Biomechanics and Modeling in Mechanobiology (6), 925–937
    https://doi.org/10.1007/s10237-010-0284-9

    :param Array: 3D numpy array (2D + RGB)
    :param N: Number of ROIs to extract (int)
    :param Plot: Plot the results (bool)
    :return: ROIs
    """

    Threshold = 0.88

    # Segment bone and extract coordinate
    Bone = SegmentBone(Array, Plot=False)
    GridSize = int(Arguments.ROI_S / Arguments.Pixel_S * 1.2)
    BoneVA = ValidArea(Bone, GridSize, Threshold, Plot=True)
    Y, X = np.where(BoneVA)

    # Record time
    Tic = time.time()
    print('\nBegin ' + str(N) + ' ROIs extraction ...')

    # Set ROI pixel size
    ROISize = int(round(Arguments.ROI_S / Arguments.Pixel_S))

    # Filter positions too close to the border
    F1 = X > ROISize / 2
    F2 = X < Bone.shape[1] - ROISize / 2
    FilteredX = X[F1 & F2]
    FilteredY = Y[F1 & F2]

    F1 = FilteredY > ROISize / 2
    F2 = FilteredY < Bone.shape[0] - ROISize / 2
    FilteredY = FilteredY[F1 & F2]
    FilteredX = FilteredX[F1 & F2]

    # Perform semi-random ROI selection
    ROIs = np.zeros((N,ROISize,ROISize,3)).astype('int')
    Xs = np.zeros((N,2)).astype('int')
    Ys = np.zeros((N,2)).astype('int')

    for i in range(N):

        print('Extract ROI number ' + str(i+1))

        RandX, RandY = RandCoords([FilteredX, FilteredY], i, N)
        X1, X2 = RandX - int(ROISize / 2), RandX + int(ROISize / 2)
        Y1, Y2 = RandY - int(ROISize / 2), RandY + int(ROISize / 2)
        BoneROI = Bone[Y1:Y2, X1:X2]
        BVTV = BoneROI.sum() / BoneROI.size
        ROI = True

        j = 0
        while BVTV < Threshold:
            RandX, RandY = RandCoords([FilteredX, FilteredY], i, N)
            X1, X2 = RandX - int(ROISize / 2), RandX + int(ROISize / 2)
            Y1, Y2 = RandY - int(ROISize / 2), RandY + int(ROISize / 2)
            BoneROI = Bone[Y1:Y2, X1:X2]
            BVTV = BoneROI.sum() / BoneROI.size

            # Limit the number of iterations to find a "good" ROI
            j += 1
            if j == 100:
                print('No ROI found after 100 iterations')
                ROI = False
                break

        if Plot:

            Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
            Axis.imshow(ROIs[i])
            Axis.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            plt.show()

        # Store ROI and remove coordinates to no select the same
        if ROI:

            ROIs[i] += Array[Y1:Y2, X1:X2]
            Xs[i] += [X1, X2]
            Ys[i] += [Y1, Y2]

            XRemove = (FilteredX > X1) & (FilteredX < X2)
            YRemove = (FilteredY > Y1) & (FilteredY < Y2)
            FilteredX = FilteredX[~(XRemove & YRemove)]
            FilteredY = FilteredY[~(XRemove & YRemove)]

    if Plot:
        Shape = np.array(Array.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Array)

        for i in range(len(Xs)):
            Axis.plot([Xs[i,0], Xs[i,1]], [Ys[i,0], Ys[i,0]], color=(1, 0, 0))
            Axis.plot([Xs[i,1], Xs[i,1]], [Ys[i,0], Ys[i,1]], color=(1, 0, 0))
            Axis.plot([Xs[i,1], Xs[i,0]], [Ys[i,1], Ys[i,1]], color=(1, 0, 0))
            Axis.plot([Xs[i,0], Xs[i,0]], [Ys[i,1], Ys[i,0]], color=(1, 0, 0))
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Print elapsed time
    Toc = time.time()
    PrintTime(Tic,Toc)


    return ROIs.astype('uint8'), [Xs, Ys]

# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Scripts' / 'Pipeline' / 'Data')
        self.Path = str(Path.cwd() / 'Scripts' / 'Variability' / 'ROIs')
        self.N = 5
        self.Pixel_S = 1.0460251046025104
        self.ROI_S = 500
Arguments = ArgumentsClass()

def Main(Arguments):

    # List pictures
    DataDirectory = Arguments.Data
    Pictures = [P for P in os.listdir(DataDirectory)]
    Pictures.sort()

    # Build data frame
    Data = pd.DataFrame()
    for Index, Name in enumerate(Pictures):
        Data.loc[Index, 'DonorID'] = Name[:3]
        Data.loc[Index, 'Side'] = Name[3]
        Data.loc[Index, 'Site'] = Name[4]
    Donors = Data['DonorID'].unique()
    Sides = Data['Side'].unique()
    Sites = Data['Site'].unique()
    ROIs = np.arange(Arguments.N)+1
    Indices = pd.MultiIndex.from_product([Donors,Sides,Sites,ROIs], names=['Donor ID', 'Side', 'Site', 'ROI Number'])
    Data = pd.DataFrame(index=Indices, columns=['X1', 'X2', 'Y1', 'Y2'])

    # Perform ROI selection
    FilePath = Path(Arguments.Path)
    for Index, Name in enumerate(Pictures):
        Array = io.imread(str(Path(DataDirectory, Name)))
        ROIs, Coords = ExtractROIs(Array, Arguments.N, Plot=False)

        for iROI, ROI in enumerate(ROIs):

            # Save ROI
            if ROI.sum() > 0:
                H, W = ROI.shape[:-1]
                Figure, Axis = plt.subplots(1, 1, figsize=(H / 100, W / 100))
                Axis.imshow(ROI)
                Axis.axis('off')
                plt.subplots_adjust(0, 0, 1, 1)
                plt.savefig(str(FilePath / str(Name[:-4] + '_' + str(iROI+1) + '.png')))
                plt.show()

                Data.loc[Name[:3],Name[3],Name[4],iROI+1]['X1'] = Coords[0][iROI][0]
                Data.loc[Name[:3],Name[3],Name[4],iROI+1]['X2'] = Coords[0][iROI][1]
                Data.loc[Name[:3],Name[3],Name[4],iROI+1]['Y1'] = Coords[1][iROI][0]
                Data.loc[Name[:3],Name[3],Name[4],iROI+1]['Y2'] = Coords[1][iROI][1]

    # Save data frame
    Data.dropna().to_csv(str(FilePath / 'Coordinates.csv'))


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