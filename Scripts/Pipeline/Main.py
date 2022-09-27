#!/usr/bin/env python3

import os
import sys
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from skimage import io, future, exposure, morphology

Version = '01'

# Define the script description
Description = """
    This script runs the analysis of cement line densities of the test samples in the curse
    of the FEXHIP project.
    
    It uses the random forest classification trained with manually segmented picture for the
    segmentation of cement lines. 3 regions of interest (ROI) of 500 um side length are rand-
    omly selected on each picture. Then, statistical comparisons between superior and infer-
    ior side are performed.
    
    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern
    
    Date: September 2022
    """

# Import functions from other scripts
sys.path.insert(0, str(Path.cwd() / 'Scripts' / 'Pipeline'))
from RandomForest import ExtractFeatures

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
    print('Define valid area ...')

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
        Shape = np.array(Bone.shape[:-1]) / 1000
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone.Image)
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
    Bone = SegmentBone(Array, Plot=True)
    LessBone = morphology.binary_erosion(Bone,morphology.disk(50))
    LessBone2 = morphology.remove_small_objects(LessBone,50000)
    Bone = morphology.binary_dilation(LessBone2,morphology.disk(50))
    PlotImage(Bone)
    Y, X = np.where(Bone)

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
                break

        if Plot:

            Xs[i] += [X1, X2]
            Ys[i] += [Y1, Y2]

            Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
            Axis.imshow(ROIs[i])
            Axis.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            plt.show()

        # Store ROI and coordinates for plotting
        ROIs[i] += Array[Y1:Y2, X1:X2]

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


    return ROIs.astype('uint8')

def PlotOverlay(ROI,Seg, FileName=None):

    H, W = Seg.shape
    SegImage = np.zeros((H, W, 4))

    Colors = [(0,0,0,0),(1,0,0,0.25)]
    for iValue, Value in enumerate(np.unique(Seg)):
        Filter = Seg == Value
        SegImage[Filter] = Colors[iValue]

    Figure, Axis = plt.subplots(1,1, figsize=(H/100,W/100))
    Axis.imshow(ROI)
    Axis.imshow(SegImage, interpolation='none')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    if FileName:
        plt.savefig(FileName)
    plt.show()

# Perform linear mixed-effect fit
def LMEFIT(DataFrame, Plot=True):

    ColumnNames = DataFrame.columns
    DataFrame.columns = ['X','Y','Group']

    # Fit model
    Model = smf.mixedlm('Y ~ X', groups=DataFrame['Group'], re_formula='~ X', data=DataFrame)
    Free = MixedLMParams.from_components(fe_params=np.ones(2),
                                         cov_re=np.eye(2))
    FitResults = Model.fit(free=Free, method=['lbfgs'])
    print(FitResults.summary())

    # Calculate R^2, p-value, 95% CI, SE, N
    Y_Obs = FitResults.model.endog
    Y_Fit = FitResults.fittedvalues

    E = Y_Obs - Y_Fit

    N = int(FitResults.nobs)

    CI_il = FitResults.conf_int()[0][0]
    CI_ir = FitResults.conf_int()[1][0]
    CI_l = FitResults.conf_int()[0][1]
    CI_r = FitResults.conf_int()[1][1]

    X_Raw = np.matrix(FitResults.model.exog)
    X = np.unique(X_Raw,axis=0)
    X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))
    C = np.matrix(FitResults.cov_params())[:X.shape[1],:X.shape[1]]
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    Alpha = 0.95
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    Y_Pre = FitResults.predict(exog=X, transform=False)
    CI_Line_u = Y_Pre + t_Alpha[0] * B_0
    CI_Line_o = Y_Pre + t_Alpha[1] * B_0
    Sorted_CI_u = CI_Line_u[np.argsort(X[:,1])]
    Sorted_CI_o = CI_Line_o[np.argsort(X[:,1])]

    NoteYPos = 0.925
    NoteYShift = 0.075

    Re = (FitResults.params * FitResults.scale)[2:]
    Se = FitResults.resid.std(ddof=-1)

    if Plot:
        Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
        for i, gData in enumerate(DataFrame.groupby('Group')):
            Color = plt.cm.winter((int(i) + 1) / len(DataFrame.groupby('Group')))
            Color = (Color[0], Color[1], Color[2], 0.5)
            gData[1].plot(x='X', y='Y', ax=Axes, label=gData[0],
                          color=Color, marker='o', fillstyle='none', linestyle='--', linewidth=1)
        Axes.plot(X[:, 1], Y_Pre, color=(1, 0, 0), label='Fit', linewidth=2)
        Axes.fill_between(X_Obs, Sorted_CI_o, Sorted_CI_u, color=(0, 0, 0), alpha=0.2,
                          label=str(int(Alpha * 100)) + '% CI')
        Axes.annotate('Intercept 95% CI [' + str(CI_il.round(2)) + r'$,$ ' + str(CI_ir.round(2)) + ']',
                      xy=(0.05, NoteYPos), xycoords='axes fraction')
        Axes.annotate('Slope 95% CI [' + str(CI_l.round(2)) + r'$,$ ' + str(CI_r.round(2)) + ']',
                      xy=(0.05, NoteYPos - NoteYShift), xycoords='axes fraction')
        Axes.annotate(r'$\sigma_{i}^2$ : ' + str(Re[0].round(3)), xy=(0.05, NoteYPos - NoteYShift * 2),
                      xycoords='axes fraction')
        Axes.annotate(r'$\sigma_{s}^2$ : ' + str(Re[2].round(3)), xy=(0.05, NoteYPos - NoteYShift * 3),
                      xycoords='axes fraction')
        Axes.annotate(r'$\sigma^2$ : ' + str(round(Se,2)), xy=(0.05, NoteYPos - NoteYShift * 4),
                      xycoords='axes fraction')
        DataFrame.columns = ColumnNames
        Axes.set_ylabel(DataFrame.columns[1])
        Axes.set_xlabel(DataFrame.columns[0])
        plt.subplots_adjust(left=0.2, bottom=0.15)
        plt.legend(loc='lower right',ncol=2)
        plt.show()

    # Add fitted values and residuals to data
    DataFrame = pd.concat([DataFrame,pd.DataFrame(Y_Fit,columns=['Fitted Value'])], axis=1)
    DataFrame = pd.concat([DataFrame,pd.DataFrame(E,columns=['Residuals'])], axis=1)

    return DataFrame, FitResults

# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Scripts' / 'Pipeline' / 'Data')
        self.Path = str(Path.cwd() / 'Scripts' / 'Pipeline')
        self.N = 3
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

    # Perform segmentation
    Classifier = joblib.load(str(Path(Arguments.Path, 'RFC.joblib')))
    # Reference = io.imread(str(Path(Arguments.Path, 'Reference.png')))
    S_ROIs = {}
    for Index, Name in enumerate(Pictures):
        Array = io.imread(str(Path(DataDirectory, Name)))
        ROIs = ExtractROIs(Array, Arguments.N, Plot=True)

        for iROI, ROI in enumerate(ROIs):
            # E_ROI = exposure.match_histograms(ROI, Reference, multichannel=True)
            Features = ExtractFeatures(ROI)[0]
            S_ROI = future.predict_segmenter(Features, Classifier)
            CL = S_ROI == 1
            Results = morphology.remove_small_objects(CL, 200)

            # Save segmentation results
            FilePath = Path(Arguments.Path, 'SegmentationResults')
            FileName = str(FilePath / str(Name + '_' + str(iROI) + '.png'))
            PlotOverlay(ROI,Results,FileName)

            S_ROIs[Name[:-4] + ' ' + str(iROI + 1)] = S_ROI
            Data.loc[Index,'ROI ' + str(iROI + 1)] = np.sum(S_ROI == 1) / S_ROI.size

    # Build data frame for analysis
    Data2Fit = pd.DataFrame()
    i = 0
    for Index in Data.index:
        for ROI in range(3):
            Data2Fit.loc[i, 'Donor'] = Data.loc[Index,'DonorID']
            Data2Fit.loc[i, 'Side'] = Data.loc[Index,'Side']
            Data2Fit.loc[i, 'Site'] = Data.loc[Index,'Site']
            Data2Fit.loc[i, 'Density'] = Data.loc[Index,'ROI ' + str(ROI + 1)]
            i += 1

    # Replace variables by numerical values
    Donors = Data2Fit['Donor'].unique()
    Data2Fit['Donor'] = Data2Fit['Donor'].replace(Donors, np.arange(len(Donors))+1)
    Data2Fit['Side'] = Data2Fit['Side'].replace(['R', 'L'], [-1, 1])
    Data2Fit['Site'] = Data2Fit['Site'].replace(['M', 'L'], [-1, 1])

    # Perform statistical analysis
    LME = smf.mixedlm('Density ~ Site',
                      vc_formula={'Side': '0 + C(Side)'},
                      re_formula='1',
                      data=Data2Fit,
                      groups=Data2Fit['Donor']).fit(reml=True)
    print(LME.summary())

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
