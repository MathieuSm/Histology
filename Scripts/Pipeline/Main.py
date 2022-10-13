#%%
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
from skimage import io, future, filters, morphology, measure, color
from statsmodels.regression.mixed_linear_model import MixedLMParams

#%%
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
    Segment bone area using simple threshold
    """

    # Mark areas where there is bone
    Filter1 = Image[:, :, 0] < 190
    Filter2 = Image[:, :, 1] < 190
    Filter3 = Image[:, :, 2] < 235
    Bone = Filter1 & Filter2 & Filter3

    if Plot:
        PlotImage(~Bone)

    # Erode and dilate to remove small bone parts
    if Bone.shape[1] > 2500:
        Disk = morphology.disk(2)
        Dilated = morphology.binary_dilation(Bone, Disk)
        Bone = morphology.binary_erosion(Dilated, Disk)
    else:
        Bone = morphology.remove_small_objects(~Bone, 15)
        Bone = ~morphology.binary_closing(Bone, morphology.disk(25))

    if Plot:
        PlotImage(Bone)

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
    ROISize = int(round(Arguments.ROI_S / Arguments.Pixel_S)) + Arguments.Margin

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
    Bones = np.zeros((N,ROISize,ROISize)).astype('int')
    BVTVs = np.zeros((N))
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
            Xs[i] += [X1, X2]
            Ys[i] += [Y1, Y2]

            Figure, Axis = plt.subplots(1, 1, figsize=(10, 10))
            Axis.imshow(ROIs[i])
            Axis.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            plt.show()

        # Store ROI and remove coordinates to no select the same
        if ROI:
            ROIs[i] += Array[Y1:Y2, X1:X2]
            Bones[i] += SegmentBone(ROIs[i])
            BVTVs[i] = BVTV
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


    return ROIs.astype('uint8'), BVTVs, Bones.astype('bool')

def PlotOverlay(ROI,Seg, FileName=None):

    H, W = Seg.shape
    SegImage = np.zeros((H, W, 4))

    Colors = [(1,0,0,0.25),(0,0,1,0.25),(0,1,0,0.25),(1,1,1,0.25)]
    for iValue, Value in enumerate(np.unique(Seg)):
        Filter = Seg == Value
        SegImage[Filter] = Colors[iValue]

    Figure, Axis = plt.subplots(1,1, figsize=(H/100,W/100))
    Axis.imshow(ROI)
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    if FileName:
        plt.savefig(FileName)
    plt.show()

    Figure, Axis = plt.subplots(1, 1, figsize=(H / 100, W / 100))
    Axis.imshow(ROI)
    Axis.imshow(SegImage, interpolation='none')
    Axis.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    if FileName:
        plt.savefig(FileName[:-4] + '_Seg.png')
    plt.show()

def PlotImage(Array):

    Figure, Axis = plt.subplots(1,1,figsize=(10,10))
    if Array.shape[-1] == 3:
        Axis.imshow(Array)
    else:
        Axis.imshow(Array, cmap='binary_r')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()

def PlotData(Data, Donors, Sides):

    Colors = {Donors[0]: (1, 0, 0),
              Donors[1]: (0, 0, 1),
              Donors[2]: (0, 1, 0)}
    Marker = {Sides[0]: 'o',
              Sides[1]: 'x'}

    for Variable in Data.columns:
        Figure, Axis = plt.subplots(1, 1)
        i = 0
        for L1, G1 in Data.groupby('Site'):
            for L2, G2 in G1.groupby('Side'):
                if i == 1:
                    Axis.plot([], color=(0, 0, 0), marker=Marker[L2], linestyle='none', fillstyle='none', label=L2)
                for L3, G3 in G2.groupby('Donor ID'):
                    x, y = np.ones(len(G3)) * i, G3[Variable].values
                    Axis.plot(x, y, color=Colors[L3], marker=Marker[L2], linestyle='none', fillstyle='none')
            i += 1
        for iD, D in enumerate(Donors):
            Axis.plot([], color=Colors[D], label=Donors[iD])
        Axis.set_xticks([0, 1], ['Inferior', 'Superior'])
        Axis.set_xlim([-1, 2])
        Axis.set_xlabel('Site (-)')
        Axis.set_ylabel(Variable)
        plt.subplots_adjust(0.15, 0.15, 0.8, 0.9)
        plt.legend(loc='center right', ncol=1, bbox_to_anchor=(1.3, 0.5), frameon=False)
        plt.show()

    return

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

#%%
class PCNNClass:

    def __init__(self):
        self.NSegments = 10
        self.Verbose = 1

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

    def GaussianKernel(self, Length=5, Sigma=1.):
        """
        Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
        """
        Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
        Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
        Kernel = np.outer(Gauss, Gauss)
        return Kernel / sum(sum(Kernel))

    def GetNeighbours(self, Array, N=1, Map=False):

        """
        Function used to get values of the neighbourhood pixels
        :param Array: Row x Column numpy array
        :param N: Number of neighbours offset
        :param Map: return rows and columns indices of neigbours (bool)
        :return: Neighbourhood pixels values, Map
        """

        L = 2*N+1
        S = Array.shape
        iRows, iCols = np.arange(S[0]), np.arange(S[1])
        Cols, Rows = np.meshgrid(iRows,iCols)
        Cols = np.repeat(Cols, L).reshape((S[0], S[0], L))
        Cols = np.repeat(Cols, L).reshape((S[0], S[0], L, L))
        Rows = np.repeat(Rows, L).reshape((S[0], S[0], L))
        Rows = np.repeat(Rows, L).reshape((S[0], S[0], L, L))

        Range = np.arange(L) - N
        ColRange = np.repeat(Range,L).reshape((L,L))
        RowRange = np.tile(Range,L).reshape((L,L))
        iCols = Cols + ColRange
        iRows = Rows + RowRange

        Pad = np.pad(Array,N)
        Neighbours = Pad[iRows+N,iCols+N]

        if Map:
            return Neighbours, [iRows, iCols]

        else:
            return Neighbours

    def NormalizeValues(self, Image):
        """
        Normalize image values, used in PCNN for easier parameters handling
        :param Image: Original grayscale image
        :return: N_Image: Image with 0,1 normalized values
        """

        N_Image = (Image - Image.min()) / (Image.max() - Image.min())

        return N_Image

    def Enhance(self, Image, SRange = (0.01, 1.0)):

        """
        Enhance image using PCNN, single neuron firing and fast linking implementation
        Based on:
        Zhan, K., Shi, J., Wang, H. et al.
        Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
        Arch Computat Methods Eng 24, 573–588 (2017).
        https://doi.org/10.1007/s11831-016-9182-3

        :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
        :param Delta: Linear decay factor for threshold level
        :param VT: Dynamic threshold amplitude
        :param Nl_max: Max number of iteration for fast linking
        :return: H: Image histogram in numpy array
        """


        def FLM(Input):
            S = (Input - Input.min()) / (Input.max() - Input.min()) + 1 / 255
            W = np.array([[0.7, 1, 0.7], [1, 0, 1], [0.7, 1, 0.7]])
            Y = np.zeros(S.shape)
            U = np.zeros(S.shape)
            T = np.zeros(S.shape)
            SumY = 0
            N = 0

            Laplacian = np.array([[1 / 6, 2 / 3, 1 / 6], [2 / 3, -10 / 3, 2 / 3], [1 / 6, 2 / 3, 1 / 6]])
            Theta = 1 + np.sum(self.GetNeighbours(S) * Laplacian, axis=(3,2))
            f = 0.75 * np.exp(-S**2 / 0.16) + 0.05
            L = 3
            G = self.GaussianKernel(2*L+1, 1)
            f = np.sum(self.GetNeighbours(f,L) * G, axis=(3,2))

            h = 2E10
            d = 2
            g = 0.9811
            Alpha = 0.01
            Beta = 0.03

            while SumY < S.size:
                N += 1

                K = np.sum(self.GetNeighbours(Y) * W, axis=(3, 2))
                Wave = Alpha * K + Beta * S * (K - d)
                U = f * U + S + Wave
                Theta = g * Theta + h * Y
                Y = (U > Theta) * 1
                T += N * Y
                SumY += sum(sum(Y))

            T_inv = T.max() + 1 - T
            Time = self.NormalizeValues(T_inv)

            return Time

        Tic = time.time()
        print('\nEnhance image using PCNN')
        if len(Image.shape) == 2:
            Output = FLM(Image)

        else:
            Output = np.zeros(Image.shape)
            for i in range(Image.shape[-1]):
                Output[:,:,i] = FLM(Image[:,:,i])

        Hist, Bins = np.histogram(Output, bins=1000, density=True)
        Low = np.cumsum(Hist) / np.cumsum(Hist).max() > SRange[0]
        High = np.cumsum(Hist) / np.cumsum(Hist).max() < SRange[1]
        Values = Bins[1:][Low & High]
        Output = exposure.rescale_intensity(Output, in_range=(Values[0], Values[-1]))

        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output

    def Segment(self,Image, Beta=2, Delta=1 / 255, VT=100):

        """
        Segment image using simplified PCNN, single neuron firing and fast linking implementation
        Based on:
        Zhan, K., Shi, J., Wang, H. et al.
        Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
        Arch Computat Methods Eng 24, 573–588 (2017).
        https://doi.org/10.1007/s11831-016-9182-3

        :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
        :param Delta: Linear decay factor for threshold level
        :param VT: Dynamic threshold amplitude
        :param Nl_max: Max number of iteration for fast linking
        :return: H: Image histogram in numpy array
        """

        if self.Verbose == 2:
            Tic = time.time()
            print('\nImage segmentation...')

        if Image.shape[-1] == 3:
            Input = color.rgb2gray(Image)
        else:
            Input = Image

        # Initialize parameters
        S = self.NormalizeValues(Input)
        Rows, Columns = S.shape
        Y = np.zeros((Rows, Columns))
        T = np.zeros((Rows, Columns))
        W = np.array([[0.5, 1, 0.5], [1, 0, 1], [0.5, 1, 0.5]])
        Theta = np.ones((Rows, Columns))

        FiredNumber = 0
        N = 0

        # Perform segmentation
        while FiredNumber < S.size:
            N += 1
            F = S
            L = np.sum(self.GetNeighbours(Y) * W, axis=(3,2))
            Theta = Theta - Delta + VT * Y
            U = F * (1 + Beta * L)
            Y = (U > Theta) * 1

            T = T + N * Y
            FiredNumber = FiredNumber + sum(sum(Y))

        Output = 1 - self.NormalizeValues(T)

        if Image.shape[-1] == 3:
            Segments = np.zeros(Image.shape)
            for Value in np.unique(Output):
                Binary = Output == Value
                Color = np.mean(Image[Binary], axis=0)
                Segments[Binary] = Color
            Output = Segments

        if Image.dtype == 'uint8':
            Output = np.round(Output).astype('uint8')

        # Print time elapsed
        if self.Verbose == 2:
            Toc = time.time()
            PrintTime(Tic, Toc)

        return Output
PCNN = PCNNClass()
#%%
# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.Data = str(Path.cwd() / 'Data')
        self.Path = str(Path.cwd())
        self.N = 5
        self.Pixel_S = 1.0460251046025104
        self.ROI_S = 500

        # Add margin to ROI to minimize border effects in cleaning morphological operations
        self.Clean = True
        self.Margin = 100
Arguments = ArgumentsClass()

#%%
def Main(Arguments):


#%%
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
ROIs = np.arange(Arguments.N) + 1
Indices = pd.MultiIndex.from_product([Donors, Sides, Sites, ROIs], names=['Donor ID', 'Side', 'Site', 'ROI Number'])
Data = pd.DataFrame(index=Indices, columns=['BV/TV', 'CLd', 'On', 'HCn', 'HCd', 'HCa'])

#%%
# Extract ROIs
ROIsData = []
BonesData = []
BVTVData = []
for Index, Name in enumerate(Pictures):
    Array = io.imread(str(Path(DataDirectory, Name)))
    ROIs, BVTVs, Bones = ExtractROIs(Array, Arguments.N, Plot=False)
    ROIsData.append(ROIs)
    BonesData.append(Bones)
    BVTVData.append(BVTVs)
ROIsData = np.array(ROIsData)
BonesData = np.array(BonesData)
BVTVData = np.array(BVTVData)
    
#%%

# Extract features

Smin = 0.5
Smax = 8
Snum = 3

print('\nExtract manual segmentation features')
SamplesFeatures = []
for iS, SampleROIs in enumerate(ROIsData):
    Tic = time.time()
    Features, FNames = ExtractFeatures(SampleROIs, ['RGB'], ['I', 'E', 'H1', 'H2'], [Smin, Smax], nSigma=Snum)

    # Add distance variable
    Distances = np.zeros((Features.shape[0], Features.shape[1], Features.shape[2], Snum))
    for iROI, ROI in enumerate(BonesData[iS]):
        Distance = morphology.medial_axis(ROI, return_distance=True)[1]
        for iSigma, Sigma in enumerate(np.linspace(Smin,Smax,Snum)):
            fDistance = filters.gaussian(Distance,sigma=Sigma)
            Distances[iROI,:,:,iSigma] = fDistance
            if iROI == 0:
                FNames.Names.append('Distance ' + str(Sigma))

    Features = np.concatenate([Features, Distances], axis=-1)

    # Add PCNN segmentation variable
    Seg = np.zeros((Features.shape[0], Features.shape[1], Features.shape[2], Snum))
    for iROI, ROI in enumerate(SampleROIs):
        for iSigma, Sigma in enumerate(np.linspace(Smin,Smax,Snum)):
            Seg[iROI, :, :, iSigma] = color.rgb2gray(PCNN.Segment(ROI,Delta=1/10))
            if iROI == 0:
                FNames.Names.append('PCNN ' + str(Sigma))

    Features = np.concatenate([Features, Seg], axis=-1)

    SamplesFeatures.append(Features)

    Toc = time.time()
    PrintTime(Tic, Toc)


#%%
    # Perform segmentation
    Classifier = joblib.load(str(Path(Arguments.Path, 'RFC.joblib')))
    for Index, Name in enumerate(Pictures):
        Array = io.imread(str(Path(DataDirectory, Name)))
        ROIs, BVTVs = ExtractROIs(Array, Arguments.N, Plot=False)

        for iROI, ROI in enumerate(ROIs):

            if ROI.sum() > 0:
                Features = ExtractFeatures(ROI)[0]


                print('\nExtract manual segmentation features')
                Tic = time.time()
                Features, FNames = ExtractFeatures(Data, ['RGB'], ['I', 'E', 'H1', 'H2'], [Smin, Smax], nSigma=Snum)

                # Add distance variable
                Distances = np.zeros((Features.shape[0], Features.shape[1], Features.shape[2], Snum))
                for iROI, ROI in enumerate(Bones):
                    Distance = morphology.medial_axis(ROI, return_distance=True)[1]
                    for iSigma, Sigma in enumerate(np.linspace(Smin,Smax,Snum)):
                        fDistance = filters.gaussian(Distance,sigma=Sigma)
                        Distances[iROI,:,:,iSigma] = fDistance
                        if iROI == 0:
                            FNames.Names.append('Distance ' + str(Sigma))

                Features = np.concatenate([Features, Distances], axis=-1)

                # Add PCNN segmentation variable
                Seg = np.zeros((Features.shape[0], Features.shape[1], Features.shape[2], Snum))
                for iROI, ROI in enumerate(Data):
                    for iSigma, Sigma in enumerate(np.linspace(Smin,Smax,Snum)):
                        Seg[iROI, :, :, iSigma] = color.rgb2gray(PCNN.Segment(ROI,Delta=1/10))
                        if iROI == 0:
                            FNames.Names.append('PCNN ' + str(Sigma))

                Features = np.concatenate([Features, Seg], axis=-1)

                Toc = time.time()
                PrintTime(Tic, Toc)
                S_ROI = future.predict_segmenter(Features, Classifier)

                # Crop margin
                M = int(Arguments.Margin / 2)
                ROI = ROI[M:-M, M:-M]

                # File for saving image
                FilePath = Path(Arguments.Path, 'SegmentationResults')
                FileName = str(FilePath / str(Name[:-4] + '_' + str(iROI + 1) + '.png'))

                if Arguments.Clean:

                    PlotOverlay(ROI, S_ROI[M:-M, M:-M])

                    # Clean segmentation
                    HCl = measure.label(S_ROI == 4)
                    HCp = pd.DataFrame(measure.regionprops_table(HCl, properties=['equivalent_diameter']))
                    HCp = HCp[HCp['equivalent_diameter'] > 30]
                    HC = np.zeros(S_ROI.shape,'bool')
                    for i in HCp.index:
                        HC += morphology.binary_closing(HCl == i+1, morphology.disk(25))
                    # PlotImage(HC)

                    # # Clean segmentation
                    # HC = S_ROI == 2
                    # HC = morphology.binary_dilation(HC,morphology.disk(5))
                    # HC = morphology.remove_small_objects(HC, 500)
                    # HC = ~morphology.remove_small_objects(~HC, 600)
                    # HC = morphology.binary_closing(~HC,morphology.disk(25))
                    # # PlotImage(HC)

                    CL = S_ROI == 1
                    CL = morphology.binary_dilation(CL, morphology.disk(3))
                    CL = morphology.remove_small_objects(CL,1000)
                    CL = morphology.binary_erosion(CL,morphology.disk(3))
                    # PlotImage(~CL)

                    OC = S_ROI == 3
                    OC = morphology.binary_erosion(OC, morphology.disk(2))
                    OC = morphology.binary_dilation(OC, morphology.disk(2))
                    OC = morphology.remove_small_objects(OC, 25)
                    # PlotImage(OC)

                    Results = np.ones(S_ROI.shape) * 2
                    Results[OC] = 3
                    Results[CL] = 1
                    Results[HC] = 4
                    Results = Results[M:-M,M:-M]

                    # Save segmentation results
                    PlotOverlay(ROI,Results,FileName)

                else:

                    Results = S_ROI[M:-M, M:-M]
                    PlotOverlay(ROI, Results, FileName)

                # Compute different variables
                On = np.max(measure.label(Results == 3))
                HCl = measure.label(Results == 4)
                Properties = ['equivalent_diameter', 'major_axis_length', 'minor_axis_length']
                HCp = pd.DataFrame(measure.regionprops_table(HCl, properties=Properties))
                HCp = HCp[HCp['equivalent_diameter'] > 30]
                HCn = len(HCp)
                HCp['Anisotropy'] = HCp['major_axis_length'] / HCp['minor_axis_length']


                # Store results in data frame
                Data.loc[Name[:3], Name[3], Name[4], iROI+1]['BV/TV'] = BVTVs[iROI]
                Data.loc[Name[:3], Name[3], Name[4], iROI+1]['CLd'] = np.sum(Results == 1) / ROI.size
                Data.loc[Name[:3], Name[3], Name[4], iROI+1]['On'] = On
                Data.loc[Name[:3], Name[3], Name[4], iROI+1]['HCn'] = HCn
                Data.loc[Name[:3], Name[3], Name[4], iROI+1]['HCd'] = HCp['equivalent_diameter'].mean()
                Data.loc[Name[:3], Name[3], Name[4], iROI+1]['HCa'] = HCp['Anisotropy'].mean()
    Data = Data.dropna().astype('float')


    # Plot data
    PlotData(Data, Donors, Sides)

    # Perform statistical analysis
    Data2Fit = Data.reset_index()

    Model = smf.mixedlm('CLd ~ Site',
                      # vc_formula={'Side': '0 + Side'},
                      re_formula='1',
                      data=Data2Fit,
                      groups=Data2Fit['Donor ID'])
    Free = MixedLMParams.from_components(fe_params=np.ones(2),cov_re=np.eye(1))
    LME = Model.fit(reml=True,free=Free)
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

