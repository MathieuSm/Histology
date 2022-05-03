"""
Code for testing PCNN-PSO-AT with different inputs or fitness function
"""

from pathlib import Path
import time

import matplotlib.colors
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import exposure, morphology, filters, feature, segmentation, measure, color
import matplotlib as mpl

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)


# Define utilities functions
def PlotArray(Array, Title, ColorBar=False):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    CBar = Axes.imshow(Array, cmap='gray')
    if ColorBar:
        plt.colorbar(CBar)
    plt.title(Title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def PlotChanels(MChanelsArray, ChanelA, ChanelB, ChanelC):

    A = MChanelsArray[:, :, 0]
    B = MChanelsArray[:, :, 1]
    C = MChanelsArray[:, :, 2]

    Figure, Axes = plt.subplots(1, 3)
    Axes[0].imshow(A, cmap='gray')
    Axes[0].set_title(ChanelA + ' Channel')
    Axes[0].axis('off')
    Axes[1].imshow(B, cmap='gray')
    Axes[1].set_title(ChanelB + ' Channel')
    Axes[1].axis('off')
    Axes[2].imshow(C, cmap='gray')
    Axes[2].set_title(ChanelC + ' Channel')
    Axes[2].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def PlotSegment(SegmentedImage, SegmentNumber):

    SegmentValues = np.unique(SegmentedImage)
    Segment = SegmentValues[SegmentNumber]
    Filter = SegmentedImage != Segment

    PlottedArray = SegmentedImage.copy()
    PlottedArray[Filter] = -1
    PlottedArray[PlottedArray >= 0] = 0
    PlotArray(PlottedArray, 'Segment Number ' + str(SegmentNumber))

    return PlottedArray + 1
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
def NormalizeValues(Image):

    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    N_Image = (Image - Image.min()) / (Image.max()-Image.min())

    return N_Image
def BetweenClassVariance(GrayScale, Segmented):

    Ignited_Neurons = Segmented == 1

    N0 = np.sum(~Ignited_Neurons)
    N1 = np.sum(Ignited_Neurons)

    if N0 == 0 or N1 == 0:
        Vb = 0

    else:
        w0 = N0 / Segmented.size
        w1 = N1 / Segmented.size

        u0 = GrayScale[~Ignited_Neurons].mean()
        u1 = GrayScale[Ignited_Neurons].mean()

        Vb = w0 * w1 * (u0 - u1) ** 2

    return Vb


# Define PCNN class
class PCNN:

    """
    Define a class of Pulse-Connected Neural Network (PCNN) for image analysis
    Initially aimed to be used for cement lines segmentation
    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern
    Date: November 2021

    Package needed:
    time
    numpy
    correlate from scipy.ndimage

    !!! No computational cost efficiency implementation -> start with small images for tests !!!
    """

    def __init__(self):
        return


    # Set PCNN attributes
    def Set_Image(self,Image):
        """
        Define single image to be processed by PCNN
        :param Image: Grayscale intensity image in numpy array
        """
        self.Image = Image
        return

    def Images2Fuse(self,Image1,Image2,Image3=None):
        """
        Define image set to be processed by mPCNN
        :param Image1: Grayscale intensity image in numpy array
        :param Image2: Grayscale intensity image in numpy array
        :param Image3: Grayscale intensity image in numpy array
        """
        self.Image1 = Image1
        self.Image2 = Image2
        if Image3:
            self.Image3 = Image3
        return



    # Define PCNN functions
    def Histogram(self,NBins=256,Plot=False):

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
        MaxS = self.Image.max()
        S = NormalizeValues(self.Image)
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

    def PCNN_Segmentation(self,Beta=2,AlphaF=1.,VF=0.5,AlphaL=1.,VL=0.5,AlphaT=0.05,VT=100,AT=False,FastLinking=False,Nl_max=1E4):

        """
        Segment image using single neuron firing and fast linking implementation
        Based on:
        Zhan, K., Shi, J., Wang, H. et al.
        Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
        Arch Computat Methods Eng 24, 573–588 (2017).
        https://doi.org/10.1007/s11831-016-9182-3

        :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
        :param Delta: Linear decay factor for threshold level
        :param Vt: Dynamic threshold amplitude
        :return: H: Image histogram in numpy array
        """

        Tic = time.time()
        print('\nImage segmentation...')

        # Initialize parameters
        S = NormalizeValues(self.Image)
        Rows, Columns = S.shape
        F = np.zeros((Rows, Columns))
        L = np.zeros((Rows, Columns))
        Y = np.zeros((Rows, Columns))
        T = np.zeros((Rows, Columns))
        W = np.array([[0.5, 1, 0.5],
                      [1, 0, 1],
                      [0.5, 1, 0.5]])
        Theta = np.ones((Rows, Columns))


        FiredNumber = 0
        N = 0

        if AT:
            Vb, New_Vb = 0, 0
            Condition = New_Vb >= Vb
        else:
            Condition = FiredNumber < S.size

        # Perform segmentation
        while Condition:

            N += 1
            F = S + F * np.exp(-AlphaF) + VF * correlate(Y, W, output='float', mode='reflect')
            L = L * np.exp(-AlphaL) + VL * correlate(Y, W, output='float', mode='reflect')
            Theta = Theta * np.exp(-AlphaT) + VT * Y

            if FastLinking:
                Fire = 1
                Nl = 0
                while Fire == 1:

                    Q = Y
                    U = F * (1 + Beta * L)
                    Y = (U > Theta) * 1
                    if np.array_equal(Q, Y):
                        Fire = 0
                    else:
                        L = L * np.exp(-AlphaL) + VL * correlate(Y, W, output='float', mode='reflect')

                    Nl += 1
                    if Nl > Nl_max:
                        print('Fast linking too long, stopped')
                        break
            else:
                U = F * (1 + Beta * L)
                Y = (U > Theta) * 1

            if AT:
                # Update variance
                Vb = New_Vb
                New_Vb = BetweenClassVariance(S, Y)
                Condition = New_Vb >= Vb

                if New_Vb >= Vb:
                    Best_Y = Y

            else:
                T = T + N * Y
                FiredNumber = FiredNumber + sum(sum(Y))
                Condition = FiredNumber < S.size

        if AT:
            Output = Best_Y
        else:
            Output = 1 - NormalizeValues(T)

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output

    def SPCNN_Segmentation(self,Beta=2,Delta=1/255,VT=100,AT=False,FastLinking=False,Nl_max=1E4):

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

        Tic = time.time()
        print('\nImage segmentation...')

        # Initialize parameters
        S = NormalizeValues(self.Image)
        Y = np.zeros(S.shape)
        T = np.zeros(S.shape)
        W = np.array([[0.5, 1, 0.5],
                      [1, 0, 1],
                      [0.5, 1, 0.5]])
        Theta = np.ones(S.shape)

        FiredNumber = 0
        N = 0

        if AT:
            Vb, New_Vb = 0, 0
            Condition = New_Vb >= Vb
        else:
            Condition = FiredNumber < S.size

        # Perform segmentation
        while Condition:

            N += 1
            F = S
            L = correlate(Y, W, output='float', mode='reflect')
            Theta = Theta - Delta + VT * Y

            if FastLinking:
                Fire = 1
                Nl = 0
                while Fire == 1:

                    Q = Y
                    U = F * (1 + Beta * L)
                    Y = (U > Theta) * 1
                    if np.array_equal(Q, Y):
                        Fire = 0
                    else:
                        L = correlate(Y, W, output='float', mode='reflect')

                    Nl += 1
                    if Nl > Nl_max:
                        print('Fast linking too long, stopped')
                        break
            else:
                U = F * (1 + Beta * L)
                Y = (U > Theta) * 1

            if AT:
                # Update variance
                Vb = New_Vb
                New_Vb = BetweenClassVariance(S, Y)
                Condition = New_Vb >= Vb

                if New_Vb >= Vb:
                    Best_Y = Y

            else:
                T = T + N * Y
                FiredNumber = FiredNumber + sum(sum(Y))
                Condition = FiredNumber < S.size

        if AT:
            Output = Best_Y
        else:
            Output = 1 - NormalizeValues(T)

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output

    def Watershed_Segmentation(self,Peaks,Minima):

        """
        Segment image histogram watershed to compute Beta
        Based on:
        Min Li and Wei Cai and Xiao-Yan Li (2006)
        An Adaptive Image Segmentation Method Based on a Modified Pulse Coupled Neural Network
        LNCS (4221), 471-474

        :param Peaks: Histogram peaks, Peaks index < Minima index
        :param Minima: Histogram local minima, Minima index > Peaks index
        :return: Output: Segmented image as numpy array
        """

        Tic = time.time()
        print('\nImage segmentation...')

        # Initialize parameters
        S = NormalizeValues(self.Image)
        Y = np.zeros(S.shape)
        T = np.zeros(S.shape)
        W = GaussianKernel(3, 1)

        ## Feeding and linkin inputs
        F = S
        L = Y

        # Loop
        N = 0
        for m in range(len(Minima)):
            N += 1
            L = correlate(Y, W, output='float', mode='reflect')
            Beta = Minima[m] / Peaks[m] - 1
            U = F * (1 + Beta * L)
            Y = (U > Minima[m]) * 1
            T += N * Y

        Output = NormalizeValues(T)

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output

    def TimeSeries(self,N=40):

        """
        Extract invariant image features using Spiking Cortical Model (SCM)
        Based on:
        Zhan, K., Shi, J., Wang, H. et al.
        Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
        Arch Computat Methods Eng 24, 573–588 (2017).
        https://doi.org/10.1007/s11831-016-9182-3

        :param N: Dynamic threshold amplitude
        :return: TS: Image time series in numpy array
        """

        Tic = time.time()
        print('\nCompute image time series...')

        # Initialization
        S = NormalizeValues(self.Image)
        W = np.array([[0.1091, 0.1409, 0.1091],
                      [0.1409, 0, 0.1409],
                      [0.1091, 0.1409, 0.1091]])
        Y = np.zeros(S.shape)
        U = Y
        E = Y + 1
        TS = np.zeros(N)

        # Analysis
        for Time in range(N):
            U = 0.2 * U + S * correlate(Y, W, output='float', mode='reflect') + S
            E = 0.9 * E + 20 * Y
            X = 1 / (1 + np.exp(E - U))
            Y = (X > 0.5) * 1
            TS[Time] = sum(sum(Y))

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return TS

    def Restoration(self, f=0.03, g=0.99, Gamma=1., Delta=0.02):

        """
        Perform image restoration, filtering to remove noise using Spiking Cortical Model (SCM)
        Based on:
        Zhan, K., Shi, J., Wang, H. et al.
        Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
        Arch Computat Methods Eng 24, 573–588 (2017).
        https://doi.org/10.1007/s11831-016-9182-3

        :param: f: Decay factor for external input U(n) = S + f * U(n-1)
        :param: g: Decay factor for threshold E(n) = g * E(n-1) * (1-B)
        :param: Gamma: Amplitude factor for activation intensity
        :param: Delta: Weight factor for value averaging
        :return: Y: Restored Image
        """

        Tic = time.time()
        print('\nPerform image restoration...')

        # Initialization
        S = NormalizeValues(self.Image)
        Rows, Columns = S.shape
        U = np.zeros(S.shape)
        E = U + 1
        Y = np.zeros(S.shape)
        B = np.zeros(S.shape)
        T = np.zeros(S.shape)
        N = 0

        # Analysis - T image of similar gray intensities
        while sum(sum(B)) < S.size:

            N += 1

            for i in range(Rows):
                for j in range(Columns):

                    U[i, j] = S[i, j] + f * U[i, j]
                    if E[i, j] > 1E2:
                        Q = 0
                    else:
                        Q = 1 / (1 + np.exp(-Gamma * (U[i, j] - E[i, j])))

                    if Q > 0.5 or E[i, j] < 0.08:
                        Y[i, j] = 1
                        B[i, j] = 1
                        T[i, j] = N
                        E[i, j] = 100000
                    else:
                        Y[i, j] = 0

            E[B != 1] = g * E[B != 1]

        # Analysis - Average T values
        Y = np.zeros(S.shape)
        S = np.pad(S, 1, 'symmetric')
        T = np.pad(T, 1, 'symmetric')

        for i in range(1, Rows + 1):
            for j in range(1, Columns + 1):

                K = T[i - 1:i + 2, j - 1:j + 2].flatten()

                if len(np.unique(K)) == 1:
                    Y[i - 1, j - 1] = 1 / 9 * sum(sum(S[i - 1:i + 2, j - 1:j + 2]))
                else:
                    Sorted = np.sort(K)

                    if Sorted[4] == K[4]:
                        Y[i - 1, j - 1] = S[i, j]
                    elif Sorted[0] == K[4]:
                        Y[i - 1, j - 1] = S[i, j] - Delta
                    elif Sorted[-1] == K[4]:
                        Y[i - 1, j - 1] = S[i, j] + Delta
                    else:
                        Y[i - 1, j - 1] = np.median(S[i - 1:i + 2, j - 1:j + 2])

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return Y

    def Enhancement(self, d=2, h=2E10, g=0.9811, Alpha=0.01, Beta=0.03):

        """
        Perform image enhancement using Feature Linking Model (FLM)
        Based on:
        Zhan, K., Shi, J., Wang, H. et al.
        Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
        Arch Computat Methods Eng 24, 573–588 (2017).
        https://doi.org/10.1007/s11831-016-9182-3

        :param: d: Inhibition factor for linkin wave
        :param: h: Amplitude factor for threshold excitation
        :param: g: Decay factor for threshold
        :param: Alpha: Decay factor for linkin input
        :param: Beta: Decay factor for external input
        :return: Y: Restored Image
        """

        Tic = time.time()
        print('\nPerform image enhancement...')

        # Initialization
        S = NormalizeValues(self.Image) + 1/255
        W = np.array([[0.7, 1, 0.7], [1, 0, 1], [0.7, 1, 0.7]])
        Y = np.zeros(S.shape)
        U = np.zeros(S.shape)
        T = np.zeros(S.shape)
        SumY = 0
        N = 0

        Laplacian = np.array([[1 / 6, 2 / 3, 1 / 6], [2 / 3, -10 / 3, 2 / 3], [1 / 6, 2 / 3, 1 / 6]])
        Theta = 1 + correlate(S, Laplacian, mode='reflect')
        f = 0.75 * np.exp(-S ** 2 / 0.16) + 0.05
        G = GaussianKernel(7, 1)
        f = correlate(f, G, mode='reflect')

        # Analysis
        while SumY < S.size:
            N += 1

            K = correlate(Y, W, mode='reflect')
            Wave = Alpha * K + Beta * S * (K - d)
            U = f * U + S + Wave
            Theta = g * Theta + h * Y
            Y = (U > Theta) * 1
            T += N * Y
            SumY += sum(sum(Y))

        T_inv = T.max() + 1 - T

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return T_inv

    def Fusion(self,Beta1,Beta2,Beta3=0,dT=1/255,Sigma=1/255):

        """
        Perform image fusion using multi-channel PCNN
        Based on:
        Wang, Z., Ma, Y. (2008)
        Medical image fusion using m-PCNN
        Information Fusion, 9(2), 176–185
        https://doi.org/10.1016/j.inffus.2007.04.003

        :param: Beta1: Image 1 weight coefficient
        :param: Beta2: Image 2 weight coefficient
        :param: Beta3: Image 3 weight coefficient
        :param: dT: Linear threshold decay
        :param: Sigma: Fix neuron internal activity shift
        :return: T: Fused Image
        """

        Tic = time.time()
        print('\nPerform image fusion...')

        # Initialization
        S1 = NormalizeValues(self.Image1)
        S2 = NormalizeValues(self.Image2)
        if Beta3:
            S3 = NormalizeValues(self.Image3)
        else:
            S3 = np.zeros(S1.shape)

        W = np.array([[0.7, 1, 0.7], [1, 0, 1], [0.7, 1, 0.7]])

        BetaWeights = np.array([Beta1, Beta2, Beta3])
        Beta1, Beta2, Beta3 = BetaWeights / BetaWeights.sum()

        Y = np.zeros(S1.shape)
        T = np.zeros(S1.shape)
        Theta = np.ones(S1.shape)

        Vt = 100
        FireNumber = 0
        N = 0
        while FireNumber < S1.size:
            N += 1

            H1 = correlate(Y, W, output='float', mode='reflect') + S1
            H2 = correlate(Y, W, output='float', mode='reflect') + S2
            H3 = correlate(Y, W, output='float', mode='reflect') + S3

            U = (1 + Beta1 * H1) * (1 + Beta2 * H2) * (1 + Beta3 * H3) + Sigma
            U = U / U.max()

            Theta = Theta - dT + Vt * Y

            Y = (U > Theta) * 1

            T += N * Y

            FireNumber += sum(sum(Y))

        T = 1 - NormalizeValues(T)

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return T

    def SPCNN_Filtering(self,Beta=2,Delta=1/255,VT=100):

        """
        Filter image using simplified PCNN and single neuron firing
        Based on:
        Chong Shen and Ding Wang and Shuming Tang and Huiliang Cao and Jun Liu (2017)
        Hybrid image noise reduction algorithm based on genetic ant colony and PCNN
        Visual Computer (33) 11, 1373-1384

        :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
        :param Delta: Linear decay factor for threshold level
        :param VT: Dynamic threshold amplitude
        :return: H: Image histogram in numpy array
        """

        Tic = time.time()
        print('\nImage filtering...')

        # Initialize parameters
        S = NormalizeValues(self.Image)
        Rows, Columns = S.shape
        Y = np.zeros((Rows, Columns))
        T = np.zeros((Rows, Columns))
        W = np.array([[0.5, 1, 0.5],
                      [1, 0, 1],
                      [0.5, 1, 0.5]])
        Theta = np.ones((Rows, Columns))

        FiredNumber = 0
        N = 0

        # Perform segmentation
        while FiredNumber < S.size:

            N += 1
            F = S
            L = correlate(Y, W, output='float', mode='reflect')
            Theta = Theta - Delta + VT * Y

            U = F * (1 + Beta * L)
            Y = (U > Theta) * 1

            FiredNumber = FiredNumber + sum(sum(Y))

            MedianFilter = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9
            Y = correlate(Y,MedianFilter,output='int',mode='reflect')

            T = T + N * Y


        Output = 1 - NormalizeValues(T)

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output

    def SPCNN_Edges(self,Beta=2,Delta=1/255,VT=100):

        """
        Image edge detection using simplified PCNN and single neuron firing
        Based on:
        Shi, Z., Hu, J. (2010)
        Image edge detection method based on A simplified PCNN model with anisotropic linking mechanism
        Proceedings of the 2010 10th International Conference on Intelligent Systems Design and Applications, ISDA’10, 330–335
        https://doi.org/10.1109/ISDA.2010.5687242

        :param Beta: Linking strength parameter used for internal neuron activity U = F * (1 + Beta * L)
        :param Delta: Linear decay factor for threshold level
        :param VT: Dynamic threshold amplitude
        :return: H: Image histogram in numpy array
        """

        Tic = time.time()
        print('\nImage filtering...')

        # Initialize parameters
        S = NormalizeValues(self.Image)
        Rows, Columns = S.shape
        Y = np.zeros((Rows, Columns))
        T = np.zeros((Rows, Columns))
        W = np.array([[0.5, 1, 0.5],
                      [1, 0, 1],
                      [0.5, 1, 0.5]])
        Theta = np.ones((Rows, Columns))

        FiredNumber = 0
        N = 0

        # Perform analysis
        while FiredNumber < S.size:

            N += 1
            F = S
            L = correlate(Y, W, output='float', mode='reflect')
            Theta = Theta - Delta + VT * Y

            U = F * (1 + Beta * L)
            Y = (U > Theta) * 1

            FiredNumber = FiredNumber + sum(sum(Y))

            MedianFilter = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9
            Y = correlate(Y,MedianFilter,output='int',mode='reflect')

            T = T + N * Y


        Output = 1 - NormalizeValues(T)

        # Print time elapsed
        Toc = time.time()
        PrintTime(Tic, Toc)

        return Output

# Set path
CurrentDirectory = Path.cwd()
ImageDirectory = CurrentDirectory / 'Scripts/PCNN/'
ImageDirectory = CurrentDirectory / 'Tests/Osteons/HumanBone/'

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / 'Stained1_Registered.png'))
Array = sitk.GetArrayFromImage(Image)[5:-5,5:-5,:3]
PlotArray(Array, 'RGB Image')
PlotChanels(Array, 'R', 'G', 'B')
Lab = color.rgb2lab(Array)
PlotChanels(Lab, 'L', 'a', 'b')

# Use PCNN tool
PCNN_Tools = PCNN()

# Find boundaries
Match = exposure.match_histograms(Lab[:,:,2],Lab[:,:,1])
PlotArray(Match,'Matched')
PCNN_Tools.Set_Image(Match)
Y_Seg = PCNN_Tools.SPCNN_Segmentation(Delta=1/3)
PlotArray(Y_Seg,'Segmented')

Boundaries = PlotSegment(Y_Seg,2)


# Find Harvesian canals
PCNN_Tools.Set_Image(Lab[:,:,1])
Y_Seg = PCNN_Tools.SPCNN_Segmentation(Delta=1/5)
PlotArray(Y_Seg,'Segmented')
Segment = PlotSegment(Y_Seg,1)


Binary = Segment.copy()
Disk = morphology.disk(5)
for i in range(3):
    Binary = morphology.binary_dilation(Binary, Disk)
Disk = morphology.disk(10)
for i in range(3):
    Binary = morphology.binary_erosion(Binary, Disk)
Disk = morphology.disk(15)
Binary = morphology.binary_dilation(Binary, Disk)

PlotArray(Binary,'Segmented Image')

Markers = measure.label(Binary)
Props = ['area','label']
Regions = pd.DataFrame(measure.regionprops_table(Markers,properties=Props))
Filter = Regions['area'] > 2000
Harvesian = np.isin(Markers, Regions[Filter]['label'])
PlotArray(Harvesian,'Harvesian canals')

# Compute distances from Harvesian canals
MedialAxis, Distances = morphology.medial_axis(1-Harvesian, return_distance=True)
PlotArray(Distances, 'Distances')

Combine = (1-Boundaries) * NormalizeValues(Distances)
PlotArray(1-Combine,'Combined')

W_Seg = segmentation.watershed(Boundaries,Markers,connectivity=1)
PlotArray(W_Seg,'Watershed segmentation')


Otsu = filters.threshold_otsu(Combine)
Limits = Combine.copy()
Limits[Combine < Otsu] = 0
Limits[Combine >= Otsu] = 1
PlotArray(Limits,'Limits')

Gradient = np.round(HSV[:,:,1] * 255).astype('uint8')
Gradient = filters.rank.gradient(Gradient, morphology.disk(1))
Gradient = NormalizeValues(Gradient)
PlotArray(Gradient,'Gradient')

Edges = feature.canny(Gradient, sigma=1) * 1
PlotArray(Edges,'Edges')



Combine = (1-Y_Stretched) * Distances / Distances.max() + Limits + Gradient
PlotArray(Combine,'Combined')

W_Seg = segmentation.watershed(Combine,Markers,connectivity=1,mask=1-Limits)
PlotArray(W_Seg,'Watershed segmentation')


def WatershedFlood(Image, Labels, Vmax):
    """
    Insipired from master thesis of Josephson
    Does not work, for the moment
    """

    # Number of initial labels
    N = len(np.unique(Labels))

    # Initialize priority queue
    Keys = np.arange(256).astype('int')
    Q = {Key: [] for Key in Keys}

    # Transform image in 8bits integer image
    Image = np.round(NormalizeValues(Image) * 255).astype('uint8')

    for i in range(N):
        Y, X = np.where(Labels == i + 1)

        for j in range(len(X)):
            Key = Image[Y[j], X[j]]
            Values = [Y[j], X[j]]

            if Q[Key]:
                Q[Key] = [Q[Key][0], Values]
            else:
                Q[Key] = [Values]

    EmptyQ = False

    while not EmptyQ:

        for i in range(256):

            if not Q[i]:
                EmptyQ = True
                continue

            EmptyQ = False

            for j in range(len(Q[i])):
                Y, X = Q[i].pop(0)

                if X == 0:
                    cX = [X, X + 1]
                elif X == Image.shape[1] - 1:
                    cX = [X - 1, X]
                else:
                    cX = [X - 1, X, X + 1]
                if Y == 0:
                    cY = [Y, Y + 1]
                elif Y == Image.shape[0] - 1:
                    cY = [Y - 1, Y]
                else:
                    cY = [Y - 1, Y, Y + 1]

                Marker = Labels[Y, X]

                for x in cX:
                    for y in cY:

                        C1 = x == X
                        C2 = y == Y
                        C3 = Labels[y, x] > 0
                        C4 = Image[y, x] > Vmax

                        if not (C1 * C2 + C3 + C4):
                            Labels[y, x] = Marker
                            Key = Image[y, x]
                            Values = [y, x]

                            if Q[Key]:
                                Q[Key] = [Q[Key][0], Values]
                            else:
                                Q[Key] = [Values]

                break

            break

    return Labels


Test = WatershedFlood(Combine,Markers,250)
PlotArray(Test == 3,'WatershedTest')


## Segment combined array, label it and try watershed

PCNN_Tools.Set_Image(Y_Seg)
H, Bins = PCNN_Tools.Histogram(Plot=True)



Disk = morphology.disk(4)
Segments = morphology.binary_erosion(Segment,Disk)
PlotArray(Segments,'Segmented Image')



G = GaussianKernel(5,2)
Gauss = correlate(HSV[:,:,1],G)
PlotArray(Gauss, 'Gauss Image')

W = np.array([[1,1,1],[1,1,1],[1,1,1]])
Gradient = filters.rank.gradient(np.round(HSV[:,:,1]*255).astype('uint8'), W)
PlotArray(Gradient, 'Gradient Image')


Threshold = Gradient.max() * 0.4
Gradient[Gradient < Threshold] = 0
Gradient[Gradient >= Threshold] = 1
PlotArray(Gradient, 'Gradient Image')




PCNN_Tools.Set_Image(Gradient)
Test = PCNN_Tools.SPCNN_Filtering(Beta=0.289,Delta=0.1,VT=0.9)
PlotArray(Test,'Filtered')

# Open manually segmented image
SegmentedImage = sitk.ReadImage(ImageDirectory + 'PCNN_Test_Seg.png')
SegmentedArray = sitk.GetArrayFromImage(SegmentedImage)
Segments = GetSegments(SegmentedArray)
PlotArray(Segments[:,:-5],'Segments')

# Compute optimal parameters
PlotArray(Y_Stretched[:-4,8:],'Cropped')
OptimalParameters = PSO_PCNN(Y_Stretched[:-4,8:],Segments[:,:-5],AT=True,FastLinking=True)
Beta = OptimalParameters['Beta']
AlphaT = OptimalParameters['AlphaT']
VT = OptimalParameters['VT']
AlphaL = OptimalParameters['AlphaL']
VL = OptimalParameters['VL']
AlphaF = OptimalParameters['AlphaF']
VF = OptimalParameters['VF']


Y_Seg = PCNN_Tools.PCNN_Segmentation(Beta=Beta,AlphaT=AlphaT,VT=VT,
                                     AlphaL=AlphaL,VL=VL,
                                     AlphaF=AlphaF,VF=VF,
                                     AT=True,FastLinking=True)
Y_Seg = PCNN_Tools.PCNN_Segmentation(Beta=0.857,AlphaT=0.064,VT=0.63,
                                     AlphaL=0.653,VL=0.379,
                                     AlphaF=0.999,VF=0.734)
PlotArray(Y_Seg,'Segmented Image')



PCNN_Tools.Images2Fuse(U,1-S)
Y = PCNN_Tools.Fusion(Beta1=1/2,Beta2=1/2)
S2 = S**2
PlotArray(S2, 'Fused Image')




PCNN_Tools.Set_Image(Y_Stretched)
Y_Seg = PCNN_Tools.SPCNN_Segmentation()
PlotArray(Y_Seg,'Segmented Image')


PCNN_Tools.Set_Image(Y_Seg)
H2, Bins = PCNN_Tools.Histogram()
PCNN_Tools.Set_Image(Y_Stretched)
H3, Bins = PCNN_Tools.Histogram()

Figure, Axes = plt.subplots(1,1,figsize=(5.5,4.5))
# Axes.bar(x=np.arange(0,256), height=H3 / H3.sum(), width=1, color=(1,0,1),label='Stretched')
# Axes.bar(x=np.arange(0,256), height=H2 / H2.sum(), width=1, color=(1,0,0),label='Enhanced')
Axes.bar(x=np.arange(0,256), height=H1 / H1.sum(), width=1, color=(0,0,1),label='Original')
plt.legend(loc='best')
plt.show()
plt.close(Figure)

Peaks = np.where(H3 != 0)[0]
Minima = Peaks[:-1] + np.round(np.diff(Peaks) / 2).astype('int')

PCNN_Tools.Set_Image(Y)
Y_Seg = PCNN_Tools.Watershed_Segmentation(Bins[Peaks],Bins[Minima])
PlotArray(Y_Seg, 'Segmented Image')

for b in np.unique(Y_Seg):
    a = Y_Seg.copy()
    a[a != b] = 0
    a[a == b] = 1
    PlotArray(a,'Single gray level')




PCNN_Tools.Set_Image(Y_Seg)
Y = PCNN_Tools.SPCNN_Filtering()
PlotArray(Y,'Filtered')

# V = C1 * np.random.uniform(-1, 1, (Ps, Dimensions)) * RangeAmplitudes \
#     + C2 * np.random.uniform(-1, 1) * RangeAmplitudes


for v in np.unique(Y_Seg):
    a = Y_Seg.copy()
    a[a != v] = 0
    a[a == v] = 1
    PlotArray(Y_Seg,'Segmented Image')




Threshold = 1
Y_Thresh = np.zeros(Y_Stretched.shape)
Y_Thresh[Y_Stretched == Threshold] = 1
PlotArray(Y_Thresh,'Thresholded Image')


Figure, Axes = plt.subplots(1,1,figsize=(5.5,4.5))
Axes.imshow(Y_Stretched,cmap='gray')
Axes.imshow(Segments,cmap='jet',alpha=0.5)
plt.axis('off')
plt.title('Overlay')
plt.tight_layout()
plt.show()
plt.close(Figure)







Segments = CementLines + Harvesian + Osteocytes
Disk = morphology.disk(2)
Segments = morphology.binary_erosion(Harvesian,Disk)
Disk = morphology.disk(5)
Segments = morphology.binary_dilation(CementLines,Disk)
PlotArray(Segments, 'Manual Segmentation')

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / 'Stained1_Registered.png'))
Array = sitk.GetArrayFromImage(Image)
PlotArray(Array, 'RGB Image')

# Match histograms for better phase difference and clarity and rescale
R, G, B = Array[:,:,0], Array[:,:,1], Array[:,:,2]
GS = exposure.match_histograms(R, B)
GS_Rescaled = NormalizeArray(GS)
PlotArray(GS_Rescaled, 'Grayscale Image')


Gamma, Gain = 2, 1
GS_Contrast = exposure.adjust_gamma(GS_Rescaled,Gamma,Gain)
PlotArray(GS_Contrast, 'Gamma:' + str(Gamma) + ' Gain:' + str(Gain))

Sigma = 5
GS_Gauss = filters.gaussian(GS_Rescaled,sigma=Sigma)
GS_Gauss = NormalizeArray(GS_Gauss)
PlotArray(GS_Gauss, 'Gauss sigma:' + str(Sigma))


# Particle Swarm Optimization (PSO) algorithm
Ps = 20         # Population size
t = 0           # Iteration number
Max_times = 10  # Max iteration number
Omega = 0.9 - 0.5 * t/Max_times     # Inertia factor
Average_FV_std = 1E-3   # Second PSO termination condition
Image = 1-R / R.max()

# PSO step 1 - Initialization
AlphaF_Range = np.array([-1,1])*3
AlphaL_Range = np.array([-1,1])*3
AlphaT_Range = np.array([-1,1])*3

VF_Range = np.array([-1,1])*3
VL_Range = np.array([-1,1])*3
VT_Range = np.array([-1,1])*3

Beta_Range = np.array([-1,1])*3

Ranges = np.array([AlphaF_Range,AlphaL_Range,AlphaT_Range,
                   VF_Range,VL_Range,VT_Range,Beta_Range])
Dimensions = len(Ranges)

C1, C2 = 2, 2

RangeAmplitudes = Ranges[:,1] - Ranges[:,0]
X = np.random.uniform(0,1,(Ps,Dimensions)) * RangeAmplitudes + Ranges[:,0]
V = C1 * np.random.uniform(-1,1,(Ps,Dimensions)) * RangeAmplitudes \
  + C2 * np.random.uniform(-1,1) * RangeAmplitudes



# PSO step 2 - Initial evaluation
ParameterList = ['AlphaF','AlphaL','AlphaT','VF','VL','VT','Beta']
Initial_DSCs = np.zeros((Ps, 1))
for ParticleNumber in range(Ps):
    ParametersDictionary = {}
    for ParameterNumber in range(Dimensions):
        ParameterName = ParameterList[ParameterNumber]
        ParameterValue = X[ParticleNumber,ParameterNumber]
        ParametersDictionary[ParameterName] = ParameterValue

    Y = PCNN_Segmentation(Image,ParametersDictionary)

    # Compute dice similarity coefficient with manual segmentation
    Initial_DSCs[ParticleNumber, 0] = DiceCoefficient(Y, Segments)


# Set initial best values
G_Best_Value = Initial_DSCs.max()
G_Best_Index = np.where(Initial_DSCs == G_Best_Value)[0][0]
G_Best = X[G_Best_Index]

P_Best_Values = Initial_DSCs.copy()
P_Best = X.copy()


## Start loop
Average_FVs = np.array([1,0,0])
NIteration = 0
while NIteration < 100 and Average_FVs.std() >= Average_FV_std:

    ## PSO step 3 - Update positions and velocities
    R1, R2 = np.random.uniform(0, 1, 2)
    V = Omega * V + C1 * R1 * (P_Best - X) + C2 * R2 * (G_Best - X)
    X = X + V
    # If new position exceed limits, set to limit
    X[X < Ranges[:,0]] = np.tile(Ranges[:,0],Ps).reshape((Ps,Dimensions))[X < Ranges[:,0]]
    X[X > Ranges[:,1]] = np.tile(Ranges[:,1],Ps).reshape((Ps,Dimensions))[X > Ranges[:,1]]



    ## PSO step 4 - Evaluation of the updated population
    New_DSCs = np.zeros((Ps, 1))
    for ParticleNumber in range(Ps):
        ParametersDictionary = {}
        for ParameterNumber in range(Dimensions):
            ParameterName = ParameterList[ParameterNumber]
            ParameterValue = X[ParticleNumber,ParameterNumber]
            ParametersDictionary[ParameterName] = ParameterValue

        Y = PCNN_Segmentation(Image,ParametersDictionary)

        # Compute dice similarity coefficient with manual segmentation
        New_DSCs[ParticleNumber, 0] = DiceCoefficient(Y, Segments)

    # Update best values if better than previous
    if New_DSCs.max() > G_Best_Value:
        G_Best_Value = New_DSCs.max()
        G_Best_Index = np.where(New_DSCs == G_Best_Value)[0][0]
        G_Best = X[G_Best_Index]

    ImprovedValues = New_DSCs > P_Best_Values
    P_Best_Values[ImprovedValues] = New_DSCs[ImprovedValues]
    Reshaped_IP = np.tile(ImprovedValues,Dimensions).reshape((Ps,Dimensions))
    P_Best[Reshaped_IP] = X[Reshaped_IP]



    ## PSO step 5 - Update and check if terminal condition is satisfied
    NIteration += 1
    Average_FVs = np.concatenate([Average_FVs[1:],np.array(G_Best_Value).reshape(1)])


## PSO step 6 - Output results
ParametersDictionary = {}
for ParameterNumber in range(Dimensions):
    ParameterName = ParameterList[ParameterNumber]
    ParameterValue = G_Best[ParameterNumber]
    ParametersDictionary[ParameterName] = ParameterValue
Y = PCNN_Segmentation(Image,ParametersDictionary)
PlotArray(Y, 'PCNN Segmentation')

ParametersDictionary = {'AlphaF': -0.2381907530225938,
                        'AlphaL': -0.2398532814016563,
                        'AlphaT': 0.7220370952618878,
                        'VF': -0.30581795422704827,
                        'VL': 1.0,
                        'VT': 0.5418764530883855,
                        'Beta': -1.0}

# Parameters for 1 osteon
ParametersDictionary = {'AlphaF': -0.2381907530225938,
                        'AlphaL': -0.2398532814016563,
                        'AlphaT': 0.7220370952618878,
                        'VF': -0.30581795422704827,
                        'VL': 1.0,
                        'VT': 0.5418764530883855,
                        'Beta': -1.0}
# Parameters for multiple osteons
ParametersDictionary = {'AlphaF': -1.0,
                        'AlphaL': -1.0,
                        'AlphaT': 1.0,
                        'VF': -0.6907362822709551,
                        'VL': -1.0,
                        'VT': 1.0,
                        'Beta': 1.0}
ParametersDictionary = {'AlphaF': -1.0,
                        'AlphaL': -0.5171884715426962,
                        'AlphaT': 1.0,
                        'VF': -1.0,
                        'VL': -1.0,
                        'VT': 0.8293069163951987,
                        'Beta': 0.9486743832771634}
ParametersDictionary = {'AlphaF': -1.4568293766698421,
                        'AlphaL': 1.7664468143804637,
                        'AlphaT': 0.90684945230546,
                        'VF': -1.2138776497934514,
                        'VL': -2.430799349519539,
                        'VT': -1.9142988177549285,
                        'Beta': 0.2516870868740857}
ParametersDictionary = {'AlphaF': -0.4358389450874143,
                        'AlphaL': 0.42548990174264945,
                        'AlphaT': -0.0026282587881454322,
                        'VF': 0.12978216128591713,
                        'VL': -0.5343658289990307,
                        'VT': -0.6003009584815666,
                        'Beta': -0.5078775611128168}
# Parameters for Harvesian channels
ParametersDictionary = {'AlphaF': 0.559958004152834,
                        'AlphaL': 0.6362288702766017,
                        'AlphaT': -0.2964094323025681,
                        'VF': -0.6665908949885424,
                        'VL': 1.0,
                        'VT': -1.0,
                        'Beta': 1.0}
ParametersDictionary = {'AlphaF': -0.77116675769926,
                        'AlphaL': 0.3906772796273532,
                        'AlphaT': -1.0,
                        'VF': -0.48348777605091137,
                        'VL': 0.7318494869324401,
                        'VT': -0.026910000212507557,
                        'Beta': 1.0}

Disk = morphology.disk(2)
BW_Erode = morphology.binary_erosion(Y,Disk)
Disk = morphology.disk(2+10+5)
BW_Dilate = morphology.binary_dilation(BW_Erode,Disk)
PlotArray(BW_Dilate, 'Dilated segmentation')

Labels = measure.label(BW_Dilate,connectivity=2)
Properties = ('label', 'area', 'orientation', 'euler_number')
PropertiesTable = pd.DataFrame(measure.regionprops_table(Labels,properties=Properties))

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(PropertiesTable['euler_number'], marker='o',linestyle='none',color=(1,0,0),fillstyle='none')
plt.show()
plt.close(Figure)

PropertiesTable.sort_values('euler_number').iloc[:11]['label'].values
RegionsLabels = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]

for Region in RegionsLabels:
    PlotROI(Y, Region, Labels)






# Find PCNN parameters for harvesian channels
# Compute distances from harvesian channels
# Find PCNN parameters for cement lines
# Watershed from harvesian with cement lines limits and distances
###############################################################




# Compare with initial segments
C1 = np.array([[0, 0, 0, 0], [1, 0, 0, 1]])
ColorMap1 = mpl.colors.ListedColormap(C1)
C2 = np.array([[0, 0, 0, 0], [0, 1, 0, 1]])
ColorMap2 = mpl.colors.ListedColormap(C2)
C3 = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
ColorMap3 = mpl.colors.ListedColormap(C3)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
# Axes.imshow(ImageArray,cmap='gray')
Axes.imshow(CementLines, cmap=ColorMap1, vmin=0.2, vmax=0.8)
Axes.imshow(Harvesian, cmap=ColorMap2, vmin=0.2, vmax=0.8)
Axes.imshow(Osteocytes, cmap=ColorMap3, vmin=0.2, vmax=0.8)
Axes.axis('off')
Axes.set_title('Segments')
plt.tight_layout()
plt.show()
plt.close(Figure)












GS_Gradient = filters.rank.gradient(GS_Rescaled, morphology.disk(2))
PlotArray(GS_Gradient, 'Grayscale Gradient')

Threshold = filters.threshold_otsu(GS_Gradient)
GS_Gradient_Seg = (GS_Gradient > Threshold) * 1
PlotArray(GS_Gradient_Seg, 'Otsu segmentation')

Combine = (Y + GS_Gradient_Seg) / 2
Combine[Combine < 1] = 0
PlotArray(Combine, 'Combine segmentation')




GS_Enhanced = exposure.match_histograms(GS, Y)
GS_Enhanced = NormalizeArray(GS_Enhanced)
PlotArray(GS_Enhanced, 'Grayscale Image')


Threshold = 250
BW = T.copy()
BW = (BW >= Threshold) * 1
PlotArray(BW, 'Threshold Segmentation')

MedialAxis, Distances = morphology.medial_axis(1-BW, return_distance=True)
PlotArray(Distances, 'Distances', ColorBar=True)

Combine = (256-T)*Distances/Distances.max()
Combine = NormalizeArray(Combine)
PlotArray(Combine, 'T x Distances', ColorBar=True)

Threshold = filters.threshold_otsu(Combine)
Combine_Seg = (Combine > Threshold*0.8) * 255
PlotArray(Combine_Seg, 'Otsu segmentation')

WS_Limits = (Combine_Seg + Combine) / 2
# WS_Limits[WS_Limits > 255] = 255
PlotArray(WS_Limits, 'Test', True)

Labels = BW
WS = segmentation.watershed(WS_Limits,Labels,connectivity=2)
PlotArray(WS, 'Watershed segmentation')
WS_Edges = segmentation.find_boundaries(WS) * 255







# 3D visualization of the picture
XGrid = np.arange(GS_Rescaled.shape[1])
YGrid = np.arange(GS_Rescaled.shape[0])
XGrid, YGrid = np.meshgrid(XGrid,YGrid)

Figure, Axes = plt.subplots(subplot_kw={"projection": "3d"})
Axes.plot_surface(XGrid,YGrid,GS_Rescaled,cmap='jet')
plt.show()
plt.close(Figure)

W = GaussianKernel(10,7)
GS_Smooth = correlate(GS_Rescaled, W, output='float', mode='reflect')
PlotArray(GS_Smooth,'Grayscale Filtered')

Figure, Axes = plt.subplots(subplot_kw={"projection": "3d"})
Axes.plot_surface(XGrid, YGrid, GS_Smooth, cmap='gray')
plt.show()

GS_Gradient = filters.rank.gradient(GS_Rescaled, morphology.disk(2))
PlotArray(GS_Gradient, 'Grayscale Gradient')

GS_Edges_F = filters.frangi(GS_Rescaled, sigmas=range(1,5,1), mode='reflect')
PlotArray(GS_Edges_F, 'Grayscale Edges (Frangi)')

GS_Edges_S = filters.sobel(GS_Rescaled, mode='reflect')
PlotArray(GS_Edges_S, 'Grayscale Edges (Sobel)')

GS_Edges_C = feature.canny(GS_Rescaled,sigma=2)
PlotArray(GS_Edges_C, 'Grayscale Edges (Canny)')

GS_Edges_R = filters.roberts(GS_Rescaled)
PlotArray(GS_Edges_R, 'Grayscale Edges (Roberts)')

Threshold = filters.threshold_otsu(GS_Rescaled)
GS_Seg = (GS_Edges_F > Threshold) * 1
PlotArray(GS_Seg, 'Otsu segmentation')

Thresholds = filters.threshold_multiotsu(GS_Gradient,classes=2)
GS_Seg = np.digitize(GS_Gradient, bins=Thresholds)
PlotArray(GS_Seg, '3 Otsu segmentation')



XR_min = 256 - 150
XR_max = 256 - 40

XB_min = 256 - 215
XB_max = 256 - 100

Wl = np.array([[0.25, 1, 0.25],[1, 0, 1],[0.25, 1, 0.25]])

LR_min = np.sum(Wl[:int(round(Wl.shape[0]/2)),:int(round(Wl.shape[1]/2))])
LB_min = np.sum(Wl[:int(round(Wl.shape[0]/2)),:int(round(Wl.shape[1]/2))])
LB_max = np.sum(Wl[:int(round(Wl.shape[0]/2)),:int(round(Wl.shape[1]/2))])

Beta_max = ((XR_max/XB_max)-1)/LB_max
Beta_min = max(((XR_max/XR_min)-1)/LR_min, ((XB_max/XB_min)-1)/LB_min)

Wh = Wl

T_S = PCNN_Segmentation(GS_Contrast, Wl, Wh, Beta=0.7, Gamma=0.05, dT=1)
PlotArray(T_S, 'Inhibitted PCNN')

def PCNN_Segmentation(GS_Image, Wl, Wh, Beta=0.7, Gamma=0.00, dT=1):

    S = GS_Image

    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    T = Y

    F = S

    Theta = 255 * np.ones((Rows, Columns))
    Vt = 400

    FireNumber = 0
    N = 0
    while FireNumber < S.size:

        N += 1

        L = correlate(Y, Wl, output='float', mode='reflect')
        H = correlate(Y, Wh, output='float', mode='reflect')
        Theta = Theta - dT + Vt * Y
        Fire = 1

        while Fire == 1:
            Q = Y
            U = F * (1 + Beta * L) * (1 - Gamma * H)
            Y = (U > Theta) * 1

            if np.array_equal(Q, Y):
                Fire = 0
            else:
                L = correlate(Y, Wl, output='float', mode='reflect')
                # H = correlate(Y, Wh, output='float', mode='reflect')

        T = T + N * Y
        FireNumber += sum(sum(Y))

    T = 256 - T

    return T





## New technique using euclidean distances as threshold
Image = sitk.ReadImage(str(ImageDirectory / 'Stained1_Registered.png'))
Array = sitk.GetArrayFromImage(Image)[:,:,:3]
PlotArray(Array, 'RGB Image')

Norms = NormalizeValues(np.linalg.norm(Array, axis=2))
PlotArray(Norms,'Norms')

# Very simplified PCNN - Initialization
Threshold = np.ones(Norms.shape)
Theta = np.ones(Norms.shape)
T = np.zeros(Norms.shape)
N = 0
W = np.array([[0.5, 1, 0.5],
              [1, 0, 1],
              [0.5, 1, 0.5]])
FiredNumber = 0

# Decrease threshold
Delta_T = 1/5
Condition = FiredNumber < Norms.size
while Condition:
    Threshold = Threshold - Delta_T
    Y = (Norms > Threshold) * 1
    Delta = 0.15

    # Get vectors of firing pixels and compute euclidian distances
    Vectors = np.unique(Array[Norms > Threshold],axis=0)

    Delta = 1/255
    VT = 100
    Beta = 2

    for Vector in Vectors:
        N += 1
        F = 1 - NormalizeValues(np.linalg.norm(Array - Vector, axis=2))
        L = correlate(Y, W, output='float', mode='reflect')
        Theta = Theta - Delta + VT * Y

        U = F * (1 + Beta * L)
        Y = (U > Theta) * 1
        T = T + N * Y

    FiredNumber = FiredNumber + sum(sum((T > 0) * 1))
    Condition = FiredNumber < Norms.size


Output = 1 - NormalizeValues(T)
PlotArray(Output,'Output')



# Test pipeline
HSV = RGB2HSV(Array)
PlotChanels(HSV, 'H', 'S', 'V')

Y_Stretched = GrayStretch(HSV[:,:,1],0.95)
PlotArray(Y_Stretched, 'Stretched Image')

Gradient = np.round(HSV[:,:,1] * 255).astype('uint8')
Gradient = filters.rank.gradient(Gradient, morphology.disk(20))
Gradient = NormalizeValues(Gradient)
PlotArray(Gradient,'Gradient')

Gradient = Array.copy()
for i in range (3):
    G = filters.rank.gradient(Array[:,:,i], morphology.disk(20))
    Gradient[:,:,i] = G
Gradient = NormalizeValues(Gradient)

GradientNorm = np.linalg.norm(Gradient, axis=2)
PlotArray(GradientNorm,'Gradient Norm')

# Use PCNN tool
PCNN_Tools = PCNN()

# Find Harvesian canals
PCNN_Tools.Set_Image(GradientNorm)
H, Bins = PCNN_Tools.Histogram(Plot=True)


Otsu = filters.threshold_otsu(GradientNorm)
# Otsu = 0.05
Limits = GradientNorm.copy()
Limits[GradientNorm < Otsu] = 0
Limits[GradientNorm >= Otsu] = 1
PlotArray(Limits,'Limits')


BinArray = np.zeros(Gradient.shape)
BinArray[Gradient > Bins[-2]] = 1
PlotArray(BinArray,'Bin')

Vector = np.array([125,125,50])
PlotArray(np.abs(Gradient - Vector),'Test')
PlotArray(YUV[2300:3300,2300:3300,1],'Test')

# Find Harvesian canals
PCNN_Tools.Set_Image(YUV[:,:,1])
Y_Seg = PCNN_Tools.SPCNN_Segmentation(Delta=1/7)
PlotArray(Y_Seg,'Segmented')

Segments = PlotSegment(Y_Seg,4)
Markers = measure.label(Segments)
RegionsProps = measure.regionprops(Markers.astype('int'))



