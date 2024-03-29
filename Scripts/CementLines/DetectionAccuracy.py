import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import exposure, morphology, color, filters
import statsmodels.formula.api as smf


# Set path and variables
CurrentDirectory = Path.cwd()
ImageDirectory = CurrentDirectory / 'Tests/Osteons/Sensitivity/'

PixelLength = 1.0460251046025104 # Computed with 418 RM
ROISize = 1000 # Size in um
SemiLength = int(round(ROISize/PixelLength/2))

DataFrame = pd.read_csv(str(ImageDirectory / 'Data.csv'))
N = 2
SampleData = DataFrame.loc[N]
Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '_Seg.jpg')

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / Name))
Array = sitk.GetArrayFromImage(Image)[:,:,:3]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array)
plt.show()

# Extract ROI
Point = [2000,6000] # y and xx coordinate, respectively
Area = [[Point[0] - SemiLength, Point[0] + SemiLength],
        [Point[1] - SemiLength, Point[1] + SemiLength]]
ROI = Array[Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(ROI)
plt.show()

# Mark areas where there is bone
Filter1 = ROI[:,:,0] < 190
Filter2 = ROI[:,:,1] < 190
Filter3 = ROI[:,:,2] < 235
Bone = Filter1 & Filter2 & Filter3

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Bone,cmap='binary')
plt.show()

# Erode and dilate to remove small bone parts
Disk = morphology.disk(2)
Dilated = morphology.binary_dilation(Bone,Disk)
Bone = morphology.binary_erosion(Dilated,Disk)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Bone,cmap='binary')
plt.show()



# Filter image to extract manual segmentation
Filter1 = ROI[:,:,0] > 100
Filter2 = ROI[:,:,1] < 90
Filter3 = ROI[:,:,2] > 100

Bin = np.zeros(Filter1.shape)
Bin[Filter1 & Filter2 & Filter3] = 1

# Dilate to link extracted segmentation
Disk = morphology.disk(5)
BinDilate = morphology.binary_dilation(Bin,Disk)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(BinDilate,cmap='binary')
plt.show()

# Skeletonize to obtain 1 pixel thickness
Skeleton = morphology.skeletonize(BinDilate)
Figure, Axis = plt.subplots(1,1)
Axis.imshow(Skeleton,cmap='binary')
plt.show()

# Compute ROI cement line density
CMD = Skeleton.sum() / Bone.sum()


# Read non-segmented image
Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '.jpg')

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / Name))
Array = sitk.GetArrayFromImage(Image)[:,:,:3][Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array)
plt.show()


# Define utilities functions
def PlotArray(Array, Title, CMap='gray', ColorBar=False):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    CBar = Axes.imshow(Array, cmap=CMap)
    if ColorBar:
        plt.colorbar(CBar)
    plt.title(Title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def PlotChanels(MChanelsArray, ChanelA, ChanelB, ChanelC, CMap='gray', CBar=False):

    A = MChanelsArray[:, :, 0]
    B = MChanelsArray[:, :, 1]
    C = MChanelsArray[:, :, 2]

    Figure, Axes = plt.subplots(1, 3)
    Axes[0].imshow(A, cmap=CMap)
    Axes[0].set_title(ChanelA + ' Channel')
    Axes[0].axis('off')
    Axes[1].imshow(B, cmap=CMap)
    Axes[1].set_title(ChanelB + ' Channel')
    Axes[1].axis('off')
    Image = Axes[2].imshow(C, cmap=CMap)
    Axes[2].set_title(ChanelC + ' Channel')
    Axes[2].axis('off')

    if CBar:
        CBar_Axis = Figure.add_axes([0.1, 0.15, 0.8, 0.05])
        Figure.colorbar(Image, cax=CBar_Axis, orientation="horizontal")

    plt.subplots_adjust(0.05,0.05,0.95,0.95,0.05)
    plt.show()
    plt.close(Figure)

    return
def PlotSegments(SegmentedImage, SegmentsNumber):

    SegmentValues = np.unique(SegmentedImage)
    SegmentsArray = np.zeros(SegmentedImage.shape)
    SegmentsNumbers = np.arange(min(SegmentsNumber),max(SegmentsNumber)+1)
    for SegmentNumber in SegmentsNumbers:
        Segment = SegmentValues[SegmentNumber]
        Filter = SegmentedImage != Segment

        PlottedArray = SegmentedImage.copy()
        PlottedArray[Filter] = -1
        PlottedArray[PlottedArray >= 0] = 0

        SegmentsArray += PlottedArray + 1

    PlotArray(SegmentsArray, 'Segments ' + ' '.join(str(SegmentNumber) for SegmentNumber in SegmentsNumber) )

    return SegmentsArray
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
def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))
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
class PCNNTools:

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

PlotChanels(Array, 'R', 'G', 'B', CBar=True)
Lab = color.rgb2lab(Array)
PlotChanels(Lab, 'L', 'a', 'b', CBar=True)


Red = Lab[:,:,2]
v_min, v_max = np.percentile(Red, (10, 90))
Stretched_Red = exposure.rescale_intensity(Red, in_range=(v_min, v_max))
PlotArray(Stretched_Red,'Stretched')


Otsus = filters.threshold_multiotsu(Stretched_Red,classes=3)
Binary = np.ones(Stretched_Red.shape)
Binary[Stretched_Red < Otsus[0]] = 0
Binary[Stretched_Red >= Otsus[1]] = 2
PlotArray(Binary,'Otsu')


PCNN = PCNNTools()
PCNN.Set_Image(Stretched_Red)
Segments = PCNN.SPCNN_Segmentation(Delta=1/2)
Segment = PlotSegments(2-Segments,[0])