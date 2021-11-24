"""
Code for testing PCNN-PSO-AT with different inputs or fitness function
"""""

import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import exposure, morphology, filters, feature, segmentation, measure
import matplotlib as mpl

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

def RGB2Gray(RGBImage):
    """
    This function convert color image to gray scale image
    based on matplotlib linear approximation
    """

    R, G, B = RGBImage[:,:,0], RGBImage[:,:,1], RGBImage[:,:,2]
    Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    return np.round(Gray).astype('uint8')
def SaveImage(Image,Name):

    from skimage import io
    io.imsave(Name, Image.astype(np.uint8))

    return
def NormalizeArray(Array):

    Normalized_Array = (Array - Array.min()) / (Array.max() - Array.min()) * 255

    return np.round(Normalized_Array).astype('uint8')
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
def PCNN_Segmentation(Image,ParametersDictionary,MaxIteration=10):

    # Initialization
    AlphaF = ParametersDictionary['AlphaF']
    AlphaL = ParametersDictionary['AlphaL']
    AlphaT = ParametersDictionary['AlphaT']

    VF = ParametersDictionary['VF']
    VL = ParametersDictionary['VL']
    VT = ParametersDictionary['VT']

    Beta = ParametersDictionary['Beta']

    # Input parameters
    S = Image / Image.max()

    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    Vb, New_Vb = 0, 0
    # W = GaussianKernel(1, 1)
    W = np.array([[0.5,1,0.5],[1,0,1],[0.5,1,0.5]])

    ## Feeding input
    F = S
    WF = W

    ## Linking input
    L = Y
    WL = W

    # Dynamic threshold
    Theta = np.ones((Rows, Columns))

    N = 0
    while New_Vb >= Vb and N < MaxIteration:

        N += 1
        F = S + F * np.exp(-AlphaF) + VF * correlate(Y, WF, output='float', mode='reflect')
        L = L * np.exp(-AlphaL) + VL * correlate(Y, WL, output='float', mode='reflect')
        U = F * (1 + Beta*L)

        Theta = Theta * np.exp(-AlphaT) + VT * Y
        Y = (U > Theta) * 1

        # Update variance
        Vb = New_Vb
        New_Vb = BetweenClassVariance(S, Y)

        if New_Vb >= Vb:
            Best_Y = Y

    return Best_Y
def Mutual_Information(Image1, Image2):

    """ Mutual information for joint histogram """

    Hist2D = np.histogram2d(Image1.ravel(), Image2.ravel(), bins=20)[0]

    # Convert bins counts to probability values
    pxy = Hist2D / float(np.sum(Hist2D))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
def DiceCoefficient(Image1,Image2):

    Dice = 2 * np.sum(Image1 * Image2) / np.sum(Image1 + Image2)

    return Dice
def PlotROI(ImageArray, RegionLabel, Labels):

    C = np.array([[0, 0, 0, 0], [1, 0, 0, 0.5]])
    ColorMap = mpl.colors.ListedColormap(C)
    Region = (Labels == RegionLabel)

    E = measure.EllipseModel()
    X, Y = np.where(Labels == RegionLabel)
    XY = np.concatenate(np.array([X, Y]).T, axis=0).reshape((len(X), 2))
    E.estimate(XY)
    Y0, X0, R1, R2, OrientationAngle = E.params

    Radians = np.linspace(0, 2 * np.pi, 100)
    Ellipse = np.array([R2 * np.cos(Radians), R1 * np.sin(Radians)])
    R = np.array([[np.cos(OrientationAngle), -np.sin(OrientationAngle)],
                  [np.sin(OrientationAngle), np.cos(OrientationAngle)]])
    Ellipse_R = np.dot(R, Ellipse)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(ImageArray, cmap='gray')
    Axes.imshow(Region, cmap=ColorMap, vmin=0.25, vmax=0.75)
    Axes.plot([], color=(1,0,0,0.5), linestyle='none', marker='s', label='Region')
    Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='centroid')
    Axes.plot(X0 + Ellipse_R[0, :], Y0 - Ellipse_R[1, :], color=(0, 1, 0), label='Fitted ellipse')
    plt.title('Region ' + str(RegionLabel))
    plt.axis('off')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    plt.show()
    plt.close(Figure)

    return

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

    print('\nProcess executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))
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
def GrayStretch(Image, StretchPercentile):

    """
    Crop and keep a ratio of grayscale
    :param Image: Initial grayscale image
    :param StretchPercentile: percentile to stretch gray values
    :return: GC: Gray scale image with stretched gray values
    """

    m, M = FindingMm(Image, StretchPercentile)
    GC = (Image - m) / (M - m)
    GC[GC > 1] = 1
    GC[GC < 0] = 0

    return GC
def FindingMm(Image, Per):

    PCNN_Tools = PCNN()
    PCNN_Tools.Set_Image(Image)
    H, Bins = PCNN_Tools.Histogram()
    All = sum(H)
    H_Per = H / All
    mth_ceil = BoundFinding(H_Per, Per)
    H_PerM = H_Per[::-1]
    Mth_floor = BoundFinding(H_PerM, Per)
    Mth_floor = len(H) + 1 - Mth_floor + 1
    Difference = np.zeros((256, 256)) + np.inf
    for m in range(mth_ceil, 0, -1):
        for M in range(Mth_floor, len(H)):
            if H[m] > 0 and H[M] > 0:
                if np.sum(H[m:M + 1]) / All >= Per:
                    Difference[m, M] = M - m

    minD = Difference.min()
    m, M = np.where(Difference == minD)
    minI = m[0]
    MaxI = M[0]

    return [minI, MaxI]
def BoundFinding(H_Per, Per):
    Cum_Per = np.cumsum(H_Per)
    N = 1
    Residual_Per = 1 - Per
    while Cum_Per[N - 1] < Residual_Per:
        N += 1

    return N

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
        Define PCNN attributes
        :param Image: Grayscale intensity image in numpy array
        """
        self.Image = Image
        return


    # Define PCNN functions
    def Histogram(self,NBins=256):

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

        return H, Bins

    def Segmentation(self,Beta=2,Delta=1,Vt=400):

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

        # Initialize parameters
        S = NormalizeValues(self.Image)
        Rows, Columns = S.shape
        Y = np.zeros((Rows, Columns))
        T = np.zeros((Rows, Columns))
        W = np.array([[0.5, 1, 0.5],
                      [1, 0, 1],
                      [0.5, 1, 0.5]])
        F = S
        Theta = 255 * np.ones((Rows, Columns))

        FiredNumber = 0
        N = 0

        # Perform segmentation
        while FiredNumber < S.size:

            N += 1
            L = correlate(Y, W, output='float', mode='reflect')
            Theta = Theta - Delta + Vt * Y
            Fire = 1

            while Fire == 1:

                Q = Y
                U = F * (1 + Beta * L)
                Y = (U > Theta) * 1
                if np.array_equal(Q, Y):
                    Fire = 0
                else:
                    L = correlate(Y, W, output='float', mode='reflect')

            T = T + N * Y
            FiredNumber = FiredNumber + sum(sum(Y))

        Output = 256 - T

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


# Set path
CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Scripts/PCNN/'

# Open image to segment
Image = sitk.ReadImage(ImageDirectory + 'PCNN_Test2.png')
Array = sitk.GetArrayFromImage(Image)
PlotArray(Array[:,:,2], 'RGB Image')
GS = RGB2Gray(Array)
PlotArray(GS, 'Grayscale Image')

# Use PCNN tool
PCNN_Tools = PCNN()
PCNN_Tools.Set_Image(GS)
Y = PCNN_Tools.Enhancement()
Y_Stretched = GrayStretch(Y,0.90)
PlotArray(Y_Stretched, 'Enhanced Image')


H, Bins = PCNN_Tools.Histogram(NBins)
Indices = np.where(H == 0)[0]
Suite = np.concatenate([np.diff(Indices),np.array([0])])
Low, High = np.array([0]), np.array([])
i = 0
while i < len(H):
    n = 0
    while H[i+n] == 0:
        n += 1
    if n > 2:
        High = np.concatenate([High,np.array([i])]).astype('int')
        Low = np.concatenate([Low,np.array([i+n-1])]).astype('int')
        i += n
    else:
        i += 1
High = np.concatenate([High,np.array([255])])

Seg = np.zeros(Y_Stretched.shape)
for n in range(len(High)):
    F1 = Y_Stretched > Bins[Low[n]]
    F2 = Y_Stretched < Bins[High[n]]
    Seg[F1*F2] = n
Seg[Y_Stretched == Bins[High[n]]] = n + 1

a = Seg.copy()
b = np.unique(Seg)
Layer = b[-1]
a[a != Layer] = 0
a[a == Layer] = 1
PlotArray(a,'Seg')



PCNN_Tools.Set_Image(Y_Stretched)
Y_Seg = PCNN_Tools.Segmentation()
PlotArray(Y_Seg,'Segmented Image')



Threshold = 1
Y_Thresh = np.zeros(Y_Stretched.shape)
Y_Thresh[Y_Stretched == Threshold] = 1
PlotArray(Y_Thresh,'Thresholded Image')

Disk = morphology.disk(4)
Segments = morphology.binary_erosion(Y_Thresh,Disk)
Disk = morphology.disk(30)
Segments = morphology.binary_dilation(Segments,Disk)
PlotArray(Segments,'Segmented Image')

Figure, Axes = plt.subplots(1,1,figsize=(5.5,4.5))
Axes.imshow(Y_Stretched,cmap='gray')
Axes.imshow(Segments,cmap='jet',alpha=0.5)
plt.axis('off')
plt.title('Overlay')
plt.tight_layout()
plt.show()
plt.close(Figure)







# Open manually segmented image
SegmentedImage = sitk.ReadImage(ImageDirectory + 'PCNN_Test2_Seg.png')
SegmentedArray = sitk.GetArrayFromImage(SegmentedImage)
# SegmentedArray = SegmentedArray[3839:3839+1182,4724:4724+1182]

CementLines = SegmentedArray[:,:,0].copy()
Harvesian = SegmentedArray[:, :, 1].copy()
Osteocytes = SegmentedArray[:,:,2].copy()
Threshold = 200
CementLines[CementLines < Threshold] = 0
CementLines[CementLines >= Threshold] = 1
Harvesian[Harvesian < Threshold] = 0
Harvesian[Harvesian >= Threshold] = 1
Osteocytes[Osteocytes < Threshold] = 0
Osteocytes[Osteocytes >= Threshold] = 1

F0 = SegmentedArray[:,:,0] > 180
F1 = SegmentedArray[:,:,1] > 180
F2 = SegmentedArray[:,:,2] > 180
CementLines[F1] = 0
CementLines[F2] = 0
Harvesian[F0] = 0
Harvesian[F2] = 0
Osteocytes[F0] = 0
Osteocytes[F1] = 0

Segments = CementLines + Harvesian + Osteocytes
Disk = morphology.disk(2)
Segments = morphology.binary_erosion(Harvesian,Disk)
Disk = morphology.disk(5)
Segments = morphology.binary_dilation(CementLines,Disk)
PlotArray(Segments, 'Manual Segmentation')

# Open image to segment
Image = sitk.ReadImage(ImageDirectory + 'PCNN_Test2.png')
Array = sitk.GetArrayFromImage(Image)
PlotArray(Array, 'RGB Image')

# Decompose RGB image and equalize histogram
R, G, B = Array[:,:,0], Array[:,:,1], Array[:,:,2]

Figure, Axes = plt.subplots(1, 3)
Axes[0].imshow(R, cmap='gray')
Axes[0].set_title('R channel')
Axes[0].axis('off')
Axes[1].imshow(G, cmap='gray')
Axes[1].set_title('G channel')
Axes[1].axis('off')
Axes[2].imshow(B, cmap='gray')
Axes[2].set_title('B channel')
Axes[2].axis('off')
plt.tight_layout()
plt.show()
plt.close(Figure)

# Match histograms for better phase difference and clarity and rescale
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
