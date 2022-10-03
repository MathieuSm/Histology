"""
This script aims to test different Pulse-Coupled Neural Network (PCNN) algorithm
Based on :
Zhan, K., Shi, J., Wang, H. et al.
Computational Mechanisms of Pulse-Coupled Neural Networks: A Comprehensive Review.
Arch Computat Methods Eng 24, 573–588 (2017).
https://doi.org/10.1007/s11831-016-9182-3
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate, convolve
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from skimage.filters import median, laplace, gaussian

def PlotArray(Array,Title):

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(Array, cmap='gray')
    plt.axis('off')
    plt.title(Title)
    plt.tight_layout()
    plt.show()
    plt.close(Figure)

    return
def NormalizeArray(Array):

    N_Array = (Array - Array.min()) / (Array.max()-Array.min())

    return N_Array
def RGB2Gray(RGBImage):
    """
    This function convert color image to gray scale image
    based on matplotlib linear approximation
    """

    R, G, B = RGBImage[:,:,0], RGBImage[:,:,1], RGBImage[:,:,2]
    Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    return np.round(Gray).astype('uint8')
def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))
def CropCenter(Image,CropX,CropY):
    Y, X = Image.shape
    StartX = X//2-(CropX//2)
    StartY = Y//2-(CropY//2)
    return Image[StartY:StartY+CropY,StartX:StartX+CropX]

CurrentDirectory = os.getcwd()
DataDirectory = os.path.join(CurrentDirectory,'Scripts/PCNN/')

Image = sitk.ReadImage(DataDirectory + 'Lena.jpg')
Array = sitk.GetArrayFromImage(Image)
Input = RGB2Gray(Array)
PlotArray(Input,'Input')

S = (Input-Input.min()) / (Input.max()-Input.min()) * 255
S = np.round(S).astype('uint8')

# Algorithm 4, image histogram
Theta = 255
Delta = 1
Vt = 256
Y = np.zeros(S.shape)
H = np.zeros(256)

# Perform analysis
for N in range(256):
    U = S
    Theta = Theta - Delta + Vt * Y
    Y = np.where((U - Theta) > 0, 1, 0)
    H[255 - N] = Y.sum()

npHist, npEdges = np.histogram(S, bins=255)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(npEdges[:-1]+0.5,npHist,color=(0,0,1), label='Numpy')
Axes.plot(np.arange(0,256), H,linestyle='none',marker='x',color=(1,0,0), label='PCNN')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.close(Figure)


# Algorithm 5, image segmentation
Input = np.ones((256, 256)) * 230
Input[64:192, 64:192] = 205
Input[:, 128:] = np.round(Input[:, 128:] * 0.5)
PlotArray(Input,'Input')


S = Input
Rows, Columns = S.shape
Y = np.zeros((Rows, Columns))
T = np.zeros((Rows, Columns))
W = GaussianKernel(Length=7, Sigma=1.)
F = S
Beta = 2
Theta = 255 * np.ones((Rows, Columns))
Delta = 1
Vt = 400
FiredNumber = 0
N = 0

while FiredNumber < S.size:

    N += 1
    L = correlate(Y, W, output='float', mode='reflect')
    Theta = Theta - Delta + Vt * Y
    Fire = 1

    while Fire == 1:

        Q = Y
        U = F * (1 + Beta*L)
        Y = (U > Theta) * 1
        if np.array_equal(Q,Y):
            Fire = 0
        else:
            L = correlate(Y, W, output='float', mode='reflect')

    T = T + N*Y
    FiredNumber = FiredNumber + sum(sum(Y))

Output = 256 - T
PlotArray(Output,'Output')



# Algorithm x, feature extraction
def PCNN_Histogram(S):

    Theta = 255
    Delta = 1
    Vt = 256
    Y = np.zeros(S.shape)
    H = np.zeros(256)

    # Perform analysis
    for N in range(256):
        U = S
        Theta = Theta - Delta + Vt * Y
        Y = np.where((U - Theta) > 0, 1, 0)
        H[255 - N] = Y.sum()

    return H
def SpikingCorticalModel(S):

    """
    SCM used for invariant texture retrieval and image processing
    """

    W = np.array([[0.1091, 0.1409, 0.1091],
                  [0.1409, 0,      0.1409],
                  [0.1091, 0.1409, 0.1091]])

    Y = np.zeros(S.shape)
    U = Y
    E = Y + 1

    k = 40
    TS = np.zeros(k)
    for t in range(k):
        U = 0.2 * U + S * correlate(Y, W, output='float', mode='reflect') + S
        E = 0.9 * E + 20 * Y
        X = 1 / (1 + np.exp(E - U))
        Y = (X > 0.5) * 1
        TS[t] = sum(sum(Y))

    return TS

Image = sitk.ReadImage(DataDirectory + 'D12.png')
Array = sitk.GetArrayFromImage(Image)
Input = RGB2Gray(Array)
PlotArray(Input,'Input')


NScaling, NRotation = 2, 3
Figure, Axes = plt.subplots(NScaling*NRotation, 3, figsize=(5.5*NScaling, 4.5*NRotation))
CropY, CropX = np.array(np.array(Input.shape)/2).astype('int')
Line = 0
for Scaling in range(1,NScaling+1):

    ScaledInput = rescale(Input,1/(0.2*Scaling+0.6))

    for Rotation in range(1,NRotation+1):
        RotatedInput = rotate(ScaledInput,Rotation*30-30,resize=True)
        RotatedInput = CropCenter(RotatedInput, CropX, CropY)
        RotatedInput_Normalized = NormalizeArray(RotatedInput)
        H = PCNN_Histogram(RotatedInput_Normalized)
        TS = SpikingCorticalModel(RotatedInput)

        # Plot
        Axes[Line,0].imshow(RotatedInput,cmap='gray')
        Axes[Line,0].set_xticks([])
        Axes[Line,0].set_yticks([])
        Axes[Line,1].plot(H, color=(1, 0, 0))
        Axes[Line,1].set_xticks([])
        Axes[Line,1].set_yticks([])
        Axes[Line,2].plot(TS, color=(1, 0, 0))
        Axes[Line,2].set_xticks([])
        Axes[Line,2].set_yticks([])
        Line += 1
plt.show()
plt.close(Figure)


# Algorithm 6 - Image restoration
Image = sitk.ReadImage(DataDirectory + 'Lena.jpg')
Array = sitk.GetArrayFromImage(Image)
Input = RGB2Gray(Array)
PlotArray(Input,'Input')


Rows, Columns = Input.shape
Size = Input.size

J = random_noise(Input, mode='gaussian', seed=None, clip=True, mean=0, var=0.01)
J = NormalizeArray(J)
PlotArray(J,'Noised')


Input_Median = median(J,mode='reflect')
Input_Median = NormalizeArray(Input_Median)
PlotArray(Input_Median,'Median')
Input_Mean = correlate(J,np.ones((3,3))/9,mode='reflect')
Input_Mean = NormalizeArray(Input_Mean)
PlotArray(Input_Mean,'Mean')

U = np.zeros(Input.shape)
E = U + 1
Y = np.zeros(Input.shape)
B = np.zeros(Input.shape)
T = np.zeros(Input.shape)

f = 0.3
g = 0.99
gamma = 1
N = 0

while sum(sum(B)) < Size:

    N += 1

    for i in range(Rows):
        for j in range(Columns):

            U[i, j] = J[i, j] + f * U[i, j]
            if E[i, j] > 1E2:
                Q = 0
            else:
                Q = 1 / (1 + np.exp(-gamma * (U[i, j] - E[i, j])))

            if Q > 0.5 or E[i, j] < 0.08:
                Y[i, j] = 1
                B[i, j] = 1
                T[i, j] = N
                E[i, j] = 100000
            else:
                Y[i, j] = 0

    E[B != 1] = g * E[B != 1]

S = np.pad(J,1,'symmetric')
T = np.pad(T,1,'symmetric')
I_SCM = np.zeros(Input.shape)
Delta = 0.02

for i in range(1,Rows+1):
    for j in range(1,Columns+1):

        K = T[i-1:i+2,j-1:j+2].flatten()

        if len(np.unique(K)) == 1:
            I_SCM[i-1, j-1] = 1/9 * sum(sum(S[i-1:i+2, j-1:j+2]))
        else:
            Sorted = np.sort(K)

            if Sorted[4] == K[4]:
                I_SCM[i-1, j-1] = S[i, j]
            elif Sorted[0] == K[4]:
                I_SCM[i-1, j-1] = S[i, j] - Delta
            elif Sorted[-1] == K[4]:
                I_SCM[i-1, j-1] = S[i, j] + Delta
            else:
                I_SCM[i-1, j-1] = np.median(S[i-1:i+2, j-1:j+2])
I_SCM = NormalizeArray(I_SCM)
PlotArray(I_SCM,'SCM')


# Algorithm y - Image enhancement (Feature Linking Model FLM)
def GrayStretch(I,Per):

    m, M = FindingMm(I,Per)
    GS = ((I-m) / (M-m) * 255).astype('uint8')

    return GS
def FindingMm(I,Per):
    h = PCNN_Histogram(I)
    All = sum(h)
    ph = h / All
    mth_ceiling = BoundFinding(ph,Per)
    Mph = ph[::-1]
    Mth_floor = BoundFinding(Mph,Per)
    Mth_floor = 256 - Mth_floor + 1
    Difference = np.zeros((256,256)) + np.inf
    for m in range(mth_ceiling,0,-1):
        for M in range(Mth_floor,256):
            if h[m] > 0 and h[M] > 0:
                if np.sum(h[m:M+1]) / All >= Per:
                    Difference[m,M] = M - m

    minD = Difference.min()
    m, M = np.where(Difference == minD)
    minI = m[0]
    MaxI = M[0]

    return [minI,MaxI]
def BoundFinding(ph,Per):
    cumP = np.cumsum(ph)
    n = 1
    residualP = 1 - Per
    while cumP[n-1] < residualP:
        n += 1

    m_ceiling = n

    return m_ceiling
def PCNN_Histogram(S):

    Theta = 255
    Delta = 1
    Vt = 256
    Y = np.zeros(S.shape)
    H = np.zeros(256)

    # Perform analysis
    for N in range(256):
        U = S
        Theta = Theta - Delta + Vt * Y
        Y = np.where((U - Theta) > 0, 1, 0)
        H[255 - N] = Y.sum()

    return H


Image = sitk.ReadImage(DataDirectory + 'Lena.jpg')
Array = sitk.GetArrayFromImage(Image)
Input = RGB2Gray(Array)
PlotArray(Input,'Input')

S = (Input - Input.min()) / (Input.max()-Input.min()) + 1/255
W = np.array([[0.7, 1, 0.7],[1, 0, 1],[0.7, 1, 0.7]])
Y = np.zeros(S.shape)
U = np.zeros(S.shape)
T = np.zeros(S.shape)
SumY = 0
N = 0

Laplacian = np.array([[1/6, 2/3, 1/6],[2/3, -10/3, 2/3],[1/6, 2/3, 1/6]])
Theta = 1 + correlate(S,Laplacian,mode='reflect')
# Theta = 1 + laplace(S,ksize=3)
f = 0.75 * np.exp(-S**2 / 0.16) + 0.05
G = GaussianKernel(7,1)
f = correlate(f, G, mode='reflect')
# f = gaussian(f,sigma=1,mode='reflect')

h = 2E10
d = 2
g = 0.9811
Alpha = 0.01
Beta = 0.03

while SumY < S.size:

    N += 1

    K = correlate(Y,W,mode='reflect')
    Wave = Alpha * K + Beta * S * (K - d)
    U = f * U + S + Wave
    Theta = g * Theta + h * Y
    Y = (U > Theta) * 1
    T += N * Y
    SumY += sum(sum(Y))
    # print(SumY)

T_inv = T.max() + 1 - T
Time = (NormalizeArray(T_inv) * 255).astype('uint8')
Stretched = GrayStretch(Time,0.99)
PlotArray(Stretched,'Output')



Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.hist(Input.flatten()*255, bins=np.arange(0,256), density=True, color=(0,0,1), label='Original')
Axes.hist(Stretched.flatten(), bins=np.arange(0,256), density=True, color=(1,0,0), label='Enhanced')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.close(Figure)

H = PCNN_Histogram(Stretched)
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

Seg = np.zeros(Stretched.shape)
for n in range(len(High)):
    F1 = Stretched > Low[n]
    F2 = Stretched < High[n]
    Seg[F1*F2] = n

a = Seg.copy()
b = np.unique(Seg)
a[a != b[6]] = 100
PlotArray(a,'Seg')
