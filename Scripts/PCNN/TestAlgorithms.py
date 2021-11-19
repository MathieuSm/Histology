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
from scipy.ndimage import correlate
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from skimage.filters import median


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

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Input')
plt.show()

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

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Input')
plt.show()

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

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Output,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Output')
plt.show()


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

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Input')
plt.show()

NScaling, NRotation = 2, 3
Figure, Axes = plt.subplots(NScaling*NRotation, 3, figsize=(5.5*NScaling, 4.5*NRotation))
CropY, CropX = np.array(np.array(Input.shape)/2).astype('int')
Line = 0
for Scaling in range(1,NScaling+1):

    ScaledInput = rescale(Input,1/(0.2*Scaling+0.6))

    for Rotation in range(1,NRotation+1):
        RotatedInput = rotate(ScaledInput,Rotation*30-30,resize=True)
        RotatedInput = CropCenter(RotatedInput, CropX, CropY)
        RotatedInput_Normalized = (RotatedInput-RotatedInput.min()) / (RotatedInput.max()-RotatedInput.min()) * 255
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

Rows, Columns = Input.shape
Size = Input.size

J = random_noise(Input, mode='gaussian', seed=None, clip=True, mean=0, var=0.001)
Input_Median = median(J,mode='reflect')
Input_mean = correlate(J,np.ones((3,3))/9,mode='reflect')

U = np.zeros(Input.shape)
E = U + 1
Y = U
B = U
T = U

f = 0.03
g = 0.99
gamma = 1
N = 0

while sum(sum(B)) < Size:

    N += 1

    for i in range(Rows):
        for j in range(Columns):

            U[i, j] = J[i, j] + f * U[i, j]
            Q = 1 / (1 + np.exp(-gamma * (U[i, j] - E[i, j])))

            if Q > 0.5 | E[i, j] < 0.08:
                Y[i, j] = 1
                B[i, j] = 1
                T[i, j] = N
                E[i, j] = 100000
            else:
                Y[i, j] = 0

    E[B != 1] = g * E[B != 1]

S = np.pad(J,1,'symmetric')

S = padarray(J,[1 1],'symmetric');
Time_matrix = padarray(T,[1 1],'symmetric');
I_SCM = zeros(r,c);
Delta = 0.02;
for ii = 2:r+1
    for jj = 2:c+1
        K = Time_matrix(ii-1:ii+1,jj-1:jj+1);
        if isequal(K(1),K(2),K(3),K(4),K(5),K(6),K(7),K(8),K(9))
            I_SCM(ii-1,jj-1) = 1/9*(sum(sum(S(ii-1:ii+1,jj-1:jj+1))));
        else
            tmp = sort(K);
            if tmp(5) == K(5)
                I_SCM(ii-1,jj-1) = S(ii,jj);
            elseif tmp(1) == K(5)
                I_SCM(ii-1,jj-1) = S(ii,jj) - Delta;
            elseif tmp(9) == K(5)
                I_SCM(ii-1,jj-1) = S(ii,jj) + Delta;
            else
                kk = medfilt2(S(ii-1:ii+1,jj-1:jj+1));
                I_SCM(ii-1,jj-1) = kk(5);
            end
        end
    end
end
%_______________________________________
PSNR_SCM = psnr_mse_maxerr(255*I,255*I_SCM)



# Algorithm y - Image enhancement (Feature Linking Model FLM)
I = imread('tire.tif');

funInverse = @(x) max(max(x)) + 1 - x;
funNormalize = @(x) ( x-min(min(x)))/( max(max(x))-min(min(x)) + eps);
[r,c] = size(I);  rc = r*c;
S = funNormalize(double(I)) + 1/255;
W=[0.7 1 0.7; 1  0  1 ; 0.7 1 0.7;];
Y=zeros(r,c);   U = Y;  Time = Y; sumY = 0;  n = 0;
%_____________________Theta_0______________
Lap = fspecial('laplacian',0.2);
Theta = 1 + imfilter(S,Lap,'symmetric');
%_____________________f____________________
f = 0.75 * exp(-(S).^2 / 0.16) + 0.05;
G = fspecial('gaussian',7,1);
f = imfilter(f,G,'symmetric');
%___________________Parameters____________
h = 2e10; d = 2; g = 0.9811; alpha = 0.01; beta = 0.03;
while sumY < rc
    n = n + 1;
    K = conv2(Y,W,'same');
    Wave = alpha * K + beta .* S .* (K - d);
    U = f.*U + S + Wave;
    Theta = g*Theta + h.*Y;
    Y = double(U > Theta);
    Time = Time + n .* Y;
    sumY = sumY + sum(Y(:));
end
Rep = funInverse(Time);
Rep = funNormalize(Rep);
Rep = uint8(Rep * 255);
Rep1gs = GrayStretch(Rep,0.98);

imshow([I, J])


