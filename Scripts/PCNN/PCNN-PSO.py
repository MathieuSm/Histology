"""
This script aims to test different Pulse-Coupled Neural Network (PCNN) algorithm
Based on :
Xinzheng, X., Shifei, D., Zhongzhi, S., Zuopeng, Z., & Hong, Z. (2011).
Particle swarm optimization for automatic parameters determination of pulse coupled neural network.
Journal of Computer, 6 (8), 1546–1553.
https://doi.org/10.4304/jcp.6.8.1546-1553
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

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

    return Gray
def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))
def PCNN_Segmentation(Image,ParametersDictionary,IterationNumber=5):

    # Initialization
    AlphaF = ParametersDictionary['AlphaF']
    AlphaL = ParametersDictionary['AlphaL']
    AlphaT = ParametersDictionary['AlphaT']

    VF = ParametersDictionary['VF']
    VL = ParametersDictionary['VL']
    VT = ParametersDictionary['VT']

    Beta = ParametersDictionary['Beta']

    # Input parameters
    S = Image / Image.max() * 255

    Rows, Columns = S.shape
    Y = np.zeros((Rows, Columns))
    W = GaussianKernel(1, 1)

    ## Feeding input
    F = S
    WF = W

    ## Linking input
    L = Y
    WL = W

    # Dynamic threshold
    Theta = 255 * np.ones((Rows, Columns))

    N = 0
    for Iteration in range(IterationNumber):

        N += 1
        F = S + F * np.exp(-AlphaF) + VF * correlate(Y, WF, output='float', mode='reflect')
        L = L * np.exp(-AlphaL) + VL * correlate(Y, WL, output='float', mode='reflect')
        U = F * (1 + Beta*L)
        Y = (U > Theta) * 1
        Theta = Theta * np.exp(-AlphaT) + VT * Y

    return Y
def PlotResults(Image, Threshold):
    Segmented_Image = Image / Image.max()
    Segmented_Image[Segmented_Image < Threshold] = 0
    Segmented_Image[Segmented_Image >= Threshold] = 1

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(Segmented_Image, cmap='gray')
    plt.axis('off')
    plt.title('Segmented Image')
    plt.show()
    plt.close(Figure)

    return


CurrentDirectory = os.getcwd()
DataDirectory = os.path.join(CurrentDirectory,'Scripts/PCNN/')

# Read input
Input_Image = sitk.ReadImage(DataDirectory + 'Lena.jpg')
Input_Array = sitk.GetArrayFromImage(Input_Image)
Input = RGB2Gray(Input_Array)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Input,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.title('Input')
plt.show()
plt.close(Figure)



# Particle Swarm Optimization (PSO) algorithm
Ps = 20         # Population size
t = 0           # Iteration number
Max_times = 10  # Max iteration number
Omega = 0.9 - 0.5 * t/Max_times     # Inertia factor
Average_FV = 0.99   # Second PSO termination condition




# PSO step 1 - Initialization
AlphaF_Range = [-10,10]
AlphaL_Range = [-10,10]
AlphaT_Range = [-10,10]

VF_Range = [-10,10]
VL_Range = [-10,10]
VT_Range = [-255,255]

Beta_Range = [-10,10]

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
InitialEntropies = np.zeros((Ps,1))
for ParticleNumber in range(Ps):
    ParametersDictionary = {}
    for ParameterNumber in range(Dimensions):
        ParameterName = ParameterList[ParameterNumber]
        ParameterValue = X[ParticleNumber,ParameterNumber]
        ParametersDictionary[ParameterName] = ParameterValue

    Y = PCNN_Segmentation(Input,ParametersDictionary,IterationNumber=5)

    # Compute entropy and store it
    N0 = np.count_nonzero(Y == 0)
    N1 = np.count_nonzero(Y == 1)
    P1 = N0 / Y.size
    P2 = N1 / Y.size
    if P1 == 0:
        H = - P2 * np.log2(P2)
    elif P2 == 0:
        H = -P1 * np.log2(P1)
    else:
        H = -P1 * np.log2(P1) - P2 * np.log2(P2)

    InitialEntropies[ParticleNumber,0] = H


# Set initial best values
G_Best_Value = InitialEntropies.max()
G_Best_Index = np.where(InitialEntropies == G_Best_Value)[0][0]
G_Best = X[G_Best_Index]

P_Best_Values = InitialEntropies.copy()
P_Best = X.copy()


## Start loop
Average_FVs = np.zeros(3)
NIteration = 0
while NIteration < 100 and Average_FVs.mean() <= Average_FV:

    ## PSO step 3 - Update positions and velocities
    R1, R2 = np.random.uniform(0, 1, 2)
    V = Omega * V + C1 * R1 * (P_Best - X) + C2 * R2 * (G_Best - X)
    X = X + V
    # If new position exceed limits, set to limit
    X[X < Ranges[:,0]] = np.tile(Ranges[:,0],Ps).reshape((Ps,Dimensions))[X < Ranges[:,0]]
    X[X > Ranges[:,1]] = np.tile(Ranges[:,1],Ps).reshape((Ps,Dimensions))[X > Ranges[:,1]]



    ## PSO step 4 - Evaluation of the updated population
    NewEntropies = np.zeros((Ps,1))
    for ParticleNumber in range(Ps):
        ParametersDictionary = {}
        for ParameterNumber in range(Dimensions):
            ParameterName = ParameterList[ParameterNumber]
            ParameterValue = X[ParticleNumber,ParameterNumber]
            ParametersDictionary[ParameterName] = ParameterValue

        Y = PCNN_Segmentation(Input,ParametersDictionary,IterationNumber=5)

        # Compute entropy and store it
        N0 = np.count_nonzero(Y == 0)
        N1 = np.count_nonzero(Y == 1)
        P1 = N0 / Y.size
        P2 = N1 / Y.size
        if P1 == 0:
            H = - P2 * np.log2(P2)
        elif P2 == 0:
            H = -P1 * np.log2(P1)
        else:
            H = -P1 * np.log2(P1) - P2 * np.log2(P2)

        NewEntropies[ParticleNumber,0] = H

    # Update best values if better than previous
    if NewEntropies.max() > G_Best_Value:
        G_Best_Value = NewEntropies.max()
        G_Best_Index = np.where(NewEntropies == G_Best_Value)[0][0]
        G_Best = X[G_Best_Index]

    ImprovedValues = NewEntropies > P_Best_Values
    P_Best_Values[ImprovedValues] = NewEntropies[ImprovedValues]
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
Y = PCNN_Segmentation(Input,ParametersDictionary,IterationNumber=5)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Y,cmap='gray')
plt.axis('off')
plt.title('Segmented Image')
plt.show()
plt.close(Figure)