"""
This script aims to test Pulse-Coupled Neural Network (PCNN) algorithm
Optimized using particle swarm optimization and an adaptive threshold
Based on :
Hage, I., Hamade, R. (2015)
Automatic Detection of Cortical Bones Haversian Osteonal Boundaries.
AIMS Medical Science, 2(4), 328–346.
https://doi.org/10.3934/medsci.2015.4.328

Xinzheng, X., Shifei, D., Zhongzhi, S., Zuopeng, Z., & Hong, Z. (2011).
Particle swarm optimization for automatic parameters determination of pulse coupled neural network.
Journal of Computer, 6 (8), 1546–1553.
https://doi.org/10.4304/jcp.6.8.1546-1553

Pai, Y. T., Chang, Y. F., Ruan, S. J. (2010)
Adaptive thresholding algorithm: Efficient computation technique
based on intelligent block detection for degraded document images
Pattern Recognition, 43 (9), 3177–3187
https://doi.org/10.1016/j.patcog.2010.03.014
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage import measure
import pandas as pd

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
    Normalized_Gray = Gray / Gray.max()

    return Normalized_Gray
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
def GaussianKernel(Length=5, Sigma=1.):
    """
    Creates gaussian kernel with side length `Length` and a sigma of `Sigma`
    """
    Array = np.linspace(-(Length - 1) / 2., (Length - 1) / 2., Length)
    Gauss = np.exp(-0.5 * np.square(Array) / np.square(Sigma))
    Kernel = np.outer(Gauss, Gauss)
    return Kernel / sum(sum(Kernel))
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
def PlotImage(Image):
    Spacing = Image.GetSpacing()

    Image_Array = sitk.GetArrayFromImage(Image)

    X_Positions = np.arange(Image_Array.shape[1]) * Spacing[1]
    Y_Positions = np.arange(Image_Array.shape[0]) * Spacing[0]

    N_XTicks = round(len(X_Positions) / 5)
    N_YTicks = round(len(Y_Positions) / 5)
    TicksSize = min(N_XTicks,N_YTicks)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(Image_Array)
    Axes.set_xlim([0, Image_Array.shape[1]])
    Axes.set_ylim([0, Image_Array.shape[0]])
    Axes.set_xlabel('X ($\mu$m)')
    Axes.set_ylabel('Y ($\mu$m)')
    plt.xticks(np.arange(0, Image_Array.shape[1])[::TicksSize], np.round(X_Positions[::TicksSize]).astype('int'))
    plt.yticks(np.arange(0, Image_Array.shape[0])[::TicksSize], np.round(Y_Positions[::TicksSize]).astype('int'))
    plt.show()

    return Image_Array
def PlotArray(Array,Spacing):

    X_Positions = np.arange(Array.shape[1]) * Spacing[1]
    Y_Positions = np.arange(Array.shape[0]) * Spacing[0]

    N_XTicks = round(len(X_Positions) / 5)
    N_YTicks = round(len(Y_Positions) / 5)
    TicksSize = min(N_XTicks, N_YTicks)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.imshow(Array,cmap='bone')
    Axes.set_xlim([0, Array.shape[1]])
    Axes.set_ylim([0, Array.shape[0]])
    Axes.set_xlabel('X ($\mu$m)')
    Axes.set_ylabel('Y ($\mu$m)')
    plt.xticks(np.arange(0, Array.shape[1])[::TicksSize], np.round(X_Positions[::TicksSize]).astype('int'))
    plt.yticks(np.arange(0, Array.shape[0])[::TicksSize], np.round(Y_Positions[::TicksSize]).astype('int'))
    plt.show()

    return


CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Tests/Osteons/'
Images = [File for File in os.listdir(ImageDirectory) if File.endswith('.jpg')]
Images.sort()

Image = sitk.ReadImage(ImageDirectory+Images[1])
Image_Array = PlotImage(Image)

# Crop image (size in um) at random position
X_Crop_Size = 100
Y_Crop_Size = 100
Crop_X = round(X_Crop_Size / Image.GetSpacing()[1] + 0.5)
Crop_Y = round(Y_Crop_Size / Image.GetSpacing()[0] + 0.5)
Random_X = round(np.random.uniform(0,Image.GetSize()[1]-Crop_X-1))
Random_Y = round(np.random.uniform(0,Image.GetSize()[0]-Crop_Y-1))
Random_X = round(325 / Image.GetSpacing()[1])
Random_Y = round(400 / Image.GetSpacing()[0])
Cropping = (Image.GetSize()[0]-Random_Y-Crop_Y,Image.GetSize()[1]-Random_X-Crop_X)
SubImage = sitk.Crop(Image,(Random_Y,Random_X),Cropping)
SubImage_Array = PlotImage(SubImage)

# Read input
RGB_Array = sitk.GetArrayFromImage(SubImage)
GrayScale_Array = RGB2Gray(RGB_Array)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(GrayScale_Array,cmap='gray',vmin=0,vmax=1)
Axes.set_ylim([0,GrayScale_Array.shape[1]])
plt.axis("off")
plt.title('Input')
# plt.subplots_adjust(0,0,1,1)
plt.show()
plt.close(Figure)

# # Save grayscale image
# GrayScale_Array = RGB2Gray(Image_Array)
I = GrayScale_Array * 255
# from skimage import io
io.imsave('TrainingImage.png', I.astype(np.uint8))


# Compute best threshold
OtsuFilter = sitk.OtsuThresholdImageFilter()
OtsuFilter.SetInsideValue(1)
OtsuFilter.SetOutsideValue(0)
OtsuFilter.Execute(sitk.GetImageFromArray(GrayScale_Array))
Best_Threshold = OtsuFilter.GetThreshold()

Segmented_Array = GrayScale_Array.copy()
Segmented_Array[Segmented_Array < Best_Threshold] = 0
Segmented_Array[Segmented_Array >= Best_Threshold] = 1

Best_Vb = BetweenClassVariance(GrayScale_Array, Segmented_Array)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Segmented_Array,cmap='gray',vmin=0,vmax=1)
Axes.set_ylim([0,Segmented_Array.shape[1]])
plt.axis('off')
plt.title('Otsu segmentation')
plt.show()
plt.close(Figure)



# Particle Swarm Optimization (PSO) algorithm
Ps = 20         # Population size
t = 0           # Iteration number
Max_times = 10  # Max iteration number
Omega = 0.9 - 0.5 * t/Max_times     # Inertia factor
Average_FV = 0.99   # Second PSO termination condition



# PSO step 1 - Initialization
AlphaF_Range = [-1,1]
AlphaL_Range = [-1,1]
AlphaT_Range = [-1,1]

VF_Range = [-1,1]
VL_Range = [-1,1]
VT_Range = [-1,1]

Beta_Range = [-1,1]

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

    Y = PCNN_Segmentation(GrayScale_Array,ParametersDictionary)

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

        Y = PCNN_Segmentation(GrayScale_Array,ParametersDictionary)

        # Compute entropy and store it
        N0 = np.count_nonzero(Y == 0)
        N1 = np.count_nonzero(Y == 1)
        P0 = N0 / Y.size
        P1 = N1 / Y.size
        if P0 == 0:
            H = - P1 * np.log2(P1)
        elif P1 == 0:
            H = -P0 * np.log2(P0)
        else:
            H = -P1 * np.log2(P1) - P0 * np.log2(P0)

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
Y = PCNN_Segmentation(GrayScale_Array,ParametersDictionary)

Vb = BetweenClassVariance(GrayScale_Array,Y)

print('Obtained variance / Otsu variance: %.3f'%(Vb/Best_Vb))

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Y,cmap='gray')
Axes.set_ylim([0,Y.shape[1]])
plt.axis('off')
plt.title('Segmented Image')
plt.show()
plt.close(Figure)

PCNN_MI = Mutual_Information(GrayScale_Array, Y)
OTSU_MI = Mutual_Information(GrayScale_Array, Segmented_Array)

print('Obtained segmented image MI / Otsu segmentation MI: %.3f'%(PCNN_MI / OTSU_MI))



###############################################
# -> Write fitness function to maximize
#    number of feature identified having
#    similar orientation


# Region properties
Labels = measure.label(Segmented_Array)
Properties = measure.regionprops(Labels,GrayScale_Array)
Table_Props = pd.DataFrame(measure.regionprops_table(Labels,GrayScale_Array,properties=('centroid',
                                                 'orientation',
                                                 'major_axis_length',
                                                 'minor_axis_length', 'perimeter')))
M = 166
Table_Props.sort_values(by='perimeter')

Figure, Axes = plt.subplots(1,1,figsize=(4.5,5.5),dpi=100)
Axes.plot(Table_Props.sort_values(by='perimeter')['perimeter'].reset_index(),linestyle='none',marker='o',color=(1,0,0),fillstyle='none')
plt.show()

props = Properties[M]
y0, x0 = props.centroid
orientation = props.orientation
x1 = x0 + np.cos(orientation) * 0.5 * props.minor_axis_length
y1 = y0 - np.sin(orientation) * 0.5 * props.minor_axis_length
x2 = x0 - np.sin(orientation) * 0.5 * props.major_axis_length
y2 = y0 - np.cos(orientation) * 0.5 * props.major_axis_length

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Segmented_Array,cmap='gray')
Axes.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
Axes.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
Axes.plot(x0, y0, '.g', markersize=15)

minr, minc, maxr, maxc = props.bbox
bx = (minc, maxc, maxc, minc, minc)
by = (minr, minr, maxr, maxr, minr)
Axes.plot(bx, by, '-b', linewidth=2.5)

Axes.set_ylim([0,Segmented_Array.shape[1]])
plt.axis('off')
plt.title('Segmented Image')
plt.show()
plt.close(Figure)





u, v = props.centroid
a=props.major_axis_length
b=props.minor_axis_length
t_rot=props.orientation

t = np.linspace(0, 2*np.pi, 100)
Ell = np.array([a*np.cos(t) , b*np.sin(t)])
     #u,v removed to keep the same center location
R_rot = np.array([[np.cos(t_rot) , -np.sin(t_rot)],[np.sin(t_rot) , np.cos(t_rot)]])
     #2-D rotation matrix

Ell_rot = np.zeros((2,Ell.shape[1]))
for i in range(Ell.shape[1]):
    Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

plt.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'darkorange' )    #rotated ellipse


RegionLabel = props.label
Contour = measure.find_contours(Labels == RegionLabel, 0.2)[0]
y, x = Contour.T
plt.plot( x, y,'black' )    #rotated ellipse
plt.show()

