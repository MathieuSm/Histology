import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import morphology, measure, exposure
import pandas as pd
# import statsmodels.stats.api as st



def RGBThreshold(RGBArray, Threshold):
    R, G, B = RGBArray[:, :, 0], RGBArray[:, :, 1], RGBArray[:, :, 2]

    R_Filter = R < Threshold[0]
    G_Filter = G < Threshold[1]
    B_Filter = B > Threshold[2]

    BinArray = np.zeros((RGBArray.shape[0], RGBArray.shape[1]))
    BinArray[R_Filter & G_Filter & B_Filter] = 1

    return BinArray

# Set path
CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Tests/Osteons/HumanBone/'

# Read images
P1 = sitk.ReadImage(ImageDirectory + 'Stained1_Registered.png')
P2 = sitk.ReadImage(ImageDirectory + 'Stained2_Registered.png')
S1 = sitk.ReadImage(ImageDirectory + 'Stained1_Seg.png')

# Load arrays
P1_Array = sitk.GetArrayFromImage(P1)
P2_Array = sitk.GetArrayFromImage(P2)
S1_Array = sitk.GetArrayFromImage(S1)

# Build binary segmented image
S1_Bin = np.zeros(S1_Array.shape)
F_R = S1_Array[:,:,0] == 255
F_G = S1_Array[:,:,1] == 0
F_B = S1_Array[:,:,2] == 0
S1_Bin[F_R & F_G & F_B] = 1

Disk = morphology.disk(5)
S1_Bin = morphology.binary_dilation(S1_Bin[:,:,0],Disk)
S1_Bin = morphology.binary_erosion(S1_Bin,Disk)

Figure, Axes = plt.subplots(1,1,figsize=(55.44,55.97))
Axes.imshow(S1_Array)
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

Figure, Axes = plt.subplots(1,1,figsize=(20,20))
Axes.imshow(S1_Array[600:2600,:2000])
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

Figure, Axes = plt.subplots(1,1,figsize=(15,30))
Axes.imshow(S1_Array[1400:1700,1600:1750])
plt.subplots_adjust(0,0,1,1)
Axes.axis('off')
plt.show()

# Collect protocols values
V1 = P1_Array * np.repeat(S1_Bin,4).reshape((S1_Bin.shape[0],S1_Bin.shape[1],4))
V2 = P2_Array * np.repeat(S1_Bin,4).reshape((S1_Bin.shape[0],S1_Bin.shape[1],4))

Figure, Axes = plt.subplots(1,1)
Axes.imshow(V2)
Axes.axis('off')
plt.show()

V1_DF = pd.DataFrame({'R':V1[:,:,0].flatten(),
                      'G':V1[:,:,1].flatten(),
                      'B':V1[:,:,2].flatten(),
                      'A':V1[:,:,3].flatten()})
V2_DF = pd.DataFrame({'R':V2[:,:,0].flatten(),
                      'G':V2[:,:,1].flatten(),
                      'B':V2[:,:,2].flatten(),
                      'A':V2[:,:,3].flatten()})

# Filter dataframes
FR_Min = V1_DF['R'] > 0
FG_Min = V1_DF['G'] > 0
FB_Min = V1_DF['B'] > 0

FR_Max = V1_DF['R'] < 255
FG_Max = V1_DF['G'] < 255
FB_Max = V1_DF['B'] < 255

F_V1 = V1_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]
F_V2 = V2_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]

# Boxplot of the values
WhiskerProps = dict(linestyle='--', linewidth=1, color=(0,0,0))
FlierProps = dict(marker='o', markerfacecolor=(0,0,0,0), markersize=5, markeredgecolor=(0,0,0))
Positions = np.array([1,2,3])
Offset = 0.2

Figure, Axes = plt.subplots(1,1)
BoxProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
MedianProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
Axes.boxplot(F_V1[['R','G','B']], positions=Positions-Offset,
             boxprops=BoxProps, flierprops=FlierProps,
             medianprops=MedianProps, whiskerprops=WhiskerProps)
BoxProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
MedianProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
Axes.boxplot(F_V2[['R','G','B']], positions=Positions+Offset,
             boxprops=BoxProps, flierprops=FlierProps,
             medianprops=MedianProps, whiskerprops=WhiskerProps)
Axes.plot([],color=(0,0,1), label='Protocol 1')
Axes.plot([],color=(1,0,0), label='Protocol 2')
Axes.set_xticks(Positions,['R','G','B'])
Axes.set_ylim([0,255])
plt.legend(loc='lower right')
plt.show()


## Step 2: take neighboring area to compare values

T1 = np.zeros(P1_Array.shape).astype('int')
T1[:,:,0][S1_Bin == 1] = 255
T1[:,:,3][S1_Bin == 1] = 50

Figure, Axes = plt.subplots(1, 1, figsize=(15,30))
Axes.imshow(P1_Array[1400:1700, 1600:1750])
Axes.imshow(T1[1400:1700, 1600:1750])
Axes.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.show()

for i in [5,10,15]:

    Disk = morphology.disk(i)
    S1_Big = morphology.binary_dilation(S1_Bin,Disk)
    S1_Big = S1_Big*1 - S1_Bin*1

    T1 = np.zeros(P1_Array.shape).astype('int')
    T1[:, :, 0][S1_Big == 1] = 255
    T1[:, :, 3][S1_Big == 1] = 50

    Figure, Axes = plt.subplots(1, 1, figsize=(15,30))
    Axes.imshow(P1_Array[1400:1700, 1600:1750])
    Axes.imshow(T1[1400:1700, 1600:1750])
    Axes.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    plt.show()

    # Collect protocols values
    V1_B = P1_Array * np.repeat(S1_Big,4).reshape((S1_Big.shape[0],S1_Big.shape[1],4))
    V2_B = P2_Array * np.repeat(S1_Big,4).reshape((S1_Big.shape[0],S1_Big.shape[1],4))

    # Figure, Axes = plt.subplots(1,1)
    # Axes.imshow(V2)
    # Axes.axis('off')
    # plt.show()

    V1_B_DF = pd.DataFrame({'R':V1_B[:,:,0].flatten(),
                            'G':V1_B[:,:,1].flatten(),
                            'B':V1_B[:,:,2].flatten(),
                            'A':V1_B[:,:,3].flatten()})
    V2_B_DF = pd.DataFrame({'R':V2_B[:,:,0].flatten(),
                            'G':V2_B[:,:,1].flatten(),
                            'B':V2_B[:,:,2].flatten(),
                            'A':V2_B[:,:,3].flatten()})

    # Filter dataframes
    FR_Min = V1_B_DF['R'] > 0
    FG_Min = V1_B_DF['G'] > 0
    FB_Min = V1_B_DF['B'] > 0

    FR_Max = V1_B_DF['R'] < 255
    FG_Max = V1_B_DF['G'] < 255
    FB_Max = V1_B_DF['B'] < 255

    F_V1_B = V1_B_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]
    F_V2_B = V2_B_DF[FR_Min & FG_Min & FB_Min & FR_Max & FG_Max & FB_Max]

    # Boxplot of the values
    Figure, Axes = plt.subplots(1,1)
    BoxProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
    MedianProps = dict(linestyle='-', linewidth=1, color=(0,0,1))
    Axes.boxplot(F_V1_B[['R','G','B']], positions=Positions-Offset,
                 boxprops=BoxProps, flierprops=FlierProps,
                 medianprops=MedianProps, whiskerprops=WhiskerProps)
    BoxProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
    MedianProps = dict(linestyle='-', linewidth=1, color=(1,0,0))
    Axes.boxplot(F_V2_B[['R','G','B']], positions=Positions+Offset,
                 boxprops=BoxProps, flierprops=FlierProps,
                 medianprops=MedianProps, whiskerprops=WhiskerProps)
    Axes.plot([],color=(0,0,1), label='Protocol 1')
    Axes.plot([],color=(1,0,0), label='Protocol 2')
    Axes.set_xticks(Positions,['R','G','B'])
    Axes.set_ylim([0,255])
    plt.legend(loc='lower right')
    # Axes.axis('off')
    plt.show()

    # Compute Euclidian distance between mean vectors
    MV1 = F_V1.mean()
    MV2 = F_V2.mean()
    MV1_B = F_V1_B.mean()
    MV2_B = F_V2_B.mean()

    D1 = np.linalg.norm(MV1 - MV1_B)
    D2 = np.linalg.norm(MV2 - MV2_B)

    Figure, Axes = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
    Axes.plot([MV1[0], MV1_B[0]],
              [MV1[1], MV1_B[1]],
              [MV1[2], MV1_B[2]], color=(0,0,1), marker='o', label='Protocol 1')
    Axes.plot([MV2[0], MV2_B[0]],
              [MV2[1], MV2_B[1]],
              [MV2[2], MV2_B[2]], color=(1,0,0), marker='o', label='Protocol 2')
    Axes.set_xlim([80, 130])
    Axes.set_ylim([80, 130])
    Axes.set_zlim([160-15, 180+15])
    Axes.set_xlabel('R')
    Axes.set_ylabel('G')
    Axes.set_zlabel('B')
    # Axes.set_zticks([160,170,180])
    plt.legend()
    plt.show()

    # Statistical analysis
    V1_MC, V2_MC = {}, {}
    for Channel in ['R','G','B']:
        MC = st.CompareMeans(st.DescrStatsW(F_V1_B[Channel]),st.DescrStatsW(F_V1[Channel]))
        V1_MC[Channel] = MC.tconfint_diff(alpha=0.5)
        MC = st.CompareMeans(st.DescrStatsW(F_V2_B[Channel]),st.DescrStatsW(F_V2[Channel]))
        V2_MC[Channel] = MC.tconfint_diff(alpha=0.5)

    Pos = np.repeat(np.arange(1,4),2)
    Delta = 0.1
    Figure, Axes = plt.subplots(1,1,figsize=(5,6))
    Axes.plot(Pos[:2] - Delta, V1_MC['R'], color=(0, 0, 1), linewidth=2)
    Axes.plot(Pos[0] - Delta, np.mean(V1_MC['R']), color=(0, 0, 1), marker='o', markersize=5)
    Axes.plot(Pos[2:4] - Delta, V1_MC['G'], color=(0, 0, 1), linewidth=2)
    Axes.plot(Pos[2] - Delta, np.mean(V1_MC['G']), color=(0, 0, 1), marker='o', markersize=5)
    Axes.plot(Pos[4:] - Delta, V1_MC['B'], color=(0, 0, 1), linewidth=2)
    Axes.plot(Pos[4] - Delta, np.mean(V1_MC['B']), color=(0, 0, 1), marker='o', markersize=5)
    Axes.plot(Pos[:2] + Delta, V2_MC['R'], color=(1, 0, 0), linewidth=2)
    Axes.plot(Pos[0] + Delta, np.mean(V2_MC['R']), color=(1, 0, 0), marker='o', markersize=5)
    Axes.plot(Pos[2:4] + Delta, V2_MC['G'], color=(1, 0, 0), linewidth=2)
    Axes.plot(Pos[2] + Delta, np.mean(V2_MC['G']), color=(1, 0, 0), marker='o', markersize=5)
    Axes.plot(Pos[4:] + Delta, V2_MC['B'], color=(1, 0, 0), linewidth=2)
    Axes.plot(Pos[4] + Delta, np.mean(V2_MC['B']), color=(1, 0, 0), marker='o', markersize=5)
    Axes.plot([], color=(0, 0, 1), linewidth=2, marker='o', markersize=5, label='Protocol 1')
    Axes.plot([], color=(1, 0, 0), linewidth=2, marker='o', markersize=5, label='Protocol 2')
    Axes.set_xticks([1,2,3],['R','G','B'])
    Axes.set_xlim([0.4,3.6])
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.25),ncol=1, frameon=False)
    plt.subplots_adjust(0.2,0.1,0.8,0.8)
    plt.show()




# Signal analysis
C1 = sitk.ReadImage(ImageDirectory + 'Stained1_Signal.png')
C1_Array = sitk.GetArrayFromImage(C1)

# Build binary segmented image
C1_Bin = np.zeros(C1_Array.shape).astype('int')
F_R = C1_Array[:,:,0] > 230
F_G = C1_Array[:,:,1] < 50
F_B = C1_Array[:,:,2] < 50
C1_Bin[F_R & F_G & F_B] = 1

C1_Labels = morphology.label(C1_Bin[:,:,0])
RegionProperties = measure.regionprops(C1_Labels)

R, G, B = {}, {}, {}
for Region in RegionProperties:
    R[str(Region.label)] = P1_Array[Region.coords[:,0],Region.coords[:,1],0]
    G[str(Region.label)] = P1_Array[Region.coords[:,0],Region.coords[:,1],1]
    B[str(Region.label)] = P1_Array[Region.coords[:,0],Region.coords[:,1],2]

R_P1 = R
G_P1 = G
B_P1 = B


i = 4
Figure, Axes = plt.subplots(1,1)
Axes.imshow(P2_Array)
Axes.plot(RegionProperties[i-1].coords[:,1],RegionProperties[i-1].coords[:,0],color=(0,0,0))
Axes.plot(RegionProperties[i-1].coords[0,1],RegionProperties[i-1].coords[0,0],
             marker='o',linestyle='none',color=(1,0,0),label='Start')
Axes.plot(RegionProperties[i-1].coords[-1,1],RegionProperties[i-1].coords[-1,0],
             marker='o',linestyle='none',color=(0,1,0),label='Stop')
Axes.set_xlim([RegionProperties[i-1].coords[-1,1]*0.9,RegionProperties[i-1].coords[0,1]*1.1])
Axes.set_ylim([RegionProperties[i-1].coords[-1,0]*1.1,RegionProperties[i-1].coords[0,0]*0.8])
plt.legend(loc='lower right')
plt.show()

from pathlib import Path
FigPath = Path(CurrentDirectory[:-12])
FigPath = FigPath / '02_Meetings/03_MicroMeso/Pictures/StainingProtocol'
plt.rcParams['font.size'] = '16'

Figure, Axes = plt.subplots(1,1)
Axes.plot(R[str(i)][::-1],color=(1,0,0))
Axes.plot(G[str(i)][::-1],color=(0,1,0))
Axes.plot(B[str(i)][::-1],color=(0,0,1))
# Axes.plot(R_P1[str(i)][::-10],color=(1,0,0),linestyle='--')
# Axes.plot(G_P1[str(i)][::-10],color=(0,1,0),linestyle='--')
# Axes.plot(B_P1[str(i)][::-10],color=(0,0,1),linestyle='--')
Axes.set_ylim([0,255])
Axes.set_xticks([])
plt.box(False)
plt.savefig(FigPath / str('Signal' + str(i) + '_P2.png'), transparent=True)
plt.show()

Figure, Axes = plt.subplots(1,1,figsize=(12.69,10.26))
Axes.imshow(P2_Array)
# Axes.plot(RegionProperties[i-1].coords[:,1],RegionProperties[i-1].coords[:,0],color=(0,0,0))
Axes.plot(RegionProperties[i-1].coords[0,1],RegionProperties[i-1].coords[0,0],
             marker='o',linestyle='none',color=(0,0,0),label='Start')
Axes.plot(RegionProperties[i-1].coords[-1,1],RegionProperties[i-1].coords[-1,0],
             marker='o',linestyle='none',color=(0,0,0),label='Stop')
Axes.set_xlim([RegionProperties[i-1].coords[-1,1]*0.7,RegionProperties[i-1].coords[0,1]*1.3])
Axes.set_ylim([RegionProperties[i-1].coords[-1,0]*1.05,RegionProperties[i-1].coords[0,0]*0.95])
# plt.legend(loc='lower right')
plt.subplots_adjust(0,0,1,1)
plt.axis('off')
plt.show()


# Get mean signal to evaluate contrast
RG_P1 = {}
RG_P2 = {}
C_P1 = {}
C_P2 = {}
for i in range(1,6):
    RG_P1[str(i)] = (R_P1[str(i)].astype('int') + G_P1[str(i)].astype('int')) / 2
    RG_P2[str(i)] = (R_P2[str(i)].astype('int') + G_P2[str(i)].astype('int')) / 2
    C_P1[str(i)] = B_P1[str(i)] - RG_P1[str(i)]
    C_P2[str(i)] = B_P2[str(i)] - RG_P2[str(i)]

Figure, Axes = plt.subplots(1,1)
Axes.plot(C_P1[str(i)][::-1]/max(C_P1[str(i)]),color=(0,0,0),label='Protocol 1')
# Axes.plot(C_P2[str(i)][::-1]/max(C_P2[str(i)]),color=(0,0,0),label='Protocol 2')
Axes.set_ylim([0,1])
# Axes.set_yscale('log')
Axes.set_xticks([])
plt.box(False)
plt.savefig(FigPath / str('Signal' + str(i-1) + '_Diff_P1.png'), transparent=True)
# plt.legend()
plt.show()

