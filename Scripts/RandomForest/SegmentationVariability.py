import sys
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, morphology
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPCNN'))
from Utilities import *

# Load dictionary
CWD = Path.cwd() / 'Scripts' / 'RandomForest'
FileName = CWD / 'ROIs.pkl'
with open(str(FileName), 'rb') as f:
    Dict = pickle.load(f)

# Select image to compare
Key, ROINumber = 0, 0
Key, ROINumber = 4, 2
Skeleton = Dict[Key]['Skeletons'][ROINumber]
Skeleton = morphology.binary_dilation(Skeleton,morphology.disk(1))

# Newly segmented image
FileName = CWD / str('Sample' + str(Key) + '_Seg' + str(ROINumber) + '.png')
Seg_ROI = io.imread(str(FileName))
Label = Seg_ROI[:,:,0] == 255
Label[Seg_ROI[:,:,2] == 255] = False
PlotImage(Label)


plt.get_cmap('binary')(np.linspace(0,2))

CMapDict = {'red':((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),
            'green': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),
            'blue': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
CMap = LinearSegmentedColormap('MyMap',CMapDict)

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Skeleton, cmap='binary_r')
Axis.imshow(Label*1, cmap=CMap, alpha=0.3)
Axis.plot([], color=(1,1,1), label='1$^{st}$ segmentation',
          lw=1, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
Axis.plot([], color=(1,0,0), lw=1, label='2$^{nd}$ segmentation')
Axis.axis('off')
plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.15))
plt.tight_layout()
plt.show()

# Dilate both segmentation and capture common parts
Disk = morphology.disk(5)
SkeletonD = morphology.binary_dilation(Skeleton,Disk)
LabelD = morphology.binary_dilation(Label,Disk)

C1 = np.sum(Skeleton * LabelD) / Skeleton.sum()
print('Pixel ratio segmented twice:')
print(round(C1,3))

C2 = 1 - np.sum(SkeletonD * Label) / Label.sum()
print('Pixel ratio not segmented at first attempt:')
print(round(C2,3))