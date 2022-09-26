from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import color, morphology

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

# Set path and variables
CurrentDirectory = Path.cwd()
ImageDirectory = CurrentDirectory / 'Tests/Osteons/Sensitivity/'

PixelLength = 1.0460251046025104 # Computed with 418 RM
ROISize = 1000 # Size in um
SemiLength = int(round(ROISize/PixelLength/2))

DataFrame = pd.read_csv(str(ImageDirectory / 'Data.csv'))
N = 2
SampleData = DataFrame.loc[N]

# Read non-segmented image
Name = str(str(SampleData['Sample']) + SampleData['Side'][0] + SampleData['Cortex'][0] + '.jpg')

# Open image to segment
Image = sitk.ReadImage(str(ImageDirectory / Name))
Array = sitk.GetArrayFromImage(Image)[:,:,:3][Area[0][0]:Area[0][1],Area[1][0]:Area[1][1]]

Figure, Axis = plt.subplots(1,1)
Axis.imshow(Array)
plt.show()


