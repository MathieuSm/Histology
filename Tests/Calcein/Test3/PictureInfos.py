import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

Path = 'Tests/Calcein/Test3/'
Files = os.listdir(Path)
JPG = [File for File in Files if File.endswith('.JPG')]
TIF = [File for File in Files if File.endswith('.TIF')]

Image = sitk.ReadImage(Path + TIF[3])
Array = sitk.GetArrayFromImage(Image)

Figure, Axes = plt.subplots(1,1,figsize=(5.5,4.5))
Axes.imshow(Array)
plt.axis('off')
plt.show()


Image.GetSpacing()
Image.GetSize()
