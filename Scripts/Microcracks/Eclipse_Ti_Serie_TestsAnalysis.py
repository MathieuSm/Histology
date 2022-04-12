import os
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt


CurrentDirectory = Path.cwd()
TestDirectory = CurrentDirectory / 'Tests/Calcein/Nikon/418_L'

Files = [File for File in os.listdir(TestDirectory) if File.endswith('.tif')]
Files.sort()

File = Files[-2]
Image = sitk.ReadImage(str(TestDirectory / File))
Array = sitk.GetArrayFromImage(Image)
Shape = Array.shape

Figure, Axes = plt.subplots(1,1, figsize=(Shape[1]/1000, Shape[0]/1000), dpi=100)
Axes.imshow(Array)
Axes.axis('off')
plt.subplots_adjust(0, 0, 1, 1)
plt.show()

