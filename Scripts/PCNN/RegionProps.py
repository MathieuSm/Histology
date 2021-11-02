
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd

desired_width = 500
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)

CurrentDirectory = os.getcwd()
ImageDirectory = CurrentDirectory + '/Scripts/PCNN/Training/'
Files = os.listdir(ImageDirectory)
Files.sort()

Image = sitk.ReadImage(ImageDirectory + Files[-1])
ImageArray = sitk.GetArrayFromImage(Image)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(ImageArray,cmap='gray')
plt.axis('off')
plt.title('Image')
plt.show()
plt.close(Figure)

ROIs = ImageArray.copy()
ROI_Files = [File for File in Files if File.endswith('csv')]

ROI = 0
Coordinates = pd.read_csv(ImageDirectory + ROI_Files[ROI])

Image_ROI = np.zeros(ImageArray.shape)
X = Coordinates.round().astype('int')['X'].values-1
Y = Coordinates.round().astype('int')['Y'].values-1
Image_ROI[Y,X] = 1

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.imshow(Image_ROI,cmap='gray')
plt.axis('off')
plt.title('ROI')
plt.show()
plt.close(Figure)




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

