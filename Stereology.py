import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PlotImage(Image,TicksSize,ColorBar=False):


    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    CMap = Axes.imshow(Image, cmap='bone_r',vmax=15)
    Axes.set_xlim([0, Image.shape[1]])
    Axes.set_ylim([0, Image.shape[0]])
    Axes.set_xlabel('X ($\mu$m)')
    Axes.set_ylabel('Y ($\mu$m)')
    plt.xticks(np.arange(0, Image.shape[1])[::TicksSize], X_Positions[::TicksSize])
    plt.yticks(np.arange(0, Image.shape[0])[::TicksSize], Y_Positions[::TicksSize])
    if ColorBar:
        plt.colorbar(CMap)
    plt.show()

    return
def WriteRaw(ImageArray, OutputFileName, PixelType):

    if PixelType == 'uint':

        MinValue = ImageArray.min()

        if MinValue < 0:
            ShiftedImageArray = ImageArray + abs(MinValue)
            MaxValue = ShiftedImageArray.max()
        elif MinValue > 0:
            ShiftedImageArray = ImageArray - MinValue
            MaxValue = ShiftedImageArray.max()
        else :
            ShiftedImageArray = ImageArray
            MaxValue = ShiftedImageArray.max()

        ScaledImageArray = ShiftedImageArray / MaxValue * 255

        CastedImageArray = ScaledImageArray.astype(np.uint8)

    elif PixelType == 'short':
        CastedImageArray = ImageArray.astype(np.short)
    elif PixelType == 'float':
        CastedImageArray = ImageArray.astype('float32')

    File = np.memmap(OutputFileName, dtype=CastedImageArray.dtype, mode='w+', shape=CastedImageArray.shape)
    File[:] = CastedImageArray[:]
    del File

    return
def WriteMHD(ImageArray, Spacing, Offset, Path, FileName, PixelType='uint'):

    if PixelType == 'short' or PixelType == 'float':
        if len(ImageArray.shape) == 2:

            Array_3D = np.zeros((1,ImageArray.shape[0],ImageArray.shape[1]))

            for j in range(ImageArray.shape[0]):
                for i in range(ImageArray.shape[1]):
                    Array_3D[0,j,i] = ImageArray[j,i]

            ImageArray = Array_3D

    nz, ny, nx = np.shape(ImageArray)

    lx = float(Spacing[0])
    ly = float(Spacing[1])
    lz = float(Spacing[2])

    TransformMatrix = '1 0 0 0 1 0 0 0 1'
    X_o, Y_o, Z_o = float(Offset[0]), float(Offset[1]), float(Offset[2])
    CenterOfRotation = '0 0 0'
    AnatomicalOrientation = 'LPS'

    outs = open(os.path.join(Path, FileName) + '.mhd', 'w')
    outs.write('ObjectType = Image\n')
    outs.write('NDims = 3\n')
    outs.write('BinaryData = True\n')
    outs.write('BinaryDataByteOrderMSB = False\n')
    outs.write('CompressedData = False\n')
    outs.write('TransformMatrix = %s \n' % TransformMatrix)
    outs.write('Offset = %g %g %g\n' % (X_o, Y_o, Z_o))
    outs.write('CenterOfRotation = %s \n' % CenterOfRotation)
    outs.write('AnatomicalOrientation = %s \n' % AnatomicalOrientation)
    outs.write('ElementSpacing = %g %g %g\n' % (lx, ly, lz))
    outs.write('DimSize = %i %i %i\n' % (nx, ny, nz))

    if PixelType == 'uint':
        outs.write('ElementType = %s\n' % 'MET_UCHAR')
    elif PixelType == 'short':
        outs.write('ElementType = %s\n' % 'MET_SHORT')
    elif PixelType == 'float':
        outs.write('ElementType = %s\n' % 'MET_FLOAT')

    outs.write('ElementDataFile = %s\n' % (FileName + '.raw'))
    outs.close()

    WriteRaw(ImageArray, os.path.join(Path, FileName) + '.raw', PixelType)

    return

# Define original array
Size = np.array([1000,1000,1000])    # Image size x, y (um)
Resolution = np.array([10, 10, 10])  # x, y spacing (um)

X_Positions = np.arange(0,Size[0]+Resolution[0],Resolution[0])
Y_Positions = np.arange(0,Size[1]+Resolution[1],Resolution[1])
X_y, Y_x = np.meshgrid(X_Positions,Y_Positions)
Z_Positions = np.arange(0,Size[2]+Resolution[2],Resolution[2])

Image = np.zeros((len(Y_Positions),len(X_Positions)))
PlotImage(Image,TicksSize=20)


# Generate random osteons
N = 10      # Number of osteons
d = 100     # Osteon mean diameter (um)
S = 3*d     # Minimum distance between osteons
l = 25      # Lamellar thickness (um

Osteons = pd.DataFrame()
while len(Osteons) < N:

    TooClose = True
    while TooClose:

        ## Select random osteon
        Random_Position = round(np.random.uniform(0,X_y.shape[0]*X_y.shape[1]))
        Random_X = X_y.reshape(len(X_Positions)*len(Y_Positions))[Random_Position]
        Random_Y = Y_x.reshape(len(X_Positions)*len(Y_Positions))[Random_Position]
        X_Index = np.where(X_Positions == Random_X)[0][0]
        Y_Index = np.where(Y_Positions == Random_Y)[0][0]
        Image[Y_Index,X_Index] = 1
        # PlotImage(Image, TicksSize=20)

        # Generate circle quadrant coordinates
        Cx = np.arange(Random_X,Random_X+round(S/2)+Resolution[1],Resolution[1])
        Cy = np.arange(Random_Y,Random_Y+round(S/2)+Resolution[0],Resolution[0])

        CirclePoints = pd.DataFrame()
        for Px in Cx:
            for Py in Cy:
                R2 = (Px-Random_X)**2 + (Py-Random_Y)**2
                if R2 < (S/2)**2:
                    CirclePoints = CirclePoints.append({'Cx':Px,'Cy':Py},ignore_index=True).astype('int')
                else:
                    Cy = Cy[:-1]

        ## Mirror quadrants 1->2
        New_Cx = 2 * Random_X - CirclePoints['Cx'].values
        New_Cy = CirclePoints['Cy'].values
        New_Points = pd.DataFrame({'Cx':New_Cx,'Cy':New_Cy})
        CirclePoints = CirclePoints.append(New_Points,ignore_index=True).astype('int')

        ## Mirror quadrants 1&2 -> 3&4
        New_Cx = CirclePoints['Cx'].values
        New_Cy = 2 * Random_Y - CirclePoints['Cy'].values
        New_Points = pd.DataFrame({'Cx': New_Cx, 'Cy': New_Cy})
        CirclePoints = CirclePoints.append(New_Points, ignore_index=True).astype('int')

        ## Filter position out of the image
        CirclePoints = CirclePoints.drop_duplicates()
        for Px, Py in CirclePoints[['Cx', 'Cy']].values:
            if not np.isin(Px, X_Positions) or not np.isin(Py, Y_Positions):
                Filter1 = CirclePoints['Cx'] == Px
                Filter2 = CirclePoints['Cy'] == Py
                CirclePoints = CirclePoints.drop(CirclePoints[Filter1 & Filter2].index)

        ## Verify for no other close osteon
        PointIndex = 0
        for Px, Py in CirclePoints[['Cx','Cy']].values:

            PointIndex += 1
            X_Index = np.where(X_Positions == Px)[0][0]
            Y_Index = np.where(Y_Positions == Py)[0][0]

            if Image[Y_Index, X_Index] == 2:
                print('New osteon too close, selecting a new one')
                TooClose = True
                break
            elif PointIndex == len(CirclePoints):
                TooClose = False

    ## If no other close osteon, label area
    for Px, Py in CirclePoints[['Cx', 'Cy']].values:

        X_Index = np.where(X_Positions == Px)[0][0]
        Y_Index = np.where(Y_Positions == Py)[0][0]

        Image[Y_Index, X_Index] = 2

    Osteons = Osteons.append({'X': Random_X, 'Y': Random_Y}, ignore_index=True).astype('int')

    PlotImage(Image, TicksSize=20)

Image = np.zeros((len(Y_Positions),len(X_Positions)))
R = d/2
Label = 1
while np.isin(0,Image):
    for Index in Osteons.index:

        X, Y = Osteons.loc[Index,['X','Y']]

        # Generate circle quadrant coordinates
        Cx = np.arange(X, X + R + Resolution[1], Resolution[1])
        Cy = np.arange(Y, Y + R + Resolution[0], Resolution[0])

        CirclePoints = pd.DataFrame()
        for Px in Cx:
            for Py in Cy:
                R2 = (Px - X) ** 2 + (Py - Y) ** 2
                if R2 < R ** 2:
                    CirclePoints = CirclePoints.append({'Cx': Px, 'Cy': Py}, ignore_index=True).astype('int')
                else:
                    Cy = Cy[:-1]

        # Drop points of previous circle
        if R > d/2:
            for PointIndex in CirclePoints.index:
                Point = CirclePoints.loc[PointIndex]
                R2 = (Point['Cx'] - X) ** 2 + (Point['Cy'] - Y) ** 2
                if R2 < (R-l)**2:
                    CirclePoints = CirclePoints.drop(PointIndex)

        ## Mirror quadrants 1->2
        New_Cx = 2 * X - CirclePoints['Cx'].values
        New_Cy = CirclePoints['Cy'].values
        New_Points = pd.DataFrame({'Cx': New_Cx, 'Cy': New_Cy})
        CirclePoints = CirclePoints.append(New_Points, ignore_index=True).astype('int')

        ## Mirror quadrants 1&2 -> 3&4
        New_Cx = CirclePoints['Cx'].values
        New_Cy = 2 * Y - CirclePoints['Cy'].values
        New_Points = pd.DataFrame({'Cx': New_Cx, 'Cy': New_Cy})
        CirclePoints = CirclePoints.append(New_Points, ignore_index=True).astype('int')

        ## Filter position out of the image and duplicates
        CirclePoints = CirclePoints.drop_duplicates()
        for Px, Py in CirclePoints[['Cx', 'Cy']].values:
            if not np.isin(Px, X_Positions) or not np.isin(Py, Y_Positions):
                Filter1 = CirclePoints['Cx'] == Px
                Filter2 = CirclePoints['Cy'] == Py
                CirclePoints = CirclePoints.drop(CirclePoints[Filter1 & Filter2].index)

        ## If no previous label, label area
        for Px, Py in CirclePoints[['Cx', 'Cy']].values:
            X_Index = np.where(X_Positions == Px)[0][0]
            Y_Index = np.where(Y_Positions == Py)[0][0]

            PointValue = Image[Y_Index, X_Index]

            if PointValue == 0:
                Image[Y_Index, X_Index] = Label

            elif PointValue == Label:
                Image[Y_Index, X_Index] = 100

    R += l
    Label += 1

    PlotImage(Image, TicksSize=20)
PlotImage(Image,TicksSize=20,ColorBar=True)

# Save image in 3D
Image_3D = np.repeat(Image,len(Z_Positions))
Image_3D = Image_3D.reshape((len(X_Positions),len(Y_Positions),len(Z_Positions)))
Offset = np.array([0,0,0])
Path = os.path.join(os.getcwd(),'Scripts/')
WriteMHD(Image_3D, Resolution, Offset, Path, 'Osteons', PixelType='uint')
