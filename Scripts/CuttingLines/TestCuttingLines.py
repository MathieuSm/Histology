#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.widgets import Slider, RadioButtons, Button

desired_width = 500
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', desired_width)
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width,suppress=True,formatter={'float_kind':'{:3}'.format})
plt.rc('font', size=12)


# Define directories
CurrentDir = os.getcwd()
MHDDirectory = CurrentDir.replace('06_Histology','08_uCT')

# Create cube of 1s
Array = np.zeros((20,20,20))
Array[6:14+1,6:14+1,6:14+1] = 1
Array[8,10,8] = 5
Scan = sitk.GetImageFromArray(Array)
Nx, Ny, Nz = Scan.GetSize()
Center = np.array(Scan.GetSize()) / 2


def ChangePlane(label):

    Position = int(Mid_Position_Slider.val)

    X = np.array([X11_Slider.val,
                  X21_Slider.val,
                  X31_Slider.val,
                  X11_Slider.val])
    Y = np.array([X12_Slider.val,
                  X22_Slider.val,
                  X32_Slider.val,
                  X12_Slider.val])
    Z = np.array([X13_Slider.val,
                  X23_Slider.val,
                  X33_Slider.val,
                  X13_Slider.val])

    if label == 'YZ Plane':
        Plane = Array[:, :, Position]
        CuttingPlane.set_xdata(Y)
        CuttingPlane.set_ydata(Z)
        Axes.set_xlabel('Y')
        Axes.set_ylabel('Z ', rotation=0)

    elif label == 'XZ Plane':
        Plane = Array[:, Position, :]
        CuttingPlane.set_xdata(X)
        CuttingPlane.set_ydata(Z)
        Axes.set_xlabel('X')
        Axes.set_ylabel('Z ', rotation=0)

    elif label == 'XY Plane':
        Plane = Array[Position, :, :]
        CuttingPlane.set_xdata(X)
        CuttingPlane.set_ydata(Y)
        Axes.set_xlabel('X')
        Axes.set_ylabel('Y ', rotation=0)

    Show.set_data(Plane)
    Figure.canvas.draw_idle()

def MovePlane(val):

    Position = int(val)
    Label = Buttons.value_selected

    if Label == 'YZ Plane':
        Plane = Array[:, :, Position]

    elif Label == 'XZ Plane':
        Plane = Array[:, Position, :]

    elif Label == 'XY Plane':
        Plane = Array[Position, :, :]

    Show.set_data(Plane)
    Figure.canvas.draw_idle()

def MoveCuttingPlane(val):

    X = np.array([X11_Slider.val,
                  X21_Slider.val,
                  X31_Slider.val,
                  X11_Slider.val])
    Y = np.array([X12_Slider.val,
                  X22_Slider.val,
                  X32_Slider.val,
                  X12_Slider.val])
    Z = np.array([X13_Slider.val,
                  X23_Slider.val,
                  X33_Slider.val,
                  X13_Slider.val])

    if Buttons.value_selected == 'YZ Plane':
        CuttingPlane.set_xdata(Y)
        CuttingPlane.set_ydata(Z)
        Axes.set_xlabel('Y')
        Axes.set_ylabel('Z ', rotation=0)

    elif Buttons.value_selected == 'XZ Plane':
        CuttingPlane.set_xdata(X)
        CuttingPlane.set_ydata(Z)
        Axes.set_xlabel('X')
        Axes.set_ylabel('Z ', rotation=0)

    elif Buttons.value_selected == 'XY Plane':
        CuttingPlane.set_xdata(X)
        CuttingPlane.set_ydata(Y)
        Axes.set_xlabel('X')
        Axes.set_ylabel('Y ', rotation=0)

    Figure.canvas.draw_idle()



# Plot chosen plane in interactive mode
N = max(Nx,Ny,Nz)
Mid_Position = int(round(Ny / 2))
Plane = Array[:, Mid_Position, :]

X1 = [6, 10, 6]
X2 = [14, 12, 8]
X3 = [12, 12, 14]

X = np.array([X1[0],X2[0],X3[0],X1[0]])
Y = np.array([X1[1],X2[1],X3[1],X1[1]])
Z = np.array([X1[2],X2[2],X3[2],X1[2]])


Figure, Axes = plt.subplots(1, 1, figsize=(10, 8), dpi=100)
Show = Axes.imshow(Plane, cmap='bone', alpha=1)
CuttingPlane, = Axes.plot(X,Z,color=(1,0,0),linewidth=1,marker='o',markersize=5)
Axes.set_xlabel('X')
Axes.set_ylabel('Z ', rotation=0)

SliderAxis = plt.axes([0.35, 0.05, 0.55, 0.03])
Mid_Position_Slider = Slider(SliderAxis, 'Plane Position', 0, N, valinit=Mid_Position, valstep=2, color=(0,0,0))

Radio_Axis = plt.axes([0.05, 0.7, 0.2, 0.25])
Buttons = RadioButtons(Radio_Axis, ('YZ Plane', 'XZ Plane', 'XY Plane'),activecolor='black',active=1)

X11Axis = plt.axes([0.05, 0.6, 0.2, 0.02])
X11_Slider = Slider(X11Axis, 'X1$_{X}$', 0, Nx, valinit=X1[0], valstep=1, color=(0,0,0))
X12Axis = plt.axes([0.05, 0.55, 0.2, 0.02])
X12_Slider = Slider(X12Axis, 'X1$_{Y}$', 0, Ny, valinit=X1[1], valstep=1, color=(0,0,0))
X13Axis = plt.axes([0.05, 0.5, 0.2, 0.02])
X13_Slider = Slider(X13Axis, 'X1$_{Z}$', 0, Nz, valinit=X1[2], valstep=1, color=(0,0,0))

X21Axis = plt.axes([0.05, 0.4, 0.2, 0.02])
X21_Slider = Slider(X21Axis, 'X2$_{X}$', 0, Nx, valinit=X2[0], valstep=1, color=(0,0,0))
X22Axis = plt.axes([0.05, 0.35, 0.2, 0.02])
X22_Slider = Slider(X22Axis, 'X2$_{Y}$', 0, Ny, valinit=X2[1], valstep=1, color=(0,0,0))
X23Axis = plt.axes([0.05, 0.3, 0.2, 0.02])
X23_Slider = Slider(X23Axis, 'X2$_{Z}$', 0, Nz, valinit=X2[2], valstep=1, color=(0,0,0))

X31Axis = plt.axes([0.05, 0.2, 0.2, 0.02])
X31_Slider = Slider(X31Axis, 'X3$_{X}$', 0, Nx, valinit=X3[0], valstep=1, color=(0,0,0))
X32Axis = plt.axes([0.05, 0.15, 0.2, 0.02])
X32_Slider = Slider(X32Axis, 'X3$_{Y}$', 0, Ny, valinit=X3[1], valstep=1, color=(0,0,0))
X33Axis = plt.axes([0.05, 0.1, 0.2, 0.02])
X33_Slider = Slider(X33Axis, 'X3$_{Z}$', 0, Nx, valinit=X3[2], valstep=1, color=(0,0,0))

plt.subplots_adjust(left=0.4, bottom=0.2)
Buttons.on_clicked(ChangePlane)
Mid_Position_Slider.on_changed(MovePlane)
X11_Slider.on_changed(MoveCuttingPlane)
X12_Slider.on_changed(MoveCuttingPlane)
X13_Slider.on_changed(MoveCuttingPlane)
X21_Slider.on_changed(MoveCuttingPlane)
X22_Slider.on_changed(MoveCuttingPlane)
X23_Slider.on_changed(MoveCuttingPlane)
X31_Slider.on_changed(MoveCuttingPlane)
X32_Slider.on_changed(MoveCuttingPlane)
X33_Slider.on_changed(MoveCuttingPlane)
plt.show()


# Transform image according to plan
X1 = np.array([X11_Slider.val, X12_Slider.val,  X13_Slider.val])
X2 = np.array([X21_Slider.val, X22_Slider.val,  X23_Slider.val])
X3 = np.array([X31_Slider.val, X32_Slider.val,  X33_Slider.val])

X = np.array([X1[0],X2[0],X3[0],X1[0]])
Y = np.array([X1[1],X2[1],X3[1],X1[1]])
Z = np.array([X1[2],X2[2],X3[2],X1[2]])

W = (X3 - X2) / np.linalg.norm(X3 - X2)
V = (X2 - X1) / np.linalg.norm(X2 - X1)
U = np.cross(W,V) / np.linalg.norm(np.cross(W,V))
V = np.cross(U,W) / np.linalg.norm(np.cross(U,W))


# Define rotation
RotationMatrix = np.zeros((3,3))
# Inverse of orthogonal rotation matrix is its transpose thus transpose(R.T) = R
RotationMatrix[0] = V
RotationMatrix[1] = U
RotationMatrix[2] = W

RotationCenter = Center * np.array(Scan.GetSpacing())


Figure = plt.figure()
Axis = Figure.gca(projection='3d')
Axis.voxels(Array,facecolors=(0,0,0,0.2))
Axis.plot(X,Y,Z,marker='o',color=(0,1,1))
Axis.quiver(5, 5, 5, 2, 0, 0, color=(1,0,0,0.2))
Axis.quiver(5, 5, 5, 0, 2, 0, color=(0,1,0,0.2))
Axis.quiver(5, 5, 5, 0, 0, 2, color=(0,0,1,0.2))
Axis.quiver(5, 5, 5, RotationMatrix[0,0]*2, RotationMatrix[0,1]*2, RotationMatrix[0,1]*2, color=(1,0,0))
Axis.quiver(5, 5, 5, RotationMatrix[1,0]*2, RotationMatrix[1,1]*2, RotationMatrix[1,2]*2, color=(0,1,0))
Axis.quiver(5, 5, 5, RotationMatrix[2,0]*2, RotationMatrix[2,1]*2, RotationMatrix[2,2]*2, color=(0,0,1))
Axis.set_xlabel('X')
Axis.set_ylabel('Y')
Axis.set_zlabel('Z')
plt.show()

# Define transform
Transform = sitk.AffineTransform(3)
Transform.SetMatrix(RotationMatrix.ravel())
Transform.SetCenter(RotationCenter)

# Resample image
Resampler = sitk.ResampleImageFilter()
Resampler.SetReferenceImage(Scan)
Resampler.SetTransform(Transform.GetInverse())
Resampled_Scan = Resampler.Execute(Scan)
sitk.WriteImage(Resampled_Scan,os.path.join(MHDDirectory,'Resampled.mhd'))

Resampled_Array = sitk.GetArrayFromImage(Resampled_Scan)

R_X1 = Center + np.dot(RotationMatrix,X1 - Center)
R_X2 = Center + np.dot(RotationMatrix,X2 - Center)
R_X3 = Center + np.dot(RotationMatrix,X3 - Center)

X = np.array([R_X1[0],R_X2[0],R_X3[0],R_X1[0]])
Y = np.array([R_X1[1],R_X2[1],R_X3[1],R_X1[1]])
Z = np.array([R_X1[2],R_X2[2],R_X3[2],R_X1[2]])


Figure = plt.figure()
Axis = Figure.gca(projection='3d')
Axis.voxels(Resampled_Array,facecolors=(0,0,0,0.2))
Axis.plot(X,Y,Z,marker='o',color=(0,1,1))
Axis.set_xlabel('X')
Axis.set_ylabel('Y')
Axis.set_zlabel('Z')
plt.show()


M = (R_X1 + R_X2) / 2


Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
Axes.imshow(Resampled_Array[:,int(round(R_X1[1])),:], cmap='bone')
Axes.plot(R_X1[0],R_X1[2],marker='o',color=(1,0,0))
Axes.plot(R_X2[0],R_X2[2],marker='o',color=(1,0,0))
Axes.plot(R_X3[0],R_X3[2],marker='o',color=(1,0,0))
Axes.plot(M[0],M[2],marker='o',color=(0,1,0))
plt.show()



# Compute mid point between X1 and X2
N = (R_X1 - R_X2) / np.linalg.norm(R_X1 - R_X2)

Alpha = np.arccos(N.dot(np.array([0,0,1])))
R = np.array([[np.cos(-Alpha), 0, -np.sin(-Alpha)],
              [0, 1, 0],
              [np.sin(-Alpha), 0, np.cos(-Alpha)]])

Transform.SetMatrix(R.ravel())

# Resample image
Resampler = sitk.ResampleImageFilter()
Resampler.SetReferenceImage(Resampled_Scan)
Resampler.SetTransform(Transform.GetInverse())
NeckAligned = Resampler.Execute(Resampled_Scan)

NeckAligned_Array = sitk.GetArrayFromImage(NeckAligned)

A_X1 = Center + np.dot(R,R_X1 - Center)
A_X2 = Center + np.dot(R,R_X2 - Center)
A_X3 = Center + np.dot(R,R_X3 - Center)

X = np.array([A_X1[0],A_X2[0],A_X3[0],A_X1[0]])
Y = np.array([A_X1[1],A_X2[1],A_X3[1],A_X1[1]])
Z = np.array([A_X1[2],A_X2[2],A_X3[2],A_X1[2]])


Figure = plt.figure()
Axis = Figure.gca(projection='3d')
Axis.voxels(Resampled_Array,facecolors=(0,0,0,0.2))
Axis.plot(X,Y,Z,marker='o',color=(0,1,1))
Axis.set_xlabel('X')
Axis.set_ylabel('Y')
Axis.set_zlabel('Z')
plt.show()



M = (A_X1 + A_X2) / 2

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
Axes.imshow(NeckAligned_Array[:,int(round(R_X1[1])),:], cmap='bone')
Axes.plot(A_X1[0],A_X1[2],marker='o',color=(1,0,0))
Axes.plot(A_X2[0],A_X2[2],marker='o',color=(1,0,0))
Axes.plot(A_X3[0],A_X3[2],marker='o',color=(1,0,0))
Axes.plot(M[0],M[2],marker='o',color=(0,1,0))
plt.show()


# Extract Neck
X_Start = int(round(A_X2[0] - 22.5 / NeckAligned.GetSpacing()[0]))
X_Stop = int(round(A_X2[0] + 22.5 / NeckAligned.GetSpacing()[0]))
Y_Start = int(round(A_X2[1] - 22.5 / NeckAligned.GetSpacing()[1]))
Y_Stop = int(round(A_X2[1] + 22.5 / NeckAligned.GetSpacing()[1]))
Z_Start = int(round(A_X2[2]))
Z_Stop = int(round(A_X1[2]))

Neck = sitk.Slice(NeckAligned,(X_Start,Y_Start,Z_Start),(X_Stop,Y_Stop,Z_Stop))
Neck_Array = sitk.GetArrayFromImage(Neck)


Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
Axes.imshow(Neck_Array[:,int(round(A_X1[1])),:], cmap='bone')
Axes.plot(A_X1[0],A_X1[2]-A_X2[2],marker='o',color=(1,0,0))
Axes.plot(A_X2[0],A_X2[2]-A_X2[2],marker='o',color=(1,0,0))
Axes.plot(A_X3[0],A_X3[2]-A_X2[2],marker='o',color=(1,0,0))
Axes.plot(M[0],M[2]-A_X2[2],marker='o',color=(0,1,0))
plt.show()











# Generate coordinate system orthogonal to N
O = np.random.randn(3)
O -= O.dot(N) * N
O /= np.linalg.norm(O)
P = np.cross(N,O)



Resampler = sitk.ResampleImageFilter()
Resampler.SetReferenceImage(Scan)
Resampler.SetOutputSpacing((1,1,1))
Scan = Resampler.Execute(Scan)