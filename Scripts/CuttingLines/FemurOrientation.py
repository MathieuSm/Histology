#!/usr/bin/env python3

import argparse
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from skimage import morphology, measure
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    This script runs the interactive registration of proximal femur scan according to 3 points
    
    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern
    
    Date: January 2022
    """


def Main(Arguments, FigColor=(0.17, 0.18, 0.22)):
    print('\nStart femur orientation ...\n')

    # Get argument attributes
    print('Read arguments ...')
    MHDDirectory = Arguments.id
    File = Arguments.Proximal
    TxtDirectory = Arguments.od

    FigColor3D = (FigColor[0], FigColor[1], FigColor[2], 0)

    # Load file
    print('Load file ...')
    Scan = sitk.ReadImage(os.path.join(MHDDirectory,'Proximal',File))
    Nx, Ny, Nz = Scan.GetSize()
    Center = np.array(Scan.GetSize()) / 2
    Array = sitk.GetArrayFromImage(Scan)

    # If file exist, read point coordinates. Otherwise, given start
    TxtFileName = os.path.join(TxtDirectory,File[:8]+'_PC.txt')

    if os.path.isfile(TxtFileName):
        X1, X2, X3 = np.loadtxt(TxtFileName).astype('int')

    else:
        X1 = [145, 190, 80]
        X2 = [260, 170, 190]
        X3 = [250, 200, 380]


    # 3D interactive plot for plane selection
    print('Segment image using multiple Otsu (2) ...')
    OtsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
    OtsuFilter.SetNumberOfThresholds(2)
    OtsuFilter.Execute(Scan)
    OtsuThresholds = OtsuFilter.GetThresholds()

    BinArray = np.zeros(Array.shape)
    BinArray[Array > OtsuThresholds[1]] = 1
    BinScan = sitk.GetImageFromArray(BinArray)
    BinScan.SetSpacing(Scan.GetSpacing())

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Axes.imshow(BinArray[:, int(round(BinArray.shape[1] / 2)), :], cmap='bone')
    # Axes.plot([X1[0], X2[0], X3[0]], [X1[2], X2[2], X3[2]], marker='o', color=(1, 0, 0))
    # plt.show()

    print('Smooth image using Gauss filtering ...')
    GaussFilter = sitk.DiscreteGaussianImageFilter()
    GaussFilter.SetVariance((1, 1, 1))
    FilteredScan = GaussFilter.Execute(BinScan)
    FilteredArray = sitk.GetArrayFromImage(FilteredScan)

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Axes.imshow(FilteredArray[:, int(round(FilteredArray.shape[1] / 2)), :], cmap='bone')
    # Axes.plot([X1[0], X2[0], X3[0]], [X1[2], X2[2], X3[2]], marker='o', color=(1, 0, 0))
    # plt.show()

    print('Segment image again for binary hole filling ...')
    Histogram, BinEdges = np.histogram(FilteredArray, bins=256, range=(0, 1))
    Minimums = argrelextrema(Histogram, np.less)
    Threshold = BinEdges[Minimums[0][0]]

    BinArray = np.zeros(Array.shape)
    BinArray[FilteredArray > Threshold] = 1
    BinScan = sitk.GetImageFromArray(FilteredArray)
    BinScan.SetSpacing(Scan.GetSpacing())

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Axes.imshow(BinArray[:, int(round(BinArray.shape[1] / 2)), :], cmap='bone')
    # Axes.plot([X1[0], X2[0], X3[0]], [X1[2], X2[2], X3[2]], marker='o', color=(1, 0, 0))
    # plt.show()

    print('Fill holes using binary dilation ...')
    Radius = 10
    Sampling = 5
    PadArray = np.pad(BinArray,pad_width=Sampling*Radius+1)
    Ball = morphology.ball(Radius)
    DilatedArray = morphology.binary_dilation(PadArray[::Sampling,::Sampling,::Sampling],Ball)
    ErodedArray = morphology.binary_erosion(DilatedArray,Ball)
    UnPaddedArray = ErodedArray[Radius+1:-Radius,Radius+1:-Radius,Radius+1:-Radius]

    P1 = np.array([X1[0], X2[0], X3[0]]) / Sampling
    P2 = np.array([X1[1], X2[1], X3[1]]) / Sampling
    P3 = np.array([X1[2], X2[2], X3[2]]) / Sampling

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Axes.imshow(UnPaddedArray[:, int(round(UnPaddedArray.shape[1] / 2)), :], cmap='bone')
    # Axes.plot(P1, P3, marker='o', color=(1, 0, 0))
    # plt.show()

    print('Plane selection using interactive plot ...')
    SubSampling3D = 3
    Array3D = np.transpose(UnPaddedArray,(2,1,0))[::SubSampling3D,::SubSampling3D,::SubSampling3D]
    Vertices, Faces, Normals, Values = measure.marching_cubes(Array3D, 0.5)

    Figure = plt.figure(figsize=(15,9),facecolor=FigColor)
    Figure.suptitle('Plane Interactive Definition', color=(0,0,0))
    Axes = [Figure.add_subplot(221, projection='3d',facecolor=FigColor),
            Figure.add_subplot(222,facecolor=FigColor),
            Figure.add_subplot(223,facecolor=FigColor),
            Figure.add_subplot(224,facecolor=FigColor)]

    Mesh = Poly3DCollection(Vertices[Faces],alpha=0.1)
    Mesh.set_facecolor([0.5, 0.5, 1])
    Axes[0].add_collection3d(Mesh)
    CuttingPlane, = Axes[0].plot(P1/SubSampling3D,P2/SubSampling3D,P3/SubSampling3D,color=(1,0,0),marker='o')
    # Make the panes transparent
    Axes[0].xaxis.set_pane_color(FigColor3D)
    Axes[0].yaxis.set_pane_color(FigColor3D)
    Axes[0].zaxis.set_pane_color(FigColor3D)
    # Make the grid lines transparent
    Axes[0].xaxis._axinfo["grid"]['color'] = FigColor3D
    Axes[0].yaxis._axinfo["grid"]['color'] = FigColor3D
    Axes[0].zaxis._axinfo["grid"]['color'] = FigColor3D
    # Modify ticks
    Axes[0].set_xticks([])
    Axes[0].set_yticks([])
    Axes[0].set_zticks([])
    Axes[0].set_xlabel('X', labelpad=1)
    Axes[0].set_ylabel('Y', labelpad=1)
    Axes[0].set_zlabel('Z', labelpad=1)
    Axes[0].set_xlim(0, Array3D.shape[2])
    Axes[0].set_ylim(0, Array3D.shape[1])
    Axes[0].set_zlim(Array3D.shape[0], 0)
    # scaling hack
    Bbox_min = np.min([0, 0, 0])
    Bbox_max = np.max([Array3D.shape[2], Array3D.shape[1], Array3D.shape[0]])
    Axes[0].auto_scale_xyz([Bbox_min, Bbox_max], [Bbox_min, Bbox_max], [Bbox_min, Bbox_max])

    ShowXY = Axes[1].imshow(Array[X1[2],:,:], cmap='bone')
    Axes[1].set_xlabel('X')
    Axes[1].set_ylabel('Y')
    Axes[1].set_xticks([])
    Axes[1].set_yticks([])
    ShowYZ = Axes[2].imshow(Array[:,:,X1[0]], cmap='bone')
    Axes[2].set_xlabel('Y')
    Axes[2].set_ylabel('Z')
    Axes[2].set_xticks([])
    Axes[2].set_yticks([])
    ShowXZ = Axes[3].imshow(Array[:,X1[1],:], cmap='bone')
    Axes[3].set_xlabel('X')
    Axes[3].set_ylabel('Z')
    Axes[3].set_xticks([])
    Axes[3].set_yticks([])

    PlotXY, = Axes[1].plot(X1[0],X1[1],color=(1,0,0),marker='o')
    PlotYZ, = Axes[2].plot(X1[1],X1[2],color=(1,0,0),marker='o')
    PlotXZ, = Axes[3].plot(X1[0],X1[2],color=(1,0,0),marker='o')

    PlotXY_T, = Axes[1].plot([X1[0], X2[0], X3[0]], [X1[1], X2[1], X3[1]], color=(1, 0, 0, 0.25), marker='o')
    PlotYZ_T, = Axes[2].plot([X1[1], X2[1], X3[1]], [X1[2], X2[2], X3[2]], color=(1, 0, 0, 0.25), marker='o')
    PlotXZ_T, = Axes[3].plot([X1[0], X2[0], X3[0]], [X1[2], X2[2], X3[2]], color=(1, 0, 0, 0.25), marker='o')

    X11Axis = plt.axes([0.05, 0.9, 0.2, 0.02], facecolor=FigColor)
    X11_Slider = Slider(X11Axis, 'X1$_{X}$', 0, Nx, valinit=X1[0], valstep=1, color=(0, 0, 0))
    X12Axis = plt.axes([0.05, 0.85, 0.2, 0.02], facecolor=FigColor)
    X12_Slider = Slider(X12Axis, 'X1$_{Y}$', 0, Ny, valinit=X1[1], valstep=1, color=(0, 0, 0))
    X13Axis = plt.axes([0.05, 0.8, 0.2, 0.02], facecolor=FigColor)
    X13_Slider = Slider(X13Axis, 'X1$_{Z}$', 0, Nz, valinit=X1[2], valstep=1, color=(0, 0, 0))

    X21Axis = plt.axes([0.05, 0.55, 0.2, 0.02], facecolor=FigColor)
    X21_Slider = Slider(X21Axis, 'X2$_{X}$', 0, Nx, valinit=X2[0], valstep=1, color=(0, 0, 0))
    X22Axis = plt.axes([0.05, 0.5, 0.2, 0.02], facecolor=FigColor)
    X22_Slider = Slider(X22Axis, 'X2$_{Y}$', 0, Ny, valinit=X2[1], valstep=1, color=(0, 0, 0))
    X23Axis = plt.axes([0.05, 0.45, 0.2, 0.02], facecolor=FigColor)
    X23_Slider = Slider(X23Axis, 'X2$_{Z}$', 0, Nz, valinit=X2[2], valstep=1, color=(0, 0, 0))

    X31Axis = plt.axes([0.05, 0.2, 0.2, 0.02], facecolor=FigColor)
    X31_Slider = Slider(X31Axis, 'X3$_{X}$', 0, Nx, valinit=X3[0], valstep=1, color=(0, 0, 0))
    X32Axis = plt.axes([0.05, 0.15, 0.2, 0.02], facecolor=FigColor)
    X32_Slider = Slider(X32Axis, 'X3$_{Y}$', 0, Ny, valinit=X3[1], valstep=1, color=(0, 0, 0))
    X33Axis = plt.axes([0.05, 0.1, 0.2, 0.02], facecolor=FigColor)
    X33_Slider = Slider(X33Axis, 'X3$_{Z}$', 0, Nx, valinit=X3[2], valstep=1, color=(0, 0, 0))

    def MovePoints(val):

        X = np.array([X11_Slider.val,
                      X21_Slider.val,
                      X31_Slider.val])
        Y = np.array([X12_Slider.val,
                      X22_Slider.val,
                      X32_Slider.val])
        Z = np.array([X13_Slider.val,
                      X23_Slider.val,
                      X33_Slider.val])

        CuttingPlane.set_xdata(X/SubSampling3D/Sampling)
        CuttingPlane.set_ydata(Y/SubSampling3D/Sampling)
        CuttingPlane.set_3d_properties(Z/SubSampling3D/Sampling)

        Figure.canvas.draw_idle()

    def MovePointX1(val):

        ShowXY.set_data(Array[int(X13_Slider.val),:,:])
        ShowXZ.set_data(Array[:,int(X12_Slider.val),:])
        ShowYZ.set_data(Array[:,:,int(X11_Slider.val)])

        PlotXY.set_xdata(X11_Slider.val)
        PlotXY.set_ydata(X12_Slider.val)
        PlotXY_T.set_xdata([X11_Slider.val, X21_Slider.val, X31_Slider.val])
        PlotXY_T.set_ydata([X12_Slider.val, X22_Slider.val, X32_Slider.val])

        PlotYZ.set_xdata(X12_Slider.val)
        PlotYZ.set_ydata(X13_Slider.val)
        PlotYZ_T.set_xdata([X12_Slider.val, X22_Slider.val, X32_Slider.val])
        PlotYZ_T.set_ydata([X13_Slider.val, X23_Slider.val, X33_Slider.val])

        PlotXZ.set_xdata(X11_Slider.val)
        PlotXZ.set_ydata(X13_Slider.val)
        PlotXZ_T.set_xdata([X11_Slider.val, X21_Slider.val, X31_Slider.val])
        PlotXZ_T.set_ydata([X13_Slider.val, X23_Slider.val, X33_Slider.val])

        Figure.canvas.draw_idle()

    def MovePointX2(val):

        ShowXY.set_data(Array[int(X23_Slider.val), :, :])
        ShowXZ.set_data(Array[:, int(X22_Slider.val), :])
        ShowYZ.set_data(Array[:, :, int(X21_Slider.val)])

        PlotXY.set_xdata(X21_Slider.val)
        PlotXY.set_ydata(X22_Slider.val)
        PlotXY_T.set_xdata([X11_Slider.val, X21_Slider.val, X31_Slider.val])
        PlotXY_T.set_ydata([X12_Slider.val, X22_Slider.val, X32_Slider.val])


        PlotYZ.set_xdata(X22_Slider.val)
        PlotYZ.set_ydata(X23_Slider.val)
        PlotYZ_T.set_xdata([X12_Slider.val, X22_Slider.val, X32_Slider.val])
        PlotYZ_T.set_ydata([X13_Slider.val, X23_Slider.val, X33_Slider.val])

        PlotXZ.set_xdata(X21_Slider.val)
        PlotXZ.set_ydata(X23_Slider.val)
        PlotXZ_T.set_xdata([X11_Slider.val, X21_Slider.val, X31_Slider.val])
        PlotXZ_T.set_ydata([X13_Slider.val, X23_Slider.val, X33_Slider.val])

        Figure.canvas.draw_idle()

    def MovePointX3(val):

        ShowXY.set_data(Array[int(X33_Slider.val), :, :])
        ShowXZ.set_data(Array[:, int(X32_Slider.val), :])
        ShowYZ.set_data(Array[:, :, int(X31_Slider.val)])

        PlotXY.set_xdata(X31_Slider.val)
        PlotXY.set_ydata(X32_Slider.val)
        PlotXY_T.set_xdata([X11_Slider.val, X21_Slider.val, X31_Slider.val])
        PlotXY_T.set_ydata([X12_Slider.val, X22_Slider.val, X32_Slider.val])

        PlotYZ.set_xdata(X32_Slider.val)
        PlotYZ.set_ydata(X33_Slider.val)
        PlotYZ_T.set_xdata([X12_Slider.val, X22_Slider.val, X32_Slider.val])
        PlotYZ_T.set_ydata([X13_Slider.val, X23_Slider.val, X33_Slider.val])

        PlotXZ.set_xdata(X31_Slider.val)
        PlotXZ.set_ydata(X33_Slider.val)
        PlotXZ_T.set_xdata([X11_Slider.val, X21_Slider.val, X31_Slider.val])
        PlotXZ_T.set_ydata([X13_Slider.val, X23_Slider.val, X33_Slider.val])

        Figure.canvas.draw_idle()

    plt.subplots_adjust(left=0.4)
    X11_Slider.on_changed(MovePoints)
    X12_Slider.on_changed(MovePoints)
    X13_Slider.on_changed(MovePoints)

    X11_Slider.on_changed(MovePointX1)
    X12_Slider.on_changed(MovePointX1)
    X13_Slider.on_changed(MovePointX1)

    X21_Slider.on_changed(MovePoints)
    X22_Slider.on_changed(MovePoints)
    X23_Slider.on_changed(MovePoints)

    X21_Slider.on_changed(MovePointX2)
    X22_Slider.on_changed(MovePointX2)
    X23_Slider.on_changed(MovePointX2)

    X31_Slider.on_changed(MovePoints)
    X32_Slider.on_changed(MovePoints)
    X33_Slider.on_changed(MovePoints)

    X31_Slider.on_changed(MovePointX3)
    X32_Slider.on_changed(MovePointX3)
    X33_Slider.on_changed(MovePointX3)

    plt.show()

    # Save point coordinates into a txt file
    print('Save points coordinates into text file ...')
    X1 = np.array([X11_Slider.val, X12_Slider.val, X13_Slider.val])
    X2 = np.array([X21_Slider.val, X22_Slider.val, X23_Slider.val])
    X3 = np.array([X31_Slider.val, X32_Slider.val, X33_Slider.val])
    np.savetxt(TxtFileName,(X1,X2,X3),delimiter='\t',newline='\n',fmt='%3i')

    # Transform image according to interactively defined plan
    print('Compute rotation to align custom plane with ZX plane ...')
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

    # Plot resulting rotation
    FigColor3D = (0.1, 0.11, 0.15, 0)
    Figure = plt.figure(facecolor=FigColor)
    Axis = Figure.gca(projection='3d', facecolor=FigColor)
    Axis.set_title('Scan rotation', color=(0,0,0))
    Axis.quiver(0, 0, 0, 1, 0, 0, color=(1,0,0,0.4))
    Axis.quiver(0, 0, 0, 0, 1, 0, color=(0,1,0,0.4))
    Axis.quiver(0, 0, 0, 0, 0, 1, color=(0,0,1,0.4))
    Axis.quiver(0, 0, 0, RotationMatrix[0,0], RotationMatrix[0,1], RotationMatrix[0,1], color=(1,0,0))
    Axis.quiver(0, 0, 0, RotationMatrix[1,0], RotationMatrix[1,1], RotationMatrix[1,2], color=(0,1,0))
    Axis.quiver(0, 0, 0, RotationMatrix[2,0], RotationMatrix[2,1], RotationMatrix[2,2], color=(0,0,1))
    # Make the panes transparent
    Axis.xaxis.set_pane_color(FigColor3D)
    Axis.yaxis.set_pane_color(FigColor3D)
    Axis.zaxis.set_pane_color(FigColor3D)
    # Make the grid lines transparent
    Axis.xaxis._axinfo["grid"]['color'] = FigColor3D
    Axis.yaxis._axinfo["grid"]['color'] = FigColor3D
    Axis.zaxis._axinfo["grid"]['color'] = FigColor3D
    # Modify ticks
    MinX, MaxX = -1, 1
    MinY, MaxY = -1, 1
    MinZ, MaxZ = -1, 1
    Axis.set_xlim([MinX, MaxX])
    Axis.set_ylim([MinY, MaxY])
    Axis.set_zlim([MinZ, MaxZ])
    Axis.set_xticks([MinX, 0, MaxX])
    Axis.set_yticks([MinY, 0, MaxY])
    Axis.set_zticks([MinZ, 0, MaxZ])
    Axis.xaxis.set_ticklabels([MinX, 0, MaxX])
    Axis.yaxis.set_ticklabels([MinY, 0, MaxY])
    Axis.zaxis.set_ticklabels([MinZ, 0, MaxZ])
    Axis.set_xlabel('X')
    Axis.set_ylabel('Y')
    Axis.set_zlabel('Z')
    plt.show()

    # Define transform
    print('Image rotation to align custom plane...')
    Transform = sitk.AffineTransform(3)
    Transform.SetMatrix(RotationMatrix.ravel())
    Transform.SetCenter(RotationCenter)

    # Resample image
    Resampler = sitk.ResampleImageFilter()
    Resampler.SetReferenceImage(Scan)
    Resampler.SetTransform(Transform.GetInverse())
    Resampled_Scan = Resampler.Execute(Scan)

    # Plot rotated sample and corresponding points of the plane
    R_X1 = Center + np.dot(RotationMatrix,X1 - Center)
    R_X2 = Center + np.dot(RotationMatrix,X2 - Center)
    R_X3 = Center + np.dot(RotationMatrix,X3 - Center)

    # Resampled_Array = sitk.GetArrayFromImage(Resampled_Scan)
    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Axes.set_title('Scan plane', color=(0,0,0))
    # Axes.imshow(Resampled_Array[:,int(round(R_X1[1])),:], cmap='bone')
    # Axes.plot([R_X1[0],R_X2[0]],[R_X1[2],R_X2[2]],marker='o',color=(1,0,0))
    # plt.show()


    # Compute vector normal to neck and align with z axis
    print('Align neck axis with Z direction ...')
    N = (R_X1 - R_X2) / np.linalg.norm(R_X1 - R_X2)

    Alpha = np.arccos(N.dot(np.array([0,0,1])))
    R = np.array([[np.cos(-Alpha), 0, -np.sin(-Alpha)],
                  [0, 1, 0],
                  [np.sin(-Alpha), 0, np.cos(-Alpha)]])

    Transform.SetMatrix(R.ravel())

    # Again resample image and plot
    Resampler = sitk.ResampleImageFilter()
    Resampler.SetReferenceImage(Resampled_Scan)
    Resampler.SetTransform(Transform.GetInverse())
    NeckAligned = Resampler.Execute(Resampled_Scan)
    NeckAligned_Array = sitk.GetArrayFromImage(NeckAligned)

    A_X1 = Center + np.dot(R,R_X1 - Center)
    A_X2 = Center + np.dot(R,R_X2 - Center)
    A_X3 = Center + np.dot(R,R_X3 - Center)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    Axes.set_title('Neck aligned in Z direction', color=(0,0,0))
    Axes.imshow(NeckAligned_Array[:,int(round(R_X1[1])),:], cmap='bone')
    Axes.plot([A_X1[0],A_X2[0]],[A_X1[2],A_X2[2]],marker='o',color=(1,0,0))
    plt.show()

    # Compute neck center for extraction
    A_X12 = (A_X1[2] + A_X2[2]) / 2
    Slice = NeckAligned_Array[int(round(A_X12)),:,:]

    OtsuFilter.SetNumberOfThresholds(2)
    OtsuFilter.Execute(sitk.GetImageFromArray(Slice))
    OtsuThresholds = OtsuFilter.GetThresholds()

    BinArray = np.zeros(Slice.shape)
    BinArray[Slice > OtsuThresholds[1]] = 1

    Radius = 20
    PadArray = np.pad(BinArray, pad_width=Radius + 1)
    Disk = morphology.disk(Radius)
    DilatedArray = morphology.binary_dilation(PadArray, Disk)
    ErodedArray = morphology.binary_erosion(DilatedArray, Disk)
    UnPaddedArray = ErodedArray[Radius + 1:-Radius, Radius + 1:-Radius]

    RegionProperties = measure.regionprops(UnPaddedArray * 1)[0]
    Y0, X0 = RegionProperties.centroid
    Length = 45 / np.array(NeckAligned.GetSpacing())[0]
    XSquare = np.array([X0 - Length * 0.5,
                        X0 + Length * 0.5,
                        X0 + Length * 0.5,
                        X0 - Length * 0.5,
                        X0 - Length * 0.5])
    YSquare = np.array([Y0 - Length * 0.5,
                        Y0 - Length * 0.5,
                        Y0 + Length * 0.5,
                        Y0 + Length * 0.5,
                        Y0 - Length * 0.5])

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Axes.imshow(UnPaddedArray, cmap='bone')
    # Axes.imshow(Slice,cmap='bone',alpha=0.5)
    # Axes.plot(X0,Y0,marker='x',color=(0,1,0))
    # Axes.plot(XSquare,YSquare,color=(1,0,0))
    # Axes.plot()
    # plt.show()


    # Extract Neck
    print('Extract neck data ...')
    X_Start = int(round(X0 - Length * 0.5))
    X_Stop = int(round(X0 + Length * 0.5))
    Y_Start = int(round(Y0 - Length * 0.5))
    Y_Stop = int(round(Y0 + Length * 0.5))
    Z_Start = int(round(A_X2[2]))
    Z_Stop = int(round(A_X1[2]))

    Neck_Array = NeckAligned_Array[Z_Start:Z_Stop,Y_Start:Y_Stop,X_Start:X_Stop]
    Neck = sitk.GetImageFromArray(Neck_Array)
    Neck.SetSpacing(NeckAligned.GetSpacing())

    return Neck

if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('id', help='Set the input file directory (required)', type=str)
    Parser.add_argument('Proximal', help='Set the file name of proximal uCT scan (required)', type=str)
    Parser.add_argument('od', help='Set output file directory (required)', type=str)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)