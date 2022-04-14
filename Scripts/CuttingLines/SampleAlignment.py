#!/usr/bin/env python3

import argparse
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from skimage import morphology, measure

plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    This script runs the interactive registration of sample and draw the cutting lines according to a given angle

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: January 2022
    """

# For testing purposes
# class Arguments:
#     pass
# Arguments.id = '/home/mathieu/Documents/PhD/08_uCT/'
# Arguments.Neck = 'C0001094_reso_0.274_DOWNSCALED.mhd'
# Arguments.Sample = 'C0002074.mhd'
# Arguments.Angle = 60
# Arguments.od = '/home/mathieu/Documents/PhD/06_Histology/Cutting Lines/'

def Main(Arguments, Neck=None, FigColor=(0.17, 0.18, 0.22)):
    print('\nStart sample alignment ...\n')

    # Get argument attributes
    print('Read arguments ...')
    MHDDirectory = Arguments.id
    File = Arguments.Sample
    Angle = Arguments.Angle
    OutputDirectory = Arguments.od

    print('Load file ...')
    if Neck:
        Scan = Neck
    else:
        Scan = sitk.ReadImage(os.path.join(MHDDirectory,'Proximal', Arguments.Neck))

    # Read corresponding sample uct
    Sample_Scan = sitk.ReadImage(os.path.join(MHDDirectory,'Neck', File))

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Show = Axes.imshow(sitk.GetArrayFromImage(Sample_Scan)[-1, :, :], cmap='bone', alpha=1)
    # Axes.axis('off')
    # plt.show()

    print('Resample scan for similar resolution ...')
    Resampler = sitk.ResampleImageFilter()
    Resampler.SetReferenceImage(Sample_Scan)
    Resampler.SetOutputSpacing(Scan.GetSpacing())
    Resampled_Sample = Resampler.Execute(Sample_Scan)

    Sample_Array = sitk.GetArrayFromImage(Resampled_Sample)
    Scan_Array = sitk.GetArrayFromImage(Scan)
    ZPos_Start = int(Scan_Array.shape[0] / 2)

    # Mid_ZPos = int(Sample_Array.shape[0] / 2)
    #
    # def MoveZPlane(val):
    #
    #     Position = int(val)
    #     Plane = Sample_Array[Position, :, :]
    #     Show.set_data(Plane)
    #     Figure.canvas.draw_idle()
    #
    # Figure, Axes = plt.subplots(1, 1, figsize=(10, 8), dpi=100)
    # Show = Axes.imshow(Sample_Array[Mid_ZPos,:,:], cmap='bone', alpha=1)
    # Axes.set_xlabel('X')
    # Axes.set_ylabel('Y ', rotation=0)
    #
    # SliderAxis = plt.axes([0.25, 0.05, 0.50, 0.03])
    # Mid_Position_Slider = Slider(SliderAxis, 'Plane Position', 0, Sample_Array.shape[0], valinit=Mid_ZPos, valstep=1, color=(0,0,0))
    #
    # plt.subplots_adjust(bottom=0.2)
    # Mid_Position_Slider.on_changed(MoveZPlane)
    # plt.show()

    print('Crop and pad array to correspond to neck uct ...')

    # Crop
    CropX = np.ceil(Sample_Scan.GetSize()[0] * Sample_Scan.GetSpacing()[0] / Scan.GetSpacing()[0])
    CropY = np.ceil(Sample_Scan.GetSize()[1] * Sample_Scan.GetSpacing()[1] / Scan.GetSpacing()[1])
    CropZ = np.ceil(Sample_Scan.GetSize()[2] * Sample_Scan.GetSpacing()[2] / Scan.GetSpacing()[2])
    Sample_Crop = Sample_Array[:int(CropZ),:int(CropY),:int(CropX)]

    # Pad
    PadWidth = np.array(Scan_Array.shape) - np.array(Sample_Crop.shape)
    PadWidth = np.repeat(np.round(PadWidth/2),2).astype('int')
    Sample_Pad = np.pad(Sample_Crop,PadWidth.reshape((3,2)))


    # If file exist, read point coordinates. Otherwise, given start
    TxtFileName = os.path.join(OutputDirectory,File[:8]+'_R.txt')

    if os.path.isfile(TxtFileName):
        Rotation, X_Translation, Y_Translation, Z_Translation = np.loadtxt(TxtFileName).astype('int')
    else:
        Rotation, X_Translation, Y_Translation, Z_Translation = [0, 0, 0, ZPos_Start]

    # Compute corresponding sample position
    R_Rotation = Rotation / 180 * np.pi
    R = np.array([[np.cos(-R_Rotation), np.sin(-R_Rotation), 0],
                  [-np.sin(-R_Rotation), np.cos(-R_Rotation), 0],
                  [0, 0, 1]])
    RotationCenter = np.array(Sample_Pad.shape) / 2
    Transform = sitk.AffineTransform(3)
    Transform.SetMatrix(R.ravel())
    Transform.SetCenter(RotationCenter[::-1])
    Transform.SetTranslation((int(X_Translation), int(Y_Translation), 0))

    # Resample image
    Sample_Scan = sitk.GetImageFromArray(Sample_Pad)
    Resampler = sitk.ResampleImageFilter()
    Resampler.SetReferenceImage(Sample_Scan)
    Resampler.SetTransform(Transform.GetInverse())
    Scan_Pad = Resampler.Execute(Sample_Scan)
    Sample_Pad = sitk.GetArrayFromImage(Scan_Pad)

    print('Plot for interactive registration ...')
    ProximalSlice = Sample_Pad[PadWidth[0]+2, :, :]
    DistalSlice = Sample_Pad[-PadWidth[0]-3, :, :]

    # Create custom color map
    import matplotlib as mpl
    C = np.array([[0, 0, 0], [255, 0, 0]])
    ColorMap = mpl.colors.ListedColormap(C / 255.0)

    # Write updating functions
    def MoveZPlane(val):

        Plane1 = Scan_Array[int(val + CropZ), :, :]
        Plane2 = Scan_Array[int(val), :, :]
        Show1.set_data(Plane1)
        Show2.set_data(Plane2)
        Figure.canvas.draw_idle()

    def UpdateSample(val):

        Rotation = int(Rotation_Slider.val) / 180 * np.pi
        X_Translation = int(X_Slide_Slider.val)
        Y_Translation = int(Y_Slide_Slider.val)

        R = np.array([[np.cos(-Rotation), np.sin(-Rotation), 0],
                      [-np.sin(-Rotation), np.cos(-Rotation), 0],
                      [0, 0, 1]])
        RotationCenter = np.array(Sample_Pad.shape) / 2
        Transform = sitk.AffineTransform(3)
        Transform.SetMatrix(R.ravel())
        Transform.SetCenter(RotationCenter[::-1])
        Transform.SetTranslation((int(X_Translation), int(Y_Translation), 0))

        # Resample image
        Resampler = sitk.ResampleImageFilter()
        Resampler.SetReferenceImage(Sample_Scan)
        Resampler.SetTransform(Transform.GetInverse())
        R_Sample_Scan = Resampler.Execute(Sample_Scan)
        R_Sample_Array = sitk.GetArrayFromImage(R_Sample_Scan)

        if Flip_CheckBox.get_status()[0]:
            R_Sample_Array = np.rot90(R_Sample_Array[::-1,:,:],2,axes=(1,2))

        ProximalSlice = R_Sample_Array[PadWidth[0] + 2, :, :]
        DistalSlice = R_Sample_Array[-PadWidth[0] - 3, :, :]

        Plane1 = Scan_Array[int(ZPos_Slider.val + CropZ) - 2, :, :]
        Plane2 = Scan_Array[int(ZPos_Slider.val) + 2, :, :]
        Show1.set_data(Plane1)
        Show2.set_data(Plane2)
        Show_Proximal.set_data(ProximalSlice)
        Show_Distal.set_data(DistalSlice)
        Figure.canvas.draw_idle()

    # Plot
    Figure, Axes = plt.subplots(1, 2, figsize=(8, 5.5), facecolor=FigColor, sharey=True)
    Show1 = Axes[0].imshow(Scan_Array[int(Z_Translation + CropZ - 2),:,:], cmap='bone', alpha=1)
    Show_Proximal = Axes[0].imshow(ProximalSlice, cmap=ColorMap, alpha=0.5)
    Show2 = Axes[1].imshow(Scan_Array[Z_Translation + 2, :, :], cmap='bone', alpha=1)
    Show_Distal = Axes[1].imshow(DistalSlice, cmap=ColorMap, alpha=0.5)
    Axes[0].set_ylabel('Y ', rotation=0)
    Axes[0].set_xlabel('X')
    Axes[1].set_xlabel('X')
    Axes[0].set_title('Proximal Slice')
    Axes[1].set_title('Distal Slice')

    SliderAxis = plt.axes([0.25, 0.025, 0.55, 0.03], facecolor=FigColor)
    ZPos_Slider = Slider(SliderAxis, 'Plane Position', 0, Scan_Array.shape[0], valinit=Z_Translation+2, valstep=1, color=(0,0,0))

    Rotation_Axis = plt.axes([0.25, 0.1, 0.35, 0.03], facecolor=FigColor)
    Rotation_Slider = Slider(Rotation_Axis, 'Rotation', -180, 180, valinit=Rotation, valstep=1, color=(0,0,0))

    X_Slide_Axis = plt.axes([0.25, 0.2, 0.35, 0.03], facecolor=FigColor)
    Limits = int(Sample_Pad.shape[0] / 2)
    X_Slide_Slider = Slider(X_Slide_Axis, 'X Translation', -Limits, Limits, valinit=X_Translation, valstep=1, color=(0,0,0))

    Y_Slide_Axis = plt.axes([0.25, 0.15, 0.35, 0.03], facecolor=FigColor)
    Limits = int(Sample_Pad.shape[1] / 2)
    Y_Slide_Slider = Slider(Y_Slide_Axis, 'Y Translation', -Limits, Limits, valinit=Y_Translation, valstep=1, color=(0,0,0))

    Flip_Axis = plt.axes([0.7, 0.1, 0.2, 0.15], facecolor=FigColor)
    Flip_CheckBox = CheckButtons(Flip_Axis, ['Flip sample'])

    plt.subplots_adjust(bottom=0.35)
    ZPos_Slider.on_changed(MoveZPlane)
    Rotation_Slider.on_changed(UpdateSample)
    X_Slide_Slider.on_changed(UpdateSample)
    Y_Slide_Slider.on_changed(UpdateSample)
    Flip_CheckBox.on_clicked(UpdateSample)
    plt.show()

    # Show aligned sample with cutting lines
    print('Compute slice registration ...')
    Rotation = int(Rotation_Slider.val)
    X_Translation = int(X_Slide_Slider.val)
    Y_Translation = int(Y_Slide_Slider.val)
    Z_Translation = int(ZPos_Slider.val)

    if Flip_CheckBox.get_status()[0]:
        Sample_Pad = np.rot90(Sample_Pad,2,axes=(1,2))[::-1, :, :]

    # Save registration parameters into a txt file
    np.savetxt(TxtFileName, (Rotation, X_Translation, Y_Translation, Z_Translation), delimiter='\t', newline='\n', fmt='%3i')

    # Generate scan from array
    R_Sample_Array = Sample_Pad
    R_Sample_Scan = sitk.GetImageFromArray(Sample_Pad)
    R_Sample_Scan.SetSpacing(Scan_Pad.GetSpacing())
    # Rotation = Rotation / 180 * np.pi
    # R = np.array([[np.cos(-Rotation), np.sin(-Rotation)],
    #               [-np.sin(-Rotation), np.cos(-Rotation)]])
    #
    # # Define transform
    # RotationCenter = np.array(Sample_Pad.shape) / 2
    # Transform = sitk.AffineTransform(2)
    # Transform.SetMatrix(R.ravel())
    # Transform.SetCenter(RotationCenter)
    # Transform.SetTranslation((X_Translation,Y_Translation))
    #
    # # Resample image
    # Resampler = sitk.ResampleImageFilter()
    # Resampler.SetReferenceImage(sitk.GetImageFromArray(Sample_Pad))
    # Resampler.SetTransform(Transform.GetInverse())
    # R_Sample_Scan = Resampler.Execute(sitk.GetImageFromArray(Sample_Pad))
    # R_Sample_Array = sitk.GetArrayFromImage(R_Sample_Scan)

    # Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor=FigColor)
    # Show = Axes.imshow(Scan_Array[Mid_ZPos,:,:], cmap='bone', alpha=1)
    # Show_Sample = Axes.imshow(R_Sample_Array, cmap=ColorMap, alpha=0.5)
    # Axes.set_xlabel('X')
    # Axes.set_ylabel('Y ', rotation=0)
    # plt.show()

    # Binarize image to find area centroid
    print('Binarize using Otsu and fill holes ...')
    OtsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
    OtsuFilter.SetNumberOfThresholds(2)
    OtsuFilter.Execute(R_Sample_Scan)
    OtsuThresholds = OtsuFilter.GetThresholds()

    BinArray = np.zeros(R_Sample_Array.shape)
    # F1 = R_Sample_Array > OtsuThresholds[0]
    # F2 = R_Sample_Array < OtsuThresholds[1]
    # BinArray[F1 & F2] = 1
    BinArray[R_Sample_Array > OtsuThresholds[1]] = 1

    # Figure, Axes = plt.subplots(1,1,figsize=(4.5,5.5))
    # Axes.imshow(BinArray[Z_Translation,:,:],cmap='binary')
    # plt.show()

    # Perform binary dilation and erosion to find center for proximal and distal slices
    Radius = 20
    PadArray = np.pad(BinArray, pad_width=Radius + 1)
    Disk = morphology.disk(Radius)

    for iSlice in range(2):
        if iSlice == 0:
            Slice = PadArray[PadWidth[0] + Radius + 1 + 2, :, :]
            Image = R_Sample_Array[PadWidth[0] + 2, :, :]
        else:
            Slice = PadArray[-(PadWidth[0] + Radius + 1) - 1 - 2,:,:]
            Image = R_Sample_Array[-PadWidth[0] - 2, :, :]

        DilatedArray = morphology.binary_dilation(Slice, Disk)
        ErodedArray = morphology.binary_erosion(DilatedArray, Disk)
        UnPaddedArray = ErodedArray[Radius + 1:-Radius, Radius + 1:-Radius]

        Coord = np.argwhere(Slice[Radius + 1:-Radius, Radius + 1:-Radius] == 1)
        Cg = np.mean(Coord, axis=0)[::-1]

        # Figure, Axes = plt.subplots(1, 1, figsize=(4.5, 5.5), facecolor=FigColor)
        # Axes.imshow(Slice[Radius + 1:-Radius, Radius + 1:-Radius], cmap='bone')
        # Axes.plot(Cg[0], Cg[1], marker='x', color=(0,1,0))
        # Axes.imshow(UnPaddedArray, cmap=ColorMap, alpha=0.5)
        # plt.show()

        # Compute ellipse to find section center
        print('Compute slice geometric center ...')
        RegionProperties = measure.regionprops(UnPaddedArray*1)[0]

        Y0, X0 = RegionProperties.centroid
        R1 = RegionProperties.major_axis_length * 0.5
        R2 = RegionProperties.minor_axis_length * 0.5

        OrientationVector = Cg - np.array([X0,Y0])
        OrientationAngle = np.arctan(-OrientationVector[1]/OrientationVector[0])

        Radians = np.linspace(0, 2 * np.pi, 100)
        Ellipse = np.array([R1 * np.cos(Radians), R2 * np.sin(Radians)])
        R = np.array([[np.cos(OrientationAngle), -np.sin(OrientationAngle)],
                      [np.sin(OrientationAngle), np.cos(OrientationAngle)]])
        Ellipse_R = np.dot(R, Ellipse)

        # Rotate sample according to its main axis
        RotationCenter = np.array(Image.shape) / 2
        Transform = sitk.AffineTransform(2)
        Transform.SetMatrix(R.ravel())
        Transform.SetCenter(RotationCenter[::-1])
        Transform.SetTranslation((0, 0, 0))

        # Resample image
        Resampler = sitk.ResampleImageFilter()
        Resampler.SetReferenceImage(sitk.GetImageFromArray(Image))
        Resampler.SetTransform(Transform.GetInverse())
        R_Image = Resampler.Execute(sitk.GetImageFromArray(Image))
        Image = sitk.GetArrayFromImage(R_Image)

        # Figure, Axes = plt.subplots(1, 1, figsize=FigureSize, dpi=int(DPI), facecolor=FigColor)
        # Axes.imshow(Image2, cmap='bone')
        # Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='Centroid')
        # Axes.plot(X0 + Ellipse_R[0, :], Y0 - Ellipse_R[1, :], color=(0, 1, 0), label='Fitted ellipse')
        # Axes.plot(X0 + X_Line, Y0 + Y_Line, color=(1, 0, 0))
        # Axes.plot(X0 - X_Line, Y0 + Y_Line, color=(1, 0, 0))
        # Axes.plot(X0 + X_Line, Y0 - Y_Line, color=(1, 0, 0))
        # Axes.plot(X0 - X_Line, Y0 - Y_Line, color=(1, 0, 0), label='Cutting lines')
        # Axes.set_xlim([0, R_Sample_Array.shape[2]])
        # Axes.set_ylim([R_Sample_Array.shape[1], 0])
        # Axes.axis('off')
        # plt.subplots_adjust(RealMargins[0], RealMargins[1], RealMargins[2], RealMargins[3])
        # plt.show()

        # Compute cutting lines
        print('Plot cutting lines ...')
        X_Line = np.linspace(0,R_Sample_Array.shape[2],100)
        Y_Line = X_Line * np.tan(Angle/2 * np.pi/180)

        # Print image with correct size
        ImageSize = Image.shape[::-1]
        ImageSpacing = np.array(Scan.GetSpacing())[:2]
        PhysicalSize = ImageSize * ImageSpacing

        Inch = 25.4
        mm = 1 / Inch
        Subsampling = 1
        DPI = np.round(1 / ImageSpacing / Subsampling * Inch)[0]
        Margins = np.array([0., 0.])
        FigureSize = ImageSpacing * ImageSize * mm

        RealMargins = [Margins[0] / 2 * PhysicalSize[0] / (FigureSize[0] * Inch),
                       Margins[1] / 2 * PhysicalSize[1] / (FigureSize[1] * Inch),
                       1 - Margins[0] / 2 * PhysicalSize[0] / (FigureSize[0] * Inch),
                       1 - Margins[1] / 2 * PhysicalSize[1] / (FigureSize[1] * Inch)]

        Figure, Axes = plt.subplots(1,1, figsize=FigureSize, dpi=int(DPI), facecolor=FigColor)
        Axes.imshow(Image, cmap='bone')
        Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='Centroid')
        Axes.plot(X0 + Ellipse_R[0, :], Y0 - Ellipse_R[1, :], color=(0, 1, 0), label='Fitted ellipse')
        Axes.plot(X0 + X_Line, Y0 + Y_Line, color=(1,0,0))
        Axes.plot(X0 - X_Line, Y0 + Y_Line, color=(1, 0, 0))
        Axes.plot(X0 + X_Line, Y0 - Y_Line, color=(1, 0, 0))
        Axes.plot(X0 - X_Line, Y0 - Y_Line, color=(1, 0, 0), label='Cutting lines')
        Axes.set_xlim([0,R_Sample_Array.shape[2]])
        Axes.set_ylim([R_Sample_Array.shape[1],0])
        Axes.axis('off')

        if iSlice == 0:
            plt.subplots_adjust(RealMargins[0], RealMargins[1], RealMargins[2], RealMargins[3])
            plt.savefig(TxtFileName[:-6] + '_Proximal.png', dpi=int(DPI))
            plt.show()

        else:

            plt.subplots_adjust(RealMargins[0], RealMargins[1], RealMargins[2], RealMargins[3])
            plt.savefig(TxtFileName[:-6] + '_Distal.png',dpi=int(DPI))
            plt.show()

if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('id', help='Set the input file directory (required)', type=str)
    Parser.add_argument('Neck', help='Set the file name of proximal uCT neck scan registered (required)', type=str)
    Parser.add_argument('Sample', help='Set slice (sample) scan file number (required)', type=str)
    Parser.add_argument('od', help='Set the output file directory (required)', type=str)
    Parser.add_argument('-a', '--Angle', help='Set angle of the cutting lines in degrees', type=int, default=60)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)