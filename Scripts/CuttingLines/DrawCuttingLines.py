#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
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
# Arguments.id = Path.cwd() / '../08_uCT/Neck/'
# Arguments.Sample = 'C0002074.mhd'
# Arguments.Angle = 60
# Arguments.od = Path.cwd() / 'Cutting Lines'

def Main(Arguments, FigColor=(0.17, 0.18, 0.22)):

    # Get argument attributes
    print('Read arguments ...')
    MHDDirectory = Arguments.id
    File = Arguments.Sample
    Angle = Arguments.Angle
    OutputDirectory = Arguments.od

    # Read corresponding sample uct
    Scan = sitk.ReadImage(str(MHDDirectory / File))
    Array = sitk.GetArrayFromImage(Scan)

    # Binarize image to find area centroid
    print('Binarize using Otsu and fill holes ...')
    OtsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
    OtsuFilter.SetNumberOfThresholds(2)
    OtsuFilter.Execute(Scan)
    OtsuThresholds = OtsuFilter.GetThresholds()

    BinArray = np.zeros(Array.shape)
    BinArray[Array > OtsuThresholds[1]] = 1

    # Figure, Axes = plt.subplots(1,1,figsize=(4.5,5.5))
    # Axes.imshow(BinArray[0,:,:],cmap='binary')
    # plt.show()

    # Perform binary dilation and erosion to find center for proximal and distal slices
    Radius = 20
    PadArray = np.pad(BinArray, pad_width=Radius + 1)
    Disk = morphology.disk(Radius)

    # Use distal slice for orientation computation
    Slice = PadArray[Radius + 1, :, :]

    DilatedArray = morphology.binary_dilation(Slice, Disk)
    ErodedArray = morphology.binary_erosion(DilatedArray, Disk)
    UnPaddedArray = ErodedArray[Radius + 1:-Radius, Radius + 1:-Radius]

    Coord = np.argwhere(Slice[Radius + 1:-Radius, Radius + 1:-Radius] == 1)
    Cg = np.mean(Coord, axis=0)[::-1]

    # Create custom color map
    import matplotlib as mpl
    C = np.array([[0, 0, 0], [255, 0, 0]])
    ColorMap = mpl.colors.ListedColormap(C / 255.0)

    Figure, Axes = plt.subplots(1, 1, figsize=(4.5, 5.5), facecolor=FigColor)
    Axes.imshow(Slice[Radius + 1:-Radius, Radius + 1:-Radius], cmap='bone')
    Axes.plot(Cg[0], Cg[1], marker='x', color=(0, 1, 0))
    Axes.imshow(UnPaddedArray, cmap=ColorMap, alpha=0.5)
    plt.show()

    # Compute ellipse to find section center
    print('Compute slice geometric center ...')
    RegionProperties = measure.regionprops(UnPaddedArray * 1)[0]

    Y0, X0 = RegionProperties.centroid
    R1 = RegionProperties.major_axis_length * 0.5
    R2 = RegionProperties.minor_axis_length * 0.5

    OrientationVector = Cg - np.array([X0, Y0])
    OrientationAngle = np.arctan(-OrientationVector[1] / OrientationVector[0])

    Radians = np.linspace(0, 2 * np.pi, 100)
    Ellipse = np.array([R1 * np.cos(Radians), R2 * np.sin(Radians)])
    R = np.array([[np.cos(OrientationAngle), -np.sin(OrientationAngle)],
                  [np.sin(OrientationAngle), np.cos(OrientationAngle)]])

    # Rotate slices and draw lines
    for iSlice in range(2):

        if iSlice == 0:
            Image = Array[0, :, :]
        else:
            Image = Array[-1, :, :]

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

        # Compute cutting lines
        print('Plot cutting lines ...')
        X_Line = np.linspace(0,Array.shape[2],100)
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
        # Axes.plot(X0, Y0, marker='x', color=(0, 0, 1), linestyle='none', markersize=10, mew=2, label='Centroid')
        Axes.plot(X0 + Ellipse[0, :], Y0 - Ellipse[1, :], color=(0, 1, 0), linewidth=0.5, label='Fitted ellipse')
        Axes.plot(X0 + X_Line, Y0 + Y_Line, color=(1,0,0))
        Axes.plot(X0 - X_Line, Y0 + Y_Line, color=(1, 0, 0))
        Axes.plot(X0 + X_Line, Y0 - Y_Line, color=(1, 0, 0))
        Axes.plot(X0 - X_Line, Y0 - Y_Line, color=(1, 0, 0), label='Cutting lines')
        Axes.set_xlim([0,Array.shape[2]])
        Axes.set_ylim([Array.shape[1],0])
        Axes.axis('off')

        if iSlice == 0:
            plt.subplots_adjust(RealMargins[0], RealMargins[1], RealMargins[2], RealMargins[3])
            plt.savefig(str(OutputDirectory / (File[:-4] + '_Distal.png')), dpi=int(DPI))
            plt.show()

        else:

            plt.subplots_adjust(RealMargins[0], RealMargins[1], RealMargins[2], RealMargins[3])
            plt.savefig(str(OutputDirectory / (File[:-4] + '_Proximal.png')), dpi=int(DPI))
            plt.show()

if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('Sample', help='Set slice (sample) scan file number (required)', type=str)
    Parser.add_argument('-a', '--Angle', help='Set angle of the cutting lines in degrees', type=int, default=60)

    # Define paths
    InputDirectory = Path.cwd() / '../08_uCT/Neck/'
    OutputDirectory = Path.cwd() / 'Cutting Lines'
    Parser.add_argument('-id', help='Set input directory', type=str, default=InputDirectory)
    Parser.add_argument('-od', help='Set output directory', type=str, default=OutputDirectory)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments, FigColor=(1, 1, 1))