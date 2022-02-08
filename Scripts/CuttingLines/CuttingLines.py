#!/usr/bin/env python3

import argparse
import FemurOrientation
import SampleAlignment

Version = '01'

# Define the script description
Description = """
    This script runs the analysis of sample cut for histology in the curse of the FEMHALS project.
    Meant to be run locally from the CuttingLines folder
    
    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern
    
    Date: January 2022
    """

def Main(Arguments, FigColor=(0.17, 0.18, 0.22)):

    # Get Arguments attributes
    Proximal = Arguments.Proximal
    Sample = Arguments.Sample
    Angle = Arguments.Angle

    print('\n')
    print('Proximal file name: ' + Proximal)
    print('Sample file name: ' + Sample)
    print('Angle of cutting lines: %i degrees' % Angle)
    print('\n')

    Neck = FemurOrientation.Main(Arguments, FigColor=FigColor)
    print('\nAligned neck extracted!\n')
    SampleAlignment.Main(Arguments,Neck,FigColor=FigColor)
    print('\nDone!\n')

if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('Proximal', help='Set proximal scan file number (required)', type=str)
    Parser.add_argument('Sample', help='Set slice (sample) scan file number (required)', type=str)
    Parser.add_argument('-a', '--Angle', help='Set angle of the cutting lines in degrees', type=int, default=60)

    # Define paths
    InputDirectory = r'C:\Users\mathi\OneDrive\Documents\PhD\08_uCT'
    OutputDirectory = r'C:\Users\mathi\OneDrive\Documents\PhD\06_Histology\Cutting Lines'
    Parser.add_argument('-id', help='Set input directory', type=str, default=InputDirectory)
    Parser.add_argument('-od', help='Set output directory', type=str, default=OutputDirectory)


    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments,FigColor=(1,1,1))






#
#
# ScanSlice = sitk.Slice(Scan,(0,int(np.floor(Ny/2)),0),(Nx,int(np.floor(Ny/2)+1),Nz))
# Array = sitk.GetArrayFromImage(ScanSlice)
# Array = Array[:,0,:]
#
# Figure, Axes = plt.subplots(1,1,figsize=(4.5,5.5))
# Axes.imshow(Array,cmap='bone')
# plt.show()
#
#
# GaussFilter = sitk.DiscreteGaussianImageFilter()
# GaussFilter.SetVariance((1,1,1))
# FilteredScan = GaussFilter.Execute(Scan)
# FilteredArray = sitk.GetArrayFromImage(FilteredScan)
#
# Figure, Axes = plt.subplots(1,1,figsize=(4.5,5.5))
# Axes.imshow(FilteredArray[:,int(Ny/2),:],cmap='bone')
# plt.show()
#
# OtsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
# OtsuFilter.SetNumberOfThresholds(2)
# OtsuFilter.Execute(FilteredScan)
# OtsuThresholds = OtsuFilter.GetThresholds()
#
# BinArray = np.zeros(FilteredArray.shape)
# F1 = FilteredArray > OtsuThresholds[0]
# F2 = FilteredArray < OtsuThresholds[1]
# BinArray[F1 & F2] = 1
# BinArray[FilteredArray > OtsuThresholds[1]] = 2
#
# Figure, Axes = plt.subplots(1,1,figsize=(4.5,5.5))
# Axes.imshow(BinArray[:,int(Ny/2),:],cmap='binary')
# plt.show()
#
# Labels = measure.label(BinArray,background=2)
# np.unique(Labels)