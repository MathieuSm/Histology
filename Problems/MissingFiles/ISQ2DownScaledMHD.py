#!/usr/bin/env python3

import argparse
import ISQReader
import Downscale_Image
import SimpleITK as sitk

Version = '01'

# Define the script description
Description = """
    This script read ISQ files, downscales them and write corresponding MHDs.
    Meant to be run locally from the MissingFiles folder

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: January 2022
    """


def Main(Arguments):

    # Get Arguments attributes
    File = Arguments.File
    Factor = Arguments.Factor

    print('\n')
    print('ISQ file name: ' + File)
    print('Downscaling with a factor %.2f' % Factor)
    print('\n')

    ImageArray, AdditionalData = ISQReader.Main(Arguments)

    SitkImage = sitk.GetImageFromArray(ImageArray)
    SitkImage.SetOrigin(AdditionalData['Offset'])
    SitkImage.SetSpacing(AdditionalData['ElementSpacing'])
    SitkImage.SetDirection(AdditionalData['TransformMatrix'])

    DownScaled_Image = Downscale_Image.Main(SitkImage=SitkImage, Factor=Factor)
    AdditionalData['ElementSpacing'] = DownScaled_Image.GetSpacing()
    AdditionalData['Offset'] = DownScaled_Image.GetOrigin()

    DownScaled_ImageArray = sitk.GetArrayFromImage(DownScaled_Image)

    ISQReader.WriteMHD(DownScaled_ImageArray, AdditionalData, File[:-4], PixelType='short')
    print('\nDone!\n')


if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add required arguments
    Parser.add_argument('File', help='ISQ file (required)', type=str)

    # Add long and short optional arguments
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--BMD', default=False, help='Convert gray values to BMD (bool) !!! Depends on voltage, current and time !!!', type=bool)
    Parser.add_argument('--Echo', default=True, help='Print out current operation and results (bool)', type=bool)
    Parser.add_argument('--Factor', default=8., help='Set downscaling factor (float)', type=float)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)
