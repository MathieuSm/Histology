#!/usr/bin/env python3

import argparse
import os
import numpy as np
import SimpleITK as sitk

Version = '01'

# Define the script description
Description = """
    This script downscales SimpleITK image

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: January 2022
    """


def Main(Arguments=None, SitkImage=None, Factor=1.):

    """
    Downscale image (SimpleITK)
    Adapted from https://stackoverflow.com/questions/48065117/simpleitk-resize-images
    """

    print('Downscale image...')

    if Arguments:
        File = Arguments.File
        Factor = Arguments.Factor

        if Arguments.d:
            Path = Arguments.d
        else:
            Path = os.getcwd()

        SitkImage = sitk.ReadImage(os.path.join(Path,File), sitk.sitkInt32)

    elif not SitkImage:
        print('Argument SITK image is missing!')

    # Get reference image data
    print('Analyze initial image')
    Dimension = SitkImage.GetDimension()
    Reference_Physical_Size = np.zeros(SitkImage.GetDimension())
    Reference_Physical_Size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(SitkImage.GetSize(), SitkImage.GetSpacing(), Reference_Physical_Size)]

    Reference_Origin = SitkImage.GetOrigin()
    Reference_Direction = SitkImage.GetDirection()

    Reference_Size = [round(sz / Factor) for sz in SitkImage.GetSize()]
    Reference_Spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(Reference_Size, Reference_Physical_Size)]

    # Generate reference coordinate system
    print('Generate reference coordinate system')
    Reference_Image = sitk.Image(Reference_Size, SitkImage.GetPixelIDValue())
    Reference_Image.SetOrigin(Reference_Origin)
    Reference_Image.SetSpacing(Reference_Spacing)
    Reference_Image.SetDirection(Reference_Direction)

    Reference_Center = np.array(
        Reference_Image.TransformContinuousIndexToPhysicalPoint(np.array(Reference_Image.GetSize()) / 2.0))

    # Set centering transform parameters for downscaled image
    print('Perform centering image downscaling')
    Transform = sitk.AffineTransform(Dimension)
    Transform.SetMatrix(SitkImage.GetDirection())
    Transform.SetTranslation(np.array(SitkImage.GetOrigin()) - Reference_Origin)

    Centering_Transform = sitk.TranslationTransform(Dimension)
    Image_Center = np.array(SitkImage.TransformContinuousIndexToPhysicalPoint(np.array(SitkImage.GetSize()) / 2.0))
    Centering_Transform.SetOffset(np.array(Transform.GetInverse().TransformPoint(Image_Center) - Reference_Center))
    Centered_Transform = sitk.CompositeTransform(Transform)
    Centered_Transform.AddTransform(Centering_Transform)

    # Perform downscaling and return new image
    DownScaled_Image = sitk.Resample(SitkImage, Reference_Image, Centered_Transform, sitk.sitkLinear, 0.0)

    print('Done!\n')

    return DownScaled_Image


if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add required arguments
    Parser.add_argument('File', help='Image file to downscale (required)', type=str)
    Parser.add_argument('Factor', help='Downscaling factor (required)', type=float)

    # Add long and short optional arguments
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('-d', '--Directory', help='Set file directory', type=str)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)