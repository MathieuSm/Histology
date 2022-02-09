#!/usr/bin/env python3

import argparse
import time
import numpy as np
import struct
import os

Version = '01'

# Define the script description
Description = """
    This script read an ISQ file from Scanco and return values in a numpy array.
    
    Adapted from https://github.com/mdoube/BoneJ/blob/master/src/org/bonej/io/ISQReader.java
    
    Little endian byte order (the least significant bit occupies the lowest memory position.
     00   char    check[16];              // CTDATA-HEADER_V1
     16   int     data_type;
     20   int     nr_of_bytes;
     24   int     nr_of_blocks;
     28   int     patient_index;          //p.skip(28);
     32   int     scanner_id;				//p.skip(32);
     36   int     creation_date[2];		//P.skip(36);
     44   int     dimx_p;					//p.skip(44);
     48   int     dimy_p;
     52   int     dimz_p;
     56   int     dimx_um;				//p.skip(56);
     60   int     dimy_um;
     64   int     dimz_um;
     68   int     slice_thickness_um;		//p.skip(68);
     72   int     slice_increment_um;		//p.skip(72);
     76   int     slice_1_pos_um;
     80   int     min_data_value;
     84   int     max_data_value;
     88   int     mu_scaling;             //p.skip(88);  /* p(x,y,z)/mu_scaling = value [1/cm]
     92	int     nr_of_samples;
     96	int     nr_of_projections;
     100  int     scandist_um;
     104  int     scanner_type;
     108  int     sampletime_us;
     112  int     index_measurement;
     116  int     site;                   //coded value
     120  int     reference_line_um;
     124  int     recon_alg;              //coded value
     128  char    name[40]; 		 		//p.skip(128);
     168  int     energy;        /* V     //p.skip(168);
     172  int     intensity;     /* uA    //p.skip(172);
     ...
     508 int     data_offset;     /* in 512-byte-blocks  //p.skip(508);
     * So the first 16 bytes are a string 'CTDATA-HEADER_V1', used to identify
     * the type of data. The 'int' are all 4-byte integers.
     *
     * dimx_p is the dimension in pixels, dimx_um the dimensions in micrometer
     *
     * So dimx_p is at byte-offset 40, then dimy_p at 44, dimz_p (=number of
     * slices) at 48.
     *
     * The microCT calculates so called 'x-ray linear attenuation' values. These
     * (float) values are scaled with 'mu_scaling' (see header, e.g. 4096) to
     * get to the signed 2-byte integers values that we save in the .isq file.
     *
     * e.g. Pixel value 8192 corresponds to lin. att. coeff. of 2.0 [1/cm]
     * (8192/4096)
     *
     * Following to the headers is the data part. It is in 2-byte short integers
     * (signed) and starts from the top-left pixel of slice 1 to the left, then
     * the next line follows, until the last pixel of the last sclice in the
     * lower right.

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: January 2022
    """


def Main(Arguments):

    """
    Read Isq File (Scanco)
    """

    print('\nRead ISQ file...')

    File = Arguments.File
    BMD = Arguments.BMD
    Echo = Arguments.Echo

    Time1 = time.time()

    try:
        f = open(File, 'rb')
    except IOError:
        print("\n **ERROR**: ISQReader: intput file ' % s' not found!\n\n" % File)
        print('\n E N D E D  with ERRORS \n\n')

    for Index in np.arange(0, 200, 4):
        f.seek(Index)
        print('Index %s :          %s' % (Index, struct.unpack('i', f.read(4))[0]))
        f.seek(Index)
        try:
            print('Index %s :          %s' % (Index, struct.unpack('c', f.read(4))[0]))
        except:
            print('')

    f.seek(32)
    CT_ID = struct.unpack('i', f.read(4))[0]
    print('scanner ID:                 ', CT_ID)

    if CT_ID != 6020:
        print('!!! unknown muCT -> no Slope and Intercept known !!!')

    f.seek(28)
    #    sample_nb = struct.unpack('i', f.read(4))[0]

    f.seek(108)
    Scanning_time = struct.unpack('i', f.read(4))[0] / 1000
    print('Scanning time in ms:         ', Scanning_time)

    f.seek(168)
    Energy = struct.unpack('i', f.read(4))[0] / 1000.
    print('Energy in keV:              ', Energy)

    f.seek(172)
    Current = struct.unpack('i', f.read(4))[0]
    print('Current in muA:             ', Current)

    f.seek(44)
    X_pixel = struct.unpack('i', f.read(4))[0]
    print('Nb X pixel:                 ', X_pixel)

    f.seek(48)
    Y_pixel = struct.unpack('i', f.read(4))[0]
    print('Nb Y pixel:                 ', Y_pixel)

    f.seek(52)
    Z_pixel = struct.unpack('i', f.read(4))[0]
    print('Nb Z pixel:                 ', Z_pixel)

    f.seek(56)
    Res_General_X = struct.unpack('i', f.read(4))[0]
    print('Resolution general X in mu: ', Res_General_X)

    f.seek(60)
    Res_General_Y = struct.unpack('i', f.read(4))[0]
    print('Resolution general Y in mu: ', Res_General_Y)

    f.seek(64)
    Res_General_Z = struct.unpack('i', f.read(4))[0]
    print('Resolution general Z in mu: ', Res_General_Z)

    Res_X = Res_General_X / float(X_pixel)
    print('Pixel resolution X in mu:    %.2f' % Res_X)

    Res_Y = Res_General_Y / float(Y_pixel)
    print('Pixel resolution Y in mu:    %.2f' % Res_Y)

    Res_Z = Res_General_Z / float(Z_pixel)
    print('Pixel resolution Z in mu:    %.2f' % Res_Z)

    Header_Txt = ['scanner ID:                 %s' % CT_ID,
                  'scaning time in ms:         %s' % Scanning_time,
                  'scaning time in ms:         %s' % Scanning_time,
                  'Energy in keV:              %s' % Energy,
                  'Current in muA:             %s' % Current,
                  'nb X pixel:                 %s' % X_pixel,
                  'nb Y pixel:                 %s' % Y_pixel,
                  'nb Z pixel:                 %s' % Z_pixel,
                  'resolution general X in mu: %s' % Res_General_X,
                  'resolution general Y in mu: %s' % Res_General_Y,
                  'resolution general Z in mu: %s' % Res_General_Z,
                  'pixel resolution X in mu:   %.2f' % Res_X,
                  'pixel resolution Y in mu:   %.2f' % Res_Y,
                  'pixel resolution Z in mu:   %.2f' % Res_Z]
    #    np.savetxt(inFileName.split('.')[0]+'.txt', Header_Txt)

    Write_File = open(File.split('.')[0] + '_info.txt', 'w')
    for Item in Header_Txt:
        Write_File.write("%s\n" % Item)
    Write_File.close()

    f.seek(44)
    Header = np.zeros(6)
    for i in range(0, 6):
        Header[i] = struct.unpack('i', f.read(4))[0]
    print(Header)

    ElementSpacing = [Header[3] / Header[0] / 1000, Header[4] / Header[1] / 1000, Header[5] / Header[2] / 1000]
    f.seek(508)

    HeaderSize = 512 * (1 + struct.unpack('i', f.read(4))[0])
    f.seek(HeaderSize)


    VoxelModel = np.fromfile(f, dtype='i2')
    # VoxelModel = np.fromfile(f, dtype=np.float)

    NDim = [int(Header[0]), int(Header[1]), int(Header[2])]
    LDim = [float(ElementSpacing[0]), float(ElementSpacing[1]), float(ElementSpacing[2])]
    AdditionalData = {'ElementSpacing': LDim,
                      'DimSize': NDim,
                      'HeaderSize': HeaderSize,
                      'TransformMatrix': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      'CenterOfRotation': [0.0, 0.0, 0.0],
                      'Offset': [0.0, 0.0, 0.0],
                      'AnatomicalOrientation': 'LPS',
                      'ElementType': 'int16',
                      'ElementDataFile': File}

    Time2 = time.time()
    print('                          t1')
    if Echo:
        print('     -> read    data in     : %8.2f sec\n' % (Time2 - Time1))

    Time3 = time.time()

    try:
        VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
        f.close()
        del f

    except:
        # if the length does not fit the dimensions (len(VoxelModel) != NDim[2] * NDim[1] * NDim[0]),
        # add an offset with seek to reshape the image -> actualise length, delta *2 = seek

        Offset = (len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))
        f.seek(0)
        VoxelModel = np.fromfile(f, dtype='i2')

        print('len(VoxelModel) = ', len(VoxelModel))
        print('Should be ', (NDim[2] * NDim[1] * NDim[0]))
        print('Delta:', len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))

        f.seek((len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0])) * 2)
        VoxelModel = np.fromfile(f, dtype='i2')
        f.close()
        del f

        VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
        # the image is flipped by the Offset --> change the order to obtain the continuous image:
        VoxelModel = np.c_[VoxelModel[:, :, -Offset:-1], VoxelModel[:, :, :(VoxelModel.shape[2] - Offset)]]

    if Echo:
        print('     -> reshape data in     : %8.2f sec\n' % (Time3 - Time2))

    if CT_ID == 6020 and BMD is True:
        # BE CAREFULL, THIS IS FOR BMD CONVERSION:
        print('muCT 100 of ISTB detected, IS IT CORRECT?')
        Slope = 369.154  # ! ATTENTION, dependent on voltage, Current and time!!!
        Intercept = -191.56
        try:
            VoxelModel = VoxelModel.astype('i4')
            VoxelModel *= Slope
            VoxelModel += Intercept
        except:
            print('\n********* memory not sufficient for BMD values ************\n')
    # Density: Slope                         3.69153992e+02
    # Density: Intercept                    -1.91559998e+02
    #         Slope = 1.2703
    #         Intercept = -84.7
    #                       (double(data)./Scaling)*Slope + densityIntercept
    #         VoxelModel = VoxelModel * Slope + Intercept
    # return VoxelModel, AdditionalData
    # print('before sitk get image')
    # voxelModel_sitk = sitk.GetImageFromArray(VoxelModel)  # .transpose(2, 1, 0))
    # del VoxelModel
    # print('\n LDim:', LDim)
    # voxelModel_sitk.SetSpacing(LDim)
    # return voxelModel_sitk, AdditionalData

    print('Done!\n')

    return VoxelModel, AdditionalData

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

def WriteMHD(ImageArray, AdditionalData, FileName, Path=None, PixelType='uint'):

    print('\nWrite MHD...')
    Spacing = AdditionalData['ElementSpacing']
    Offset = AdditionalData['Offset']

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

    if not Path:
        Path = os.getcwd()

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

    print('Done!\n')

    return


if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add required arguments
    Parser.add_argument('File', help='ISQ file (required)', type=str)

    # Add long and short optional arguments
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('--BMD', default=False, help='Convert gray values to BMD (bool), !!! Depends on voltage, current and time !!!', type=bool)
    Parser.add_argument('--Echo', default=True, help='Print out current operation and results (bool)', type=bool)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)