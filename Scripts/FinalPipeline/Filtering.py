#!/usr/bin/env python3

import time
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    This script can be used to perform signal filtering

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: July 2022
    """

def PrintTime(Tic, Toc):
    """
    Print elapsed time in seconds to time in HH:MM:SS format
    :param Tic: Actual time at the beginning of the process
    :param Toc: Actual time at the end of the process
    """

    Delta = Toc - Tic

    Hours = np.floor(Delta / 60 / 60)
    Minutes = np.floor(Delta / 60) - 60 * Hours
    Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

    print('Process executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))
def FFT2D(Image,CutOff,Sharpness,PassType,Plot=False):

    Tic = time.time()
    print('Perform 2D FFT filtering')

    if Plot:
        Figure, Axes = plt.subplots(1, 1)
        Axes.imshow(Image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(Figure)

    # Filter by FFT
    FFT = np.fft.fft2(Image)
    Center = np.fft.fftshift(FFT)

    if Plot:
        Figure, Axes = plt.subplots(1, 1)
        Axes.imshow(np.log(1+np.abs(FFT)), cmap='gray')
        plt.title('Signal FFT')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(Figure)

    # Build filter using sigmoid function
    Half = np.array(FFT.shape) / 2
    Xc, Yc = np.meshgrid(np.arange(-Half[0], Half[0]), np.arange(-Half[1], Half[1]))
    Norm = np.sqrt(Xc ** 2 + Yc ** 2) / np.sqrt(Xc ** 2 + Yc ** 2).max()

    if PassType == 'Low':
        Filter = 1 - 1 / (1 + np.exp(-Sharpness * (Norm - CutOff)))
    elif PassType == 'High':
        Filter = 1 / (1 + np.exp(-Sharpness * (Norm - CutOff)))

    if Plot:
        Figure, Axes = plt.subplots(1, 1)
        Axes.imshow(Filter, cmap='gray')
        plt.title('Filter')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(Figure)

    # Apply filter
    LowPassCenter = Center * Filter
    LowPass = np.fft.ifftshift(LowPassCenter)
    Filtered = np.abs(np.fft.ifft2(LowPass))

    if Plot:
        Figure, Axes = plt.subplots(1, 1)
        Axes.imshow(Filtered, cmap='gray')
        plt.title('Filtered Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(Figure)

    Toc = time.time()
    PrintTime(Tic,Toc)

    return Filtered