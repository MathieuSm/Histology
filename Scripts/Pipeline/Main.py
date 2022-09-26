#!/usr/bin/env python3

import os
import time
import joblib
import argparse
import pandas as pd
from pathlib import Path

Version = '01'

# Define the script description
Description = """
    This script runs the analysis of cement line densities of the test samples in the curse
    of the FEXHIP project.
    
    It uses the random forest classification trained with manually segmented picture for the
    segmentation of cement lines. 3 regions of interest (ROI) of 500 um side length are rand-
    omly selected on each picture. Then, statistical comparisons between superior and infer-
    ior side are performed.
    
    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern
    
    Date: September 2022
    """


# For testing purpose
class ArgumentsClass:

    def __init__(self):
        self.data = str(Path.cwd() / 'Tests\Osteons\Sensitivity')
Arguments = ArgumentsClass()

def Main(Arguments):

    # List pictures
    DataDirectory = Arguments.data
    Pictures = [P for P in os.listdir(DataDirectory) if P.endswith('Seg.jpg')]

    # Build data frame
    Data = pd.DataFrame()
    for Index, Name in enumerate(Pictures):
        Data.loc[Index, 'DonorID'] = Name[:3]
        Data.loc[Index, 'Side'] = Name[3]
        Data.loc[Index, 'Site'] = Name[4]

    # Perform segmentation
    Classifier = joblib.load('RFC.joblib')
    Segmented = {}
    for Picture in

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
    DataDirectory = str(Path.cwd() / 'Tests\Osteons\Sensitivity')
    Parser.add_argument('-data', help='Set data directory', type=str, default=DataDirectory)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)
