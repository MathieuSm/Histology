import os
import pandas as pd

MainDirectory = r'C:\Users\mathi\OneDrive\Documents\PhD'
FileDirectory = os.path.join(MainDirectory, r'06_Histology\Scripts\CuttingLines')

Data = pd.read_excel(os.path.join(MainDirectory,'Samples.xlsx'),engine='openpyxl')
Data = Data.loc[Data['uCT Neck ID'].notna()]

Data2Analyze = pd.DataFrame()
for Index in Data.index:
    Proximal = Data.loc[Index,'uCT Proximal ID'] + '_reso_0.274_DOWNSCALED.mhd'
    File = os.path.join(MainDirectory,'08_uCT\Proximal',Proximal)
    if os.path.isfile(File):
        Neck = Data.loc[Index,'uCT Neck ID'] + '.mhd'
        Data2Append = {'Proximal':Proximal,'Neck':Neck}
        Data2Analyze = Data2Analyze.append(Data2Append,ignore_index=True)

File = open(os.path.join(FileDirectory, 'SubmitCuttingLines.py'),'w')
Command = 'os.system(\'python CuttingLines.py '

File.write('#!/usr/bin/env python3\n\nimport os\n\n')

for Line in Data2Analyze.index:

    LineData = Data2Analyze.loc[Line]
    Parameters = LineData['Proximal'] + ' ' + LineData['Neck'] + '\')\n'

    File.write(Command + Parameters)

File.close()