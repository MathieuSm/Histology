import os
from pathlib import Path

MainDirectory = Path.cwd()
FilesDirectory = MainDirectory / '../08_uCT/Neck'
ScriptsDirectory = MainDirectory / 'Scripts/CuttingLines'

Files = [File for File in os.listdir(FilesDirectory) if File.endswith('.mhd')]
Files.sort()

SubmitFile = open(str(ScriptsDirectory / 'SubmitCuttingLines.py'),'w')
Command = 'os.system(\'python  Scripts/CuttingLines/DrawCuttingLines.py '

SubmitFile.write('#!/usr/bin/env python3\n\nimport os\n\n')

for File in Files:
    Line = Command + File + '\')\n'
    SubmitFile.write(Line)

SubmitFile.close()