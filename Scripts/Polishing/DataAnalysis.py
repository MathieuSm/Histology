import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

desired_width = 320
pd.set_option('display.width', desired_width)

CurrentDirectory = Path.cwd()
DataPath = CurrentDirectory / 'Tests/Polishing/Tests.csv'

Data = pd.read_csv(DataPath)
Data['Delta [um]'] = Data['Initial Max Thickness [um]'] - Data['Final Max Thickness [um]']
Data['Distance [rev]'] = Data['Test time [min]'] * Data['Rotation velocity [rpm]']
Data['Grinding ratio [um/rev]'] = Data['Delta [um]'] / Data['Distance [rev]']

Samples = Data['Samples'].unique()
Sides = Data['Side'].unique()

Colors = [(0,0,0),(1,0,0),(1,0,1),(0,0,1),(0,1,1),(0,1,0)]

Figure, Axes = plt.subplots(1,1)
i = 0
for Sample in Samples:
    for Side in Sides:
        Filter1 = Data['Samples'] == Sample
        Filter2 = Data['Side'] == Side
        FilteredData = Data[Filter1 & Filter2]
        Axes.plot(FilteredData['Test Run'], FilteredData['Grinding ratio [um/rev]'],
                  color=Colors[i], label=Sample + ' ' + Side)
        i += 1
Axes.set_xticks([1,2,3,4,5])
Axes.set_xlabel('Test Run n° [-]')
Axes.set_ylabel('Grinding ratio [$\mu$m / rev]')
plt.legend()
plt.show()



Healthy_LMM = smf.mixedlm("LogSxy ~ Sii + Sij + Sjj + LogBVTV + Logmxy - 1",
                         data=HealthySystem, groups=HealthySystem['Scan'],
                         vc_formula={"IF": "IF-1"}).fit(reml=True)

