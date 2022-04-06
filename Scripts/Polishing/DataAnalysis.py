import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

desired_width = 320
pd.set_option('display.width', desired_width)
plt.rcParams['font.size'] = '16'


# Read data
CurrentDirectory = Path.cwd()
DataPath = CurrentDirectory / 'Tests/Polishing/Tests.csv'
Data = pd.read_csv(DataPath)

# Compute delta at each run
Data['Delta [um]'] = Data['Initial Max Thickness [um]'] - Data['Final Max Thickness [um]']

# Generate variable for per sample plot
Samples = Data['Samples'].unique()
Sides = Data['Side'].unique()
Colors = [(0,0,0),(1,0,0),(1,0,1),(0,0,1),(0,1,1),(0,1,0)]

Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
i = 0
for Sample in Samples:
    for Side in Sides:
        Filter1 = Data['Samples'] == Sample
        Filter2 = Data['Side'] == Side
        FilteredData = Data[Filter1 & Filter2]
        Axes.plot(FilteredData['Test Run'], FilteredData['Delta [um]'],
                  color=Colors[i], label=Sample + ' ' + Side)
        i += 1
Axes.set_xticks([1,2,3,4,5])
Axes.set_xlabel('Test Run n° [-]')
Axes.set_ylabel('Thickness $\Delta$ [$\mu$m]')
plt.legend(ncol=2)
plt.show()


# Compute additional data
Data['Distance [rev]'] = Data['Test time [min]'] * Data['Rotation velocity [rpm]']
Data['Grinding ratio [um/rev]'] = Data['Delta [um]'] / Data['Distance [rev]']
Filter1 = Data['Samples'] == '391 L'
Filter2 = Data['Side'] == 'Lateral'
Data['Total Distance [rev]'] = np.repeat(Data[Filter1 & Filter2]['Distance [rev]'].cumsum().values,6)

Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
i = 0
for Sample in Samples:
    for Side in Sides:
        Filter1 = Data['Samples'] == Sample
        Filter2 = Data['Side'] == Side
        FilteredData = Data[Filter1 & Filter2]
        Axes.plot(FilteredData['Total Distance [rev]'], FilteredData['Grinding ratio [um/rev]'],
                  color=Colors[i], label=Sample + ' ' + Side)
        i += 1
Axes.set_xlabel('Cumulative distance [rev]')
Axes.set_ylabel('Grinding ratio [$\mu$m / rev]')
plt.legend(ncol=2)
plt.show()

Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
i = 0
for Sample in Samples:
    for Side in Sides:
        Filter1 = Data['Samples'] == Sample
        Filter2 = Data['Side'] == Side
        FilteredData = Data[Filter1 & Filter2]
        Axes.plot(np.log(FilteredData['Total Distance [rev]']), np.log(FilteredData['Grinding ratio [um/rev]']+1),
                  color=Colors[i], label=Sample + ' ' + Side)
        i += 1
Axes.set_xlabel('Cumulative distance [rev]')
Axes.set_ylabel('Grinding ratio [$\mu$m / rev]')
plt.legend(ncol=2)
plt.show()


# Transform data for linear relationships
Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
i = 0
for Sample in Samples:
    for Side in Sides:
        Filter1 = Data['Samples'] == Sample
        Filter2 = Data['Side'] == Side
        FilteredData = Data[Filter1 & Filter2]
        Axes.plot(np.log(FilteredData['Total Distance [rev]']), np.log(FilteredData['Grinding ratio [um/rev]']+1),
                  color=Colors[i], label=Sample + ' ' + Side)
        i += 1
Axes.set_xlabel('Cumulative distance [rev]')
Axes.set_ylabel('Grinding ratio [$\mu$m / rev]')
plt.legend(ncol=2)
plt.show()

Data['LogD'] = np.log(Data['Total Distance [rev]'])
Data['LogG'] = np.log(Data['Grinding ratio [um/rev]']+1)

LM_Data = pd.DataFrame(Data.dropna(subset='LogG'))
LM_Data['Groups'] = LM_Data['Samples'] + ' ' + LM_Data['Side']



LMM = smf.mixedlm('LogG ~ LogD', data=LM_Data, groups=LM_Data['Groups']).fit(reml=True)

Healthy_LMM = smf.mixedlm("Grinding ratio [um/rev] ~ exp(-Total Distance [rev])",
                         data=Data, groups=Data[['Samples','Side']],
                         vc_formula={"IF": "IF-1"}).fit(reml=True)

