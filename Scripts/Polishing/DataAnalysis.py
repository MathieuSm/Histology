import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats.distributions import t


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
LM_Data['y'] = LM_Data['Grinding ratio [um/rev]']
LM_Data['x'] = 1/LM_Data['Total Distance [rev]']


LMM = smf.mixedlm('y ~ x - 1', data=LM_Data, groups=LM_Data['Groups']).fit(reml=True)


def PlotRegressionResults(Model, Data, Alpha=0.95):

    ## Get data from the model
    Y_Obs = Model.model.endog
    N = int(Model.nobs)
    C = np.matrix(Model.cov_params())
    X = 1 / np.matrix(Model.model.exog)

    X_Pred = np.matrix(np.linspace(X.min(),X.max())).T
    Y_Fit = Model.predict(1/X_Pred, transform=False)


    if not C.shape[0] == X.shape[1]:
        C = C[:-1,:-1]


    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / Model.df_resid)
    TSS = np.sum((Model.model.endog - Model.model.endog.mean()) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    B_0 = np.sqrt(np.diag(np.abs(X_Pred * C * X_Pred.T)))
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0

    ## Plot
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5 * 1.5, 4.5 * 1.5))
    Data.groupby('Groups').plot(x='Total Distance [rev]',y='Grinding ratio [um/rev]', label='_nolegend_',
                                ax=Axes, color=(0, 0, 0), linestyle='--', marker='o', fillstyle='none')
    Axes.plot([], color=(0, 0, 0), linestyle='--', marker='o', fillstyle='none', label='Data')
    Axes.plot(X_Pred, Y_Fit, color=(1, 0, 0), linestyle='--', label='Fit')
    # Axes.fill_between(np.linspace(X.min(),X.max()), CI_Line_u, CI_Line_o, color=(0, 0, 0, 0.1), label='95% CI')
    Axes.set_xlabel('Cumulative distance [rev]')
    Axes.set_ylabel('Grinding ratio [$\mu$m / rev]')
    Axes.annotate(r'N Groups : ' + str(len(Data.groupby('Groups'))), xy=(0.525, 0.925), xycoords='axes fraction')
    Axes.annotate(r'N Points : ' + str(N), xy=(0.525, 0.86), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.8, 0.75), xycoords='axes fraction')
    Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.8, 0.685), xycoords='axes fraction')
    plt.legend(ncol=1)
    plt.show()

    return R2, SE


Healthy_LMM = smf.mixedlm("Grinding ratio [um/rev] ~ exp(-Total Distance [rev])",
                         data=Data, groups=Data[['Samples','Side']],
                         vc_formula={"IF": "IF-1"}).fit(reml=True)

