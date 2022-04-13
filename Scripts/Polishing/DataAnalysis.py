import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats.distributions import t
import sympy as sp

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
Data['Groups'] = Data['Samples'] + ' ' + Data['Side']
Groups = Data['Groups'].unique()
Colors = [(0,0,0),(1,0,0),(1,0,1),(0,0,1),(0,1,1),(0,1,0)]

Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
i = 0
for Group in Groups:
        Filter = Data['Groups'] == Group
        FilteredData = Data[Filter]
        Axes.plot(FilteredData['Test Run'], FilteredData['Delta [um]'],
                  color=Colors[i], label=Group)
        i += 1
Axes.set_xticks([1,2,3,4,5])
Axes.set_xlabel('Test Run n° [-]')
Axes.set_ylabel('Thickness $\Delta$ [$\mu$m]')
plt.legend(ncol=2)
plt.show()


# Compute additional data
Data['Distance [rev]'] = Data['Test time [min]'] * Data['Rotation velocity [rpm]']
Data['Total Distance [rev]'] = Data.groupby('Groups')['Distance [rev]'].cumsum()
Data['Total Grinding [um]'] = Data.groupby('Groups')['Delta [um]'].cumsum()

Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
i = 0
for Group in Groups:
    Filter = Data['Groups'] == Group
    FilteredData = Data[Filter]
    Axes.plot(FilteredData['Total Distance [rev]'], FilteredData['Total Grinding [um]'],
              color=Colors[i], label=Group)
    i += 1
Axes.set_xlabel('Cumulative Distance [rev]')
Axes.set_ylabel('Cumulative Grinding [$\mu$m]')
Axes.set_xlim([0, Data['Total Distance [rev]'].max() * 1.05])
Axes.set_ylim([0,Data['Total Grinding [um]'].max()+40])
plt.legend(ncol=2,frameon=False)
plt.show()


# Fit
LM_Data = pd.DataFrame(Data.dropna(subset='Final Max Thickness [um]'))
LM_Data['x'] = np.log(LM_Data['Total Distance [rev]'])
LM_Data['y'] = LM_Data['Total Grinding [um]']

LMM = smf.mixedlm('y ~ x', data=LM_Data, groups=LM_Data['Groups']).fit(reml=True)
def PlotRegressionResults(Model, Data, Alpha=0.95):

    ## Get data from the model
    Y_Obs = Model.model.endog
    N = int(Model.nobs)
    C = np.matrix(Model.cov_params())
    X = np.matrix(Model.model.exog)

    X_Pred = np.matrix(np.ones((50,2)))
    X_Pred[:,1] = np.matrix(np.linspace(X.min(),X.max(),50)).T
    Y_Fit = Model.predict(X_Pred, transform=False)

    if not C.shape[0] == X.shape[1]:
        C = C[:-1,:-1]


    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Model.predict()
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
    Data.groupby('Groups').plot(x='Total Distance [rev]',y='Total Grinding [um]', label='_nolegend_',
                                ax=Axes, color=(0, 0, 0), linestyle='--', marker='o', fillstyle='none')
    Axes.plot([], color=(0, 0, 0), linestyle='--', marker='o', fillstyle='none', label='Data')
    Axes.plot(np.exp(X_Pred[:,1]), Y_Fit, color=(1, 0, 0), linestyle='--', label='Fit')
    # Axes.fill_between(np.linspace(X.min(),X.max()), CI_Line_u, CI_Line_o, color=(0, 0, 0, 0.1), label='95% CI')
    Axes.set_xlabel('Cumulative distance [rev]')
    Axes.set_ylabel('Cumulative grinding [$\mu$m]')
    Axes.annotate(r'N Groups : ' + str(len(Data.groupby('Groups'))), xy=(0.25, 0.925), xycoords='axes fraction')
    Axes.annotate(r'N Points : ' + str(N), xy=(0.25, 0.86), xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.05, 0.75), xycoords='axes fraction')
    Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.05, 0.685), xycoords='axes fraction')
    Axes.set_xlim([0, np.exp(X.max())*1.05])
    Axes.set_ylim([0, max(Y_Obs.max(),Y_Fit.max())+40])
    plt.legend(ncol=1, loc='upper left')
    plt.show()

    return R2, SE
PlotRegressionResults(LMM, LM_Data, Alpha=0.95)

# Build function to estimate time
x = sp.symbols('x')
TotalGrinding = LMM.params['Intercept'] + LMM.params['x'] * x
Fx = sp.lambdify(x, TotalGrinding, 'numpy')
Distances = np.arange(1E0,1E4)
GrindingProfile = Fx(np.log(Distances))

Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
Axes.plot(GrindingProfile)
Axes.set_xlabel('Cumulative Distance [rev]')
Axes.set_ylabel('Cumulative Grinding [$\mu$m]')
plt.show()

# Compute distance
MinThickness = 200
R_Data = pd.DataFrame(LM_Data[LM_Data['Test Run'] == 5])
R_Data['Thickness2Grind'] = R_Data['Final Max Thickness [um]'] - MinThickness

ActualDistance = Data['Total Distance [rev]'].max()
ActualDistances = Distances[Distances >= ActualDistance]
ActualProfile = GrindingProfile[Distances >= ActualDistance]

# Set profile and distance to 0
ActualDistances = ActualDistances - ActualDistances.min()
ActualProfile = ActualProfile - ActualProfile.min()


# Figure, Axes = plt.subplots(1,1,figsize=(5.5*1.5,4.5*1.5))
# Axes.plot(ActualDistances,ActualProfile)
# Axes.set_xlabel('Cumulative Distance [rev]')
# Axes.set_ylabel('Cumulative Grinding [$\mu$m]')
# plt.show()

## Rotation velocity [rev/min]
Rv = 50
GrindingTimes = pd.DataFrame()
Distances2Grind = pd.DataFrame()
for Index in R_Data.index:
    Sample = R_Data.loc[Index,'Groups']
    Thickness2Grind = R_Data.loc[Index,'Thickness2Grind']
    Distance2Grind = ActualDistances[np.argmin(np.abs(ActualProfile - Thickness2Grind))]
    DataFrame = pd.DataFrame([Distance2Grind],index=[Sample])
    Distances2Grind = pd.concat([Distances2Grind,DataFrame])
    GrindingTimes = pd.concat([GrindingTimes,DataFrame / Rv])


Time = 22
FinalThickness = pd.DataFrame()
for Index in R_Data.index:
    Sample = R_Data.loc[Index,'Groups']
    Thickness2Grind = R_Data.loc[Index,'Thickness2Grind']
    DistanceGrounded = Time * Rv
    ThicknessGrounded = ActualProfile[np.argmin(np.abs(ActualDistances - DistanceGrounded))]
    DataFrame = pd.DataFrame([Thickness2Grind + MinThickness - ThicknessGrounded],index=[Sample])
    FinalThickness = pd.concat([FinalThickness, DataFrame])

