import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# Paths
CurrentDirectory = Path.cwd()
DataDirectory = CurrentDirectory / 'Scripts/Statistics'

# Load and look at the data
Data = pd.read_csv(str(DataDirectory / 'Ortho.csv'))
Data.sample(5)
Data_F = Data.drop(index=Data[Data['Sex'] == 'Male'].index)

# Plot for each subject
plt.rc('font', size=12)
Figure, Axis = plt.subplots(1,1, figsize=(10,5))
for Label, DataFrame in Data_F.groupby('Subject'):
    DataFrame.plot(x='age', y='distance', ax=Axis, label=Label,
                   marker='o', fillstyle='none', linestyle='--')
Axis.set_xlabel('Age (years)')
Axis.set_ylabel('Distance (mm)')
plt.legend(title='Subject', bbox_to_anchor=(1.05,1),ncol=2)
plt.subplots_adjust(right=0.7)
plt.show()

# First fit general curve
OLS = smf.ols('distance ~ age', data=Data_F).fit(reml=True)
print(OLS.summary())

X = np.sort(Data_F['age'].unique())
Figure, Axis = plt.subplots(1,1, figsize=(10,5))
for Label, DataFrame in Data_F.groupby('Subject'):
    DataFrame.plot(x='age', y='distance', ax=Axis, label=Label,
                   marker='o', fillstyle='none', linestyle='--')
Axis.plot(X,OLS.params['Intercept'] + X * OLS.params['age'], color=(0,0,0), linewidth=2)
Axis.set_xlabel('Age (years)')
Axis.set_ylabel('Distance (mm)')
plt.legend(title='Subject', bbox_to_anchor=(1.05,1),ncol=2)
plt.subplots_adjust(right=0.7)
plt.show()

# Second fit individual curves for each subject
OLSs = {}
for Subject, DataFrame in Data_F.groupby('Subject'):
    OLSs[Subject] = smf.ols('distance ~ age', data=DataFrame).fit(reml=True)

Coefficients = {}
for Subject in OLSs:
    Coefficients[Subject] = {'Age Mean': OLSs[Subject].params['age'],
                      'Intercept Mean': OLSs[Subject].params['Intercept'],
                      'Age 0.05': OLSs[Subject].conf_int(0.05).loc['age',0],
                      'Age 0.95': OLSs[Subject].conf_int(0.05).loc['age',1],
                      'Intercept 0.05': OLSs[Subject].conf_int(0.05).loc['Intercept',0],
                      'Intercept 0.95': OLSs[Subject].conf_int(0.05).loc['Intercept',1]}
Coefficients = pd.DataFrame(Coefficients).T

Figure, Axis = plt.subplots(2,1, figsize=(15,12), sharex=True)
for Subject in Coefficients.index:
    Axis[0].plot([Subject, Subject], Coefficients.loc[Subject,['Age 0.05', 'Age 0.95']], color=(0,0,0))
Coefficients.plot(y='Age 0.05', linestyle='none', marker='v', color=(0,0,0), ax=Axis[0], legend=None)
Coefficients.plot(y='Age 0.95', linestyle='none', marker='^', color=(0,0,0), ax=Axis[0], legend=None)
Coefficients.plot(y='Age Mean', linestyle='none', marker='o', color=(1,0,0), ax=Axis[0], legend=None)
Axis[0].set_ylabel('Age coefficient')
for Subject in Coefficients.index:
    Axis[1].plot([Subject, Subject], Coefficients.loc[Subject,['Intercept 0.05', 'Intercept 0.95']], color=(0,0,0))
Coefficients.plot(y='Intercept 0.05', linestyle='none', marker='v', color=(0,0,0), ax=Axis[1], legend=None)
Coefficients.plot(y='Intercept 0.95', linestyle='none', marker='^', color=(0,0,0), ax=Axis[1], legend=None)
Coefficients.plot(y='Intercept Mean', linestyle='none', marker='o', color=(1,0,0), ax=Axis[1], legend=None)
Axis[1].set_ylabel('Intercept coefficient')
plt.show()

# Third center age
Data_F['centeredage'] = Data_F['age'] - 11
OLSs = {}
for Subject, DataFrame in Data_F.groupby('Subject'):
    OLSs[Subject] = smf.ols('distance ~ centeredage', data=DataFrame).fit(reml=True)

Coefficients = {}
for Subject in OLSs:
    Coefficients[Subject] = {'Age Mean': OLSs[Subject].params['centeredage'],
                      'Intercept Mean': OLSs[Subject].params['Intercept'],
                      'Age 0.05': OLSs[Subject].conf_int(0.05).loc['centeredage',0],
                      'Age 0.95': OLSs[Subject].conf_int(0.05).loc['centeredage',1],
                      'Intercept 0.05': OLSs[Subject].conf_int(0.05).loc['Intercept',0],
                      'Intercept 0.95': OLSs[Subject].conf_int(0.05).loc['Intercept',1]}
Coefficients = pd.DataFrame(Coefficients).T

Figure, Axis = plt.subplots(2,1, figsize=(15,12), sharex=True)
for Subject in Coefficients.index:
    Axis[0].plot([Subject, Subject], Coefficients.loc[Subject,['Age 0.05', 'Age 0.95']], color=(0,0,0))
Coefficients.plot(y='Age 0.05', linestyle='none', marker='v', color=(0,0,0), ax=Axis[0], legend=None)
Coefficients.plot(y='Age 0.95', linestyle='none', marker='^', color=(0,0,0), ax=Axis[0], legend=None)
Coefficients.plot(y='Age Mean', linestyle='none', marker='o', color=(1,0,0), ax=Axis[0], legend=None)
Axis[0].set_ylabel('Age coefficient')
for Subject in Coefficients.index:
    Axis[1].plot([Subject, Subject], Coefficients.loc[Subject,['Intercept 0.05', 'Intercept 0.95']], color=(0,0,0))
Coefficients.plot(y='Intercept 0.05', linestyle='none', marker='v', color=(0,0,0), ax=Axis[1], legend=None)
Coefficients.plot(y='Intercept 0.95', linestyle='none', marker='^', color=(0,0,0), ax=Axis[1], legend=None)
Coefficients.plot(y='Intercept Mean', linestyle='none', marker='o', color=(1,0,0), ax=Axis[1], legend=None)
Axis[1].set_ylabel('Intercept coefficient')
plt.show()


# Linear mixed-effect model with common slope but different intercept
LME_1 = smf.mixedlm('distance ~ age',
                  # vc_formula={'Subject': '1'},
                  re_formula='1',
                  data=Data_F,
                  groups=Data_F['Subject']).fit(reml=True)
print(LME_1.summary())

Parameters = LME_1.params
Scale = LME_1.scale

print(Parameters[:-1])
print((Parameters[-1:] * Scale) ** 0.5)


# Linear mixed-effect model with common slope but different intercept
LME_2 = smf.mixedlm('distance ~ age',
                  # vc_formula={'Subject': '1 + age'},
                  re_formula='1 + age',
                  data=Data_F,
                  groups=Data_F['Subject']).fit(reml=True)
print(LME_2.summary())

Parameters = LME_2.params
Scale = LME_2.scale

print(Parameters[:-1])
print((Parameters[-1:] * Scale) ** 0.5)

# Covariance structure
LME_2.cov_re

# Log-Likelihood Ratio test
from scipy.stats.distributions import chi2
LRT = 2 * (LME_2.llf - LME_1.llf)
delta_df = abs(LME_2.df_modelwc - LME_1.df_modelwc)
p = 1 - chi2.cdf(LRT,delta_df)


# Fit full data set with fixed sex * (age-11) and random intercept and (age-11)
Data['centeredage'] = Data['age'] - Data['age'].unique().mean()
LME_3 = smf.mixedlm('distance ~ Sex * centeredage',
                  # vc_formula={'Subject': '1'},
                  re_formula='1 + centeredage',
                  data=Data,
                  groups=Data['Subject']).fit(reml=True)
print(LME_3.summary())
print(LME_3.params)
print((LME_3.params * LME_3.scale) ** 0.5)
print(LME_3.cov_re)


# Check fit quality
Min = min(LME_3.fittedvalues.min(),LME_3.model.endog.min())
Max = max(LME_3.fittedvalues.max(),LME_3.model.endog.max())
Figure, Axis = plt.subplots(1,1)
Axis.plot([Min, Max], [Min, Max], color=(0,0,0), linewidth=0.5)
Axis.plot(LME_3.fittedvalues,LME_3.model.endog, linestyle='none', marker='o', color=(1,0,0), fillstyle='none')
Axis.set_xlabel('Fitted values')
Axis.set_ylabel('Observed values')
plt.show()



# Check within group error assumptions
Data['residuals'] = LME_3.model.endog - LME_3.fittedvalues

Figure, Axis = plt.subplots(1,1, figsize=(15,6))
Axis.plot([0,len(Data['Subject'].unique())], [0,0], color=(0,0,0), linestyle='--', linewidth=0.5)
Axis.plot(Data['Subject'], Data['residuals'], marker='o',
          fillstyle='none', linestyle='none', color=(1,0,0))
Axis.set_xlabel('Subject')
Axis.set_ylabel('Residuals')
plt.show()

## Residuals by sex
Data['fitted'] = LME_3.fittedvalues
M = Data[Data['Sex'] == 'Male']
F = Data[Data['Sex'] == 'Female']
Figure, Axis = plt.subplots(1,2, figsize=(15,6), sharex=True, sharey=True)
Axis[0].plot([M['fitted'].min(),M['fitted'].max()], [0,0], color=(0,0,0), linestyle='--', linewidth=0.5)
Axis[0].plot(M['fitted'], M['residuals'], marker='o',
          fillstyle='none', linestyle='none', color=(1,0,0))
Axis[1].plot([F['fitted'].min(),F['fitted'].max()], [0,0], color=(0,0,0), linestyle='--', linewidth=0.5)
Axis[1].plot(F['fitted'], F['residuals'], marker='o',
          fillstyle='none', linestyle='none', color=(1,0,0))
Axis[0].set_xlabel('Fitted values')
Axis[1].set_xlabel('Fitted values')
Axis[0].set_ylabel('Residuals')
plt.show()


## QQ plots
def QQPlot(DataValues, Alpha_CI=0.95):

    from scipy.stats.distributions import norm

    ### Based on: https://www.tjmahr.com/quantile-quantile-plots-from-scratch/
    ### Itself based on Fox book: Fox, J. (2015)
    ### Applied Regression Analysis and Generalized Linear Models.
    ### Sage Publications, Thousand Oaks, California.

    # Data analysis
    N = len(DataValues)
    X_Bar = np.mean(DataValues)
    S_X = np.std(DataValues,ddof=1)

    # Sort data to get the rank
    Data_Sorted = np.zeros(N)
    Data_Sorted += DataValues
    Data_Sorted.sort()

    # Compute quantiles
    EmpiricalQuantiles = np.arange(0.5, N + 0.5) / N
    TheoreticalQuantiles = norm.ppf(EmpiricalQuantiles, X_Bar, S_X)
    ZQuantiles = norm.ppf(EmpiricalQuantiles,0,1)

    # Compute data variance
    DataIQR = np.quantile(DataValues, 0.75) - np.quantile(DataValues, 0.25)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
    Variance = DataIQR / NormalIQR
    Z_Space = np.linspace(min(ZQuantiles), max(ZQuantiles), 100)
    Variance_Line = Z_Space * Variance + np.median(DataValues)

    # Compute alpha confidence interval (CI)
    Z_SE = np.sqrt(norm.cdf(Z_Space) * (1 - norm.cdf(Z_Space)) / N) / norm.pdf(Z_Space)
    Data_SE = Z_SE * Variance
    Z_CI_Quantile = norm.ppf(np.array([(1 - Alpha_CI) / 2]), 0, 1)

    # Create point in the data space
    Data_Space = np.linspace(min(TheoreticalQuantiles), max(TheoreticalQuantiles), 100)

    # QQPlot
    BorderSpace = max( 0.05*abs(Data_Sorted.min()), 0.05*abs(Data_Sorted.max()))
    Y_Min = Data_Sorted.min() - BorderSpace
    Y_Max = Data_Sorted.max() + BorderSpace
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.plot(Data_Space, Variance_Line, linestyle='--', color=(1, 0, 0), label='Variance :' + str(format(np.round(Variance, 2),'.2f')))
    Axes.plot(Data_Space, Variance_Line + Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1), label=str(int(100*Alpha_CI)) + '% CI')
    Axes.plot(Data_Space, Variance_Line - Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1))
    Axes.plot(TheoreticalQuantiles, Data_Sorted, linestyle='none', marker='o', mew=0.5, fillstyle='none', color=(0, 0, 0))
    plt.xlabel('Theoretical quantiles (-)')
    plt.ylabel('Empirical quantiles (-)')
    plt.ylim([Y_Min, Y_Max])
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size':10})
    plt.subplots_adjust(left=0.15, bottom=0.15)
    # plt.savefig(FigFile)
    plt.show()
    plt.close(Figure)

    return Variance
QQPlot(M['residuals'].values)
QQPlot(F['residuals'].values)


# Random effects assumptions
RandomEffect = pd.DataFrame(LME_3.random_effects).T
QQPlot(RandomEffect['Group'].values)
QQPlot(RandomEffect['centeredage'].values)

## Variance per group
SexData = Data[['Subject','Sex']].drop_duplicates()
RandomEffect['Sex'] = pd.DataFrame(SexData['Sex'].values, index=SexData['Subject'].values)

Figure, Axis = plt.subplots(1,2, figsize=(8,4))
Axis[0].plot([1-0.35, len(RandomEffect['Sex'].unique()) + 0.35], [0,0],
             linestyle='--', linewidth=0.5, color=(0,0,0))
Axis[1].plot([1-0.35, len(RandomEffect['Sex'].unique()) + 0.35], [0,0],
             linestyle='--', linewidth=0.5, color=(0,0,0))
RandomEffect.boxplot('Group', by='Sex',vert=True, widths=0.6,
             showmeans=False, grid=False, ax=Axis[0], patch_artist=True,
             boxprops=dict(linestyle='-',color=(0,0,0), facecolor=(1,1,1)),
             medianprops=dict(linestyle='-',color=(1,0,0)),
             whiskerprops=dict(linestyle='--',color=(0,0,0)),
             meanprops=dict(marker='x',markeredgecolor=(0,0,1)))
RandomEffect.boxplot('centeredage', by='Sex',vert=True, widths=0.6,
             showmeans=False, grid=False, ax=Axis[1], patch_artist=True,
             boxprops=dict(linestyle='-',color=(0,0,0), facecolor=(1,1,1)),
             medianprops=dict(linestyle='-',color=(1,0,0)),
             whiskerprops=dict(linestyle='--',color=(0,0,0)),
             meanprops=dict(marker='x',markeredgecolor=(0,0,1)))
Axis[0].set_title('')
Axis[1].set_title('')
Axis[0].set_ylabel('Intercept')
Axis[1].set_ylabel('Slope')
plt.suptitle('')
plt.subplots_adjust(wspace=0.4)
plt.show()

## Investigate variance structure
M_RE = RandomEffect[RandomEffect['Sex'] == 'Male']
F_RE = RandomEffect[RandomEffect['Sex'] == 'Female']
Figure, Axis = plt.subplots(1,2, figsize=(8,3.5), sharex=True, sharey=True)
Axis[0].plot([RandomEffect['Group'].min(),RandomEffect['Group'].max()], [0,0], color=(0,0,0), linestyle='--', linewidth=0.5)
Axis[0].plot([0,0], [RandomEffect['centeredage'].min(),RandomEffect['centeredage'].max()], color=(0,0,0), linestyle='--', linewidth=0.5)
Axis[0].plot(M_RE['Group'], M_RE['centeredage'], marker='o', fillstyle='none', linestyle='none', color=(1,0,0))
Axis[1].plot([RandomEffect['Group'].min(),RandomEffect['Group'].max()], [0,0], color=(0,0,0), linestyle='--', linewidth=0.5)
Axis[1].plot([0,0], [RandomEffect['centeredage'].min(),RandomEffect['centeredage'].max()], color=(0,0,0), linestyle='--', linewidth=0.5)
Axis[1].plot(F_RE['Group'], F_RE['centeredage'], marker='o', fillstyle='none', linestyle='none', color=(1,0,0))
Axis[0].set_xlabel('Intercept')
Axis[1].set_xlabel('Intercept')
Axis[0].set_ylabel('Slope')
plt.subplots_adjust(bottom=0.15)
plt.show()


# Look at parameters correlations
def CorrelationMatrix(Data, Variables, Title=None):

    Matrix = Data[Variables].corr(method='pearson')
    NVariables = Matrix.shape[0]
    Labels = Data[Variables].columns

    Figure, Axis = plt.subplots(1,1)
    Imshow = Axis.imshow(Matrix, cmap='jet', extent=(0,NVariables,0,NVariables))

    # create list of label positions
    LabelPosition = np.arange(0, NVariables) + 0.5

    Axis.set_yticks(LabelPosition)
    Axis.set_yticks(LabelPosition[:-1]+0.5, minor=True)
    Axis.set_yticklabels(Labels[::-1], verticalalignment='center', rotation=90)

    Axis.set_xticks(LabelPosition)
    Axis.set_xticks(LabelPosition[:-1]+0.5, minor=True)
    Axis.set_xticklabels(Labels, horizontalalignment='center')

    if Title:
        Axis.set_title(Title)

    Figure.colorbar(Imshow, use_gridspec=True)
    Figure.tight_layout()

    Axis.tick_params(which='minor', length=0)
    Axis.tick_params(direction='out', top=False, right=False)
    Axis.grid(True, which='minor', linestyle='-', color='w', lw=1)

    plt.show()

    return Matrix
CMatrix = CorrelationMatrix(RandomEffect, ['Group', 'centeredage'])


# Try to set random effects correlations to 0
LME_4 = smf.mixedlm('distance ~ Sex * centeredage',
                  vc_formula={'centeredage': '1 + centeredage'},
                  re_formula='1',
                  data=Data,
                  groups=Data['Subject']).fit(reml=True)
print(LME_4.summary())
print(LME_4.cov_re)

# Log-Likelihood Ratio test
from scipy.stats.distributions import chi2
LRT = 2 * (LME_3.llf - LME_4.llf)
delta_df = abs(LME_4.df_modelwc - LME_3.df_modelwc)
p = 1 - chi2.cdf(LRT,delta_df)
print('p-value: ' + str(round(p,3)))

RandomEffect = pd.DataFrame(LME_4.random_effects).T


# Machines data frame analysis
Machines = pd.read_csv(str(DataDirectory / 'Machines.csv'))
Machines.sample(5)

# Try to set random effects correlations to 0
LME_5 = smf.mixedlm('score ~ Machine',
                    vc_formula={'Worker': '1'},
                    re_formula='1',
                    data=Data,
                    groups=Data['Subject']).fit(reml=True)
print(LME_5.summary())
print(LME_5.cov_re)
