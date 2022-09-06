#!/usr/bin/env python3

"""
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
#sphx-glr-auto-examples-segmentation-plot-trainable-segmentation-py
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from skimage import io, future
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats.distributions import t
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import multiscale_basic_features as mbf


sys.path.insert(0, str(Path.cwd() / 'Scripts/FinalPipeline'))
from Utilities import *


def FitData(DataFrame):

    Formula = DataFrame.columns[1] + ' ~ ' + DataFrame.columns[0]
    FitResults = smf.ols(Formula, data=DataFrame).fit()

    # Calculate R^2, p-value, 95% CI, SE, N
    Y_Obs = FitResults.model.endog
    Y_Fit = FitResults.fittedvalues

    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / FitResults.df_resid)

    N = int(FitResults.nobs)
    R2 = FitResults.rsquared
    p = FitResults.pvalues[1]

    CI_l = FitResults.conf_int()[0][1]
    CI_r = FitResults.conf_int()[1][1]

    X = np.matrix(FitResults.model.exog)
    X_Obs = np.sort(np.array(X[:, 1]).reshape(len(X)))
    C = np.matrix(FitResults.cov_params())
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    Alpha = 0.95
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * B_0
    CI_Line_o = Y_Fit + t_Alpha[1] * B_0
    Sorted_CI_u = CI_Line_u[np.argsort(FitResults.model.exog[:,1])]
    Sorted_CI_o = CI_Line_o[np.argsort(FitResults.model.exog[:,1])]

    NoteYPos = 0.925
    NoteYShift = 0.075

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
    Axes.plot(X[:, 1], Y_Fit, color=(1, 0, 0), label='Fit')
    Axes.fill_between(X_Obs, Sorted_CI_o, Sorted_CI_u, color=(0, 0, 0), alpha=0.1,
                      label=str(int(Alpha * 100)) + '% CI')
    Axes.plot(X[:, 1], Y_Obs, linestyle='none', fillstyle='none', marker='o', color=(0, 0, 1), label='Data')
    Axes.annotate('Slope 95% CI [' + str(CI_l.round(2)) + r'$,$ ' + str(CI_r.round(2)) + ']',
                  xy=(0.05, NoteYPos), xycoords='axes fraction')
    # Axes.annotate(r'$N$ : ' + str(N), xy=(0.05, NoteYPos),
    #               xycoords='axes fraction')
    Axes.annotate(r'$R^2$ : ' + str(R2.round(2)), xy=(0.05, NoteYPos - NoteYShift),
                  xycoords='axes fraction')
    Axes.annotate(r'$\sigma_{est}$ : ' + str(SE.round(5)), xy=(0.05, NoteYPos - NoteYShift*2),
                  xycoords='axes fraction')
    Axes.annotate(r'$p$ : ' + str(p.round(3)), xy=(0.05, NoteYPos - NoteYShift*3),
                  xycoords='axes fraction')
    Axes.set_ylabel(DataFrame.columns[1])
    Axes.set_xlabel(DataFrame.columns[0])
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend(loc='lower right')
    plt.show()

    # Add fitted values and residuals to data
    DataFrame['Fitted Value'] = Y_Fit
    DataFrame['Residuals'] = E

    return DataFrame, FitResults, R2, SE, p, [CI_l, CI_r]

# Load images
ROI = io.imread('TestROI.png')
PlotImage(ROI)

Seg_ROI = io.imread('TestROI_Forest.png')
PlotImage(Seg_ROI)

# Extract segments
CL = Seg_ROI[:,:,0] == 255
OC = Seg_ROI[:,:,1] == 255
IT = Seg_ROI[:,:,2] > 220
HC = CL * OC * IT
PlotImage(HC)

# Label segments
Label = np.ones(HC.shape,'uint8')
Label[OC] = 1
Label[CL] = 2
Label[HC] = 1
PlotImage(Label)

# Random forest classifier
Channels = ['R','G','B']
Names = ['Intensity', 'Edges', 'Hessian 1', 'Hessian 2']
SigmaMin = 0.5
SigmaMax = 16
NumSigma = int(np.log2(SigmaMax / SigmaMin) + 1)
F_Names = []
for Channel in Channels:
    for Sigma in range(NumSigma):
        SigmaValue = SigmaMin * 2**Sigma
        if SigmaValue >= 1:
            SigmaValue = int(SigmaValue)
        for Name in Names:
            F_Names.append(Channel + ' Sigma ' + str(SigmaValue) + ' ' + Name)


Features = mbf(ROI, multichannel=True,intensity=True,edges=True,texture=True,
               sigma_min=SigmaMin,sigma_max=SigmaMax,num_sigma=NumSigma)
Classifier = RandomForestClassifier(n_jobs=-1, max_samples=0.2, class_weight='balanced')

# Exclude H2
LessFeatures = np.zeros((Features.shape[0],Features.shape[1],Features.shape[-1] - int(Features.shape[-1]/4)))
LessFeatures[:,:,0::3] = Features[:,:,0::4]
LessFeatures[:,:,1::3] = Features[:,:,1::4]
LessFeatures[:,:,2::3] = Features[:,:,2::4]

# Train model
Tic = time.time()
clf = future.fit_segmenter(Label,Features, Classifier)
Toc = time.time()
PrintTime(Tic, Toc)

# See predictions
Results = future.predict_segmenter(Features, Classifier)
PlotImage(Results)

# Assess model
CM = metrics.confusion_matrix(Label.ravel(),Results.ravel(),normalize='pred')

Figure, Axis = plt.subplots(1,1)
Axis.matshow(CM, cmap='binary', alpha=0.5)
for Row in range(CM.shape[0]):
    for Column in range(CM.shape[1]):
        Axis.text(x=Row,y=Column,s=round(CM[Row, Column],2), va='center', ha='center')
Axis.xaxis.set_ticks_position('bottom')
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.show()

# Dice coefficient
Dice(Label-1,Results-1)

Report = metrics.classification_report(Label.ravel(),Results.ravel())
print(Report)

# Feature importance
FI = pd.DataFrame(Classifier.feature_importances_,columns=['Importance'])
FI['Channel'] = [C.split()[0] for C in F_Names]
FI['Sigma'] = [C.split()[2] for C in F_Names]
FI['Feature'] = [C.split()[3:] for C in F_Names]

Sorted = FI.sort_values(by='Importance')
R = Sorted[Sorted['Channel'] == 'R']
G = Sorted[Sorted['Channel'] == 'G']
B = Sorted[Sorted['Channel'] == 'B']

Figure, Axis = plt.subplots(2, 3, sharex=True, sharey=True)
i = 0
for Sigma in FI['Sigma'].unique():

    if i < 2:
        Row = 0
        Column = i
    else:
        Row = 1
        Column = i - 2

    RF = R[R['Sigma'] == Sigma]
    GF = G[G['Sigma'] == Sigma]
    BF = B[B['Sigma'] == Sigma]

    RS = RF.sort_values(by='Feature')
    GS = GF.sort_values(by='Feature')
    BS = BF.sort_values(by='Feature')

    Axis[Row,Column].bar(np.arange(len(RS)), RS['Importance'], edgecolor=(1,0,0), facecolor=(0,0,0,0))
    Axis[Row,Column].bar(np.arange(len(GS)), GS['Importance'], edgecolor=(0,1,0), facecolor=(0,0,0,0))
    Axis[Row,Column].bar(np.arange(len(BS)), BS['Importance'], edgecolor=(0,0,1), facecolor=(0,0,0,0))

    Axis[Row,Column].set_xticks(np.arange(4), ['E', 'H1', 'H2', 'I'])
    Axis[Row,Column].set_title('Sigma = ' + Sigma)

    i += 1
plt.show()


# Load dictionary
with open('OptimizationData.pkl', 'rb') as f:
    Dict = pickle.load(f)


Data = []
for Key in Dict.keys():
    for ROINumber in range(3):
        TestROI = Dict[Key]['ROI'][ROINumber]
        F_Test = mbf(TestROI, multichannel=True, intensity=True, edges=True, texture=True,
                     sigma_min=SigmaMin, sigma_max=SigmaMax, num_sigma=NumSigma)
        R_Test = future.predict_segmenter(F_Test, Classifier)
        Data.append(np.sum(R_Test == 2) / R_Test.size)

Density = []
for Key in Dict.keys():
    for ROINumber in range(3):
        TestSeg = Dict[Key]['Skeleton'][ROINumber]
        Density.append(np.sum(TestSeg) / TestSeg.size)


Data2Fit = pd.DataFrame({'Manual':Density,'Automatic':Data})
Data2Fit = Data2Fit[Data2Fit['Manual'] > 2E-4].reset_index()
Data2Fit, FitResults, R2, SE, p, CI = FitData(Data2Fit[['Automatic','Manual']])


Figure, Axis = plt.subplots(1,1)
Axis.plot(Data2Fit['Residuals'],linestyle='none',marker='o')
plt.show()

FitData(Data2Fit.drop([3]).reset_index(drop=True))
FitData(Data2Fit.drop([1,4,5,6,9]).reset_index(drop=True))


ROINumber = 9
PlotImage(ROIs[ROINumber])

SegROI = Parameters.SegImage[Ys[ROINumber, 0]:Ys[ROINumber, 1], Xs[ROINumber, 0]:Xs[ROINumber, 1]]
PlotImage(SegROI)

F = filters.gaussian(ROIs[ROINumber],sigma=1, multichannel=True)
R = features_func(F)
Results = future.predict_segmenter(R, clf)
PlotImage(Results == 4)

B1 = []
for i in [1,4,5,6,9]:
    SegROI = Parameters.SegImage[Ys[i, 0]:Ys[i, 1], Xs[i, 0]:Xs[i, 1]]
    B1.append(SegROI)
HistoError(B1,ylim=0.015)

B2 = []
for i in [0,2,3,7,8]:
    SegROI = Parameters.SegImage[Ys[i, 0]:Ys[i, 1], Xs[i, 0]:Xs[i, 1]]
    B2.append(SegROI)
HistoError(B2,ylim=0.015)

B1 = np.concatenate(B1,axis=0)
B2 = np.concatenate(B2,axis=0)
NBins = 20
B1H, B2H = np.zeros((3,NBins)), np.zeros((3,NBins))
for i in range(3):
    B1H[i], Bins = np.histogram(B1[:, :, i], density=True, bins=NBins, range=(0, 255))
    B2H[i], Bins = np.histogram(B2[:, :, i], density=True, bins=NBins, range=(0, 255))

Width = Bins[1]
Bins = 0.5 * (Bins[1:] + Bins[:-1])

Figure, Axes = plt.subplots(1,1)
Axes.bar(Bins, B1H[0]-B2H[0], width=Width, color=(1, 1, 1, 0), edgecolor=(1,0,0))
Axes.bar(Bins, B1H[1]-B2H[1], width=Width, color=(1, 1, 1, 0), edgecolor=(0,1,0))
Axes.bar(Bins, B1H[2]-B2H[2], width=Width, color=(1, 1, 1, 0), edgecolor=(0,0,1))
plt.show()

[1,4,5,6,9]
[0,2,3,7,8]
i = 9
PlotImage(Parameters.SegImage[Ys[i,0]:Ys[i,1], Xs[i,0]:Xs[i,1]])
PlotImage(ROI)
