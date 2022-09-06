import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats.distributions import norm, t

desired_width = 500
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', desired_width)
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width, suppress=True, formatter={'float_kind': '{:3}'.format})
plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    This script contains diverse utility functions

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: August 2022
    """

# Time counts
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

# Signal processing
def NormalizeValues(Array):
    """
    Normalize image values, used in PCNN for easier parameters handling
    :param Image: Original grayscale image
    :return: N_Image: Image with 0,1 normalized values
    """

    NArray = (Array - Array.min()) / (Array.max() - Array.min())

    return NArray

def FFT2D(Image,CutOff,Sharpness,PassType,Plot=False):

    Tic = time.time()
    print('Perform 2D FFT filtering')

    # Filter by FFT

    FFT = np.fft.fft2(Image)
    Center = np.fft.fftshift(FFT)

    # Build filter using sigmoid function
    Half = np.array(FFT.shape) / 2
    Xc, Yc = np.meshgrid(np.arange(-Half[0], Half[0]), np.arange(-Half[1], Half[1]))
    Norm = np.sqrt(Xc ** 2 + Yc ** 2) / np.sqrt(Xc ** 2 + Yc ** 2).max()

    if PassType == 'Low':
        Filter = 1 - 1 / (1 + np.exp(-Sharpness * (Norm - CutOff)))
    elif PassType == 'High':
        Filter = 1 / (1 + np.exp(-Sharpness * (Norm - CutOff)))

    # Apply filter
    LowPassCenter = Center * Filter
    LowPass = np.fft.ifftshift(LowPassCenter)
    Filtered = np.abs(np.fft.ifft2(LowPass))

    if Plot:

        Figure, Axes = plt.subplots(1, 1)
        Axes.imshow(Image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(Figure)

        Figure, Axes = plt.subplots(1, 1)
        Axes.imshow(np.log(1+np.abs(FFT)), cmap='gray')
        plt.title('Signal FFT')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(Figure)

        Figure, Axes = plt.subplots(1, 1)
        Axes.imshow(Filter, cmap='gray')
        plt.title('Filter')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(Figure)

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

# Plotting
def PlotImage(Array):

    Figure, Axis = plt.subplots(1,1,figsize=(10,10))
    if Array.shape[-1] == 3:
        Axis.imshow(Array)
    else:
        Axis.imshow(Array, cmap='binary_r')
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.show()

def Histogram(Array,Variable=None, NBins=20, ylim=None):

    if len(Array.shape) > 1:
        Dim = Array.shape[-1]
    else:
        Dim = 1

    Colors = [(1,0,0),(0,1,0),(0,0,1),(0,1,1),(1,0,1)]

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)

    if Dim > 1:
        for i in range(Dim):
            Axes.hist(Array[:, :, i].ravel(), density=True, bins=NBins, edgecolor=Colors[i], color=(1, 1, 1, 0))
    else:
        Axes.hist(Array, density=True, bins=NBins, edgecolor=(0, 0, 1), color=(1, 1, 1, 0))

    if Variable:
        plt.xlabel(Variable)

    if ylim:
        Axes.set_ylim([0, ylim])
    plt.ylabel('Density (-)')
    plt.subplots_adjust(left=0.2)
    plt.show()

def HistoError(ArrayList,Variable=None, NBins=20, ylim=None):

    Dim = ArrayList[0].shape[-1]
    Colors = [(1,0,0),(0,1,0),(0,0,1),(0,1,1),(1,0,1)]

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)

    if Dim > 1:

        for i in range(Dim):
            Hists, Bins = np.histogram(ArrayList[0][:,:,i], density=True, bins=NBins, range=(0,255))
            Width = Bins[1]
            Bins = 0.5 * (Bins[1:] + Bins[:-1])

            for j in range(1,len(ArrayList)):
                V = ArrayList[j]
                Hists = np.vstack([Hists, np.histogram(V, density=True, bins=NBins, range=(0,255))[0]])

            Mean = np.mean(Hists,axis=0)
            SD = np.std(Hists,axis=0,ddof=1)

            Axes.bar(Bins, Mean, width=Width, color=(1,1,1,0), edgecolor=Colors[i], yerr=SD)

    else:
        Hists, Bins = np.histogram(ArrayList[0], density=True, bins=NBins, range=(0,255))
        Width = Bins[1]
        for i in range(1, len(ArrayList)):
            V = ArrayList[i]
            Hists = np.vstack([Hists, np.histogram(V, density=True, bins=NBins)[0]], range=(0,255))

        Mean = np.mean(Hists, axis=0)
        SD = np.std(Hists, axis=0, ddof=1)

        Axes.bar(Bins, Mean, width=Width, color=(1, 1, 1, 0), edgecolor=Colors[2], yerr=SD)

    if Variable:
        plt.xlabel(Variable)

    if ylim:
        Axes.set_ylim([0, ylim])
    plt.ylabel('Density (-)')
    plt.subplots_adjust(left=0.2)
    plt.show()


# Statistics
def DataDistribution(Data, Variable, NBins=20):

    # Get data attributes
    X = Data[Variable]
    SortedValues = np.sort(X.values)
    N = len(X)
    X_Bar = np.mean(X)
    S_X = np.std(X, ddof=1)

    # Kernel density estimation (Gaussian kernel)
    KernelEstimator = np.zeros(N)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
    DataIQR = np.abs(X.quantile(0.75)) - np.abs(X.quantile(0.25))
    KernelHalfWidth = 0.9 * N ** (-1 / 5) * min(np.abs([S_X, DataIQR / NormalIQR]))
    for Value in SortedValues:
        KernelEstimator += norm.pdf(SortedValues - Value, 0, KernelHalfWidth * 2)
    KernelEstimator = KernelEstimator / N

    # Histogram and density distribution
    TheoreticalDistribution = norm.pdf(SortedValues, X_Bar, S_X)
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.hist(X, density=True, bins=NBins, edgecolor=(0, 0, 1), color=(1, 1, 1), label='Histogram')
    Axes.plot(SortedValues, KernelEstimator, color=(1, 0, 0), label='Kernel Density')
    Axes.plot(SortedValues, TheoreticalDistribution, linestyle='--', color=(0, 0, 0), label='Normal Distribution')
    plt.xlabel(Variable)
    plt.ylabel('Density (-)')
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size': 10})
    plt.show()

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

# Optimization
def PlotState(Positions, Velocities, Values, Ranges):

    Best = Positions[np.tile(Values == Values.min(), 2)]

    Figure, Axes = plt.subplots(1,1, figsize=(5,5))
    Axes.plot(Positions[:,0],Positions[:,1],linestyle='none',marker='o',fillstyle='none',color=(0,0,0), label='Position')
    for i in range(len(Positions)):
        Axes.arrow(Positions[:,0][i],Positions[:,1][i],Velocities[:,0][i],Velocities[:,1][i], color=(0,0,1),
                   length_includes_head=True, head_width=0.05, head_length=0.1)
    Axes.plot([], color=(0,0,1), label='Velocity')
    Axes.plot([Ranges[0,0],Ranges[0,0]],[Ranges[1,0],Ranges[1,1]],color=(1,0,0),linestyle='--')
    Axes.plot([Ranges[0,0],Ranges[0,1]],[Ranges[1,1],Ranges[1,1]],color=(1,0,0),linestyle='--')
    Axes.plot([Ranges[0,1],Ranges[0,1]],[Ranges[1,1],Ranges[1,0]],color=(1,0,0),linestyle='--')
    Axes.plot([Ranges[0,1],Ranges[0,0]],[Ranges[1,0],Ranges[1,0]],color=(1,0,0),linestyle='--', label='Parameter space')
    Axes.plot(Best[0],Best[1], linestyle='none', marker='o', color=(0,1,0), label='Best')
    Axes.set_xlim(Ranges[0]*1.5)
    Axes.set_ylim(Ranges[1]*1.5)
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.17))
    plt.show()
def PlotEvolution(Xs, G_Bests, GBVs, Ranges):

    for i in range(Xs.shape[-1]):
        Figure, Axes = plt.subplots(1,1, figsize=(5.5,4.5))

        Axes2 = Axes.twinx()
        Axes2.plot(GBVs/GBVs.max(), color=(0,0,0), marker='o', markersize=5, linestyle='--', linewidth=1.5, label='Best Cost')
        Axes2.set_ylabel('Relative cost')
        Axes2.set_ylim([0, 1])

        for j in range(Xs.shape[-2]):
            Color = plt.cm.winter((j+1) / Xs.shape[-2])
            Axes.plot(Xs[:, j, i], marker='o', markersize=3, color=Color, linestyle='--', linewidth=1.5)
        Axes.plot(G_Bests[:, i], marker='o', color=(1,0,0), label='Best Parameter')
        Axes.plot([0, len(GBVs)-1],[Ranges[i,0],Ranges[i,0]], color=(0.7,0.7,0.7))
        Axes.plot([0, len(GBVs)-1],[Ranges[i,1],Ranges[i,1]], color=(0.7,0.7,0.7))
        Axes.set_xlabel('Iteration number')
        Axes.set_ylabel('Parameter ' + str(i) + ' value')

        Axes.legend(loc='upper left', bbox_to_anchor=(0,1.125))
        Axes2.legend(loc='upper right', bbox_to_anchor=(1,1.125))
        plt.subplots_adjust(0.2,0.15,0.85,0.9)
        plt.show()
def PSO(Arguments, Plot=False, Evolution=False):

    Tic = time.time()
    print('Run PSO algorithm')

    # Step 1: PSO initialization
    F = Arguments.Function
    Dim = len(Arguments.Ranges)
    PMin = Arguments.Ranges[:,0]
    PMax = Arguments.Ranges[:,1]
    C1, C2 = Arguments.Cs

    Amp = PMax - PMin
    X = np.random.uniform(0, 1, (Arguments.Population, Dim)) * Amp + Arguments.Ranges[:, 0]
    V = C1 * np.random.uniform(-1, 1, (Arguments.Population, Dim)) * Amp \
        + C2 * np.random.uniform(-1, 1) * Amp

    # PSO step 2 - Initial evaluation
    VInit = np.zeros((Arguments.Population, 1))
    for Particle in range(Arguments.Population):
        PArray = np.array([])

        for Parameter in range(Dim):
            PArray = np.append(PArray,X[Particle, Parameter])

        VInit[Particle,0] = F(PArray)

    # Set initial best values
    GBV = VInit.min()
    GBI = np.where(VInit == GBV)[0][0]
    G_Best = X[GBI]

    # Store values history
    if Evolution:
        Xs = np.zeros((Arguments.MaxIt+1, X.shape[0], X.shape[1]))
        G_Bests = np.zeros((Arguments.MaxIt+1, G_Best.shape[0]))
        GBVs = GBV

    PBV = VInit.copy()
    P_Best = X.copy()

    # Plot
    if Plot:
        PlotState(X, V, VInit, Arguments.Ranges)

    ## Start loop
    Iteration = 0
    while Iteration < Arguments.MaxIt and GBV > Arguments.STC:

        # Store values history
        if Evolution:
            Xs[Iteration] += X
            G_Bests[Iteration] += G_Best
            GBVs = np.append(GBVs, GBV)

        ## PSO step 3 - Update positions and velocities
        Omega = 0.9 - 0.5 * Iteration / Arguments.MaxIt  # Inertia factor
        V = Omega * V + C1 * (P_Best - X) + C2 * (G_Best - X)
        X = X + V

        # If new position exceed limits, set to limit
        X[X < PMin] = np.tile(PMin, Arguments.Population).reshape((Arguments.Population, Dim))[X < PMin]
        X[X > PMax] = np.tile(PMax, Arguments.Population).reshape((Arguments.Population, Dim))[X > PMax]

        ## PSO step 4 - Evaluation of the updated population
        VNew = np.zeros((Arguments.Population, 1))
        for Particle in range(Arguments.Population):
            PArray = np.array([])

            for Parameter in range(Dim):
                PArray = np.append(PArray, X[Particle, Parameter])

            VNew[Particle, 0] = F(PArray)

        # Update best values if better than previous
        if VNew.min() < GBV:
            GBV = VNew.min()
            GBI = np.where(VNew == GBV)[0][0]
            G_Best = X[GBI]

        ImprovedV = VNew < PBV
        PBV[ImprovedV] = VNew[ImprovedV]
        ImprovedV = np.tile(ImprovedV, Dim).reshape((Arguments.Population, Dim))
        P_Best[ImprovedV] = X[ImprovedV]

        ## PSO step 5 - Update terminal conditions
        Iteration += 1
        print('Iteration number: ' + str(Iteration))

        # Plot
        if Plot:
            PlotState(X, V, VNew, Arguments.Ranges)

    if Evolution:
        Xs[Iteration] += X
        G_Bests[Iteration] += G_Best
        GBVs = np.append(GBVs, GBV)

        Xs[Iteration + 1:] = np.nan
        G_Bests[Iteration + 1:] = np.nan
        PlotEvolution(Xs, G_Bests, GBVs[1:], Arguments.Ranges)

    # Print time elapsed
    print('\nOptimization ended')
    Toc = time.time()
    PrintTime(Tic, Toc)

    return G_Best

# Image processing
def Dice(Bin1, Bin2):

    Num = 2 * np.sum(Bin1 * Bin2)
    Den = np.sum([Bin1.sum(), Bin2.sum()])

    return Num / Den