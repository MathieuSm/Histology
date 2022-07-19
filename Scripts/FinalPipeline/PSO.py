#!/usr/bin/env python3

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=12)

Version = '01'

# Define the script description
Description = """
    This script runs the a Particle-Swarm-Optimization algorithm optimizing parameters to minimize a given function

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: July 2022
    """

# For testing purposes
class Arguments:

    def __init__(self):
        Arguments.Ranges = np.array([[-1, 1], [-1, 1]])
        Arguments.Population = 20
        Arguments.Cs = [0.1, 0.1]
        Arguments.MaxIt = 10
        Arguments.STC = 1E-3

    def Function(self, Parameters):
        """
        Function to find center of a coordinate system (test)
        :param self:
        :return: Euclidian norm
        """
        P1, P2 = Parameters
        return P1**2 + P2**2
Arguments = Arguments()


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

def Main(Arguments, Plot=False, Evolution=False):

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

if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add arguments
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Required arguments
    Parser.add_argument('Function', help='Function to minimize, input must be an np array (required)')
    Parser.add_argument('Ranges', help='Parameter ranges (required)', type=np.array)


    # Optional arguments
    Ps = 20
    Max = 10
    Cs = [2, 2]
    STC = 0.
    Parser.add_argument('-P','--Population', help='Population size', type=int, default=Ps)
    Parser.add_argument('-M','--MaxIt', help='Maximum number of iterations', type=int, default=Max)
    Parser.add_argument('-Cs', help='C1 and C2 coefficients for particles velocities', type=list, default=Cs)
    Parser.add_argument('-STC', help='Second PSO termination condition', type=float, default=STC)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments)
