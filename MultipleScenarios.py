# This script sets up the energy-management problem for a truck platoon
#

# Requirements:
# To run this script, you need at least the following programs installed on your machine:
# Python, IPOPT, CasADi

from casadi import *
from casadi.tools import *
import numpy as NP
from scipy import interpolate
from matplotlib.pyplot import plot, draw, show, ion
from classes import *
from EM_platoon_load_params import *

from Platoon import *
import random

###########################
#   Scenario definition   #
###########################

# Requiring solver regeneration
#       Number of trucks
#       Frequency content / Total height variation ?

# Not requiring solver regeneration
#       Driving cycles
#       Mass repartition
#       Airdrag models

Ncycles = 1
NFourrierCoeff = 30
plotresults = False

#Generate random cycles
Cycles = []
for cycle in range(Ncycles):
    Cycles.append([ [random.normalvariate(0,2) for i in range(NFourrierCoeff)],
                    [random.normalvariate(0,2) for i in range(NFourrierCoeff)] ])

#Masses: Heavy, Medium, Light
H = 4e4
M = 3e4
L = 2e4

Scenarios = [{'Ntruck'  : 3, 'Masses'  : [ H, H, H ] },
             {'Ntruck'  : 3, 'Masses'  : [ M, M, M ] },
             {'Ntruck'  : 3, 'Masses'  : [ L, L, L ] },
             
             {'Ntruck'  : 3, 'Masses'  : [ H, M, M ] },
             {'Ntruck'  : 3, 'Masses'  : [ M, H, M ] },
             {'Ntruck'  : 3, 'Masses'  : [ M, M, H ] },
             
             {'Ntruck'  : 3, 'Masses'  : [ H, M, L ] },
             {'Ntruck'  : 3, 'Masses'  : [ L, M, H ] },
             {'Ntruck'  : 3, 'Masses'  : [ H, L, M ] },
             ]

for Mass in [L,M,H]:
    for Ntruck in range(4,11):
        Scenarios.append(
                        {'Ntruck'  : Ntruck, 'Masses'  : [Mass for i in range(Ntruck)] }
                         )


Scenarios = [{'Ntruck'  : 3, 'Masses'  : [ H, H, H ] }]

ScenariosOut = []
counter = 0
for scenario in Scenarios:
    print scenario
    
    #Hugly hack...
    if counter == 0:
        print 'Regenerate solver'
        MyPlatoon = SetPlatoon(Ntruck = scenario['Ntruck'], NFourrierCoeff = NFourrierCoeff)
    else:
        if not(scenario['Ntruck'] == Scenarios[counter-1]['Ntruck']):
            print 'Regenerate solver'
            MyPlatoon = SetPlatoon(Ntruck = scenario['Ntruck'], NFourrierCoeff = NFourrierCoeff)

    counter += 1

    ########################################################
    #  Create cost function (this is a unique declaration) #
    ########################################################

    Torque            = np.array(MyPlatoon.Vtruck['Input',:,'TdE'])
    c1                = MyPlatoon.Ptruck['Parameter','c1']
    c2                = MyPlatoon.Ptruck['Parameter','c2']
    fobj              = 1e-6*MyPlatoon.Vtruck['Tf']*(sum(c1*Torque + c2))/float(MyPlatoon.nk)
    MyPlatoon.ObjFunc = SXFunction('ObjFunc',[MyPlatoon.Vtruck,MyPlatoon.Ptruck],[fobj])


    MyPlatoon.SetSolvers()

    Parameters = MyPlatoon.parameters()


    # SIMULATE
    AllGain         = []
    AllProfiles     = []
    AllCosts        = []
    AllSlopeL1Norms = []
    AllSlopeL2Norms = []
    for cycle in range(Ncycles):
    
        MyPlatoon.ProblemParameters['Sin'] = Cycles[cycle][0]
        MyPlatoon.ProblemParameters['Cos'] = Cycles[cycle][1]
    
        MyPlatoon.Optimize()
    
        AllGain.append(float(100*(MyPlatoon.SolversOut['Cost'][0]-MyPlatoon.SolversOut['Cost'][1])/MyPlatoon.SolversOut['Cost'][0]))
        AllCosts.append(MyPlatoon.SolversOut['Cost'])
    
        for label in ['Sin','Cos','Distance']:
            Parameters[label] = MyPlatoon.ProblemParameters[label]
    
        HeightPlot = []
        SlopePlot    = []
        for k in range(MyPlatoon.nk):
            pos = Parameters['Distance']*k/float(MyPlatoon.nk)
            [Height_d,Slope_d] = MyPlatoon.HeightFunc([Parameters,pos])
            HeightPlot.append(float(Height_d))
            SlopePlot.append(100*float(Slope_d))
    
        AllProfiles.append({'Slope': SlopePlot, 'Height': HeightPlot})
        AllSlopeL1Norms.append(np.sum(np.abs(AllProfiles[cycle]['Slope'])))
        AllSlopeL2Norms.append(np.sum(np.array(AllProfiles[cycle]['Slope'])**2))

#assert(counter < 2)
    ScenariosOut.append({'Gains':AllGain, 'Costs':AllCosts, 'Profiles':AllProfiles, 'ProfilesNorms':[AllSlopeL1Norms]})
    print 'Resolution ', MyPlatoon.resolution, 's'
    print 'Cost gain', AllGain ,'%'
    print 'Slopes norm', AllSlopeL1Norms

    ####################################################################################

    ###########################     PLOT PLOT PLOT      ################################

    ####################################################################################

    if plotresults:
        MyPlatoon.HazPlot()
        
        plt.figure(10)
        plt.hold('on')
        plt.plot(AllSlopeL1Norms,AllGain, linestyle = 'none', marker = '.', color = 'r')
        plt.hold('on')
        plt.plot(AllSlopeL2Norms,AllGain, linestyle = 'none', marker = '.', color = 'r')

        for cycle in range(Ncycles):
            plt.figure(7)
            plt.subplot(1,2,1)
            plt.hold('on')
            plt.plot([1e-3*Parameters['Distance']*i/MyPlatoon.nk for i in range(MyPlatoon.nk)],AllProfiles[cycle]['Slope'])
            plt.ylabel('Slope in %')
            plt.xlabel('Dist. in km')
            plt.subplot(1,2,2)
            plt.hold('on')
            plt.plot([1e-3*Parameters['Distance']*i/MyPlatoon.nk for i in range(MyPlatoon.nk)],AllProfiles[cycle]['Height'])
            plt.ylabel('Height')
            plt.xlabel('Dist. in km')


        plt.show()
