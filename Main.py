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

MyPlatoon = SetPlatoon(Ntruck = 3, NFourrierCoeff = 8)

########################################################
#  Create cost function (this is a unique declaration) #
########################################################
#
Torque            = np.array(MyPlatoon.Vtruck['Input',:,'TdE'])
c1                = MyPlatoon.Ptruck['Parameter','c1']
c2                = MyPlatoon.Ptruck['Parameter','c2']
fobj              = 1e-6*MyPlatoon.Vtruck['Tf']*(sum(c1*Torque + c2))/float(MyPlatoon.nk)
MyPlatoon.ObjFunc = SXFunction('ObjFunc',[MyPlatoon.Vtruck,MyPlatoon.Ptruck],[fobj])


MyPlatoon.SetSolvers()

#################################
#  Run the optimization problem #
#################################


MyPlatoon.ProblemParameters['Sin'] = [-1, 2,  5, -2,  3, -2, -2, 3]
MyPlatoon.ProblemParameters['Cos'] = [ 2, 1, -1,  2, -3,  5,  2, 4]

MyPlatoon.Optimize()

MyPlatoon.HazPlot()

####################################################################################

###########################     PLOT PLOT PLOT      ################################

####################################################################################

Parameters = MyPlatoon.parameters()
for label in ['Sin','Cos','Distance']:
    Parameters[label] = MyPlatoon.ProblemParameters[label]


HeightPlot = []
SlopePlot    = []
for k in range(MyPlatoon.nk):
    pos = Parameters['Distance']*k/float(MyPlatoon.nk)
    [Height_d,Slope_d] = MyPlatoon.HeightFunc([Parameters,pos])
    HeightPlot.append(Height_d)
    SlopePlot.append(100*Slope_d)

plt.figure(7)
plt.subplot(1,2,1)
plt.plot([1e-3*Parameters['Distance']*i/MyPlatoon.nk for i in range(MyPlatoon.nk)],SlopePlot)
plt.ylabel('Slope in %')
plt.xlabel('Dist. in km')
plt.subplot(1,2,2)
plt.plot([1e-3*Parameters['Distance']*i/MyPlatoon.nk for i in range(MyPlatoon.nk)],HeightPlot)
plt.ylabel('Height')
plt.xlabel('Dist. in km')

print 'Cost gain', 100*(MyPlatoon.SolversOut['Cost'][0]-MyPlatoon.SolversOut['Cost'][1])/MyPlatoon.SolversOut['Cost'][0],'%'
plt.show()
