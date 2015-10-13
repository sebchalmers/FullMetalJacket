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

# TODO: free position of the followers, force the separation in the beginning and the end to match

#
# Load parameters for EM problem (cleaner code to separate it out from this script)
# 
from EM_platoon_load_params import *


# Optimization parameters
Ntruck = 2                                              # Number of trucks

Distance           = 30e3                               # Distance travelled by the trucks
MaxTime            = 1.2*Distance/(120/3.6)             # Time allowed for travelling the cycle
MinSeparation      = 10.                                 # Minimum sepearation between the trucks

Vel0               = 100.                               # Velocity at time 0
Separation0        = 10.5                               # Separation at time 0

MinVel             = 80.                                # Minimum velocity in km/h
MaxVel             = 120.                               # Maximum velocity in km/h
MaxTorques         = 5e3

Coeff = {'sin':[-1, 2, 5, -2, 3, -2, -2, 3], 'cos':[2, 1, -1, 2, -3, 5, 2, 4]}

SolversOut = {'Cost':[], 'Status':[]}

# Parameters
rw    = np.array(5*list(rw))
r_t   = np.array(5*list(r_t))
CdA   = np.array(5*list(CdA))
a0    = np.array(5*list(a0))
cr    = np.array(5*list(cr))
c1    = np.array(5*list(c1))
c2    = np.array(5*list(c2))
m_eff = np.array(5*list(m_eff))

#m_eff /= 2

#OBS: truck parameters are defined via this dictionary only !
ParamValue = {'c1':c1,'c2':c2,'cr': cr, 'rw': rw, 'r_t': r_t, 'CdA': CdA, 'm': m_eff, 'a0':a0, 'DragRedSlope': [0., -0.45, -0.48, -0.48, -0.48, -0.48, -0.48, -0.48], 'DragRedOffset': [0., 43, 52, 52, 52, 52, 52, 52]}

# Numerical parameters
nk       = np.int(np.ceil(MaxTime/2.))  #2 sec resolution
nstep    = 2
DicIpopt = {'max_iter':3000,'tol':1e-6}

# Plot parameters
ScalePlot = {'pos':1e-3,'vel':3.6}

#
# Create basic variables
#

states = struct_symSX([
                       entry("pos"),
                       entry("vel")
    ])

inputs = struct_symSX([
                       entry("TdE"),
                       entry("Tdbrk")
    ])

parameters = struct_symSX([ entry(i) for i in ParamValue.keys()])

#
# Truck variables
#


AllPos_struct =  struct_symSX([
                               entry('pos', repeat = nstep)
                               ])


Vtruck = struct_symSX([
                       entry('State',     struct=states,      repeat=nk+1),
                       entry('Input',     struct=inputs,      repeat=nk),
                       entry('Tf')
                       ])

Ptruck = struct_symSX([
                       entry('Parameter',         struct=parameters ),
                       entry('PosLeader',         struct = AllPos_struct, repeat=nk+1),  #OBS: used only in greedy optimization, disregarded in full platoon
                       entry('HazLeader')                                                #OBS: used only in greedy optimization, disregarded in full platoon
                       ])


#
# Obtain aliases with the Ellipsis index:
#
pos, vel    = states[...]
TdE, Tdbrk  = inputs[...]

PosLeader   = SX.sym('PosLeader')
HazLeader   = SX.sym('leader')

###################################################
#  Create dynamics (this is a unique declaration) #
###################################################
CdModifLeader   = 1 - HazLeader*( parameters['DragRedSlope']*( PosLeader - pos ) + parameters['DragRedOffset'] )/100.

Faero = 0.5 * env.airdens * parameters['CdA'] * CdModifLeader * vel**2                             # Aerodyn force
Td    = TdE - Tdbrk                                                                                # Torque at gearbox output

SinSlope = 0                                                                                       # Build Fourrier series for the slope
for k in range(len(Coeff['sin'])):
    SinSlope += ( -(2*pi*k/Distance)*Coeff['cos'][k]*sin(2*pi*k*pos/Distance) + (2*pi*k/Distance)*Coeff['sin'][k]*cos(2*pi*k*pos/Distance)  )

Fg      = 9.81*parameters['m']*SinSlope                                                               # Force due to slopes
Froll   = parameters['cr']*parameters['m']*9.81*(1 - SinSlope**2)                                     # Force from roll
Facc    = Td * parameters['r_t']/parameters['rw'] - Faero - Fg - Froll                                # Acceleration force
dvel    = Facc/parameters['m']                                                                        # Velocity change

rhs = struct_SX([
    entry("pos", expr = vel),
    entry("vel", expr = dvel)
])


## Build the height for plotting
Height = 0                                                                                         # Build Fourrier series for the height
for k in range(len(Coeff['sin'])):
    Height   += (                    Coeff['cos'][k]*cos(2*pi*k*pos/Distance)                   + Coeff['sin'][k]*sin(2*pi*k*pos/Distance)  )


HeightFunc = SXFunction('height',[states],[Height,SinSlope])

states_num = states()
HeightPlot = []
SlopePlot    = []
for k in range(nk):
    states_num['pos'] = k*Distance/nk
    [Height_d,Slope_d] = HeightFunc([states_num])
    HeightPlot.append(Height_d)
    SlopePlot.append(100*Slope_d)

plt.figure(7)
plt.subplot(1,2,1)
plt.plot([1e-3*i*Distance/nk for i in range(nk)],SlopePlot)
plt.ylabel('Slope in %')
plt.xlabel('Dist. in km')
plt.subplot(1,2,2)
plt.plot([1e-3*i*Distance/nk for i in range(nk)],HeightPlot)
plt.ylabel('Height')
plt.xlabel('Dist. in km')

#plt.show()
#assert(0==1)
#########################################################
#  Create the integrator (this is a unique declaration) #
#########################################################
Tf = SX.sym('Tf')
fode = SXFunction('ode',[states,inputs,parameters,PosLeader,HazLeader],[rhs])

dt    = 1/float(nk)

k1         = states
AllPosExpr = []
for k in range(nstep):
    AllPosExpr.append(states(k1)['pos'])
    [f]  = fode([k1 , inputs, parameters, AllPos_struct['pos'][k], HazLeader])
    k1   = k1 + Tf*dt*f/float(nstep)

AllPosExpr = struct_SX([
                        entry('pos', expr = AllPosExpr)
                        ])

euler_struct = struct_SX([
                        entry('final',              expr = k1),
                        entry('intermediate_pos',   expr = AllPosExpr)
                       ])

euler = SXFunction('euler',[states, inputs, parameters, AllPos_struct, HazLeader, Tf],[euler_struct])



########################################################
#  Create cost function (this is a unique declaration) #
########################################################
#
Torque  = np.array(Vtruck['Input',:,'TdE'])
c1      = Ptruck['Parameter','c1']
c2      = Ptruck['Parameter','c2']
fobj    =  1e-6*Vtruck['Tf']*(sum(c1*Torque + c2))/float(nk)
ObjFunc = SXFunction('ObjFunc',[Vtruck,Ptruck],[fobj])


####################################################################################

###########################      BUILD SOLVERS      ################################

####################################################################################

#####################
#                   #
#   TRUCK SOLVER    #
#                   #
#####################

#
# Create shooting constraints for truck
#
shooting = []
for time in range(nk):
    state_truck     = Vtruck['State',    time]
    input_truck     = Vtruck['Input',    time]
    param_truck     = Ptruck['Parameter'     ]
    Tf              = Vtruck['Tf'            ]
    PosLeader_truck = Ptruck['PosLeader',time]
    HazLeader_truck = Ptruck['HazLeader'     ]

    [shoot]  = euler([state_truck, input_truck, param_truck, PosLeader_truck, HazLeader_truck, Tf])
    shoot    = euler_struct(shoot)

    # append continuity constraints
    shooting.append(Vtruck['State',time+1] - shoot['final']) # The state evolution gets connected through the constraint shooting

[fobj_truck] = ObjFunc([Vtruck,Ptruck])
g_truck = struct_SX([
                        entry('shooting',         expr = shooting),
                     ])

g_truck_func = SXFunction('g_truck',[Vtruck, Ptruck],[g_truck]) #OBS: for debugging purposes

nlp=SXFunction("nlp", nlpIn(x=Vtruck, p=Ptruck),nlpOut(f=fobj_truck, g = g_truck))
solver_truck = NlpSolver("solver", "ipopt", nlp,DicIpopt)


#######################
#                     #
#   PLATOON SOLVER    #
#                     #
#######################

#
# Platoon variables
#

## Platoon structure: Vplatoon['Truck', truck number ,'State'/'Input'/'Parameter', time , state label]
Vplatoon = struct_symSX([ entry('Truck', struct=Vtruck, repeat=Ntruck) ])
Pplatoon = struct_symSX([ entry('Truck', struct=Ptruck, repeat=Ntruck) ])


#
# Create shooting constraints for platooning
#
shooting = []
ordering_const = []
fobj     = 0

# Multiple shooting - Platoon
for time in range(nk):
    AllPos = AllPos_struct(0)
    
    for truck in range(Ntruck):
        state_truck = Vplatoon['Truck',truck,'State',    time]
        input_truck = Vplatoon['Truck',truck,'Input',    time]
        param_truck = Pplatoon['Truck',truck,'Parameter'     ]
        Tf          = Vplatoon['Truck',truck,'Tf'            ]
        
        if truck == 0:
            HazLeader = 0.
        else:
            HazLeader = 1.
        
        [shoot]  = euler([state_truck, input_truck, param_truck, AllPos, HazLeader, Tf])
        shoot    = euler_struct(shoot)
        AllPos   = shoot['intermediate_pos']

        # append continuity constraints
        shooting.append(Vplatoon['Truck',truck,'State',time+1] - shoot['final']) # The state evolution gets connected through the constraint shooting

#
# Create cost for platooning
#
fobj = 0
for truck in range(Ntruck):
    [fobj_truck] = ObjFunc([Vplatoon['Truck',truck],Pplatoon['Truck',truck]])
    fobj += fobj_truck

#
# Create ordering constraints for platooning
#
for truck in range(Ntruck-1):
  for time in range(nk):
      ordering_const.append(Vplatoon['Truck',truck+1,'State',time,'pos'] - Vplatoon['Truck',truck,'State',time,'pos'])

#
# Match final times
#
TfConst = []
for truck in range(Ntruck-1):
    TfConst.append(Vplatoon['Truck',truck+1,'Tf'] - Vplatoon['Truck',truck,'Tf'])

g_platoon = struct_SX([
                        entry('shooting',         expr = shooting),
                        entry('ordering',         expr = ordering_const),
                        entry('final_times',      expr = TfConst)
                     ])

nlp=SXFunction("nlp", nlpIn(x=Vplatoon, p=Pplatoon),nlpOut(f=fobj, g = g_platoon))
solver_platoon = NlpSolver("solver", "ipopt", nlp)

####################################################################################

###########################     NUMERICAL PART      ################################

####################################################################################

############################
#                          #
#   GREEDY OPTIMIZATION    #
#                          #
############################

vel_guess = 100/3.6

truck_lb   = Vtruck(-inf)
truck_ub   = Vtruck( inf)
truck_init = Vtruck()

truck_lb['Input']             = 0.
truck_ub['Input',:,'TdE']     = MaxTorques
truck_lb['State',:,'vel']     = MinVel/3.6
truck_ub['State',:,'vel']     = MaxVel/3.6
truck_lb['Tf']                = 0.1
truck_ub['Tf']                = MaxTime

truck_init['State',:,'vel']   = vel_guess

truck_init['Tf']              = MaxTime

truck_lbg = g_truck()
truck_ubg = g_truck()
Ptruck_num = Ptruck()

Sol_greedy = Vplatoon()

SolversOut['Cost'].append(0)

for truck in range(Ntruck):
    
    #Truck-specific stuff
    truck_init['State',:,'pos'] = [vel_guess*i*MaxTime*dt-Separation0*truck for i in range(nk+1)]
    
    truck_lb['State',0,'pos'] = -Separation0*truck
    truck_ub['State',0,'pos'] = -Separation0*truck
    truck_lb['State',0,'vel'] =  Vel0/3.6
    truck_ub['State',0,'vel'] =  Vel0/3.6
    
    truck_lb['State',-1,'pos']  = Distance - Separation0*truck
    
    if truck == 0:
        Ptruck_num['HazLeader'] = 0.
        Ptruck_num['PosLeader'] = truck_init['State',:,'pos']
    else:
        Ptruck_num['HazLeader'] = 1.
        #Ordering constraints as bounds
        for time in range(1,nk+1):
            truck_ub['State',time,'pos'] = Sol_greedy['Truck',truck-1,'State',time,'pos'] - MinSeparation

        #Next truck inherits final time
        truck_lb['Tf'] = Sol_greedy['Truck',truck-1,'Tf']
        truck_ub['Tf'] = Sol_greedy['Truck',truck-1,'Tf']

    for label in ParamValue.keys():
        print label,ParamValue[label]
        Ptruck_num['Parameter',label] = ParamValue[label][truck]

    solver_truck.setInput(truck_lb,"lbx")
    solver_truck.setInput(truck_ub,"ubx")
    solver_truck.setInput(Ptruck_num,"p")
    solver_truck.setInput(truck_lbg,"lbg")
    solver_truck.setInput(truck_ubg,"ubg")
    solver_truck.setInput(truck_init,"x0")
    solver_truck.evaluate()

    
    Sol_truck = Vtruck(solver_truck.getOutput())


    #Extract intermediate positions of the current truck -> leader of the next truck
    for time in range(nk):
        [shoot] = euler([Sol_truck['State',time],Sol_truck['Input',time],Ptruck_num['Parameter'],Ptruck_num['PosLeader',time],Ptruck_num['HazLeader'],Sol_truck['Tf']])
        shoot    = euler_struct(shoot)
        Ptruck_num['PosLeader',time] = shoot['intermediate_pos']

    Sol_greedy['Truck',truck] = Sol_truck.cat

    SolversOut['Cost'][0] += solver_truck.getOutput('f')


plt.figure(1)
for index_sub, label_sub in enumerate(inputs.keys()):
    plt.subplot(1,2,index_sub+1)
    for truck in range(Ntruck):
        plt.hold('on')
        plt.step(Sol_greedy['Truck',truck,'Tf']*range(nk)/float(nk),Sol_greedy['Truck',truck,'Input',:,label_sub])
        plt.ylabel(label_sub)
plt.figure(2)
for index_sub, label_sub in enumerate(states.keys()):
    plt.subplot(1,2,index_sub+1)
    for truck in range(Ntruck):
        plt.hold('on')
        plt.plot(Sol_greedy['Truck',truck,'Tf']*range(nk+1)/float(nk),ScalePlot[label_sub]*np.array(Sol_greedy['Truck',truck,'State',:,label_sub]))
        plt.ylabel(label_sub)

plt.figure(3)
plt.hold('on')
for truck in range(1,Ntruck):
    plt.plot(Sol_greedy['Truck',truck,'Tf']*range(nk+1)/float(nk),np.array(Sol_greedy['Truck',truck,'State',:,'pos'])-np.array(Sol_greedy['Truck',truck-1,'State',:,'pos']))
plt.ylabel('Separations')
#plt.show()
#assert(0==1)


##############################
#                            #
#   HOLISTIC  OPTIMIZATION   #
#                            #
##############################

#
# Define explicit bounds on variables
#
platoon_lb   = Vplatoon(-inf)
platoon_ub   = Vplatoon( inf)
platoon_init = Vplatoon()

platoon_lb['Truck',:]   = truck_lb
platoon_ub['Truck',:]   = truck_ub
platoon_init['Truck',:] = truck_init

platoon_ub['Truck',:,'State',:,'pos'] = inf

for truck in range(Ntruck):
    platoon_init['Truck',truck,'State',:,'pos'] = [vel_guess*i*MaxTime*dt-Separation0*truck for i in range(nk+1)]

# Set states and its bounds
platoon_init['Truck',:,'Tf'] = truck_init['Tf']

lbg = g_platoon()
ubg = g_platoon()

lbg['ordering'] =  -inf
ubg['ordering'] =  -MinSeparation


Pplatoon_num = Pplatoon()
for label in ParamValue.keys():
    print label,ParamValue[label]
    for truck in range(Ntruck):
        Pplatoon_num['Truck',truck,'Parameter',label] = ParamValue[label][truck]

for truck in range(Ntruck):
    platoon_lb['Truck',truck,'State',0,'pos']  = -Separation0*truck
    platoon_ub['Truck',truck,'State',0,'pos']  = -Separation0*truck
    platoon_lb['Truck',truck,'State',0,'vel']  =    Vel0/3.6
    platoon_ub['Truck',truck,'State',0,'vel']  =    Vel0/3.6
    platoon_lb['Truck',truck,'State',-1,'pos'] = Distance - Separation0*truck


solver_platoon.setInput(platoon_lb,"lbx")
solver_platoon.setInput(platoon_ub,"ubx")
solver_platoon.setInput(Pplatoon_num,"p")
solver_platoon.setInput(lbg,"lbg")
solver_platoon.setInput(ubg,"ubg")
solver_platoon.setInput(platoon_init,"x0")


solver_platoon.evaluate()

Sol_platoon = Vplatoon(solver_platoon.getOutput())

SolversOut['Cost'].append(solver_platoon.getOutput('f'))

####################################################################################

###########################     PLOT PLOT PLOT      ################################

####################################################################################

print 'Cost gain', 100*(SolversOut['Cost'][0]-SolversOut['Cost'][1])/SolversOut['Cost'][0],'%'

plt.figure(4)
for index_sub, label_sub in enumerate(inputs.keys()):
    plt.subplot(1,2,index_sub+1)
    for truck in range(Ntruck):
        plt.hold('on')
        plt.step(Sol_platoon['Truck',truck,'Tf']*range(nk)/float(nk),Sol_platoon['Truck',truck,'Input',:,label_sub])
        plt.ylabel(label_sub)
plt.figure(5)
for index_sub, label_sub in enumerate(states.keys()):
    plt.subplot(1,2,index_sub+1)
    for truck in range(Ntruck):
        plt.hold('on')
        plt.plot(Sol_platoon['Truck',truck,'Tf']*range(nk+1)/float(nk),ScalePlot[label_sub]*np.array(Sol_platoon['Truck',truck,'State',:,label_sub]))
        plt.ylabel(label_sub)
plt.figure(6)
plt.hold('on')
for truck in range(1,Ntruck):
    plt.plot(Sol_platoon['Truck',truck,'Tf']*range(nk+1)/float(nk),np.array(Sol_platoon['Truck',truck,'State',:,'pos'])-np.array(Sol_platoon['Truck',truck-1,'State',:,'pos']))
plt.ylabel('Separations')

plt.show()
